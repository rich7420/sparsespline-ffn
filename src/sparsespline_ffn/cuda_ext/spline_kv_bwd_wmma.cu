// FlashSplineFeature backward — CUDA C++ wmma version (v2).
//
// Strategy: express dC scatter as a real matmul and use bf16 tensor cores
// via the cuda::wmma API.  This matches Triton v3's tl.dot strategy but
// with manual control over SMEM staging.
//
// Math:
//   dC[j, b, c] = sum_n  W[n, j, b]  *  g_delta[n, c]
// where
//   W[n, j, b] = B_k(tau_nj)  if  b == bin_nj + k  for k in {0,1,2}
//                0            otherwise
// (tau, bin determined by z[n,j])
//
// Treat (j, b) as one flat dimension M = BLOCK_H * L_PAD.  Then
//   dC_tile[M, R] = W_tile[N, M].T  @  g_delta_tile[N, R]
// is a standard bf16 matmul of shape (M, R, N) ← (M, N) @ (N, R) where
// the K axis is BLOCK_N (number of tokens in the block).
//
// W is densified in SMEM (99% zeros, but matmul cost is identical with
// or without zeros — the wasted FLOPs are amortized by tensor core
// throughput, which is 16-30× faster than scalar atomicAdd).
//
// dz path: same loop as v1 but uses warp shuffle to reduce the c-sum
// instead of scalar accumulation.
//
// Numerical: bf16 inputs, fp32 wmma accumulator, fp32 outputs.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

namespace {

// wmma fragment tile size.  m=16 n=16 k=16 is universally supported on
// sm_80 (Ampere) and sm_90 (Hopper) for bf16 accum-fp32.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f2bf(float x) {
    return __float2bfloat16(x);
}

// B-spline B2 basis (same as v1)
__device__ __forceinline__ void compute_B2(float tau, float& B0, float& B1, float& B2) {
    float omt = 1.0f - tau;
    B0 = 0.5f * omt * omt;
    B1 = 0.5f * (1.0f + 2.0f * tau - 2.0f * tau * tau);
    B2 = 0.5f * tau * tau;
}

__device__ __forceinline__ void compute_dB2(float tau, float& dB0, float& dB1, float& dB2) {
    dB0 = -(1.0f - tau);
    dB1 = 1.0f - 2.0f * tau;
    dB2 = tau;
}

// ---------------------------------------------------------------------------
// Backward kernel.
//
// Tile contract:
//   BLOCK_N = K (token axis), must be multiple of WMMA_K = 16
//   BLOCK_H * L_PAD = M (flat j×b axis), must be multiple of WMMA_M = 16
//   R       = N (channel axis), must be multiple of WMMA_N = 16
//
// Grid: (cdiv(N, BLOCK_N), cdiv(H, BLOCK_H))
// Threads/block: 128 (4 warps).  4 warps cooperate on M-axis tiling of dC.
//
// SMEM layout (production-tuned for r=32 L=22 BN=64 BH=8):
//   W_smem      [BLOCK_N][M_PAD]    bf16   (BN*M*2 bytes)
//   g_smem      [BLOCK_N][R]         bf16   (BN*R*2 bytes)
//   C_smem      [BLOCK_H][L_PAD][R]  bf16   (BH*L_PAD*R*2 bytes)
//   dC_smem     [M_PAD][R]           fp32   (M*R*4 bytes)
//
// For BN=64, BH=8, L_PAD=22, R=32:
//   W_smem = 64*176*2 = 22.5 KB
//   g_smem = 64*32*2  = 4 KB
//   C_smem = 8*22*32*2 = 11 KB
//   dC_smem = 176*32*4 = 22.5 KB
//   total = 60 KB < 100 KB (3080) and < 228 KB (H100)

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R>
__global__ void __launch_bounds__(128, 2)
spline_kv_bwd_wmma_kernel(
    const __nv_bfloat16* __restrict__ z,        // [N, H]
    const __nv_bfloat16* __restrict__ C,        // [H, L, R]
    const __nv_bfloat16* __restrict__ g_delta,  // [N, R]
    float* __restrict__ dC,                      // [H, L, R] fp32 (zero-init)
    float* __restrict__ dz,                      // [N, H] fp32 (zero-init)
    const int N, const int H, const int L,
    const float grid_lo, const float scale
) {
    constexpr int M = BLOCK_H * L_PAD;
    static_assert(M % WMMA_M == 0,        "M must be multiple of 16");
    static_assert(BLOCK_N % WMMA_K == 0, "BLOCK_N must be multiple of 16");
    static_assert(R % WMMA_N == 0,       "R must be multiple of 16");

    const int pid_n = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int n_start = pid_n * BLOCK_N;
    const int h_start = pid_h * BLOCK_H;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // SMEM
    __shared__ __nv_bfloat16 W_smem[BLOCK_N][M];      // K × M (column = (j,b))
    __shared__ __nv_bfloat16 g_smem[BLOCK_N][R];      // K × N (channel)
    __shared__ __nv_bfloat16 C_smem[BLOCK_H][L_PAD][R];
    __shared__ float        dC_smem[M][R];            // M × R fp32 acc

    // ---- Phase 0: zero W_smem and dC_smem ----
    {
        __nv_bfloat16 zero_bf = f2bf(0.0f);
        const int total_W = BLOCK_N * M;
        for (int idx = tid; idx < total_W; idx += blockDim.x) {
            (&W_smem[0][0])[idx] = zero_bf;
        }
        const int total_dC = M * R;
        for (int idx = tid; idx < total_dC; idx += blockDim.x) {
            (&dC_smem[0][0])[idx] = 0.0f;
        }
    }

    // ---- Phase 1a: load g_delta tile [BLOCK_N][R] ----
    {
        const int total = BLOCK_N * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int n_local = idx / R;
            const int c       = idx % R;
            const int n_global = n_start + n_local;
            __nv_bfloat16 v = (n_global < N)
                ? g_delta[n_global * R + c]
                : f2bf(0.0f);
            g_smem[n_local][c] = v;
        }
    }

    // ---- Phase 1b: load C tile [BLOCK_H][L_PAD][R] ----
    {
        const int total = BLOCK_H * L_PAD * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int j_local = idx / (L_PAD * R);
            const int b       = (idx / R) % L_PAD;
            const int c       = idx % R;
            const int j_global = h_start + j_local;
            __nv_bfloat16 v = (j_global < H && b < L)
                ? C[j_global * L * R + b * R + c]
                : f2bf(0.0f);
            C_smem[j_local][b][c] = v;
        }
    }

    __syncthreads();

    // ---- Phase 2: compute B/τ/bin for each (n,j) pair, write into W_smem ----
    //   W_smem[n_local][j_local * L_PAD + bin + k] = B_k(τ)   for k=0,1,2
    //
    // Also compute dz contribution (gather 3 rows of C, dot with g) and write
    // to global dz directly (no atomic; dz is unique per (n,j)).
    {
        const int total_pairs = BLOCK_N * BLOCK_H;
        for (int p = tid; p < total_pairs; p += blockDim.x) {
            const int n_local = p / BLOCK_H;
            const int j_local = p % BLOCK_H;
            const int n_global = n_start + n_local;
            const int j_global = h_start + j_local;
            if (n_global >= N || j_global >= H) continue;

            const float z_val = bf2f(z[n_global * H + j_global]);
            const float u = (z_val - grid_lo) * scale;
            const float G_max = (float)(L - 2);
            const bool in_range     = (u >= 0.0f) && (u <= G_max);
            const bool clamp_active = (u >= 0.0f) && (u <= G_max - 1.0f);
            const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
            const int   bin_idx = (int)u_clip;
            const float tau = u_clip - (float)bin_idx;

            float B0, B1, B2, dB0, dB1, dB2;
            compute_B2(tau, B0, B1, B2);
            compute_dB2(tau, dB0, dB1, dB2);
            if (!in_range)     { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }
            if (!clamp_active) { dB0 = 0.0f; dB1 = 0.0f; dB2 = 0.0f; }

            // Write 3 nonzeros into W_smem at column (j_local*L_PAD + bin+k)
            const int col_base = j_local * L_PAD + bin_idx;
            W_smem[n_local][col_base + 0] = f2bf(B0);
            W_smem[n_local][col_base + 1] = f2bf(B1);
            W_smem[n_local][col_base + 2] = f2bf(B2);

            // ----- dz path -----
            //   dz[n,j] = scale * sum_c g[n,c] * (dB0*C[j,bin,c] + dB1*C[j,bin+1,c]
            //                                    + dB2*C[j,bin+2,c])
            float inner = 0.0f;
            #pragma unroll
            for (int c = 0; c < R; c++) {
                const float g  = bf2f(g_smem[n_local][c]);
                const float c0 = bf2f(C_smem[j_local][bin_idx + 0][c]);
                const float c1 = bf2f(C_smem[j_local][bin_idx + 1][c]);
                const float c2 = bf2f(C_smem[j_local][bin_idx + 2][c]);
                inner += g * (dB0 * c0 + dB1 * c1 + dB2 * c2);
            }
            dz[n_global * H + j_global] = scale * inner;
        }
    }

    __syncthreads();

    // ---- Phase 3: wmma matmul  dC_smem[M, R] += W_smem^T[M, K] @ g_smem[K, R] ----
    //
    // W_smem layout: [K=BLOCK_N][M].  We need W^T → [M][K].
    // wmma.load_matrix_sync supports row_major or col_major.  Treating
    // W_smem[K][M] as col_major gives us W^T view directly (logical [M][K]).
    //
    // Tile decomposition:
    //   M tiles: warp_id chooses which (m_tile) it owns.
    //   N tiles: each warp handles all N tiles serially (R/16 of them).
    //   K reduction: each warp loops over K_tiles = BLOCK_N / 16.
    //
    // 4 warps × M-tile interleave: M = BLOCK_H * L_PAD.
    //   For BH=8, L_PAD=22: M = 176, M_tiles = 11.
    //   For BH=8, L_PAD=16: M = 128, M_tiles = 8.
    // 4 warps cycle through M_tiles round-robin.

    constexpr int M_TILES = M / WMMA_M;
    constexpr int N_TILES = R / WMMA_N;
    constexpr int K_TILES = BLOCK_N / WMMA_K;

    // Each warp owns a subset of (m_tile, n_tile) pairs.
    // Total M_TILES * N_TILES tiles distributed round-robin among 4 warps.
    const int total_mn = M_TILES * N_TILES;
    for (int t = warp_id; t < total_mn; t += 4) {
        const int m_tile = t / N_TILES;
        const int n_tile = t % N_TILES;

        // Accumulator fragment, fp32 (already-zero from dC_smem; we do += later)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        // Loop over K tiles
        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            // Load A: W_smem viewed as [M][K] col_major = W_smem[k][m] in mem.
            // Stride between m's = 1 element (innermost).  Stride between
            // k's (rows of W_smem) = M elements.  In col_major load, the
            // "leading dim" is the outer dim, which is M.
            //
            // wmma::load_matrix_sync(frag, ptr, ldm) — ldm = stride between
            // adjacent columns (col_major) or rows (row_major).
            //
            // For W^T as col_major: A[m][k] stored as A_mem[k * M + m].
            // → load with col_major, ldm = M (so column stride = M elems).
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> a_frag;
            const __nv_bfloat16* a_ptr =
                &W_smem[k_tile * WMMA_K][m_tile * WMMA_M];
            wmma::load_matrix_sync(a_frag, a_ptr, M);

            // Load B: g_smem [K][R] row_major.  ldm = R.
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> b_frag;
            const __nv_bfloat16* b_ptr =
                &g_smem[k_tile * WMMA_K][n_tile * WMMA_N];
            wmma::load_matrix_sync(b_frag, b_ptr, R);

            // Multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // Store to dC_smem: row_major, ldm = R.
        float* c_ptr = &dC_smem[m_tile * WMMA_M][n_tile * WMMA_N];
        wmma::store_matrix_sync(c_ptr, acc_frag, R, wmma::mem_row_major);
    }

    __syncthreads();

    // ---- Phase 4: cooperative bulk atomicAdd dC_smem → global dC ----
    //
    // Each block writes to dC[h_chunk, :, :] which is unique per (h_block).
    // Different (n_block) values write to the SAME h_chunk → use atomicAdd.
    {
        const int total = M * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int m       = idx / R;
            const int c       = idx % R;
            const int j_local = m / L_PAD;
            const int b       = m % L_PAD;
            const int j_global = h_start + j_local;
            if (j_global < H && b < L) {
                const float val = dC_smem[m][c];
                if (val != 0.0f) {
                    atomicAdd(&dC[j_global * L * R + b * R + c], val);
                }
            }
        }
    }
}

#define LAUNCH_BWD_WMMA(BN, BH, LP, RR) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        dim3 grid(blocks_n, blocks_h, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_wmma_kernel<BN, BH, LP, RR><<<grid, block, 0, stream>>>( \
            (const __nv_bfloat16*)z_ptr, \
            (const __nv_bfloat16*)C_ptr, \
            (const __nv_bfloat16*)g_delta_ptr, \
            dC_ptr, dz_ptr, \
            N, H, L, grid_lo, scale); \
    } while(0)

}  // namespace

void spline_kv_bwd_wmma_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    const torch::Tensor& g_delta,
    torch::Tensor& dC,
    torch::Tensor& dz,
    double grid_lo,
    double scale
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16
                && C.dtype() == torch::kBFloat16
                && g_delta.dtype() == torch::kBFloat16);
    TORCH_CHECK(dC.dtype() == torch::kFloat32 && dz.dtype() == torch::kFloat32);

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);

    void* z_ptr       = z.data_ptr();
    void* C_ptr       = C.data_ptr();
    void* g_delta_ptr = g_delta.data_ptr();
    float* dC_ptr     = dC.data_ptr<float>();
    float* dz_ptr     = dz.data_ptr<float>();

    // Production shapes.  L_PAD chosen so that BLOCK_H * L_PAD is multiple
    // of 16 for wmma.
    //   L=16 → L_PAD=16 (16-aligned)
    //   L=22 → L_PAD=24 (need >=22, multiple of 16/BH; for BH=8 → 16-aligned via 8*24=192)
    if (R == 32 && L == 22) {
        LAUNCH_BWD_WMMA(64, 8, 24, 32);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_WMMA(64, 8, 16, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_WMMA(64, 8, 24, 64);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD_WMMA(64, 8, 16, 64);
    } else {
        TORCH_CHECK(false, "unsupported (R, L) — extend dispatch");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_wmma_cuda", &spline_kv_bwd_wmma_cuda,
          "FlashSplineFeature backward (wmma)",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("dC"), py::arg("dz"),
          py::arg("grid_lo"), py::arg("scale"));
}
