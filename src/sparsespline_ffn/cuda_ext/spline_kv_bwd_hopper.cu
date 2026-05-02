// FlashSplineFeature backward — Hopper-only kernel (sm_90+).
//
// Strategy: skip SMEM dC stage entirely.  After computing the wmma
// matmul, write fragment elements directly to global dC via per-element
// atomicAdd.  Use cp.async for g+C loads.
//
// This is the "wmma v3" / "Path Z step 1" version — uses cuda::wmma
// (mma.sync on Hopper) rather than the warpgroup-level wgmma.async
// because the simpler API is faster to get right; if this still loses
// to Triton v3 we'll escalate to raw wgmma+TMA inline PTX.
//
// Math: same as v2 wmma kernel.  dC[j, b, c] = sum_n W[n, m] * g[n, c]
// where m = j_local * L_PAD + b and W is densified with 3-of-L nonzeros.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f2bf(float x) {
    return __float2bfloat16(x);
}

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

// cp.async wrapper — copy 16 bytes (8 bf16) from global to SMEM, async.
__device__ __forceinline__ void cp_async_16(__nv_bfloat16* smem_dst,
                                              const __nv_bfloat16* gmem_src,
                                              bool valid) {
    unsigned smem_int = __cvta_generic_to_shared(smem_dst);
    if (valid) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(smem_int), "l"(gmem_src), "n"(16)
        );
    } else {
        // Zero-fill SMEM if source is OOB
        #pragma unroll
        for (int i = 0; i < 8; i++) smem_dst[i] = f2bf(0.0f);
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// ---------------------------------------------------------------------------
// Backward kernel — Hopper version with cp.async + no-SMEM-dC.
//
// Pipeline:
//   Phase 0: zero W_smem (BLOCK_N × M)              ~25 KB
//   Phase 1: cp.async g_smem (BLOCK_N × R)          ~4 KB
//            cp.async C_smem (BLOCK_H × L_PAD × R)  ~12 KB
//            commit_group ; wait_all
//   Phase 2: compute (n,j) pairs:
//            - B/τ/bin/dB
//            - W_smem[n][m] = B_k(τ) at 3 active columns
//            - dz scalar accumulate from C_smem (use SMEM, fp32 acc)
//            - write dz to global (single writer per (n,j))
//   __syncthreads
//   Phase 3: warps cooperate on (m_tile, n_tile) cells:
//            - each warp handles one cell at a time
//            - wmma load A=W_smem (col-major, ldm=M), B=g_smem (row-major, ldm=R)
//            - mma_sync over K_TILES = BLOCK_N / WMMA_K
//            - per-element atomicAdd of fragment → global dC (no SMEM stage)

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R>
__global__ void __launch_bounds__(128, 2)
spline_kv_bwd_hopper_kernel(
    const __nv_bfloat16* __restrict__ z,        // [N, H]
    const __nv_bfloat16* __restrict__ C,        // [H, L, R]
    const __nv_bfloat16* __restrict__ g_delta,  // [N, R]
    float* __restrict__ dC,                      // [H, L, R] fp32 (zero-init)
    float* __restrict__ dz,                      // [N, H] fp32 (zero-init)
    const int N, const int H, const int L,
    const float grid_lo, const float scale
) {
    constexpr int M = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / WMMA_M;
    constexpr int N_TILES = R / WMMA_N;
    constexpr int K_TILES = BLOCK_N / WMMA_K;
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
    __shared__ __nv_bfloat16 W_smem[BLOCK_N][M];          // K × M, col-major view
    __shared__ __nv_bfloat16 g_smem[BLOCK_N][R];          // K × N, row-major
    __shared__ __nv_bfloat16 C_smem[BLOCK_H][L_PAD][R];   // for dz reads
    __shared__ float warp_scratch[4][WMMA_M][WMMA_N];     // per-warp wmma scratch

    // ---- Phase 0: zero W_smem (essential — wmma reads ALL columns) ----
    {
        __nv_bfloat16 zero_bf = f2bf(0.0f);
        constexpr int total = BLOCK_N * M;
        // Vectorize: each thread writes 8 bf16 (1 uint4) per iter
        #pragma unroll
        for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
            // Use uint4 stores when possible.  total=BN*M; BN=64,M=128/192 → div by 8.
            uint4* p = reinterpret_cast<uint4*>(&((&W_smem[0][0])[idx]));
            *p = make_uint4(0, 0, 0, 0);
        }
    }

    // ---- Phase 1: cp.async load g_smem and C_smem ----
    // g_smem: BLOCK_N × R bf16 — cp.async 16-byte chunks (8 bf16)
    {
        constexpr int total = BLOCK_N * R;
        constexpr int CHUNK = 8;
        #pragma unroll
        for (int idx = tid * CHUNK; idx < total; idx += blockDim.x * CHUNK) {
            const int n_local = idx / R;
            const int c_start = idx % R;
            const int n_global = n_start + n_local;
            __nv_bfloat16* dst = &g_smem[n_local][c_start];
            const __nv_bfloat16* src = (n_global < N)
                ? &g_delta[n_global * R + c_start]
                : nullptr;
            cp_async_16(dst, src, n_global < N);
        }
    }
    {
        constexpr int total = BLOCK_H * L_PAD * R;
        constexpr int CHUNK = 8;
        #pragma unroll
        for (int idx = tid * CHUNK; idx < total; idx += blockDim.x * CHUNK) {
            const int j_local = idx / (L_PAD * R);
            const int b       = (idx / R) % L_PAD;
            const int c_start = idx % R;
            const int j_global = h_start + j_local;
            __nv_bfloat16* dst = &C_smem[j_local][b][c_start];
            bool valid = (j_global < H) && (b < L);
            const __nv_bfloat16* src = valid
                ? &C[j_global * L * R + b * R + c_start]
                : nullptr;
            cp_async_16(dst, src, valid);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- Phase 2: compute (n,j) pairs — write W_smem + dz ----
    {
        constexpr int total_pairs = BLOCK_N * BLOCK_H;
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

            const int col_base = j_local * L_PAD + bin_idx;
            W_smem[n_local][col_base + 0] = f2bf(B0);
            W_smem[n_local][col_base + 1] = f2bf(B1);
            W_smem[n_local][col_base + 2] = f2bf(B2);

            // dz scalar reduction
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

    // ---- Phase 3: wmma matmul + DIRECT atomicAdd to global dC ----
    //
    // No SMEM dC staging.  Each warp computes one (m_tile, n_tile) cell at a
    // time, accumulates fp32 in fragment, then atomicAdd's to global.
    //
    // Round-robin (m_tile, n_tile) across 4 warps.

    constexpr int total_cells = M_TILES * N_TILES;
    for (int t = warp_id; t < total_cells; t += 4) {
        const int m_tile = t / N_TILES;
        const int n_tile = t % N_TILES;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag,
                                    &W_smem[k_tile * WMMA_K][m_tile * WMMA_M], M);
            wmma::load_matrix_sync(b_frag,
                                    &g_smem[k_tile * WMMA_K][n_tile * WMMA_N], R);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Direct fragment-to-global atomicAdd.
        // For m16n16k16 fp32 accumulator on sm_80+, fragment layout (per
        // PTX docs §9.7.13 — wmma m16n16k16 fp32 accumulator):
        //
        //   8 elements per thread.  groupID = lane_id / 4, tigid = lane_id % 4.
        //   elem[0] @ (groupID*2 + 0,    tigid*2 + 0)
        //   elem[1] @ (groupID*2 + 0,    tigid*2 + 1)
        //   elem[2] @ (groupID*2 + 8,    tigid*2 + 0)
        //   elem[3] @ (groupID*2 + 8,    tigid*2 + 1)
        //   elem[4] @ (groupID*2 + 1,    tigid*2 + 0)
        //   elem[5] @ (groupID*2 + 1,    tigid*2 + 1)
        //   elem[6] @ (groupID*2 + 9,    tigid*2 + 0)
        //   elem[7] @ (groupID*2 + 9,    tigid*2 + 1)
        //
        // BUT the cuda::wmma API does NOT guarantee this layout — it's
        // implementation-defined.  The portable way is store_matrix_sync
        // to a per-warp SMEM scratch, then read+atomicAdd from scratch.
        //
        // Per-warp scratch: 16x16 fp32 = 1KB per warp × 4 warps = 4KB total.
        // Reused across cells.

        wmma::store_matrix_sync(&warp_scratch[warp_id][0][0],
                                  acc, WMMA_N, wmma::mem_row_major);

        // 32 threads in warp, each handles 8 elements (16x16/32 = 8).
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = lane_id + i * 32;  // 0..255
            const int row = idx / WMMA_N;       // 0..15
            const int col = idx % WMMA_N;       // 0..15

            const int m_global = m_tile * WMMA_M + row;
            const int j_local  = m_global / L_PAD;
            const int b        = m_global % L_PAD;
            const int j_global = h_start + j_local;
            const int c        = n_tile * WMMA_N + col;

            if (j_global < H && b < L) {
                const float val = warp_scratch[warp_id][row][col];
                if (val != 0.0f) {
                    atomicAdd(&dC[j_global * L * R + b * R + c], val);
                }
            }
        }
    }
}

#define LAUNCH_BWD_HOPPER(BN, BH, LP, RR) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        dim3 grid(blocks_n, blocks_h, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_hopper_kernel<BN, BH, LP, RR><<<grid, block, 0, stream>>>( \
            (const __nv_bfloat16*)z_ptr, \
            (const __nv_bfloat16*)C_ptr, \
            (const __nv_bfloat16*)g_delta_ptr, \
            dC_ptr, dz_ptr, \
            N, H, L, grid_lo, scale); \
    } while(0)

}  // namespace

void spline_kv_bwd_hopper_cuda(
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

    if (R == 32 && L == 22) {
        LAUNCH_BWD_HOPPER(64, 8, 24, 32);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_HOPPER(64, 8, 16, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_HOPPER(64, 8, 24, 64);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD_HOPPER(64, 8, 16, 64);
    } else {
        TORCH_CHECK(false, "unsupported (R, L) — extend dispatch");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_hopper_cuda", &spline_kv_bwd_hopper_cuda,
          "FlashSplineFeature backward (Hopper: cp.async + no-SMEM-dC + wmma)",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("dC"), py::arg("dz"),
          py::arg("grid_lo"), py::arg("scale"));
}
