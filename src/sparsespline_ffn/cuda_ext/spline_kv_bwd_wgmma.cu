// FlashSplineFeature backward — Hopper WGMMA + TMA kernel (sm_90+ only).
//
// Strategy:
//   1. TMA bulk async copy for g_delta and C tiles (cp.async.bulk.tensor)
//   2. wgmma.mma_async m64n32k16.f32.bf16.bf16 for densified matmul
//   3. Per-thread direct atomicAdd of fragment → global dC (no SMEM dC stage)
//   4. Single warpgroup (4 warps, 128 threads) per block
//
// Math:
//   dC[j, b, c] = sum_n W[n, m] * g[n, c]   where m = j_local*L_PAD + b
//   dz[n, j]    = scale * sum_c g[n,c] * sum_k dB_k(τ)*C[j, bin+k, c]
//
// W is densified: W[n, m] = B_k(τ_nj) at the 3 active columns m=j*L_PAD+bin+k,
// zero elsewhere.  wgmma processes the dense matmul; the wasted FLOPs from
// the L-3 zero columns are absorbed by tensor core throughput.
//
// Wgmma fragment layout for m64n32k16 fp32 accumulator (per PTX docs):
//   16 fp32 elements per thread, organized as 4 chunks of 4 elements each.
//   Each chunk covers an 8-column slice of the n=32 output.
//   Within each chunk (16x8 sub-tile per warp), the per-thread layout is
//   the same as mma m16n8k16: 4 elements at known (row, col) positions.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// Wgmma SMEM descriptor encoder.
//
// Format (Hopper PTX 8.0+, "SMEM descriptor format"):
//   bits[13:0]   start address (>>4)
//   bits[29:16]  leading byte offset (>>4)
//   bits[45:32]  stride byte offset  (>>4)
//   bits[52:49]  matrix base offset
//   bits[63:62]  swizzle mode (00=no, 01=128B, 10=64B, 11=32B)
//
// We use NoSwizzle (mode 0) for simplicity.  For our small tile sizes
// (M*2 = 256-384 bytes), bank conflicts are minimal without swizzle.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t encode_smem_desc(
    void* smem_ptr,
    uint32_t leading_byte_offset,    // byte stride within K dim (K-major)
    uint32_t stride_byte_offset,     // byte stride between cores in M/N dim
    uint32_t swizzle = 0
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr & 0x3FFFF) >> 4) << 0;
    desc |= ((uint64_t)(leading_byte_offset & 0x3FFFF) >> 4) << 16;
    desc |= ((uint64_t)(stride_byte_offset & 0x3FFFF) >> 4) << 32;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}

// -----------------------------------------------------------------------------
// wgmma fence + commit + wait + proxy fence
// -----------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}
// CRITICAL: fence between generic proxy SMEM writes and async proxy wgmma reads.
// PTX 8.5 §9.7.14.5 Step 2: must use both wgmma.fence AND fence.proxy.async.
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// -----------------------------------------------------------------------------
// wgmma m64n32k16.f32.bf16.bf16 — single op
//   acc: 16 fp32 registers per thread
//   a_desc: SMEM descriptor for A (M, K) in col-major (= K rows × M cols in mem)
//   b_desc: SMEM descriptor for B (K, N) in row-major (= K rows × N cols in mem)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_m64n32k16(
    float* acc, uint64_t a_desc, uint64_t b_desc, bool scale_d
) {
    int sd = scale_d ? 1 : 0;
    // PTX: wgmma.mma_async ... d, a-desc, b-desc, scale-d, scale-a, scale-b, trans-a, trans-b
    //   trans=0 → K-major (K innermost in mem)
    //   trans=1 → M-major (A) / N-major (B) (M/N innermost)
    // Our W_smem[K][M] has M innermost → trans_a = 1.
    // Our g_smem[K][N] has N innermost → trans_b = 1.
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %18, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, p, 1, 1, 1, 1;\n\t"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15])
        : "l"(a_desc), "l"(b_desc), "r"(sd)
    );
}

// -----------------------------------------------------------------------------
// cp.async helpers (used as fallback / for non-TMA loads).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(__nv_bfloat16* dst, const __nv_bfloat16* src, bool valid) {
    unsigned d = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __float2bfloat16(0.0f);
    }
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// -----------------------------------------------------------------------------
// B-spline B2 basis (same as before)
// -----------------------------------------------------------------------------
__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16(x); }

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

// =============================================================================
// Backward kernel — wgmma + cp.async (no TMA in v1; TMA in v2 if needed).
//
// Tile contract:
//   BLOCK_N = K dim, must be multiple of 16  (we use 64 → 4 k-iters)
//   BLOCK_H * L_PAD = M dim, must be multiple of 64  (M_TILES = M/64)
//   R = N dim, must be 32 or 64  (N_TILES = R/32)
//
// Grid: (cdiv(N, BLOCK_N), cdiv(H, BLOCK_H))
// Block: 128 threads (1 warpgroup)
// =============================================================================

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R>
__global__ void __launch_bounds__(128, 2)
spline_kv_bwd_wgmma_kernel(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ g_delta,
    float* __restrict__ dC,
    float* __restrict__ dz,
    const int N, const int H, const int L,
    const float grid_lo, const float scale
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;            // wgmma m=64
    constexpr int N_TILES = R / 32;            // wgmma n=32
    constexpr int K_TILES = BLOCK_N / 16;      // wgmma k=16
    static_assert(M % 64 == 0,        "M (BLOCK_H*L_PAD) must be multiple of 64");
    static_assert(BLOCK_N % 16 == 0,  "BLOCK_N must be multiple of 16");
    static_assert(R % 32 == 0,        "R must be multiple of 32");

    const int pid_n   = blockIdx.x;
    const int pid_h   = blockIdx.y;
    const int n_start = pid_n * BLOCK_N;
    const int h_start = pid_h * BLOCK_H;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // SMEM core-major NoSwizzle layout for wgmma.
    //
    // PTX 8.5 §9.7.14.5.1.6: "core matrices occupy contiguous space" — each
    // 8x8 core (= 128 bytes) must be a single contiguous chunk in SMEM.
    //
    // For trans_a=1 (M-major) A[m, k]: each core row = 8 M-elements = 16 bytes
    //   contiguous; 8 such rows stacked along K = 128 bytes/core contiguous.
    // Layout: cores[k_core][m_core][k_in_core=0..7][m_in_core=0..7] flat
    //   addr(k, m) in bf16 elements = (k>>3)*M_CORES*64 + (m>>3)*64
    //                                 + (k&7)*8 + (m&7)
    //
    // For trans_b=1 (N-major) B[k, n]: same pattern with N replacing M.
    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;
    constexpr int K_CORES = BLOCK_N / 8;
    static_assert(M % 8 == 0, "M must be multiple of 8 (core size)");
    static_assert(R % 8 == 0, "R must be multiple of 8 (core size)");
    static_assert(BLOCK_N % 8 == 0, "BLOCK_N must be multiple of 8 (core size)");

    __shared__ __align__(128) __nv_bfloat16 W_cores[K_CORES * M_CORES * 64];
    __shared__ __align__(128) __nv_bfloat16 g_cores[K_CORES * N_CORES * 64];
    __shared__           __nv_bfloat16 C_smem[BLOCK_H][L_PAD][R];    // dz reads

    // Macros for bf16 element offset into core-major SMEM.
    // (k >> 3) selects K core, (m >> 3) selects M/N core; cores are 64 elements
    // (= 128 bytes) flat with internal layout [k_in*8 + m_in].
    #define W_OFF(k, m)  (((k) >> 3) * M_CORES * 64 + ((m) >> 3) * 64 + ((k) & 7) * 8 + ((m) & 7))
    #define G_OFF(k, n)  (((k) >> 3) * N_CORES * 64 + ((n) >> 3) * 64 + ((k) & 7) * 8 + ((n) & 7))

    // ---- Phase 0: zero W_cores (vectorized uint4 stores; flat layout) ----
    {
        constexpr int total_elems = K_CORES * M_CORES * 64;
        #pragma unroll
        for (int idx = tid * 8; idx < total_elems; idx += blockDim.x * 8) {
            uint4* p = reinterpret_cast<uint4*>(&W_cores[idx]);
            *p = make_uint4(0, 0, 0, 0);
        }
    }

    // ---- Phase 1: cp.async load g_cores and C_smem ----
    // g_cores is core-major; load 8 contiguous bf16 (= 16 bytes) per cp.async.
    // For trans_b=1, "row of core" = 8 N-elements at fixed K.  Each thread
    // loads one such row into a core's row slot.
    {
        // total rows of cores = K_CORES * N_CORES * 8
        constexpr int total_rows = K_CORES * N_CORES * 8;  // each row=8 elem=16B
        for (int idx = tid; idx < total_rows; idx += blockDim.x) {
            const int k_core = idx / (N_CORES * 8);
            const int rem    = idx % (N_CORES * 8);
            const int n_core = rem / 8;
            const int k_in   = rem % 8;
            const int k_global = k_core * 8 + k_in;
            const int n_start_in_core = n_core * 8;
            const int n_global_idx = n_start + k_global;  // BLOCK_N = K dim of B
            // dst: g_cores at core (k_core, n_core), row k_in, cols n_start..+7
            __nv_bfloat16* dst =
                &g_cores[((k_core * N_CORES) + n_core) * 64 + k_in * 8];
            const __nv_bfloat16* src = (n_global_idx < N)
                ? &g_delta[n_global_idx * R + n_start_in_core]
                : nullptr;
            cp_async_16(dst, src, n_global_idx < N);
        }
    }
    {
        constexpr int total = BLOCK_H * L_PAD * R;
        #pragma unroll
        for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
            const int j_local  = idx / (L_PAD * R);
            const int b        = (idx / R) % L_PAD;
            const int c_start  = idx % R;
            const int j_global = h_start + j_local;
            const bool valid = (j_global < H) && (b < L);
            cp_async_16(&C_smem[j_local][b][c_start],
                        valid ? &C[j_global * L * R + b * R + c_start] : nullptr,
                        valid);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- Phase 2: compute (n,j) pairs — write W_smem + dz ----
    {
        constexpr int total_pairs = BLOCK_N * BLOCK_H;
        for (int p = tid; p < total_pairs; p += blockDim.x) {
            const int n_local  = p / BLOCK_H;
            const int j_local  = p % BLOCK_H;
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
            // W_cores layout: W_OFF(k=n_local, m=col_base+k_off)
            W_cores[W_OFF(n_local, col_base + 0)] = f2bf(B0);
            W_cores[W_OFF(n_local, col_base + 1)] = f2bf(B1);
            W_cores[W_OFF(n_local, col_base + 2)] = f2bf(B2);

            float inner = 0.0f;
            #pragma unroll
            for (int c = 0; c < R; c++) {
                const float g  = bf2f(g_cores[G_OFF(n_local, c)]);
                const float c0 = bf2f(C_smem[j_local][bin_idx + 0][c]);
                const float c1 = bf2f(C_smem[j_local][bin_idx + 1][c]);
                const float c2 = bf2f(C_smem[j_local][bin_idx + 2][c]);
                inner += g * (dB0 * c0 + dB1 * c1 + dB2 * c2);
            }
            dz[n_global * H + j_global] = scale * inner;
        }
    }
    __syncthreads();

    // CRITICAL: cross-proxy fence — generic SMEM writes (Phase 0/1/2) → async
    // proxy reads (wgmma).  Without this, wgmma sees stale/garbage data.
    fence_proxy_async_shared_cta();

    // ---- Phase 3: wgmma matmul ----
    //
    // SMEM core-major layout (PTX 8.5 §9.7.14.5.1.6 compliant):
    //   Each 8x8 core = 128 contiguous bytes.
    //   Layout: cores[k_core * (M_or_N)_CORES + (m_or_n)_core][k_in*8 + (m/n)_in]
    //
    // Stride between cores:
    //   K direction: M_CORES * 128 bytes = 16*M for A; N_CORES*128 = 16*R for B
    //   M/N direction: 128 bytes
    constexpr uint32_t LBO_A = M_CORES * 128;  // bytes — between K-cores
    constexpr uint32_t SBO_A = 128;             // bytes — between M-cores
    constexpr uint32_t LBO_B = N_CORES * 128;  // bytes — between K-cores
    constexpr uint32_t SBO_B = 128;             // bytes — between N-cores

    for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
        for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
            // 16 fp32 acc per thread (m=64, n=32 → 2048 elem / 128 thr = 16)
            float acc[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) acc[i] = 0.0f;

            wgmma_fence();

            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                // wgmma m=64, k=16 → 2 K cores tall × 8 M cores wide for tile
                // A base = core (k_tile*2, m_tile*8) — first core of the 2x8 tile
                // SMEM offset (bf16 elem) = ((k_tile*2) * M_CORES + m_tile*8) * 64
                __nv_bfloat16* a_ptr =
                    &W_cores[((k_tile * 2) * M_CORES + m_tile * 8) * 64];
                // B base = core (k_tile*2, n_tile*4) — 2 K cores × 4 N cores (n=32)
                __nv_bfloat16* b_ptr =
                    &g_cores[((k_tile * 2) * N_CORES + n_tile * 4) * 64];
                uint64_t a_desc = encode_smem_desc(a_ptr, LBO_A, SBO_A, 0);
                uint64_t b_desc = encode_smem_desc(b_ptr, LBO_B, SBO_B, 0);
                bool scale_d = (k_tile > 0);  // first op zeros, rest accumulate
                wgmma_m64n32k16(acc, a_desc, b_desc, scale_d);
            }

            wgmma_commit_group();
            wgmma_wait_group<0>();

            // ---- Phase 4: per-thread atomicAdd to global dC ----
            //
            // Wgmma m64n32 fp32 fragment layout:
            //   16 elements per thread, 4 chunks of 4 elements.
            //   Each chunk covers an 8-col slice of the n=32 output.
            //   Within each chunk (16 rows × 8 cols per warp):
            //     groupID = lane_id / 4  (0..7), tigid = lane_id % 4 (0..3)
            //     elem[0] @ (groupID,    tigid*2 + 0)
            //     elem[1] @ (groupID,    tigid*2 + 1)
            //     elem[2] @ (groupID+8,  tigid*2 + 0)
            //     elem[3] @ (groupID+8,  tigid*2 + 1)
            //   Each warp covers rows [warp_id*16, warp_id*16+16).

            #pragma unroll
            for (int chunk = 0; chunk < 4; chunk++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int frag_idx = chunk * 4 + e;
                    const int groupID  = lane_id / 4;
                    const int tigid    = lane_id % 4;
                    int row_in_warp, col_in_chunk;
                    switch (e) {
                        case 0: row_in_warp = groupID;     col_in_chunk = tigid*2 + 0; break;
                        case 1: row_in_warp = groupID;     col_in_chunk = tigid*2 + 1; break;
                        case 2: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 0; break;
                        case 3: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 1; break;
                    }
                    const int row_in_tile = warp_id * 16 + row_in_warp;     // 0..63
                    const int col_in_tile = chunk * 8 + col_in_chunk;        // 0..31
                    const int m_global    = m_tile * 64 + row_in_tile;
                    const int j_local     = m_global / L_PAD;
                    const int b           = m_global % L_PAD;
                    const int j_global    = h_start + j_local;
                    const int c           = n_tile * 32 + col_in_tile;
                    if (j_global < H && b < L) {
                        const float val = acc[frag_idx];
                        if (val != 0.0f) {
                            atomicAdd(&dC[j_global * L * R + b * R + c], val);
                        }
                    }
                }
            }
        }
    }
}

__global__ void spline_kv_bwd_postprocess_kernel(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ g_a,
    const float* __restrict__ dC_accum,
    const float* __restrict__ dz_spline,
    __nv_bfloat16* __restrict__ dC_out,
    __nv_bfloat16* __restrict__ dz_out,
    const int total_dz,
    const int total_dC,
    const int activation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_dC) {
        dC_out[idx] = f2bf(dC_accum[idx]);
    }
    if (idx < total_dz) {
        const float z_val = bf2f(z[idx]);
        const float g_val = bf2f(g_a[idx]);
        __nv_bfloat16 phi_prime_bf;
        if (activation == 0) {          // relu_sq
            phi_prime_bf = f2bf((z_val > 0.0f) ? (2.0f * z_val) : 0.0f);
        } else if (activation == 2) {   // identity
            phi_prime_bf = f2bf(1.0f);
        } else {
            phi_prime_bf = f2bf(1.0f);
        }
        const __nv_bfloat16 dz_base_bf = f2bf(g_val * bf2f(phi_prime_bf));
        dz_out[idx] = f2bf(bf2f(dz_base_bf) + dz_spline[idx]);
    }
}

#define LAUNCH_BWD_WGMMA(BN, BH, LP, RR) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        dim3 grid(blocks_n, blocks_h, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_wgmma_kernel<BN, BH, LP, RR><<<grid, block, 0, stream>>>( \
            (const __nv_bfloat16*)z_ptr, \
            (const __nv_bfloat16*)C_ptr, \
            (const __nv_bfloat16*)g_delta_ptr, \
            dC_ptr, dz_ptr, \
            N, H, L, grid_lo, scale); \
    } while(0)

}  // namespace

void spline_kv_bwd_wgmma_cuda(
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

    // Production shapes — L_PAD chosen so M=BH*L_PAD is multiple of 64.
    //   L=16 → L_PAD=16, M=128 (BH=8); 128/64=2 m-tiles
    //   L=22 → L_PAD=24, M=192 (BH=8); 192/64=3 m-tiles
    //   L=32 → L_PAD=32, M=256 (BH=8); 256/64=4 m-tiles
    //
    // Round 3 reverted: BN=256 hurt due to occupancy drop (1 block/SM at
    // 124 KB SMEM).  Round 3 result: 18.3s vs baseline 17.20s — atomic
    // contention reduction not worth occupancy loss.  Keep BN=128.
    // BN=128 (3.B.1 — verified marginally faster vs BN=64 in graph mode at
    // r=32 cells: -4% wall.  Re-applied after BN=64 ablation showed BN=128
    // was actually the better choice).
    if (R == 32 && L == 22) {
        LAUNCH_BWD_WGMMA(128, 8, 24, 32);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_WGMMA(128, 8, 16, 32);
    } else if (R == 32 && L == 32) {
        LAUNCH_BWD_WGMMA(128, 8, 32, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_WGMMA(128, 8, 24, 64);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD_WGMMA(128, 8, 16, 64);
    } else if (R == 64 && L == 32) {
        LAUNCH_BWD_WGMMA(128, 8, 32, 64);
    } else {
        TORCH_CHECK(false, "unsupported (R, L) — extend dispatch");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> spline_kv_bwd_wgmma_cuda_fused(
    const torch::Tensor& z,
    const torch::Tensor& C,
    const torch::Tensor& g_delta,
    const torch::Tensor& g_a,
    double grid_lo,
    double scale,
    int64_t activation
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda() && g_a.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16
                && C.dtype() == torch::kBFloat16
                && g_delta.dtype() == torch::kBFloat16
                && g_a.dtype() == torch::kBFloat16);
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous()
                && g_delta.is_contiguous() && g_a.is_contiguous());

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);
    TORCH_CHECK(g_a.size(0) == N && g_a.size(1) == H);

    auto accum_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat32);
    auto out_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    torch::Tensor dC_accum = torch::zeros({H, L, R}, accum_opts);
    torch::Tensor dz_accum = torch::zeros({N, H}, accum_opts);
    torch::Tensor dC_out = torch::empty({H, L, R}, out_opts);
    torch::Tensor dz_out = torch::empty({N, H}, out_opts);

    void* z_ptr       = z.data_ptr();
    void* C_ptr       = C.data_ptr();
    void* g_delta_ptr = g_delta.data_ptr();
    float* dC_ptr     = dC_accum.data_ptr<float>();
    float* dz_ptr     = dz_accum.data_ptr<float>();

    // BN=128 across all configs (Round 3 BN=256 reverted — occupancy loss
    // outweighed atomic-contention reduction benefit).
    // BN=128 (3.B.1 — verified marginally faster vs BN=64 in graph mode at
    // r=32 cells: -4% wall.  Re-applied after BN=64 ablation showed BN=128
    // was actually the better choice).
    if (R == 32 && L == 22) {
        LAUNCH_BWD_WGMMA(128, 8, 24, 32);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_WGMMA(128, 8, 16, 32);
    } else if (R == 32 && L == 32) {
        LAUNCH_BWD_WGMMA(128, 8, 32, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_WGMMA(128, 8, 24, 64);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD_WGMMA(128, 8, 16, 64);
    } else if (R == 64 && L == 32) {
        LAUNCH_BWD_WGMMA(128, 8, 32, 64);
    } else {
        TORCH_CHECK(false, "unsupported (R, L) — extend dispatch");
    }

    const int total_dz = N * H;
    const int total_dC = H * L * R;
    const int total = total_dz > total_dC ? total_dz : total_dC;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto post_stream = c10::cuda::getCurrentCUDAStream();
    spline_kv_bwd_postprocess_kernel<<<blocks, threads, 0, post_stream>>>(
        (const __nv_bfloat16*)z.data_ptr(),
        (const __nv_bfloat16*)g_a.data_ptr(),
        dC_accum.data_ptr<float>(),
        dz_accum.data_ptr<float>(),
        (__nv_bfloat16*)dC_out.data_ptr(),
        (__nv_bfloat16*)dz_out.data_ptr(),
        total_dz,
        total_dC,
        (int)activation
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dC_out, dz_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_wgmma_cuda", &spline_kv_bwd_wgmma_cuda,
          "FlashSplineFeature backward (Hopper wgmma + cp.async)",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("dC"), py::arg("dz"),
          py::arg("grid_lo"), py::arg("scale"));
    m.def("spline_kv_bwd_wgmma_cuda_fused", &spline_kv_bwd_wgmma_cuda_fused,
          "FlashSplineFeature backward fused postprocess (Hopper wgmma + cp.async)",
          py::arg("z"), py::arg("C"), py::arg("g_delta"), py::arg("g_a"),
          py::arg("grid_lo"), py::arg("scale"), py::arg("activation"));
}
