// FlashSplineFeature backward — Hopper WGMMA v2 (split-N + reduce, no atomic).
//
// Grid: (H_TILES, N_PART)
//   Each block uniquely owns dC_scratch[h_block, n_part_idx, L, R].
//   No atomic-add anywhere — each (h, b, c) element written once per partition,
//   then a small reduce kernel sums partitions in fp32 + casts to bf16.
//
// Compared to v1 (atomic-add into shared dC):
//   - v1: grid (N_TILE_768/128 + H_TILE) ≈ 768 blocks, 16-way atomic contention
//   - v2: grid (48, 8)  = 384 blocks, ZERO atomic-add
//   v2 trades off slightly fewer blocks for vastly less contention.
//
// dz computation is identical to v1 (per-(n,j) pair, written once).
//
// Math (unchanged):
//   dC[j, b, c] = sum_n W[n, m] * g[n, c]   where m = j_local*L_PAD + b
//   dz[n, j]    = scale * sum_c g[n,c] * sum_k dB_k(τ)*C[j, bin+k, c]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {

// =============================================================================
// Helpers (shared with v1).
// =============================================================================
__device__ __forceinline__ uint64_t encode_smem_desc(
    void* smem_ptr, uint32_t leading_byte_offset,
    uint32_t stride_byte_offset, uint32_t swizzle = 0
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr & 0x3FFFF) >> 4) << 0;
    desc |= ((uint64_t)(leading_byte_offset & 0x3FFFF) >> 4) << 16;
    desc |= ((uint64_t)(stride_byte_offset & 0x3FFFF) >> 4) << 32;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}

__device__ __forceinline__ void wgmma_fence()        { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group()   { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N)); }
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n32k16(
    float* acc, uint64_t a_desc, uint64_t b_desc, bool scale_d
) {
    int sd = scale_d ? 1 : 0;
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

__device__ __forceinline__ void cp_async_16(__nv_bfloat16* dst,
                                              const __nv_bfloat16* src,
                                              bool valid) {
    unsigned d = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __float2bfloat16(0.0f);
    }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

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
// v2 main kernel — split-N, fp32 scratch.
//
// Grid: (cdiv(H, BLOCK_H), N_PART, 1)
// Block: 128 threads (1 warpgroup)
//
// dz computation: per (n,j) pair.  Each (n_global, j_global) is touched by
// EXACTLY ONE block (the one owning the partition containing n_global and
// the h-tile containing j_global).  No atomic.
//
// dC accumulation: lives in SMEM during the chunk loop; final write goes to
// dC_scratch[h_block, n_part, L, R] in fp32 — unique per block, no atomic.
// =============================================================================
template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int N_PER_PART>
__global__ void __launch_bounds__(128, 1)
spline_kv_bwd_wgmma_v3_kernel(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ g_delta,
    float* __restrict__ dC_scratch,    // [H, N_PART, L, R] fp32
    float* __restrict__ dz,             // [N, H]            fp32
    const int N, const int H, const int L,
    const int N_PART,
    const float grid_lo, const float scale
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;
    constexpr int N_TILES = R / 32;
    constexpr int K_TILES = BLOCK_N / 16;
    constexpr int CHUNKS_PER_PART = N_PER_PART / BLOCK_N;
    static_assert(M % 64 == 0,            "M must be multiple of 64");
    static_assert(BLOCK_N % 16 == 0,      "BLOCK_N must be multiple of 16");
    static_assert(R % 32 == 0,            "R must be multiple of 32");
    static_assert(N_PER_PART % BLOCK_N == 0, "N_PER_PART must be multiple of BLOCK_N");

    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;
    constexpr int K_CORES = BLOCK_N / 8;

    const int pid_h    = blockIdx.x;
    const int pid_npart = blockIdx.y;
    const int h_start  = pid_h * BLOCK_H;
    const int part_n_start = pid_npart * N_PER_PART;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;

    // Per-chunk SMEM — v3: double-buffered g_cores for 2-stage cp.async pipeline.
    // W_cores stays single-buffer because it's filled in compute (Phase 3) and
    // consumed within the same chunk's wgmma (Phase 4) — no producer-consumer
    // overlap on W_cores.
    __shared__ __align__(128) __nv_bfloat16 W_cores[K_CORES * M_CORES * 64];
    __shared__ __align__(128) __nv_bfloat16 g_cores[2][K_CORES * N_CORES * 64];   // double-buffered
    __shared__ __nv_bfloat16 C_smem[BLOCK_H][L_PAD][R];

    // Persistent SMEM dC accumulator [M, R] fp32 — no atomic
    __shared__ float dC_acc[M][R];

    #define W_OFF(k, m)  (((k) >> 3) * M_CORES * 64 + ((m) >> 3) * 64 + ((k) & 7) * 8 + ((m) & 7))
    #define G_OFF(k, n)  (((k) >> 3) * N_CORES * 64 + ((n) >> 3) * 64 + ((k) & 7) * 8 + ((n) & 7))

    constexpr uint32_t LBO_A = M_CORES * 128;
    constexpr uint32_t SBO_A = 128;
    constexpr uint32_t LBO_B = N_CORES * 128;
    constexpr uint32_t SBO_B = 128;

    // ---- Phase 0a: zero dC_acc ----
    {
        constexpr int total_dc = M * R;
        #pragma unroll
        for (int idx = tid; idx < total_dc; idx += blockDim.x) {
            dC_acc[idx / R][idx % R] = 0.0f;
        }
    }

    // ---- Phase 0b: load C[h_block, :, :] (used by every chunk for dz) ----
    {
        constexpr int total = BLOCK_H * L_PAD * R;
        for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
            const int j_local  = idx / (L_PAD * R);
            const int b        = (idx / R) % L_PAD;
            const int c_start  = idx % R;
            const int j_global = h_start + j_local;
            const bool valid   = (j_global < H) && (b < L);
            cp_async_16(&C_smem[j_local][b][c_start],
                        valid ? &C[j_global * L * R + b * R + c_start] : nullptr,
                        valid);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- v3 prologue: pre-issue cp.async load of chunk[0] into g_cores[0] ----
    // 2-stage pipeline: at iteration i, we wait for chunk[i] (issued in iteration
    // i-1 or in this prologue) and pre-issue chunk[i+1].  This overlaps memory
    // load with compute.
    auto issue_g_load = [&](int chunk_idx, int buf_idx) {
        const int n_start_chunk = part_n_start + chunk_idx * BLOCK_N;
        constexpr int total_rows = K_CORES * N_CORES * 8;
        for (int idx = tid; idx < total_rows; idx += blockDim.x) {
            const int k_core = idx / (N_CORES * 8);
            const int rem    = idx % (N_CORES * 8);
            const int n_core = rem / 8;
            const int k_in   = rem % 8;
            const int k_global = k_core * 8 + k_in;
            const int n_start_in_core = n_core * 8;
            const int n_global_idx = n_start_chunk + k_global;
            __nv_bfloat16* dst =
                &g_cores[buf_idx][((k_core * N_CORES) + n_core) * 64 + k_in * 8];
            const __nv_bfloat16* src = (n_global_idx < N)
                ? &g_delta[n_global_idx * R + n_start_in_core]
                : nullptr;
            cp_async_16(dst, src, n_global_idx < N);
        }
        cp_async_commit();
    };

    // Issue load of chunk[0] into buf[0]
    issue_g_load(0, 0);

    // ---- Main loop: iterate THIS partition's chunks (2-stage pipelined) ----
    #pragma unroll 1
    for (int chunk = 0; chunk < CHUNKS_PER_PART; chunk++) {
        const int n_start = part_n_start + chunk * BLOCK_N;
        if (n_start >= N) break;  // partial last partition

        const int cur_buf  = chunk & 1;       // current chunk's buffer
        const int next_buf = (chunk + 1) & 1; // next chunk's buffer

        // ---- Phase 1: zero W_cores (still per-iter; W_cores is single-buffered) ----
        {
            constexpr int total_elems = K_CORES * M_CORES * 64;
            #pragma unroll
            for (int idx = tid * 8; idx < total_elems; idx += blockDim.x * 8) {
                uint4* p = reinterpret_cast<uint4*>(&W_cores[idx]);
                *p = make_uint4(0, 0, 0, 0);
            }
        }

        // ---- Phase 2: pre-issue load of chunk[i+1] into next_buf (overlaps with Phase 3) ----
        if (chunk + 1 < CHUNKS_PER_PART) {
            const int next_n_start = part_n_start + (chunk + 1) * BLOCK_N;
            if (next_n_start < N) {
                issue_g_load(chunk + 1, next_buf);
            } else {
                // Out-of-range: still issue commit so wait_group accounting matches
                cp_async_commit();
            }
        }

        // Wait for chunk[i] to be ready.  At this point, two cp.async groups
        // may be in flight (chunk[i] commit + chunk[i+1] commit if pre-issued).
        // wait_group<1> waits until at most 1 group remains in flight (i.e.,
        // chunk[i] is finished, chunk[i+1] still in flight).
        if (chunk + 1 < CHUNKS_PER_PART) {
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            // Last chunk: only chunk[i] in flight; wait for all to finish.
            cp_async_wait_all();
        }
        // CRITICAL: cp_async_wait_group is per-thread.  Block-wide sync needed
        // before reading g_cores[cur_buf] across threads.
        __syncthreads();

        // ---- Phase 3: per-thread compute (n,j) — write W_cores + dz ----
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
                W_cores[W_OFF(n_local, col_base + 0)] = f2bf(B0);
                W_cores[W_OFF(n_local, col_base + 1)] = f2bf(B1);
                W_cores[W_OFF(n_local, col_base + 2)] = f2bf(B2);

                float inner = 0.0f;
                #pragma unroll
                for (int c = 0; c < R; c++) {
                    const float g  = bf2f(g_cores[cur_buf][G_OFF(n_local, c)]);
                    const float c0 = bf2f(C_smem[j_local][bin_idx + 0][c]);
                    const float c1 = bf2f(C_smem[j_local][bin_idx + 1][c]);
                    const float c2 = bf2f(C_smem[j_local][bin_idx + 2][c]);
                    inner += g * (dB0 * c0 + dB1 * c1 + dB2 * c2);
                }
                dz[n_global * H + j_global] = scale * inner;
            }
        }
        __syncthreads();
        fence_proxy_async_shared_cta();

        // ---- Phase 4: wgmma matmul, accumulate into dC_acc SMEM ----
        for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
            for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
                float acc[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) acc[i] = 0.0f;

                wgmma_fence();
                for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                    __nv_bfloat16* a_ptr =
                        &W_cores[((k_tile * 2) * M_CORES + m_tile * 8) * 64];
                    __nv_bfloat16* b_ptr =
                        &g_cores[cur_buf][((k_tile * 2) * N_CORES + n_tile * 4) * 64];
                    uint64_t a_desc = encode_smem_desc(a_ptr, LBO_A, SBO_A, 0);
                    uint64_t b_desc = encode_smem_desc(b_ptr, LBO_B, SBO_B, 0);
                    bool scale_d = (k_tile > 0);
                    wgmma_m64n32k16(acc, a_desc, b_desc, scale_d);
                }
                wgmma_commit_group();
                wgmma_wait_group<0>();

                // Per-thread fragment → dC_acc SMEM (single writer per dest)
                #pragma unroll
                for (int chunk_e = 0; chunk_e < 4; chunk_e++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int frag_idx = chunk_e * 4 + e;
                        const int groupID  = lane_id / 4;
                        const int tigid    = lane_id % 4;
                        int row_in_warp, col_in_chunk;
                        switch (e) {
                            case 0: row_in_warp = groupID;     col_in_chunk = tigid*2 + 0; break;
                            case 1: row_in_warp = groupID;     col_in_chunk = tigid*2 + 1; break;
                            case 2: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 0; break;
                            case 3: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 1; break;
                        }
                        const int row_in_tile = warp_id * 16 + row_in_warp;
                        const int col_in_tile = chunk_e * 8 + col_in_chunk;
                        const int m_global    = m_tile * 64 + row_in_tile;
                        const int c           = n_tile * 32 + col_in_tile;
                        if (m_global < M && c < R) {
                            const int j_local = m_global / L_PAD;
                            const int b       = m_global % L_PAD;
                            const int j_global = h_start + j_local;
                            if (j_global < H && b < L) {
                                dC_acc[m_global][c] += acc[frag_idx];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }  // end chunk loop

    // ---- Phase 5: store dC_acc → dC_scratch[h_block, n_part_idx, L, R] (fp32) ----
    {
        constexpr int total_out = M * R;
        for (int idx = tid; idx < total_out; idx += blockDim.x) {
            const int m = idx / R;
            const int c = idx % R;
            const int j_local = m / L_PAD;
            const int b       = m % L_PAD;
            const int j_global = h_start + j_local;
            if (j_global < H && b < L) {
                // dC_scratch layout: [H, N_PART, L, R]
                const long out_idx =
                    ((long)j_global * N_PART + pid_npart) * L * R + b * R + c;
                dC_scratch[out_idx] = dC_acc[m][c];
            }
        }
    }

    #undef W_OFF
    #undef G_OFF
}

// =============================================================================
// Reduce kernel: dC_scratch[H, N_PART, L, R] fp32 → dC_out[H, L, R] bf16
// One thread per (h, b, c).  Sequential sum over N_PART.
// =============================================================================
__global__ void reduce_dC_scratch_kernel(
    const float* __restrict__ dC_scratch,
    __nv_bfloat16* __restrict__ dC_out,
    const int H, const int L, const int R, const int N_PART
) {
    const int total = H * L * R;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int h = tid / (L * R);
    const int rem = tid % (L * R);
    const int b = rem / R;
    const int c = rem % R;
    float sum = 0.0f;
    #pragma unroll 4
    for (int p = 0; p < N_PART; p++) {
        sum += dC_scratch[((long)h * N_PART + p) * L * R + b * R + c];
    }
    dC_out[h * L * R + b * R + c] = f2bf(sum);
}

#define LAUNCH_BWD_WGMMA_V3(BN, BH, LP, RR, NPP) \
    do { \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        const int blocks_n = (N + (NPP) - 1) / (NPP); \
        dim3 grid(blocks_h, blocks_n, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_wgmma_v3_kernel<BN, BH, LP, RR, NPP> \
            <<<grid, block, 0, stream>>>( \
                (const __nv_bfloat16*)z_ptr, \
                (const __nv_bfloat16*)C_ptr, \
                (const __nv_bfloat16*)g_delta_ptr, \
                dC_scratch_ptr, dz_ptr, \
                N, H, L, blocks_n, grid_lo, scale); \
        const int reduce_total = H * L * R; \
        const int reduce_block = 128; \
        const int reduce_grid  = (reduce_total + reduce_block - 1) / reduce_block; \
        reduce_dC_scratch_kernel<<<reduce_grid, reduce_block, 0, stream>>>( \
            dC_scratch_ptr, (__nv_bfloat16*)dC_ptr, H, L, RR, blocks_n); \
    } while(0)

}  // namespace

// =============================================================================
// PyTorch entry point.  Returns:
//   dC : [H, L, R] bf16
//   dz : [N, H]    fp32
// =============================================================================
std::vector<torch::Tensor> spline_kv_bwd_wgmma_v3_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    const torch::Tensor& g_delta,
    double grid_lo, double scale
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16
                && C.dtype() == torch::kBFloat16
                && g_delta.dtype() == torch::kBFloat16);
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous() && g_delta.is_contiguous());

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);

    auto bf16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat32);
    torch::Tensor dC = torch::empty({H, L, R}, bf16_opts);
    torch::Tensor dz = torch::empty({N, H}, fp32_opts);

    // Tunable: BLOCK_N=64 inner chunk, N_PER_PART=256 → 4 chunks/block,
    // grid_n = ceil(N/256). For N=2048 → 8 partitions.
    constexpr int N_PER_PART_DEFAULT = 256;
    const int n_part = (N + N_PER_PART_DEFAULT - 1) / N_PER_PART_DEFAULT;

    torch::Tensor dC_scratch = torch::empty({H, n_part, L, R}, fp32_opts);

    void* z_ptr        = z.data_ptr();
    void* C_ptr        = C.data_ptr();
    void* g_delta_ptr  = g_delta.data_ptr();
    void* dC_ptr       = dC.data_ptr();
    float* dC_scratch_ptr = dC_scratch.data_ptr<float>();
    float* dz_ptr      = dz.data_ptr<float>();

    if (R == 32 && L == 22) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 24, 32, 256);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 16, 32, 256);
    } else if (R == 32 && L == 32) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 32, 32, 256);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 24, 64, 256);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 16, 64, 256);
    } else if (R == 64 && L == 32) {
        LAUNCH_BWD_WGMMA_V3(64, 8, 32, 64, 256);
    } else {
        TORCH_CHECK(false, "v2: unsupported (R, L)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dC, dz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_wgmma_v3_cuda", &spline_kv_bwd_wgmma_v3_cuda,
          "FlashSplineFeature backward v2 (split-N + reduce, no dC atomic)",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("grid_lo"), py::arg("scale"));
}
