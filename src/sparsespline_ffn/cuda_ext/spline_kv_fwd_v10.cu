// Forward v10 — Dense-W wgmma kernel.
//
// PLAN_KERNEL_v10_DENSE_W.md §3.
//
// Math:
//   δ[n, c] = (W · C_flat)[n, c]
//   where W[n, m=j·L+b] = B_{b-bin_{n,j}}(τ_{n,j}) for b in {bin..bin+2} else 0
//   and C_flat = reshape(C, [H·L, R]).
//
// Grid: (N_TILE, H_PART)
//   - N_TILE = ceil(N, BLOCK_N=64); split-N for parallelism.
//   - H_PART = 8;                    split-H for occupancy.
//   Each block writes to dC_scratch[n_tile, h_part, BLOCK_N, R] (fp32, no atomic).
//   Reduce kernel sums h_part dim and casts to bf16.
//
// Block: 128 threads (1 warpgroup).
// BLOCK_H = 8 j's per chunk; CHUNKS_PER_PART = (H/H_PART) / BLOCK_H.
//
// Per chunk:
//   - cp.async load C[h_chunk] → C_smem (12 KB)
//   - per-thread compute bin/τ/B + fill W_smem (24 KB) — densified W
//   - wgmma m64n32k16 × M_TILES × K_TILES → +δ_acc (8 KB fp32)
//
// Equivalence: bit-equiv vs v1 within bf16 noise (§0.1 of PLAN_KERNEL_REWRITE_v9.md).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {

// =============================================================================
// Helpers (mirror v1 backward kernel).
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

// v10 wgmma variant — for fwd we compute δ = W · C (no transpose on A).
// Order of immediates: scaleA, scaleB, transA, transB.
// v1 bwd uses transA=1 (because it computes W^T · g via stored W).
// v10 fwd uses transA=0 (computes W · C directly using stored W).
__device__ __forceinline__ void wgmma_m64n32k16_NN(
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
        "%16, %17, p, 1, 1, 0, 1;\n\t"
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

// =============================================================================
// Forward v10 main kernel.
//
// Computes one tile of δ_partial = W · C_flat for (n_tile, h_part).
// Result written to dC_scratch[n_tile, h_part, BLOCK_N, R] in fp32.
// =============================================================================
template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int L, int H_PART>
__global__ void __launch_bounds__(128, 1)
spline_kv_fwd_v10_kernel(
    const __nv_bfloat16* __restrict__ z,            // [N, H]
    const __nv_bfloat16* __restrict__ C,            // [H, L, R]
    float* __restrict__ delta_scratch,              // [N_TILE, H_PART, BLOCK_N, R] fp32
    const int N, const int H,
    const float grid_lo, const float scale, const float G_max
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;       // wgmma M-dim chunks
    constexpr int N_TILES = R / 32;       // wgmma N-dim (= 1 for R=32)
    constexpr int K_TILES = M / 16;       // wgmma K-dim chunks per matmul = 12

    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;
    constexpr int K_CORES = BLOCK_N / 8;
    static_assert(BLOCK_N % 64 == 0, "BLOCK_N must be multiple of 64 (wgmma M-dim)");
    static_assert(R % 32 == 0,        "R must be multiple of 32");

    const int pid_n   = blockIdx.x;
    const int pid_hp  = blockIdx.y;
    const int n_start = pid_n * BLOCK_N;
    const int H_per_part = H / H_PART;
    const int h_part_start = pid_hp * H_per_part;
    const int CHUNKS = H_per_part / BLOCK_H;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // SMEM
    __shared__ __align__(128) __nv_bfloat16 W_smem[K_CORES * M_CORES * 64];   // [BLOCK_N, M] core-major
    __shared__ __align__(128) __nv_bfloat16 C_smem[BLOCK_H * L_PAD * R];     // flat [BH, L_PAD, R]
    __shared__ float delta_acc[BLOCK_N * R];                                    // [BLOCK_N, R] fp32
    __shared__ __nv_bfloat16 z_chunk[BLOCK_N * BLOCK_H];                       // [BLOCK_N, BLOCK_H]

    // wgmma A=W: [M_wgmma=BLOCK_N=64, K_wgmma=M_W=192].  K_dim is the
    //   m_global axis of stored W.  In W_OFF, m_in is FAST → K is contiguous
    //   within core → K-MAJOR A.
    //   For K-major A no-swizzle:
    //     LBO_A = stride between adjacent M-cores (along k_core in our terms)
    //           = M_CORES * 128 bytes  ... wait, actually for K-major:
    //   Per Colfax: K-major A: ((8,m),(T,2)):((1T,SBO),(1,LBO))
    //   LBO advances K dim cores; SBO advances M dim cores.
    //   In our W: K-cores along m_global, M-cores along n_local (BLOCK_N).
    //   K-core stride: 64 elements = 128 bytes (consecutive m_core).
    //   M-core stride: M_CORES*64 elements = M_CORES*128 bytes.
    constexpr uint32_t LBO_A = 128;            // K-major: between K-cores (m_core)
    constexpr uint32_t SBO_A = M_CORES * 128;  // K-major: between M-cores (k_core)
    // B=C_flat: [K_wgmma=M_W=192, N_wgmma=R=32].  In C_OFF, n_in is FAST → N
    // is contiguous within core → N-major B (same as v1 bwd's g).
    constexpr uint32_t LBO_B = N_CORES * 128;  // between K-cores (k_core)
    constexpr uint32_t SBO_B = 128;             // between N-cores (n_core)

    #define W_OFF(k, m)  (((k) >> 3) * M_CORES * 64 + ((m) >> 3) * 64 + ((k) & 7) * 8 + ((m) & 7))
    #define C_OFF(k, n)  (((k) >> 3) * N_CORES * 64 + ((n) >> 3) * 64 + ((k) & 7) * 8 + ((n) & 7))

    // ---- Phase 0a: zero δ_acc ----
    {
        constexpr int total = BLOCK_N * R;
        #pragma unroll
        for (int idx = tid; idx < total; idx += blockDim.x) {
            delta_acc[idx] = 0.0f;
        }
    }
    __syncthreads();

    // ---- Main loop: chunks within this h-partition ----
    #pragma unroll 1
    for (int chunk = 0; chunk < CHUNKS; chunk++) {
        const int h_start = h_part_start + chunk * BLOCK_H;

        // ---- Phase 1: zero W_smem (fresh each chunk; only 3 cols per (n,j) get written) ----
        {
            constexpr int total = K_CORES * M_CORES * 64;
            #pragma unroll
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                uint4* p = reinterpret_cast<uint4*>(&W_smem[idx]);
                *p = make_uint4(0, 0, 0, 0);
            }
        }

        // ---- Phase 2a: cp.async load C[h_start:h_start+BH, :, :] in core-major K-major layout ----
        // C is logically [BH, L, R] but for wgmma B we want core layout with K=m=j*L_PAD+b, n=c.
        // Layout: C_OFF(k = j*L_PAD+b, n = c).
        {
            // Each thread loads 8 bf16 (16 bytes) per cp.async.  Total elements = BH*L*R.
            // We zero-pad b ∈ [L, L_PAD) implicitly via cp_async_16 valid-mask path.
            constexpr int total = BLOCK_H * L_PAD * R;  // pad to L_PAD
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                const int k = idx / R;                                  // m_global = k = j*L_PAD+b
                const int c_start = idx % R;                              // n_start in core-major
                const int j_local = k / L_PAD;
                const int b       = k % L_PAD;
                const int j_global = h_start + j_local;
                const bool valid = (j_global < H) && (b < L);

                __nv_bfloat16* dst = &C_smem[C_OFF(k, c_start)];
                const __nv_bfloat16* src = valid
                    ? &C[j_global * L * R + b * R + c_start]
                    : nullptr;
                cp_async_16(dst, src, valid);
            }
        }

        // ---- Phase 2b: cooperative load z chunk into z_chunk ----
        {
            constexpr int total = BLOCK_N * BLOCK_H;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                const int n_local = idx / BLOCK_H;
                const int j_local = idx % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global < N && j_global < H) {
                    z_chunk[idx] = z[n_global * H + j_global];
                } else {
                    z_chunk[idx] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // ---- Phase 3: per-(n_local, j_local) compute bin/τ/B and fill W_smem ----
        // W layout: W_smem[W_OFF(k=n_local, m=j_local*L_PAD+bin)] = B_k(τ).
        {
            constexpr int total = BLOCK_N * BLOCK_H;
            for (int p = tid; p < total; p += blockDim.x) {
                const int n_local = p / BLOCK_H;
                const int j_local = p % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global >= N || j_global >= H) continue;

                const float z_val = bf2f(z_chunk[n_local * BLOCK_H + j_local]);
                const float u = (z_val - grid_lo) * scale;
                const bool in_range = (u >= 0.0f) && (u <= G_max);
                const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
                const int   bin_idx = (int)u_clip;
                const float tau = u_clip - (float)bin_idx;

                float B0, B1, B2;
                compute_B2(tau, B0, B1, B2);
                if (!in_range) { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }

                const int col_base = j_local * L_PAD + bin_idx;
                W_smem[W_OFF(n_local, col_base + 0)] = f2bf(B0);
                W_smem[W_OFF(n_local, col_base + 1)] = f2bf(B1);
                W_smem[W_OFF(n_local, col_base + 2)] = f2bf(B2);
            }
        }
        __syncthreads();
        fence_proxy_async_shared_cta();

        // ---- Phase 4: wgmma m64n32k16 W @ C → +δ_acc ----
        // δ_partial[n_local, c] = Σ_k W[n_local, k] · C[k, c]
        //
        // wgmma A is [M_dim=BLOCK_N=64, K_dim=K]: matches our W (k=n_local row).
        // Wait — wgmma m64nNk16 takes A:[M=64, K=16], B:[K=16, N].  Our W has shape
        // [BLOCK_N=64, M_dim=192].  In wgmma terms, our "M" (output row dim) = BLOCK_N,
        // our "K" (contraction) = M_dim of W = 192.
        //
        // Per wgmma op: A=W slice [64, 16], B=C slice [16, R=32], accumulating into
        // a 64x32 fp32 fragment.
        //
        // We need M_TILES of wgmma per accumulation slot (M_TILES=192/16=... wait no).
        // Reread: wgmma m64n32k16 has fixed M=64, N=32, K=16.
        // Our M=192 must be split into K_TILES = 192/16 = 12 wgmma operations,
        // all accumulating into the same 64x32 fragment.
        //
        // BLOCK_N=64 is one wgmma M.  R=32 is one wgmma N.
        // So we do 1 (M_TILE) × 1 (N_TILE) × 12 (K_TILE) = 12 wgmma ops, all accumulating.

        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;

        wgmma_fence();
        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            // A pointer: W_smem core at (k_core=0, m_core=k_tile*2)
            //   In core-major: linear = k_core*M_CORES*64 + m_core*64
            __nv_bfloat16* a_ptr = &W_smem[(k_tile * 2) * 64];
            // Wait, that's wrong.  Let me redo.
            //
            // W_smem layout: cores indexed as [k_core, m_core], k_core covers BLOCK_N rows,
            // m_core covers M cols.  W_smem[(k_core*M_CORES + m_core)*64 + offset].
            //
            // For wgmma A: M_dim=BLOCK_N (rows = k axis here), K_dim=K (cols = m axis here).
            // wgmma needs A as [M=64, K=16].  Our 64 rows = BLOCK_N (k_core=0..K_CORES=8).
            // Our 16 cols = m slice [k_tile*16, (k_tile+1)*16) = m_core=k_tile*2 and k_tile*2+1.
            //
            // SMEM descriptor: start_addr at k_core=0, m_core=k_tile*2.
            //   linear offset = (0*M_CORES + k_tile*2) * 64 = k_tile * 128 elements
            __nv_bfloat16* a_smem_start = &W_smem[k_tile * 2 * 64];
            uint64_t a_desc = encode_smem_desc(a_smem_start, LBO_A, SBO_A, 0);

            // B pointer: C_smem core at (k_core=k_tile*2, n_core=0)
            //   linear offset = (k_tile*2 * N_CORES + 0) * 64
            __nv_bfloat16* b_smem_start = &C_smem[k_tile * 2 * N_CORES * 64];
            uint64_t b_desc = encode_smem_desc(b_smem_start, LBO_B, SBO_B, 0);

            bool scale_d = (k_tile > 0);  // accumulate within chunk via wgmma; cross-chunk via SMEM
            wgmma_m64n32k16_NN(acc, a_desc, b_desc, scale_d);
        }
        wgmma_commit_group();
        wgmma_wait_group<0>();

        // ---- Phase 5: add wgmma fragment to delta_acc (single-writer, no atomic) ----
        // wgmma m64n32 fragment: 16 elements per thread, mapped as:
        //   chunk_e in 0..3, e in 0..3 → 16 elements
        //   row_in_warp = (e<2 ? groupID : groupID+8); col_in_chunk = tigid*2 + (e%2)
        //   row_in_tile = warp_id*16 + row_in_warp;  col_in_tile = chunk_e*8 + col_in_chunk
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
                const int n_local  = warp_id * 16 + row_in_warp;
                const int c        = chunk_e * 8 + col_in_chunk;
                if (n_local < BLOCK_N && c < R) {
                    // Accumulate ACROSS chunks.  Within-chunk accumulation is
                    // handled by wgmma's scale_d in the K_TILES loop.
                    delta_acc[n_local * R + c] += acc[frag_idx];
                }
            }
        }
        __syncthreads();
    }  // end chunk loop

    // ---- Phase 6: write δ_acc to dC_scratch[n_tile, h_part, :, :] ----
    {
        constexpr int total = BLOCK_N * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int n_local = idx / R;
            const int c       = idx % R;
            const int n_global = n_start + n_local;
            if (n_global < N) {
                const long out_idx = ((long)n_global * H_PART + pid_hp) * R + c;
                delta_scratch[out_idx] = delta_acc[idx];
            }
        }
    }
    #undef W_OFF
    #undef C_OFF
}

// =============================================================================
// Reduce kernel: dC_scratch[N, H_PART, R] fp32 → f[:, H:H+R] bf16 with λ scale.
// Also writes activation f[:, :H] = ReLU²(z).
// =============================================================================
__global__ void spline_kv_fwd_v10_finalize_kernel(
    const float* __restrict__ delta_scratch,        // [N, H_PART, R] fp32
    const __nv_bfloat16* __restrict__ z,             // [N, H]
    __nv_bfloat16* __restrict__ f,                    // [N, H+R] output
    const int N, const int H, const int R, const int H_PART, const int H_R,
    const float lambda_scale, const int activation_id
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_delta = N * R;
    if (idx < total_delta) {
        const int n = idx / R;
        const int c = idx % R;
        float sum = 0.0f;
        #pragma unroll 4
        for (int p = 0; p < H_PART; p++) {
            sum += delta_scratch[((long)n * H_PART + p) * R + c];
        }
        f[n * H_R + H + c] = __float2bfloat16(lambda_scale * sum);
    }
    // Concurrently: activation
    const int total_a = N * H;
    if (idx < total_a) {
        const int n = idx / H;
        const int j = idx % H;
        float zv = __bfloat162float(z[n * H + j]);
        float av = (activation_id == 0) ? ((zv > 0.0f) ? zv * zv : 0.0f) : zv;
        f[n * H_R + j] = __float2bfloat16(av);
    }
}

#define LAUNCH_FWD_V10(BN, BH, LP, RR, LL, HP) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        dim3 grid(blocks_n, (HP), 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_fwd_v10_kernel<BN, BH, LP, RR, LL, HP> \
            <<<grid, block, 0, stream>>>( \
                (const __nv_bfloat16*)z_ptr, \
                (const __nv_bfloat16*)C_ptr, \
                delta_scratch_ptr, \
                N, H, grid_lo, scale, G_max); \
        const int total = N * (H > R ? H : R); \
        const int finalize_grid = (total + 255) / 256; \
        spline_kv_fwd_v10_finalize_kernel<<<finalize_grid, 256, 0, stream>>>( \
            delta_scratch_ptr, \
            (const __nv_bfloat16*)z_ptr, \
            (__nv_bfloat16*)f_ptr, \
            N, H, RR, HP, H_R, lambda_scale, activation_id); \
    } while(0)

}  // namespace

// =============================================================================
// PyTorch entry point.  Returns f [N, H+R] bf16.
// =============================================================================
torch::Tensor spline_kv_fwd_v10_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    double grid_lo, double scale,
    double lambda_scale_d,
    int activation
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16);
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous());

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);
    const int H_R = H + R;
    const float G_max = (float)(L - 2);
    const float lambda_scale = (float)lambda_scale_d;
    const int activation_id = activation;

    auto bf16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat32);
    torch::Tensor f = torch::empty({N, H_R}, bf16_opts);

    constexpr int H_PART_DEFAULT = 8;
    torch::Tensor delta_scratch = torch::empty({N, H_PART_DEFAULT, R}, fp32_opts);

    void* z_ptr = z.data_ptr();
    void* C_ptr = C.data_ptr();
    void* f_ptr = f.data_ptr();
    float* delta_scratch_ptr = delta_scratch.data_ptr<float>();

    if (R == 32 && L == 22) {
        LAUNCH_FWD_V10(64, 8, 24, 32, 22, 8);
    } else if (R == 32 && L == 16) {
        LAUNCH_FWD_V10(64, 8, 16, 32, 16, 8);
    } else if (R == 32 && L == 32) {
        LAUNCH_FWD_V10(64, 8, 32, 32, 32, 8);
    } else if (R == 64 && L == 22) {
        LAUNCH_FWD_V10(64, 8, 24, 64, 22, 8);
    } else {
        TORCH_CHECK(false, "spline_kv_fwd_v10: unsupported (R, L)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_fwd_v10_cuda", &spline_kv_fwd_v10_cuda,
          "Spline-KV forward v10 (Hopper-aligned, dense-W wgmma)",
          py::arg("z"), py::arg("C"), py::arg("grid_lo"), py::arg("scale"),
          py::arg("lambda_scale"), py::arg("activation"));
}
