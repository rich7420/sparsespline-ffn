// Forward v11 — Dense-W wgmma kernel with fp16 inputs (precision fix for v10).
//
// v10 stored W = bf16(B-spline coefficients) before wgmma, losing 9 bits of
// mantissa precision in B at every (n, j).  Microbench (docs/RESULTS_2026-
// 05-02_v10_numerical_bug.md) showed v10 had 4-5x worse rel_err vs triton/v1
// (0.5% vs 0.1%) which compounded to a +0.24 nat val_loss regression at 100M.
//
// v11 fix: switch to wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16
//   - W_smem stores fp16(B) instead of bf16(B) — gains 3 mantissa bits
//   - C must be fp16 (caller casts bf16 → fp16 in Python wrapper; cheap and
//     exact for |C| < 65504)
//   - Same SMEM footprint, same wgmma throughput (f16/bf16 both 989 TFLOP/s)
//
// Expected precision: max_rel_err ~0.1% (down from 0.5%); training-equivalent.
// Expected speed: identical to v10 (1.59x over v1).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {

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

// f32.f16.f16 variant — same fragment layout as bf16 (same M, N, K, accumulator).
__device__ __forceinline__ void wgmma_m64n32k16_f16_NN(
    float* acc, uint64_t a_desc, uint64_t b_desc, bool scale_d
) {
    int sd = scale_d ? 1 : 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %18, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
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

__device__ __forceinline__ void cp_async_16(__half* dst,
                                              const __half* src,
                                              bool valid) {
    unsigned d = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __float2half(0.0f);
    }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __half f2h(float x) { return __float2half(x); }

__device__ __forceinline__ void compute_B2(float tau, float& B0, float& B1, float& B2) {
    float omt = 1.0f - tau;
    B0 = 0.5f * omt * omt;
    B1 = 0.5f * (1.0f + 2.0f * tau - 2.0f * tau * tau);
    B2 = 0.5f * tau * tau;
}

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int L, int H_PART>
__global__ void __launch_bounds__(128, 1)
spline_kv_fwd_v11_kernel(
    const __nv_bfloat16* __restrict__ z,            // [N, H]   (bf16 unchanged)
    const __half* __restrict__ C,                    // [H, L, R] (fp16 input)
    float* __restrict__ delta_scratch,              // [N, H_PART, R] fp32
    const int N, const int H,
    const float grid_lo, const float scale, const float G_max
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;
    constexpr int N_TILES = R / 32;
    constexpr int K_TILES = M / 16;

    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;
    constexpr int K_CORES = BLOCK_N / 8;
    static_assert(BLOCK_N % 64 == 0, "BLOCK_N must be multiple of 64");
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
    __shared__ __align__(128) __half W_smem[K_CORES * M_CORES * 64];   // fp16
    __shared__ __align__(128) __half C_smem[BLOCK_H * L_PAD * R];       // fp16
    __shared__ float delta_acc[BLOCK_N * R];
    __shared__ __nv_bfloat16 z_chunk[BLOCK_N * BLOCK_H];

    constexpr uint32_t LBO_A = 128;
    constexpr uint32_t SBO_A = M_CORES * 128;
    constexpr uint32_t LBO_B = N_CORES * 128;
    constexpr uint32_t SBO_B = 128;

    #define W_OFF(k, m)  (((k) >> 3) * M_CORES * 64 + ((m) >> 3) * 64 + ((k) & 7) * 8 + ((m) & 7))
    #define C_OFF(k, n)  (((k) >> 3) * N_CORES * 64 + ((n) >> 3) * 64 + ((k) & 7) * 8 + ((n) & 7))

    // Phase 0: zero delta_acc
    {
        constexpr int total = BLOCK_N * R;
        #pragma unroll
        for (int idx = tid; idx < total; idx += blockDim.x) {
            delta_acc[idx] = 0.0f;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int chunk = 0; chunk < CHUNKS; chunk++) {
        const int h_start = h_part_start + chunk * BLOCK_H;

        // Phase 1: zero W_smem
        {
            constexpr int total = K_CORES * M_CORES * 64;
            #pragma unroll
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                uint4* p = reinterpret_cast<uint4*>(&W_smem[idx]);
                *p = make_uint4(0, 0, 0, 0);
            }
        }

        // Phase 2a: cp.async load C[h_start:h_start+BH, :, :]
        {
            constexpr int total = BLOCK_H * L_PAD * R;
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                const int k = idx / R;
                const int c_start = idx % R;
                const int j_local = k / L_PAD;
                const int b       = k % L_PAD;
                const int j_global = h_start + j_local;
                const bool valid = (j_global < H) && (b < L);
                __half* dst = &C_smem[C_OFF(k, c_start)];
                const __half* src = valid
                    ? &C[j_global * L * R + b * R + c_start]
                    : nullptr;
                cp_async_16(dst, src, valid);
            }
        }

        // Phase 2b: load z chunk
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

        // Phase 3: compute B → fp16, fill W_smem
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
                W_smem[W_OFF(n_local, col_base + 0)] = f2h(B0);
                W_smem[W_OFF(n_local, col_base + 1)] = f2h(B1);
                W_smem[W_OFF(n_local, col_base + 2)] = f2h(B2);
            }
        }
        __syncthreads();
        fence_proxy_async_shared_cta();

        // Phase 4: wgmma f32.f16.f16
        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;

        wgmma_fence();
        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            __half* a_smem_start = &W_smem[k_tile * 2 * 64];
            uint64_t a_desc = encode_smem_desc(a_smem_start, LBO_A, SBO_A, 0);
            __half* b_smem_start = &C_smem[k_tile * 2 * N_CORES * 64];
            uint64_t b_desc = encode_smem_desc(b_smem_start, LBO_B, SBO_B, 0);
            bool scale_d = (k_tile > 0);
            wgmma_m64n32k16_f16_NN(acc, a_desc, b_desc, scale_d);
        }
        wgmma_commit_group();
        wgmma_wait_group<0>();

        // Phase 5: accumulate fragment to delta_acc (fp32 SMEM)
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
                    delta_acc[n_local * R + c] += acc[frag_idx];
                }
            }
        }
        __syncthreads();
    }

    // Phase 6: write delta_acc to delta_scratch[n, p, c]
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

// Finalize: same as v10 — sum across H_PART, write delta + activation
__global__ void spline_kv_fwd_v11_finalize_kernel(
    const float* __restrict__ delta_scratch,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ f,
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
    const int total_a = N * H;
    if (idx < total_a) {
        const int n = idx / H;
        const int j = idx % H;
        float zv = __bfloat162float(z[n * H + j]);
        float av = (activation_id == 0) ? ((zv > 0.0f) ? zv * zv : 0.0f) : zv;
        f[n * H_R + j] = __float2bfloat16(av);
    }
}

#define LAUNCH_FWD_V11(BN, BH, LP, RR, LL, HP) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        dim3 grid(blocks_n, (HP), 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_fwd_v11_kernel<BN, BH, LP, RR, LL, HP> \
            <<<grid, block, 0, stream>>>( \
                (const __nv_bfloat16*)z_ptr, \
                (const __half*)C_ptr, \
                delta_scratch_ptr, \
                N, H, grid_lo, scale, G_max); \
        const int total = N * (H > R ? H : R); \
        const int finalize_grid = (total + 255) / 256; \
        spline_kv_fwd_v11_finalize_kernel<<<finalize_grid, 256, 0, stream>>>( \
            delta_scratch_ptr, \
            (const __nv_bfloat16*)z_ptr, \
            (__nv_bfloat16*)f_ptr, \
            N, H, RR, HP, H_R, lambda_scale, activation_id); \
    } while(0)

}  // namespace


// =============================================================================
// PyTorch entry — z stays bf16, C must be fp16 (caller responsibility).
// =============================================================================
torch::Tensor spline_kv_fwd_v11_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    double grid_lo, double scale,
    double lambda_scale_d,
    int activation
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(z.dtype() == torch::kBFloat16, "z must be bf16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be fp16 for v11");
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous(), "inputs must be contiguous");

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
        LAUNCH_FWD_V11(64, 8, 24, 32, 22, 8);
    } else if (R == 32 && L == 16) {
        LAUNCH_FWD_V11(64, 8, 16, 32, 16, 8);
    } else if (R == 32 && L == 32) {
        LAUNCH_FWD_V11(64, 8, 32, 32, 32, 8);
    } else if (R == 64 && L == 22) {
        LAUNCH_FWD_V11(64, 8, 24, 64, 22, 8);
    } else {
        TORCH_CHECK(false, "spline_kv_fwd_v11: unsupported (R, L)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_fwd_v11_cuda", &spline_kv_fwd_v11_cuda,
          "Spline-KV forward v11 (fp16 wgmma, precision-corrected v10)",
          py::arg("z"), py::arg("C"), py::arg("grid_lo"), py::arg("scale"),
          py::arg("lambda_scale"), py::arg("activation"));
}
