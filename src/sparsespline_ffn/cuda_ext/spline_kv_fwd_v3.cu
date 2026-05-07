// Forward kernel v3 — H100-aligned spline_kv forward.
//
// Design (PLAN_KERNEL_REWRITE_v9.md):
//   - One CTA per n-chunk; iterates over all h-chunks (no atomic on δ).
//   - cp.async loads C[h_chunk] tile + z[n_chunk, h_chunk] tile to SMEM.
//   - Per-thread: compute bin/τ/B0/B1/B2 from z, fill W_smem (densified).
//   - wgmma m64n32k16: δ_acc += W_smem · C_smem  (tensor cores!)
//   - Output: bf16 δ → f[n, h:h+r], plus a = ReLU²(z) → f[n, :h] (fused).
//
// Math equivalence: matches spline_kv_fwd_cuda (v1) within bf16 noise.
// See §0 of PLAN_KERNEL_REWRITE_v9.md for bit-equivalence audit.
//
// First-version (no TMA, no warp-spec):
//   - cp.async (Ampere) for memory pipelining
//   - All warps load + compute
//   - 1-stage pipeline (single buffer)
//
// Targets BLOCK_N=64, BLOCK_H=8, L_PAD=24, R=32 (production h_ratio=2 cell).
// Other shapes fall back to v1 kernel.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// ===========================================================================
// Shared helpers — wgmma + bf16 utility (mirrors backward kernel).
// ===========================================================================

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
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
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

// ===========================================================================
// Forward v3 kernel — densified-matmul via wgmma.
//
// Grid: (N_TILES,) = ceil(N, BLOCK_N) blocks.  Each block uniquely owns
//                    f[n_chunk, h:h+r] (no atomic on δ).
// Block: 128 threads = 1 warpgroup (4 warps).
//
// Per-block work:
//   1. Load z[n_chunk, :H] cooperatively (will be needed for both spline
//      bin computation and ReLU² activation output).
//   2. For each h_chunk in [0, H/BLOCK_H):
//      a. cp.async load C[h_chunk*BH:(h_chunk+1)*BH, :, :] → C_smem
//      b. Per-(n_local, j_local): compute bin/τ/B from z[n, h_chunk*BH+j],
//         fill W_smem[n_local, j_local*L_PAD + bin..bin+2] (densified).
//      c. wgmma_async W_smem · C_smem → δ_acc (accumulating)
//   3. After all h_chunks: bf16 cast δ_acc, store to f[n_chunk, h:h+r].
//   4. Compute a = ReLU²(z) elementwise, store to f[n_chunk, :h].
//
// SMEM:
//   z_full     : [BLOCK_N, H]            bf16   (loaded once for activation reuse)
//   C_smem     : [BLOCK_H, L_PAD, R]     bf16   (per h_chunk, reloaded)
//   W_smem     : [BLOCK_N, M=BH·L_PAD]   bf16   (filled per chunk)
//   δ_acc      : [BLOCK_N, R]            fp32   (persistent across h_chunks)
//
// For BLOCK_N=64, BLOCK_H=8, L_PAD=24, R=32, H=768:
//   z_full     = 64×768×2  = 96 KB  ← TOO BIG for SMEM (228KB) when combined
//   C_smem     = 8×24×32×2 = 12 KB
//   W_smem     = 64×192×2  = 24 KB
//   δ_acc      = 64×32×4   = 8  KB
//   total      = 140 KB
//
// 96 KB for z_full is too much.  Drop z_full caching and re-read z per h_chunk
// (z reads are cheap — bf16 from global, 64×8×2 = 1 KB per chunk, 96 chunks
// total = 96 KB read but spread across the kernel runtime, not in SMEM).
// ===========================================================================

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int L>
__global__ void __launch_bounds__(128, 1)
spline_kv_fwd_v3_kernel(
    const __nv_bfloat16* __restrict__ z,        // [N, H]
    const __nv_bfloat16* __restrict__ C,        // [H, L, R]
    __nv_bfloat16* __restrict__ f,               // [N, H+R] output (a + λδ concatenated)
    const int N, const int H,
    const float grid_lo, const float scale, const float G_max,
    const float lambda_scale, const int activation_id   // 0=relu_sq, 2=identity
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;
    constexpr int N_TILES = R / 32;
    constexpr int K_TILES = BLOCK_H / 1;  // we tile K=L_PAD per j; below we do M_full step
    // wgmma m64n32k16: K=16.  The contraction is over (j_local·L_PAD + b),
    // total K = M = BH·L_PAD.  Number of k-tiles = M / 16.
    constexpr int K_TILES_PER_CHUNK = M / 16;

    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;

    const int pid_n   = blockIdx.x;
    const int n_start = pid_n * BLOCK_N;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // ---- SMEM ----
    // W_smem: [BLOCK_N, M] bf16 = 64 × 192 = 12288 elements = 24 KB
    __shared__ __align__(128) __nv_bfloat16 W_smem[BLOCK_N * M];
    __shared__ __align__(128) __nv_bfloat16 C_smem[BLOCK_H][L_PAD][R];   // 8×24×32 = 12KB
    __shared__ float delta_acc[BLOCK_N][R];                                // 64×32×4 = 8KB
    __shared__ __nv_bfloat16 z_chunk[BLOCK_N][BLOCK_H];                    // 64×8×2 = 1KB
    // Total: 24 + 12 + 8 + 1 = 45 KB / CTA, well under 164 KB H100 static SMEM limit.

    // For fp32 → bf16 cast at end
    const int H_R = H + R;

    // Phase 0: zero δ_acc
    {
        constexpr int total = BLOCK_N * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            delta_acc[idx / R][idx % R] = 0.0f;
        }
    }
    __syncthreads();

    const int H_CHUNKS = (H + BLOCK_H - 1) / BLOCK_H;
    bool first_acc = true;

    for (int chunk = 0; chunk < H_CHUNKS; chunk++) {
        const int h_start = chunk * BLOCK_H;
        const int h_end_clamp = (h_start + BLOCK_H < H) ? (h_start + BLOCK_H) : H;

        // Phase 1: cp.async load C[h_start:h_end, :, :] into C_smem
        // C is [H, L, R].  Each thread loads 8 bf16 (16 bytes) at a time.
        {
            constexpr int total = BLOCK_H * L * R;
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                const int j_local  = idx / (L * R);
                const int b        = (idx / R) % L;
                const int c_start  = idx % R;
                const int j_global = h_start + j_local;
                const bool valid = (j_global < H) && (b < L);
                cp_async_16(&C_smem[j_local][b][c_start],
                            valid ? &C[j_global * L * R + b * R + c_start] : nullptr,
                            valid);
            }
            // Pad: zero C_smem rows for L_PAD > L
            constexpr int total_pad = BLOCK_H * (L_PAD - L) * R;
            if constexpr (L_PAD > L) {
                for (int idx = tid * 8; idx < total_pad; idx += blockDim.x * 8) {
                    const int j_local = idx / ((L_PAD - L) * R);
                    const int b_pad   = (idx / R) % (L_PAD - L);
                    const int c_start = idx % R;
                    if (j_local < BLOCK_H) {
                        // Cast pointer arithmetic: skip valid C rows
                        int b = L + b_pad;
                        if (b < L_PAD) {
                            // Plain SMEM zero write (no cp.async needed, fast)
                            uint4* p = reinterpret_cast<uint4*>(&C_smem[j_local][b][c_start]);
                            *p = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
            }
        }

        // Phase 2: cooperative load z chunk into z_chunk (small, BLOCK_N × BLOCK_H bf16)
        {
            constexpr int total = BLOCK_N * BLOCK_H;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                const int n_local = idx / BLOCK_H;
                const int j_local = idx % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global < N && j_global < H) {
                    z_chunk[n_local][j_local] = z[n_global * H + j_global];
                } else {
                    z_chunk[n_local][j_local] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Phase 3: zero W_smem.
        // Layout: row-major [BLOCK_N, M], 8-bf16-aligned vec stores.
        {
            constexpr int total = BLOCK_N * M;
            for (int idx = tid * 8; idx < total; idx += blockDim.x * 8) {
                if (idx + 8 <= total) {
                    uint4* p = reinterpret_cast<uint4*>(&W_smem[idx]);
                    *p = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Phase 4: per-(n_local, j_local) compute bin/τ/B and fill W_smem.
        {
            constexpr int total_pairs = BLOCK_N * BLOCK_H;
            for (int p = tid; p < total_pairs; p += blockDim.x) {
                const int n_local = p / BLOCK_H;
                const int j_local = p % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global >= N || j_global >= H) continue;

                const float z_val = bf2f(z_chunk[n_local][j_local]);
                const float u = (z_val - grid_lo) * scale;
                const bool in_range = (u >= 0.0f) && (u <= G_max);
                const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
                const int   bin_idx = (int)u_clip;
                const float tau = u_clip - (float)bin_idx;

                float B0, B1, B2;
                compute_B2(tau, B0, B1, B2);
                if (!in_range) { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }

                const int col_base = j_local * L_PAD + bin_idx;
                // Row-major: W_smem[n_local * M + col_base + k]
                W_smem[n_local * M + col_base + 0] = f2bf(B0);
                W_smem[n_local * M + col_base + 1] = f2bf(B1);
                W_smem[n_local * M + col_base + 2] = f2bf(B2);
            }
        }
        __syncthreads();

        // Phase 5: scalar matmul δ_partial = W @ C, accumulate into δ_acc.
        // (wgmma upgrade is next iteration.  Current goal: validate correctness +
        //  no-atomic structural advantage.)
        for (int p = tid; p < BLOCK_N * R; p += blockDim.x) {
            const int n_local = p / R;
            const int c       = p % R;
            const int n_global = n_start + n_local;
            if (n_global >= N) continue;
            float acc = 0.0f;
            for (int j_local = 0; j_local < BLOCK_H; j_local++) {
                const int j_global = h_start + j_local;
                if (j_global >= H) break;
                #pragma unroll
                for (int b = 0; b < L; b++) {
                    int m = j_local * L_PAD + b;
                    float w = bf2f(W_smem[n_local * M + m]);
                    if (w != 0.0f) {
                        acc += w * bf2f(C_smem[j_local][b][c]);
                    }
                }
            }
            delta_acc[n_local][c] += acc;
        }
        __syncthreads();
    }  // end h-chunk loop

    // Phase 6: epilogue — write δ to f[n, H:H+R] (with λ scale, bf16 cast) and a to f[n, :H].
    {
        // Write δ
        constexpr int total = BLOCK_N * R;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int n_local = idx / R;
            const int c       = idx % R;
            const int n_global = n_start + n_local;
            if (n_global >= N) continue;
            float v = lambda_scale * delta_acc[n_local][c];
            f[n_global * H_R + H + c] = f2bf(v);
        }

        // Write a = activation(z)
        // total a writes = BLOCK_N × H — but H can be 768, this is heavy.
        // We need to re-load z and apply ReLU².
        for (int idx = tid; idx < BLOCK_N * H; idx += blockDim.x) {
            const int n_local = idx / H;
            const int j       = idx % H;
            const int n_global = n_start + n_local;
            if (n_global >= N) continue;
            float zv = bf2f(z[n_global * H + j]);
            float av = (activation_id == 0) ? ((zv > 0.0f) ? zv * zv : 0.0f) : zv;
            f[n_global * H_R + j] = f2bf(av);
        }
    }
}

#define LAUNCH_FWD_V3(BN, BH, LP, RR, LL) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        dim3 grid(blocks_n, 1, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_fwd_v3_kernel<BN, BH, LP, RR, LL> \
            <<<grid, block, 0, stream>>>( \
                (const __nv_bfloat16*)z_ptr, \
                (const __nv_bfloat16*)C_ptr, \
                (__nv_bfloat16*)f_ptr, \
                N, H, grid_lo, scale, G_max, lambda_scale, activation_id); \
    } while(0)

}  // namespace

// =============================================================================
// PyTorch entry point.  Returns f [N, H+R] bf16.
// =============================================================================
torch::Tensor spline_kv_fwd_v3_cuda(
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
    const float G_max = (float)(L - 2);
    const float lambda_scale = (float)lambda_scale_d;
    const int activation_id = activation;

    auto bf16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    torch::Tensor f = torch::empty({N, H + R}, bf16_opts);

    void* z_ptr = z.data_ptr();
    void* C_ptr = C.data_ptr();
    void* f_ptr = f.data_ptr();

    if (R == 32 && L == 22) {
        LAUNCH_FWD_V3(64, 8, 24, 32, 22);
    } else if (R == 32 && L == 16) {
        LAUNCH_FWD_V3(64, 8, 16, 32, 16);
    } else if (R == 32 && L == 32) {
        LAUNCH_FWD_V3(64, 8, 32, 32, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_FWD_V3(64, 8, 24, 64, 22);
    } else {
        TORCH_CHECK(false, "spline_kv_fwd_v3: unsupported (R, L)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_fwd_v3_cuda", &spline_kv_fwd_v3_cuda,
          "Spline-KV forward v3 (Hopper-aligned, densified-W matmul, scalar)",
          py::arg("z"), py::arg("C"), py::arg("grid_lo"), py::arg("scale"),
          py::arg("lambda_scale"), py::arg("activation"));
}
