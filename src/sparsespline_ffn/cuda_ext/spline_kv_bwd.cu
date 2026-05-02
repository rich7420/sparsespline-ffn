// FlashSplineFeature backward kernel — CUDA C++ implementation.
//
// Goal: minimize step time on H100 by exploiting:
//   1. SMEM cache for C[h_chunk, :, :] tile (per-token gather → SMEM read,
//      ~5 cycles vs L1 ~20-50 cycles)
//   2. SMEM accumulator for dC (block-local atomicAdd in SMEM, ~10× faster
//      than global atomicAdd on H100)
//   3. ONE global atomicAdd per (j, b, c) cell at the end (vs N global
//      atomicAdds in the Triton baseline)
//
// Memory: in-place atomicAdd to caller-allocated dC, dz buffers — no
// partial buffer, no extra VRAM beyond inputs/outputs.
//
// Numerical: fp32 internal accumulator throughout; bf16 only on inputs
// (z, C, g_delta).  Outputs (dC, dz) are fp32.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// B-spline B2 basis values evaluated at τ ∈ [0, 1].
//   B0(τ) = 0.5 (1-τ)²
//   B1(τ) = 0.5 (1 + 2τ - 2τ²)
//   B2(τ) = 0.5 τ²
__device__ __forceinline__ void compute_B2(float tau, float& B0, float& B1, float& B2) {
    float omt = 1.0f - tau;
    B0 = 0.5f * omt * omt;
    B1 = 0.5f * (1.0f + 2.0f * tau - 2.0f * tau * tau);
    B2 = 0.5f * tau * tau;
}

// B-spline B2 basis derivatives w.r.t. τ.
//   B0'(τ) = -(1-τ),  B1'(τ) = 1 - 2τ,  B2'(τ) = τ
__device__ __forceinline__ void compute_dB2(float tau, float& dB0, float& dB1, float& dB2) {
    dB0 = -(1.0f - tau);
    dB1 = 1.0f - 2.0f * tau;
    dB2 = tau;
}

// Backward kernel.
//
// Grid: (cdiv(N, BLOCK_N), cdiv(H, BLOCK_H))
// Each block:
//   1. Loads g_delta[n_chunk, :] and C[j_chunk, :, :] into SMEM
//   2. Initializes dC_local[BLOCK_H, L_PAD, R] = 0 in SMEM
//   3. Each thread processes a subset of (n_local, j_local) pairs:
//      - Computes B/dB/bin/τ from z[n_global, j_global]
//      - dz path: gather 3 rows from C_tile (SMEM), inner-product with
//        g_delta_tile (SMEM), write fp32 result to global dz
//      - dC path: 3 SMEM atomicAdd's into dC_local at (j_local, bin+k, c)
//   4. Cooperative bulk atomicAdd from SMEM dC_local to global dC.
//
// Templated for compile-time L_PAD, R, BLOCK_N, BLOCK_H.
template <int BLOCK_N, int BLOCK_H, int L_PAD, int R>
__global__ void spline_kv_bwd_kernel(
    const __nv_bfloat16* __restrict__ z,        // [N, H]
    const __nv_bfloat16* __restrict__ C,        // [H, L, R] (L can be < L_PAD)
    const __nv_bfloat16* __restrict__ g_delta,  // [N, R]
    float* __restrict__ dC,                      // [H, L, R] fp32 (zero-init)
    float* __restrict__ dz,                      // [N, H] fp32 (zero-init)
    const int N, const int H, const int L,
    const float grid_lo, const float scale
) {
    const int pid_n = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int n_start = pid_n * BLOCK_N;
    const int h_start = pid_h * BLOCK_H;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // SMEM layout (static; total = BLOCK_H*L_PAD*R*(4+2) + BLOCK_N*R*2 bytes)
    __shared__ float dC_local[BLOCK_H][L_PAD][R];
    __shared__ __nv_bfloat16 g_delta_tile[BLOCK_N][R];
    __shared__ __nv_bfloat16 C_tile[BLOCK_H][L_PAD][R];

    // ---- Phase 1: Initialize dC_local to zero ----
    {
        float* dC_flat = &dC_local[0][0][0];
        const int total = BLOCK_H * L_PAD * R;
        for (int idx = tid; idx < total; idx += block_size) {
            dC_flat[idx] = 0.0f;
        }
    }

    // ---- Phase 2a: Load g_delta tile [BLOCK_N, R] ----
    {
        __nv_bfloat16* g_flat = &g_delta_tile[0][0];
        const int total = BLOCK_N * R;
        for (int idx = tid; idx < total; idx += block_size) {
            const int n_local = idx / R;
            const int c = idx % R;
            const int n_global = n_start + n_local;
            __nv_bfloat16 v;
            if (n_global < N) {
                v = g_delta[n_global * R + c];
            } else {
                v = __float2bfloat16(0.0f);
            }
            g_flat[idx] = v;
        }
    }

    // ---- Phase 2b: Load C tile [BLOCK_H, L_PAD, R] ----
    {
        __nv_bfloat16* c_flat = &C_tile[0][0][0];
        const int total = BLOCK_H * L_PAD * R;
        for (int idx = tid; idx < total; idx += block_size) {
            const int j_local = idx / (L_PAD * R);
            const int b = (idx / R) % L_PAD;
            const int c = idx % R;
            const int j_global = h_start + j_local;
            __nv_bfloat16 v;
            if (j_global < H && b < L) {
                v = C[j_global * L * R + b * R + c];
            } else {
                v = __float2bfloat16(0.0f);
            }
            c_flat[idx] = v;
        }
    }

    __syncthreads();

    // ---- Phase 3: Process (n, j) pairs ----
    // Each thread takes a subset of (BLOCK_N * BLOCK_H) pairs.
    {
        const int total_pairs = BLOCK_N * BLOCK_H;
        for (int p = tid; p < total_pairs; p += block_size) {
            const int n_local = p / BLOCK_H;
            const int j_local = p % BLOCK_H;
            const int n_global = n_start + n_local;
            const int j_global = h_start + j_local;
            if (n_global >= N || j_global >= H) continue;

            const float z_val = bf2f(z[n_global * H + j_global]);
            const float u = (z_val - grid_lo) * scale;
            const float G_max = (float)(L - 2);  // L = G + 2
            const bool in_range    = (u >= 0.0f) && (u <= G_max);
            const bool clamp_active = (u >= 0.0f) && (u <= G_max - 1.0f);
            const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
            const int   bin_idx = (int)u_clip;
            const float tau = u_clip - (float)bin_idx;

            float B0, B1, B2, dB0, dB1, dB2;
            compute_B2(tau, B0, B1, B2);
            compute_dB2(tau, dB0, dB1, dB2);
            if (!in_range)    { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }
            if (!clamp_active){ dB0 = 0.0f; dB1 = 0.0f; dB2 = 0.0f; }

            // ----- dz path -----
            // dz[n, j] = scale * sum_c g_delta[n, c]
            //              * (dB0 * C[j, bin, c] + dB1 * C[j, bin+1, c]
            //                 + dB2 * C[j, bin+2, c])
            float inner0 = 0.0f, inner1 = 0.0f, inner2 = 0.0f;
            #pragma unroll
            for (int c = 0; c < R; c++) {
                const float g = bf2f(g_delta_tile[n_local][c]);
                inner0 += g * bf2f(C_tile[j_local][bin_idx + 0][c]);
                inner1 += g * bf2f(C_tile[j_local][bin_idx + 1][c]);
                inner2 += g * bf2f(C_tile[j_local][bin_idx + 2][c]);
            }
            const float dz_val = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2);
            // dz output is unique per (n_global, j_global) — single writer.
            dz[n_global * H + j_global] = dz_val;

            // ----- dC path: 3 atomic_add's into SMEM accumulator -----
            // dC[j, bin+k, c] += B_k * g_delta[n, c]
            #pragma unroll
            for (int c = 0; c < R; c++) {
                const float g = bf2f(g_delta_tile[n_local][c]);
                atomicAdd(&dC_local[j_local][bin_idx + 0][c], B0 * g);
                atomicAdd(&dC_local[j_local][bin_idx + 1][c], B1 * g);
                atomicAdd(&dC_local[j_local][bin_idx + 2][c], B2 * g);
            }
        }
    }

    __syncthreads();

    // ---- Phase 4: Cooperative bulk atomicAdd to global dC ----
    {
        const int total_dC = BLOCK_H * L_PAD * R;
        for (int idx = tid; idx < total_dC; idx += block_size) {
            const int j_local = idx / (L_PAD * R);
            const int b = (idx / R) % L_PAD;
            const int c = idx % R;
            const int j_global = h_start + j_local;
            if (j_global < H && b < L) {
                const float val = dC_local[j_local][b][c];
                if (val != 0.0f) {
                    atomicAdd(&dC[j_global * L * R + b * R + c], val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher: handles different (L_PAD, R, BLOCK_N, BLOCK_H) combinations.
// Keep it small — only the production-relevant shapes for now.

#define LAUNCH_BWD(BN, BH, LP, RR) \
    do { \
        const int blocks_n = (N + (BN) - 1) / (BN); \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        dim3 grid(blocks_n, blocks_h, 1); \
        dim3 block(128, 1, 1); /* 4 warps per block */ \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_kernel<BN, BH, LP, RR><<<grid, block, 0, stream>>>( \
            (const __nv_bfloat16*)z_ptr, \
            (const __nv_bfloat16*)C_ptr, \
            (const __nv_bfloat16*)g_delta_ptr, \
            dC_ptr, dz_ptr, \
            N, H, L, grid_lo, scale); \
    } while(0)

}  // namespace

// PyTorch entry point.  Returns nothing — outputs (dC, dz) are written
// in-place to caller-allocated buffers (must be zero-initialized fp32).
void spline_kv_bwd_cuda(
    const torch::Tensor& z,        // [N, H] bf16
    const torch::Tensor& C,        // [H, L, R] bf16
    const torch::Tensor& g_delta,  // [N, R] bf16
    torch::Tensor& dC,             // [H, L, R] fp32
    torch::Tensor& dz,             // [N, H] fp32
    double grid_lo,
    double scale
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(z.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16
                && g_delta.dtype() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(dC.dtype() == torch::kFloat32 && dz.dtype() == torch::kFloat32,
                "outputs must be fp32");

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);

    void* z_ptr       = z.data_ptr();
    void* C_ptr       = C.data_ptr();
    void* g_delta_ptr = g_delta.data_ptr();
    float* dC_ptr     = dC.data_ptr<float>();
    float* dz_ptr     = dz.data_ptr<float>();

    // Dispatch on (R, L_PAD).  L_PAD = next power of 2 of L.
    // For now: explicitly support R∈{32, 64}, L∈{16, 22}.
    // L_PAD = 16 if L=16, else L_PAD = 32 (covers L=22).
    if (R == 32 && L == 22) {
        LAUNCH_BWD(64, 8, 32, 32);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD(64, 8, 16, 32);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD(64, 8, 32, 64);
    } else if (R == 64 && L == 16) {
        LAUNCH_BWD(64, 8, 16, 64);
    } else {
        TORCH_CHECK(false, "unsupported (R, L) combination — extend dispatch");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_cuda", &spline_kv_bwd_cuda,
          "FlashSplineFeature backward (dC + dz) — CUDA C++",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("dC"), py::arg("dz"),
          py::arg("grid_lo"), py::arg("scale"));
}
