// Native CUDA forward for FlashSplineFeature (RL-Spline-KV path).
//
// Mirrors triton_flash_spline_feature.flash_spline_feature_forward (v4
// h-split + atomic_add). Keeps the same numerics so the bwd kernels (that
// were validated against Triton fwd) stay numerically consistent.
//
// Pipeline: 3 stream-safe kernels, all on `at::cuda::getCurrentCUDAStream()`,
// captured cleanly under torch.cuda.graph(...):
//
//   (A) activation_kernel:   f[:, :h] = activation(z)            bf16 elementwise
//   (B) delta_kernel:        delta_fp32[:, :r] += sum_j B(z_j)*C[j, bin+i, :]
//                            (atomic_add reduction across h-chunks)
//   (C) pack_kernel:         f[:, h:h+r] = (lambda * delta_fp32)  bf16 cast
//
// Output tensor `f` and scratch `delta_fp32` are allocated by the Python side
// (so torch's caching allocator owns them and CUDA-Graph capture sees stable
// pointers). `delta_fp32` MUST be zero-initialized before launch.
//
// Activation IDs follow the bwd kernel convention:  0=relu_sq, 2=identity.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ void compute_B2(
    float tau, float& B0, float& B1, float& B2
) {
    float omt = 1.0f - tau;
    B0 = 0.5f * omt * omt;
    B1 = 0.5f * (1.0f + 2.0f * tau - 2.0f * tau * tau);
    B2 = 0.5f * tau * tau;
}

// -----------------------------------------------------------------------
// (A) Activation kernel
//     Writes f[:, :h] = activation(z[:, :h])  in bf16.
//     One thread per element.
// -----------------------------------------------------------------------
__global__ void spline_kv_fwd_activation_kernel(
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ f,
    const int N, const int H, const int HR,
    const int activation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * H;
    if (idx >= total) return;
    const int n = idx / H;
    const int j = idx - n * H;
    const float z_val = bf2f(z[idx]);
    float a;
    if (activation == 0) {                 // relu_sq
        a = (z_val > 0.0f) ? z_val * z_val : 0.0f;
    } else if (activation == 2) {          // identity
        a = z_val;
    } else {
        a = z_val;                          // safe fallback
    }
    f[n * HR + j] = f2bf(a);
}

// -----------------------------------------------------------------------
// (B) Delta kernel  —  v4-style h-split + atomicAdd, optimized.
//
//     blockDim.x = BN * BR threads.  Each thread owns one (n, r) output.
//     Grid: (ceil(N/BN), ceil(R/BR), H_CHUNKS).
//
//     v2 changes (3.A.2):
//       BN: 4 → 32  (8× more outputs per block; same per-thread work)
//       BH (h chunk): 64 → 128  (halves H_CHUNKS, halving atomic contention)
//       SMEM-stage z[BN][BH] so the BR threads sharing a token re-read z
//       from L1-distance SMEM instead of competing for the same L1 line.
//
//     Atomic contention per (n, r) destination:
//       v1: H_CHUNKS=12 → 12-way contention per dest
//       v2: H_CHUNKS=6  → 6-way contention (2× less)
// -----------------------------------------------------------------------
template <int BN, int BR>
__global__ void spline_kv_fwd_delta_kernel(
    const __nv_bfloat16* __restrict__ z,    // [N, H]  bf16
    const __nv_bfloat16* __restrict__ C,    // [H, L, R]  bf16
    float* __restrict__ delta_fp32,          // [N, R]  fp32, zero-init
    const int N, const int H, const int L, const int R,
    const float grid_lo, const float scale, const float G_max
) {
    const int pid_n = blockIdx.x;
    const int pid_r = blockIdx.y;
    const int pid_h = blockIdx.z;
    const int H_CHUNKS = gridDim.z;
    const int BH = (H + H_CHUNKS - 1) / H_CHUNKS;
    const int h_start = pid_h * BH;
    const int h_end_raw = h_start + BH;
    const int h_end = h_end_raw < H ? h_end_raw : H;
    const int n_iters = h_end - h_start;

    const int tid = threadIdx.x;
    const int block_threads = BN * BR;       // = blockDim.x
    const int n_in_block = tid / BR;         // 0..BN-1
    const int r_in_block = tid - n_in_block * BR;
    const int n_global = pid_n * BN + n_in_block;
    const int r_global = pid_r * BR + r_in_block;
    const bool active = (n_global < N) && (r_global < R) && (n_in_block < BN);

    // -----------------------------------------------------------------------
    // SMEM-stage z[BN, BH].  Each thread loads multiple z values cooperatively.
    // Layout: z_smem[n_local * BH + j_local], n_local in [0,BN), j_local in [0,BH).
    // Total = BN * BH bf16 = e.g. 32 * 128 * 2 = 8 KB.
    // -----------------------------------------------------------------------
    extern __shared__ __nv_bfloat16 z_smem[];   // size = BN * BH

    // Cooperative load: each thread loads (BN * BH / block_threads) elements.
    // For BN=32 BR=32 BH=128: block_threads=1024, total=4096 → 4 loads per thread.
    const int total_z = BN * BH;
    for (int idx = tid; idx < total_z; idx += block_threads) {
        const int n_local = idx / BH;
        const int j_local = idx - n_local * BH;
        const int n_g = pid_n * BN + n_local;
        const int j_g = h_start + j_local;
        if (n_g < N && j_g < h_end) {
            z_smem[idx] = z[n_g * H + j_g];
        } else {
            z_smem[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    if (!active) return;

    float acc = 0.0f;

    #pragma unroll 1
    for (int j_local = 0; j_local < n_iters; j_local++) {
        const int j = h_start + j_local;

        // z is now from SMEM (shared across r-threads of same n)
        const float z_val = bf2f(z_smem[n_in_block * BH + j_local]);
        const float u_raw = (z_val - grid_lo) * scale;
        const bool in_range = (u_raw >= 0.0f) && (u_raw <= G_max);
        const float u = fminf(fmaxf(u_raw, 0.0f), G_max - 1.0f);
        const int bin = (int)u;
        const float tau = u - (float)bin;
        float B0, B1, B2;
        compute_B2(tau, B0, B1, B2);
        if (!in_range) { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }

        // C[j, ..., r_global]:  base = C + j * L * R
        const __nv_bfloat16* base = C + j * (L * R);
        const float c0 = bf2f(base[bin       * R + r_global]);
        const float c1 = bf2f(base[(bin + 1) * R + r_global]);
        const float c2 = bf2f(base[(bin + 2) * R + r_global]);
        acc += B0 * c0 + B1 * c1 + B2 * c2;
    }

    // Atomic reduce across h-chunks.
    atomicAdd(&delta_fp32[n_global * R + r_global], acc);
}

// -----------------------------------------------------------------------
// (C) Pack kernel
//     Writes f[:, H:HR] = bf16(lambda * delta_fp32[:, :R]).
// -----------------------------------------------------------------------
__global__ void spline_kv_fwd_pack_delta_kernel(
    const float* __restrict__ delta_fp32,
    __nv_bfloat16* __restrict__ f,
    const int N, const int H, const int R, const int HR,
    const float lambda_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * R;
    if (idx >= total) return;
    const int n = idx / R;
    const int rr = idx - n * R;
    f[n * HR + H + rr] = f2bf(lambda_scale * delta_fp32[idx]);
}

}  // namespace

// ===========================================================================
// PyTorch entry point.
// ===========================================================================
torch::Tensor spline_kv_fwd_cuda(
    const torch::Tensor& z,        // [N, H] bf16
    const torch::Tensor& C,        // [H, L, R] bf16
    double grid_lo, double scale,
    double lambda_scale,
    int64_t activation             // 0=relu_sq, 2=identity
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(z.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(z.dim() == 2 && C.dim() == 3, "z=[N,H] C=[H,L,R]");

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);
    TORCH_CHECK(C.size(0) == H, "C[0] must equal z[1]");
    const int HR = H + R;
    const float G_max = (float)(L - 2);     // L = G + spline_order(=2)

    auto out_opts = torch::TensorOptions().device(z.device())
                                            .dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device())
                                             .dtype(torch::kFloat32);
    torch::Tensor f = torch::empty({N, HR}, out_opts);
    torch::Tensor delta_fp32 = torch::empty({N, R}, fp32_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();
    // CUDA-Graph-safe zero init: cudaMemsetAsync is captured into the graph,
    // unlike a CPU-side memset.  See PyTorch CUDA Graphs notes §"Common pitfalls".
    cudaMemsetAsync(delta_fp32.data_ptr<float>(), 0,
                     (size_t)N * (size_t)R * sizeof(float), stream);

    // ----- (A) activation -----
    {
        const int total = N * H;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        spline_kv_fwd_activation_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)z.data_ptr(),
            (__nv_bfloat16*)f.data_ptr(),
            N, H, HR, (int)activation);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ----- (B) delta atomic accum -----
    // v2 (3.A.2): SMEM-staged z + larger block.  BH=128 → H_CHUNKS=6.
    //   r=32 :  BN=32, BR=32 → 1024 threads/block.  SMEM = 8 KB.
    //   r=64 :  BN=16, BR=64 → 1024 threads/block.  SMEM = 4 KB (BN halved).
    //   Atomic contention per dest: 6× (was 12× in v1).
    {
        const int H_CHUNK = 128;
        const int blocks_h = (H + H_CHUNK - 1) / H_CHUNK;
        if (R == 32) {
            const int BN = 32, BR = 32;
            const int blocks_n = (N + BN - 1) / BN;
            const int blocks_r = (R + BR - 1) / BR;
            dim3 grid(blocks_n, blocks_r, blocks_h);
            dim3 block(BN * BR, 1, 1);
            const size_t smem_bytes = BN * H_CHUNK * sizeof(__nv_bfloat16);
            spline_kv_fwd_delta_kernel<BN, BR><<<grid, block, smem_bytes, stream>>>(
                (const __nv_bfloat16*)z.data_ptr(),
                (const __nv_bfloat16*)C.data_ptr(),
                delta_fp32.data_ptr<float>(),
                N, H, L, R, (float)grid_lo, (float)scale, G_max);
        } else if (R == 64) {
            const int BN = 16, BR = 64;
            const int blocks_n = (N + BN - 1) / BN;
            const int blocks_r = (R + BR - 1) / BR;
            dim3 grid(blocks_n, blocks_r, blocks_h);
            dim3 block(BN * BR, 1, 1);
            const size_t smem_bytes = BN * H_CHUNK * sizeof(__nv_bfloat16);
            spline_kv_fwd_delta_kernel<BN, BR><<<grid, block, smem_bytes, stream>>>(
                (const __nv_bfloat16*)z.data_ptr(),
                (const __nv_bfloat16*)C.data_ptr(),
                delta_fp32.data_ptr<float>(),
                N, H, L, R, (float)grid_lo, (float)scale, G_max);
        } else {
            TORCH_CHECK(false, "spline_kv_fwd_cuda: R must be 32 or 64");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ----- (C) pack delta into f[:, H:HR] -----
    {
        const int total = N * R;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        spline_kv_fwd_pack_delta_kernel<<<blocks, threads, 0, stream>>>(
            delta_fp32.data_ptr<float>(),
            (__nv_bfloat16*)f.data_ptr(),
            N, H, R, HR, (float)lambda_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return f;
}

// ===========================================================================
// 2.3 — Fused W_out_base epilogue:
//   y = a @ W_out_a^T + lambda * delta @ W_out_d^T
// where:
//   a       = activation(z)        [N, H]   (we compute on the fly, never materialize)
//   delta   = sum_j sum_b B_b(z_j) * C[j, bin+b, :]   [N, R]
//   W_out_a = W_out[:, :H]         [d, H]
//   W_out_d = W_out[:, H:H+R]      [d, R]
//
// Strategy (v1, simple but correct):
//   1. Compute a [N, H] bf16 via activation_kernel.
//   2. Compute delta_bf16 [N, R] via delta_kernel + pack_kernel  (lambda baked in).
//   3. y = a @ W_out_a^T + delta_bf16 @ W_out_d^T  via two cuBLAS GEMMs from C++.
//
// This is "fused at the API level" — it eliminates the [N, H+R] f
// materialization (saves an HBM write of ~6 MB / layer at our shape) and
// removes a Python boundary, but it does NOT yet fuse the spline-compute
// loop with the W_out GEMM into a single kernel. v2 of this can do that.
// ===========================================================================
torch::Tensor spline_kv_fwd_fused_cuda(
    const torch::Tensor& z,        // [N, H] bf16
    const torch::Tensor& C,        // [H, L, R] bf16
    const torch::Tensor& W_out,    // [d_out, H+R] bf16  (Linear.weight orientation)
    double grid_lo, double scale,
    double lambda_scale,
    int64_t activation             // 0=relu_sq, 2=identity
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && W_out.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(z.dtype() == torch::kBFloat16
                && C.dtype() == torch::kBFloat16
                && W_out.dtype() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous() && W_out.is_contiguous(),
                "inputs must be contiguous");

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = C.size(1);
    const int R = C.size(2);
    TORCH_CHECK(C.size(0) == H, "C[0] must equal z[1]");
    const int HR = H + R;
    TORCH_CHECK(W_out.size(1) == HR, "W_out cols must equal H+R");
    const int D_OUT = W_out.size(0);
    const float G_max = (float)(L - 2);

    auto out_opts = torch::TensorOptions().device(z.device())
                                            .dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device())
                                             .dtype(torch::kFloat32);

    torch::Tensor a_buf = torch::empty({N, H}, out_opts);
    torch::Tensor delta_fp32 = torch::empty({N, R}, fp32_opts);
    torch::Tensor delta_bf16 = torch::empty({N, R}, out_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(delta_fp32.data_ptr<float>(), 0,
                     (size_t)N * (size_t)R * sizeof(float), stream);

    // (A) a = activation(z)
    {
        const int total = N * H;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        // Re-use activation kernel with HR=H so f[:, :H]=a writes into a_buf directly.
        spline_kv_fwd_activation_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)z.data_ptr(),
            (__nv_bfloat16*)a_buf.data_ptr(),
            N, H, /*HR=*/H, (int)activation);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // (B) delta atomic accum  (same v2 layout as non-fused path)
    {
        const int H_CHUNK = 128;
        const int blocks_h = (H + H_CHUNK - 1) / H_CHUNK;
        if (R == 32) {
            const int BN = 32, BR = 32;
            const int blocks_n = (N + BN - 1) / BN;
            const int blocks_r = (R + BR - 1) / BR;
            dim3 grid(blocks_n, blocks_r, blocks_h);
            dim3 block(BN * BR, 1, 1);
            const size_t smem_bytes = BN * H_CHUNK * sizeof(__nv_bfloat16);
            spline_kv_fwd_delta_kernel<BN, BR><<<grid, block, smem_bytes, stream>>>(
                (const __nv_bfloat16*)z.data_ptr(),
                (const __nv_bfloat16*)C.data_ptr(),
                delta_fp32.data_ptr<float>(),
                N, H, L, R, (float)grid_lo, (float)scale, G_max);
        } else if (R == 64) {
            const int BN = 16, BR = 64;
            const int blocks_n = (N + BN - 1) / BN;
            const int blocks_r = (R + BR - 1) / BR;
            dim3 grid(blocks_n, blocks_r, blocks_h);
            dim3 block(BN * BR, 1, 1);
            const size_t smem_bytes = BN * H_CHUNK * sizeof(__nv_bfloat16);
            spline_kv_fwd_delta_kernel<BN, BR><<<grid, block, smem_bytes, stream>>>(
                (const __nv_bfloat16*)z.data_ptr(),
                (const __nv_bfloat16*)C.data_ptr(),
                delta_fp32.data_ptr<float>(),
                N, H, L, R, (float)grid_lo, (float)scale, G_max);
        } else {
            TORCH_CHECK(false, "spline_kv_fwd_fused_cuda: R must be 32 or 64");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // (C) pack delta_fp32 → delta_bf16 (with lambda)
    {
        const int total = N * R;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        // re-use pack kernel but write to delta_bf16 with HR=R so offsets land in [0, R).
        spline_kv_fwd_pack_delta_kernel<<<blocks, threads, 0, stream>>>(
            delta_fp32.data_ptr<float>(),
            (__nv_bfloat16*)delta_bf16.data_ptr(),
            N, /*H=*/0, R, /*HR=*/R, (float)lambda_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // (D) y = a @ W_out_a^T + delta_bf16 @ W_out_d^T
    //   Slice W_out into W_out_a [D_OUT, H] and W_out_d [D_OUT, R].
    auto W_out_a = W_out.slice(/*dim=*/1, /*start=*/0, /*end=*/H);
    auto W_out_d = W_out.slice(/*dim=*/1, /*start=*/H, /*end=*/HR);

    // Use torch::matmul (calls cuBLAS BF16, captured cleanly by graph).
    // Result: [N, D_OUT] = [N, H] @ [H, D_OUT]
    torch::Tensor y = torch::matmul(a_buf, W_out_a.transpose(0, 1).contiguous());
    y.add_(torch::matmul(delta_bf16, W_out_d.transpose(0, 1).contiguous()));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_fwd_cuda", &spline_kv_fwd_cuda,
          "FlashSplineFeature forward (native CUDA, h-split + atomic)",
          py::arg("z"), py::arg("C"),
          py::arg("grid_lo"), py::arg("scale"),
          py::arg("lambda_scale"), py::arg("activation"));
    m.def("spline_kv_fwd_fused_cuda", &spline_kv_fwd_fused_cuda,
          "FlashSplineFeature forward + W_out GEMM fused at API level",
          py::arg("z"), py::arg("C"), py::arg("W_out"),
          py::arg("grid_lo"), py::arg("scale"),
          py::arg("lambda_scale"), py::arg("activation"));
}
