"""H100 Modal: Triton v3 vs CUDA C++ bwd kernel."""
import modal

IMAGE = (
    # Need full CUDA toolchain (incl. nvcc) for JIT-compiling our .cu kernel.
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                                add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                 index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-cuda-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1200)
def run_bench():
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import flash_spline_delta_backward_v3
    from sparsespline_ffn.cuda_ext import spline_kv_bwd_cuda

    print("Compiling CUDA extension...")
    # Trigger compile
    z = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(64, 16, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    g = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    spline_kv_bwd_cuda(z, C, g, -3.0, 3.0, 14)
    print("CUDA compiled.")

    def med_ms(fn, w=8, it=40):
        for _ in range(w): fn()
        torch.cuda.synchronize(); s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    print(f"\n=== H100 Triton v3 vs CUDA bwd ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  bf16 inputs, fp32 outputs")
    print()
    print(f"{'shape':<14} {'v3_ms':>9} {'cuda_ms':>9} {'v3/cuda':>9}")
    print("-" * 50)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        N, h = 2048, 768
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.bfloat16)
        flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G)
        spline_kv_bwd_cuda(z, C, g, -3.0, 3.0, G)
        t3 = med_ms(lambda: flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G))
        tc = med_ms(lambda: spline_kv_bwd_cuda(z, C, g, -3.0, 3.0, G))
        line = f"r={r:<2} L={L:<2}     {t3:>9.3f} {tc:>9.3f} {t3/tc:>8.2f}x"
        print(line)
        results.append({"r": r, "L": L, "t3_ms": t3, "tcuda_ms": tc, "speedup": t3/tc})
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print(run_bench.remote())
