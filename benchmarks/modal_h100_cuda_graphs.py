"""H100 CUDA Graphs bench: amortize launch overhead.

Compares eager vs graph-captured fwd+bwd for each FFN variant.  Expected
to compress launch-overhead-bound surrounding ops (cuBLAS GEMMs, autograd
dispatch).  We saw 1.5x speedup on 3080 with FullMix; expect similar
or better on H100 because launch overhead is relatively larger when
kernels themselves are very fast.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-cuda-graphs-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_bench(d: int = 768, b: int = 2, t: int = 1024) -> str:
    import sys, time, gc, json
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/benchmarks")
    import torch, torch.nn as nn

    from sparsespline_ffn import MLPFFN
    from sparsespline_ffn.simple_spline_mlp import SimpleSplineMLP, SimpleSplineConfig
    from sparsespline_ffn.glu_ffn import SwiGLU, GLUConfig
    from v_c_fusion_bench import CudaGraphFFN
    from ffn_full_compare import _RLKVWrap

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"\n=== H100 CUDA Graphs FFN bench ===")
    print(f"  d={d}  B={b}  T={t}  dtype=bf16")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    def med_ms(fn, w=8, it=40):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    def peak(fn, w=5, it=10):
        for _ in range(w): fn()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(it): fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2

    builders = {
        "mlp_h_4d":         lambda: MLPFFN(d=d, mlp_ratio=4),
        "mlp_h_d":          lambda: MLPFFN(d=d, mlp_ratio=1),
        "ss_h_d":           lambda: SimpleSplineMLP(SimpleSplineConfig(d=d, h_ratio=1.0, G=20, use_kernel=True)),
        "swiglu_h_d":       lambda: SwiGLU(GLUConfig(d=d, mlp_ratio=1.0)),
        "rl_kv_r32_kernel": lambda: _RLKVWrap(d=d, r=32, use_kernel=True),
    }

    print(f"\n{'name':<18} {'eager_ms':>10} {'graph_ms':>10} {'speedup':>8} "
          f"{'eager_MB':>10} {'graph_MB':>10}")
    print("-" * 80)
    rows = []
    for name, builder in builders.items():
        torch.cuda.empty_cache(); gc.collect()
        torch.manual_seed(0)
        model = builder().to(device=device, dtype=dtype).train()
        target = torch.randn(b, t, d, device=device, dtype=dtype)
        x_const = torch.randn(b, t, d, device=device, dtype=dtype)

        def eager_step():
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            loss = (y - target).pow(2).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            return loss

        try:
            ms_eager = med_ms(eager_step)
            mb_eager = peak(eager_step)
        except Exception as e:
            print(f"{name:<18}  eager FAIL: {e}")
            continue

        try:
            graph_ffn = CudaGraphFFN(builder().to(device=device, dtype=dtype),
                                       B=b, T=t, d=d, dtype=dtype, device=device,
                                       warmup_iters=5)
            ms_graph = med_ms(lambda: graph_ffn.step(x_const))
            mb_graph = peak(lambda: graph_ffn.step(x_const))
        except Exception as e:
            print(f"{name:<18}  ms_eager={ms_eager:.3f}  graph FAIL: {e}")
            rows.append({"name": name, "ms_eager": ms_eager, "ms_graph": None,
                         "mb_eager": mb_eager})
            continue

        speedup = ms_eager / ms_graph
        print(f"{name:<18}  {ms_eager:>10.3f}  {ms_graph:>10.3f}  "
              f"{speedup:>6.2f}x  {mb_eager:>10.1f}  {mb_graph:>10.1f}")
        rows.append({
            "name": name, "ms_eager": ms_eager, "ms_graph": ms_graph,
            "speedup": speedup, "mb_eager": mb_eager, "mb_graph": mb_graph,
        })
        del model, graph_ffn
        torch.cuda.empty_cache(); gc.collect()

    return json.dumps(rows, indent=2)


@app.local_entrypoint()
def main(d: int = 768, b: int = 2, t: int = 1024) -> None:
    print(run_bench.remote(d=d, b=b, t=t))
