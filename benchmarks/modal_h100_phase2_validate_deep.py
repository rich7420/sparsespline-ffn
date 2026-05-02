"""Phase 2 deep validation — adversarial / robustness / speed checks.

Builds on modal_h100_phase2_validate.py (which covers basic correctness).
Adds:

  Test D — Boundary stress
    z values clustered around bin boundaries (±ε) AND just outside
    [grid_lo, grid_hi]. Confirms fwd numerics + bwd dC don't blow up
    on edge inputs. Reference: PyTorch fwd+autograd backward.

  Test E — Multi-seed stress
    5 random seeds × 4 (fwd, bwd) combos. Catches per-seed flakiness
    (e.g. atomic-add ordering pathologies).

  Test F — Long-replay drift
    Capture a graph once, replay 1000× without re-warmup. Confirms no
    accumulator drift / dangling-pointer / silent-corruption bugs.

  Test G — Forward speed benchmark
    Wall comparison of fwd_kernel="triton" vs "wgmma_cuda" at the
    nanochat shape, with and without cuda_graph capture, 12-layer stack.
    Quantifies the speed delta we can expect in nanochat training.

  Test H — Alt shapes
    Verifies forward dispatch on shapes outside the v1 sweet spot:
      (N=1024, d=384, r=32, L=18)
      (N=4096, d=1024, r=32, L=22)
    For r != 32 we expect a clean error (kernel only supports r=32 in v1).

Run:
  modal run benchmarks/modal_h100_phase2_validate_deep.py
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                                add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/.venv/**",
                "nanochat/.nanochat-runtime/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("rlkv-phase2-deep", image=IMAGE)


@app.function(gpu="H100", timeout=2700)
def run() -> str:
    import sys, io, statistics, time
    sys.path.insert(0, "/repo/src")
    import torch
    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda,
        spline_kv_fwd_fused_cuda,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference as ref_fwd,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_forward_v4 as triton_delta_v4,
    )

    grid_lo, grid_hi = -3.0, 3.0
    G = 20
    L = G + 2
    r = 32

    # ==========================================================
    # Test D — Boundary stress
    # ==========================================================
    log("=" * 100)
    log("Test D — Boundary stress (z near bin edges + out-of-range)")
    log("=" * 100)
    test_d_pass = True
    N, h = 1024, 256
    scale = G / (grid_hi - grid_lo)

    def synth_z_at_boundary(eps_list, n_samples_each, h_dim):
        """For each eps, place samples at every bin boundary ± eps."""
        z_parts = []
        for eps in eps_list:
            for bin_lo in range(G + 1):
                z_val = grid_lo + bin_lo / scale + eps
                z_parts.append(torch.full((n_samples_each, h_dim),
                                            float(z_val), dtype=torch.bfloat16))
        return torch.cat(z_parts, dim=0)

    eps_list = [-1e-3, +1e-3, -1e-1, +1e-1]
    boundary_z = synth_z_at_boundary(eps_list, n_samples_each=8, h_dim=h)
    # Plus out-of-range samples
    oor_z = torch.cat([
        torch.full((16, h), grid_lo - 0.1, dtype=torch.bfloat16),
        torch.full((16, h), grid_lo - 1.0, dtype=torch.bfloat16),
        torch.full((16, h), grid_hi + 0.1, dtype=torch.bfloat16),
        torch.full((16, h), grid_hi + 1.0, dtype=torch.bfloat16),
    ], dim=0)
    z_d = torch.cat([boundary_z, oor_z], dim=0).cuda()
    N_d = z_d.shape[0]
    torch.manual_seed(7)
    C_d = (torch.randn(h, L, r, dtype=torch.bfloat16) * 0.1).cuda()

    # Reference: PyTorch ref forward + autograd backward
    z_ref = z_d.detach().clone().requires_grad_(True)
    C_ref = C_d.detach().clone().requires_grad_(True)
    f_ref = ref_fwd(z_ref, C_ref, grid_lo, grid_hi, G,
                     activation="relu_sq", lambda_scale=1.0)
    target = torch.randn_like(f_ref)
    loss_ref = (f_ref - target).pow(2).sum()
    loss_ref.backward()

    # CUDA fwd path
    z_cu = z_d.detach().clone().requires_grad_(True)
    C_cu = C_d.detach().clone().requires_grad_(True)
    f_cu = flash_spline_feature(
        z_cu, C_cu, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        activation="relu_sq", lambda_scale=1.0,
        use_kernel=True, fwd_kernel="wgmma_cuda", bwd_kernel="wgmma_cuda",
    )
    loss_cu = (f_cu - target).pow(2).sum()
    loss_cu.backward()

    err_f = (f_cu.float() - f_ref.float()).norm() / f_ref.float().norm().clamp_min(1e-9)
    err_dz = (z_cu.grad.float() - z_ref.grad.float()).norm() \
              / z_ref.grad.float().norm().clamp_min(1e-9)
    err_dC = (C_cu.grad.float() - C_ref.grad.float()).norm() \
              / C_ref.grad.float().norm().clamp_min(1e-9)
    nan_inf = (
        (~torch.isfinite(f_cu)).any().item()
        or (~torch.isfinite(z_cu.grad)).any().item()
        or (~torch.isfinite(C_cu.grad)).any().item()
    )
    log(f"  err_f={err_f.item():.3e}  err_dz={err_dz.item():.3e}  "
        f"err_dC={err_dC.item():.3e}  any_nan_inf={nan_inf}")
    test_d_pass = (err_f.item() < 5e-3 and err_dz.item() < 5e-3
                    and err_dC.item() < 5e-3 and not nan_inf)
    log(f"  Test D: {'PASS' if test_d_pass else 'FAIL'}")
    log("")

    # ==========================================================
    # Test E — Multi-seed stress
    # ==========================================================
    log("=" * 100)
    log("Test E — Multi-seed stress (5 seeds × 4 path combos)")
    log("=" * 100)
    N, h = 2048, 768
    combos = [
        ("triton",     "triton"),
        ("triton",     "wgmma_cuda"),
        ("wgmma_cuda", "triton"),
        ("wgmma_cuda", "wgmma_cuda"),
    ]

    seed_results: list[dict] = []
    for seed in [1, 7, 17, 42, 99]:
        torch.manual_seed(seed)
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.05)
        target = torch.randn(N, h + r, device="cuda", dtype=torch.bfloat16)

        # Reference
        zr = z.detach().clone().requires_grad_(True)
        Cr = C.detach().clone().requires_grad_(True)
        f_ref = ref_fwd(zr, Cr, grid_lo, grid_hi, G, "relu_sq", 1.0)
        loss_ref = (f_ref - target).pow(2).sum()
        loss_ref.backward()

        for (fwk, bwk) in combos:
            zt = z.detach().clone().requires_grad_(True)
            Ct = C.detach().clone().requires_grad_(True)
            ft = flash_spline_feature(
                zt, Ct, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=1.0,
                use_kernel=True, fwd_kernel=fwk, bwd_kernel=bwk,
            )
            ((ft - target).pow(2).sum()).backward()
            err_dz = (zt.grad.float() - zr.grad.float()).norm() \
                      / zr.grad.float().norm().clamp_min(1e-9)
            err_dC = (Ct.grad.float() - Cr.grad.float()).norm() \
                      / Cr.grad.float().norm().clamp_min(1e-9)
            seed_results.append({
                "seed": seed, "fwk": fwk, "bwk": bwk,
                "err_dz": err_dz.item(), "err_dC": err_dC.item(),
            })

    # Aggregate per combo
    log(f"  {'fwd':<12} {'bwd':<12} {'max err_dz':>14} {'max err_dC':>14} {'verdict':>8}")
    test_e_pass = True
    for (fwk, bwk) in combos:
        rows = [r for r in seed_results if r["fwk"] == fwk and r["bwk"] == bwk]
        max_dz = max(r["err_dz"] for r in rows)
        max_dC = max(r["err_dC"] for r in rows)
        ok = max_dz < 5e-3 and max_dC < 5e-3
        test_e_pass = test_e_pass and ok
        log(f"  {fwk:<12} {bwk:<12} {max_dz:>14.4e} {max_dC:>14.4e} "
            f"{'OK' if ok else 'FAIL':>8}")
    log(f"  Test E: {'PASS' if test_e_pass else 'FAIL'}")
    log("")

    # ==========================================================
    # Test F — Long-replay drift (1000× replay)
    # ==========================================================
    log("=" * 100)
    log("Test F — Long-replay drift (1000 replays, no re-warmup)")
    log("=" * 100)
    torch.manual_seed(0)
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.05)
    target = torch.randn(N, h + r, device="cuda", dtype=torch.bfloat16)

    static_z = z.detach().clone().requires_grad_(True)
    static_C = C.detach().clone().requires_grad_(True)
    static_z.grad = torch.zeros_like(static_z)
    static_C.grad = torch.zeros_like(static_C)
    static_target = target.detach().clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_z.grad.zero_(); static_C.grad.zero_()
            f = flash_spline_feature(
                static_z, static_C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=1.0,
                use_kernel=True, fwd_kernel="wgmma_cuda", bwd_kernel="wgmma_cuda",
            )
            ((f - static_target).pow(2).sum()).backward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    static_z.grad.zero_(); static_C.grad.zero_()
    with torch.cuda.graph(g):
        f_g = flash_spline_feature(
            static_z, static_C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=1.0,
            use_kernel=True, fwd_kernel="wgmma_cuda", bwd_kernel="wgmma_cuda",
        )
        static_loss = (f_g - static_target).pow(2).sum()
        static_loss.backward()

    # Reference: zero-grad + replay once → record snapshot
    static_z.grad.zero_(); static_C.grad.zero_()
    g.replay()
    torch.cuda.synchronize()
    ref_dz = static_z.grad.detach().clone()
    ref_dC = static_C.grad.detach().clone()
    ref_loss = static_loss.detach().clone()

    drift_max_dz = 0.0
    drift_max_dC = 0.0
    drift_max_loss = 0.0
    n_replays = 1000
    t0 = time.perf_counter()
    for i in range(n_replays):
        static_z.grad.zero_(); static_C.grad.zero_()
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    after_dz = static_z.grad.detach()
    after_dC = static_C.grad.detach()
    after_loss = static_loss.detach()

    drift_max_dz = (after_dz.float() - ref_dz.float()).abs().max().item()
    drift_max_dC = (after_dC.float() - ref_dC.float()).abs().max().item()
    drift_max_loss = (after_loss - ref_loss).abs().item()

    log(f"  {n_replays} replays in {elapsed:.3f} s "
        f"({elapsed * 1000 / n_replays:.3f} ms/replay)")
    log(f"  drift_max_dz   = {drift_max_dz:.4e}")
    log(f"  drift_max_dC   = {drift_max_dC:.4e}")
    log(f"  drift_max_loss = {drift_max_loss:.4e}")
    test_f_pass = drift_max_dz < 1e-2 and drift_max_dC < 1e-2 and drift_max_loss < 1e-2
    log(f"  Test F: {'PASS' if test_f_pass else 'FAIL'}")
    log("")
    del g

    # ==========================================================
    # Test G — Forward speed benchmark (Triton vs CUDA)
    # ==========================================================
    log("=" * 100)
    log("Test G — Forward speed benchmark (12-layer FFN stack, eager + graph)")
    log("=" * 100)
    import torch.nn as nn
    n_layers = 12
    d = 768

    class _RLKVLayer(nn.Module):
        def __init__(self, d: int, r: int, L: int):
            super().__init__()
            self.K = nn.Linear(d, d, bias=False)
            self.C = nn.Parameter(torch.zeros(d, L, r))
            self.W_out = nn.Linear(d + r, d, bias=False)
            with torch.no_grad():
                s_in = (3.0 / d) ** 0.5
                s_h = (3.0 / (d + r)) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
                nn.init.normal_(self.C, std=0.01)

    class _RLKVStack(nn.Module):
        def __init__(self, fwk: str):
            super().__init__()
            self.fwk = fwk
            self.bwk = "wgmma_cuda"
            self.layers = nn.ModuleList([
                _RLKVLayer(d, r, L) for _ in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                z = layer.K(x)
                f = flash_spline_feature(
                    z, layer.C,
                    grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                    activation="relu_sq", lambda_scale=1.0,
                    use_kernel=True,
                    fwd_kernel=self.fwk, bwd_kernel=self.bwk,
                )
                x = x + layer.W_out(f)
            return x

    def bench_eager(model, x_const, target, time_n=30, warmup=10):
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            ((y - target) ** 2).sum().backward()
        torch.cuda.synchronize()
        samples = []
        for _ in range(time_n):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            ((y - target) ** 2).sum().backward()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(samples)

    def bench_graph(model, x_const, target, time_n=30):
        static_x = torch.empty_like(x_const).requires_grad_(True)
        static_target = torch.empty_like(target)
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(11):
                for p in model.parameters():
                    p.grad.zero_()
                static_x.data.copy_(x_const)
                static_target.copy_(target)
                y = model(static_x)
                ((y - static_target) ** 2).sum().backward()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        for p in model.parameters():
            p.grad.zero_()
        with torch.cuda.graph(g):
            y_g = model(static_x)
            static_loss = ((y_g - static_target) ** 2).sum()
            static_loss.backward()

        samples = []
        for _ in range(time_n):
            static_x.data.copy_(x_const)
            static_target.copy_(target)
            for p in model.parameters():
                p.grad.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(samples)

    bench_results: dict = {}
    for fwk in ("triton", "wgmma_cuda"):
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        model = _RLKVStack(fwk).cuda().to(torch.bfloat16).train()
        x_const = torch.randn(2, 1024, d, device="cuda", dtype=torch.bfloat16).reshape(2 * 1024, d)
        target = torch.randn_like(x_const)
        e = bench_eager(model, x_const, target)
        g = bench_graph(model, x_const, target)
        bench_results[fwk] = {"eager_ms": e, "graph_ms": g}
        log(f"  {fwk:<12} eager={e:>7.3f} ms   graph={g:>7.3f} ms")
        del model
        torch.cuda.empty_cache()

    # Compare
    if "triton" in bench_results and "wgmma_cuda" in bench_results:
        e_t = bench_results["triton"]["eager_ms"]
        e_c = bench_results["wgmma_cuda"]["eager_ms"]
        g_t = bench_results["triton"]["graph_ms"]
        g_c = bench_results["wgmma_cuda"]["graph_ms"]
        log("")
        log(f"  CUDA fwd vs Triton fwd:")
        log(f"     eager:  {e_c/e_t:.3f}× ({e_t-e_c:+.3f} ms saved per step if + means slower)")
        log(f"     graph:  {g_c/g_t:.3f}× ({g_t-g_c:+.3f} ms saved per step if + means slower)")
    log(f"  Test G: complete (informational, no pass/fail)")
    log("")

    # ==========================================================
    # Test H — Alt shapes
    # ==========================================================
    log("=" * 100)
    log("Test H — Alt shapes (dispatch sanity)")
    log("=" * 100)
    test_h_pass = True
    cases = [
        # (label, N, h, r, G, expect)
        ("nominal",      2048,  768, 32, 20, "ok"),
        ("smaller_d",    1024,  384, 32, 16, "ok"),
        ("larger_d",     4096, 1024, 32, 22, "ok"),
        ("r_64_unsupp",   512,  512, 64, 20, "fallback"),
    ]
    for (label, NN, hh, rr, GG, expect) in cases:
        LL = GG + 2
        torch.manual_seed(0)
        zz = torch.randn(NN, hh, device="cuda", dtype=torch.bfloat16)
        CC = (torch.randn(hh, LL, rr, device="cuda", dtype=torch.bfloat16) * 0.05)
        try:
            f = flash_spline_feature(
                zz, CC, grid_lo=grid_lo, grid_hi=grid_hi, G=GG,
                activation="relu_sq", lambda_scale=1.0,
                use_kernel=True, fwd_kernel="auto", bwd_kernel="triton",
            )
            assert f.shape == (NN, hh + rr), f"unexpected shape {f.shape}"
            log(f"  {label:<14} N={NN} h={hh} r={rr} G={GG}: f.shape={tuple(f.shape)} OK")
        except Exception as e:
            if expect == "fallback":
                log(f"  {label:<14} N={NN} h={hh} r={rr} G={GG}: "
                    f"errored as designed → {type(e).__name__}: {e}")
            else:
                log(f"  {label:<14} N={NN} h={hh} r={rr} G={GG}: "
                    f"UNEXPECTED FAIL → {type(e).__name__}: {e}")
                test_h_pass = False
    log(f"  Test H: {'PASS' if test_h_pass else 'FAIL'}")
    log("")

    # ==========================================================
    # Final
    # ==========================================================
    log("=" * 100)
    log("DEEP VALIDATE FINAL")
    log("=" * 100)
    log(f"  Test D (boundary stress):    {'PASS' if test_d_pass else 'FAIL'}")
    log(f"  Test E (multi-seed × combos): {'PASS' if test_e_pass else 'FAIL'}")
    log(f"  Test F (1000-replay drift):  {'PASS' if test_f_pass else 'FAIL'}")
    log(f"  Test G (speed benchmark):    informational")
    log(f"  Test H (alt shapes):         {'PASS' if test_h_pass else 'FAIL'}")
    overall = test_d_pass and test_e_pass and test_f_pass and test_h_pass
    log(f"  Overall: {'✅ PASS' if overall else '❌ FAIL'}")

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
