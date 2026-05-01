"""Numerical audit for FlashSplineFeature — Task 1.

Goal: separate the *kernel's* numerical error from the *bf16 reference's*
numerical error.  Without this we can't tell whether bench rel_rms ≈ 6e-2
on the skewed workload is a kernel bug or just the reference quantizing
itself.

For each workload (uniform, skewed, collapsed) we compute four results:
  GT     = fp64 reference (ground truth)
  ref_b  = bf16 reference (current pytest oracle, lossy)
  kern_b = bf16 kernel    (Triton fwd, fp32 internal acc, bf16 output)
  kern_f = fp32 kernel    (no bf16 quantization anywhere)

and report rel-RMS error of each against GT.

Expected pattern (v7 §R.3.3.5):
  kern_f - GT  ≈ 1e-5   (algorithmic, pure noise)
  kern_b - GT  < ref_b - GT   (kernel beats reference because fp32 acc)
  ref_b  - GT ≈ 1e-2     (bf16 quantization on the entire summation)

If skewed shows kern_b - GT >> kern_f - GT, the kernel has a real bug.
If skewed shows kern_b - GT ≈ ref_b - GT, the reference is the noisy one.

Run:
  python benchmarks/flash_spline_feature_audit.py
  python benchmarks/flash_spline_feature_audit.py --shape big
"""
from __future__ import annotations

import argparse

import torch


def _gen_workload(name: str, N: int, h: int, dtype, device, *,
                   grid_lo: float = -3.0, grid_hi: float = 3.0):
    if name == "uniform":
        return torch.randn(N, h, dtype=dtype, device=device)
    if name == "skewed":
        return torch.randn(N, h, dtype=dtype, device=device) * 0.5 + 2.0
    if name == "collapsed":
        return torch.randn(N, h, dtype=dtype, device=device) * 0.05
    raise ValueError(name)


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative RMS error against ``b`` (treated as ground truth)."""
    a32 = a.detach().to(torch.float64)
    b32 = b.detach().to(torch.float64)
    rms_b = b32.pow(2).mean().sqrt().clamp_min(1e-12)
    return float(((a32 - b32).pow(2).mean().sqrt() / rms_b).item())


def _abs_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().double() - b.detach().double()).abs().max().item())


def audit_workload(
    name: str, *, N: int, h: int, r: int, G: int,
    device: torch.device,
    grid_lo: float = -3.0, grid_hi: float = 3.0,
) -> dict:
    L = G + 2

    # Generate inputs in fp32 base, then cast to bf16 for the lossy path
    z_f32 = _gen_workload(name, N, h, torch.float32, device,
                           grid_lo=grid_lo, grid_hi=grid_hi)
    C_f32 = (torch.randn(h, L, r, dtype=torch.float32, device=device) * 0.1)
    z_bf16 = z_f32.to(torch.bfloat16)
    C_bf16 = C_f32.to(torch.bfloat16)

    # Out-of-range fraction (informational)
    scale = G / (grid_hi - grid_lo)
    u = (z_f32 - grid_lo) * scale
    in_range_frac = float(((u >= 0.0) & (u <= G)).float().mean().item())

    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference as ref_fwd,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward as kern_fwd,
    )

    # Ground truth: fp64 reference
    GT = ref_fwd(
        z_f32.double(), C_f32.double(),
        grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )

    # Lossy reference (bf16)
    ref_b = ref_fwd(z_bf16, C_bf16, grid_lo=grid_lo, grid_hi=grid_hi, G=G)

    # Kernel — bf16 inputs (production-shape)
    kern_b = kern_fwd(z_bf16, C_bf16, grid_lo=grid_lo, grid_hi=grid_hi, G=G)

    # Kernel — fp32 inputs (algorithmic-only error)
    kern_f = kern_fwd(z_f32, C_f32, grid_lo=grid_lo, grid_hi=grid_hi, G=G)

    # Compare everyone against GT, splitting phi vs delta halves
    res = {"in_range_frac": in_range_frac}
    for tag, val in [("ref_b", ref_b), ("kern_b", kern_b), ("kern_f", kern_f)]:
        res[f"{tag}_phi_rel_rms"]   = _rel_rms(val[:, :h], GT[:, :h])
        res[f"{tag}_phi_abs_max"]   = _abs_max(val[:, :h], GT[:, :h])
        res[f"{tag}_delta_rel_rms"] = _rel_rms(val[:, h:], GT[:, h:])
        res[f"{tag}_delta_abs_max"] = _abs_max(val[:, h:], GT[:, h:])

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["small", "med", "big"], default="med")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.shape == "small":
        N, h, r, G = 64, 128, 16, 10
    elif args.shape == "med":
        N, h, r, G = 512, 768, 64, 20
    else:
        N, h, r, G = 2048, 1024, 64, 22

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, abort.")
        return

    print(f"\n=== FlashSplineFeature numerical audit ({args.shape}) ===")
    print(f"  N={N}  h={h}  r={r}  G={G}  device={device}")
    print("  GT = fp64 reference (ground truth)")
    print("  ref_b  = bf16 reference   (current pytest oracle)")
    print("  kern_b = bf16 kernel      (production input shape)")
    print("  kern_f = fp32 kernel      (algorithmic-only)")
    print()

    print(f"{'workload':<11} {'in_range':>9}  ", end="")
    for tag in ["ref_b", "kern_b", "kern_f"]:
        print(f"{tag+'_d_rel':>14} ", end="")
    print()
    print("-" * 90)
    for w in ["uniform", "skewed", "collapsed"]:
        r_w = audit_workload(w, N=N, h=h, r=r, G=G, device=device)
        print(f"{w:<11} {r_w['in_range_frac']:>9.1%}  ", end="")
        for tag in ["ref_b", "kern_b", "kern_f"]:
            print(f"{r_w[f'{tag}_delta_rel_rms']:>14.3e} ", end="")
        print()

    print()
    print("Diagnosis:")
    print("  - kern_f rel_rms close to fp32 noise floor (~1e-6)?  → algo correct")
    print("  - kern_b rel_rms ≤ ref_b rel_rms?                    → kernel beats lossy reference")
    print("  - skewed kern_b > uniform kern_b?                    → workload-dependent bf16 noise")
    print("  - skewed kern_f > uniform kern_f?                    → real algorithmic issue with skewed inputs")


if __name__ == "__main__":
    main()
