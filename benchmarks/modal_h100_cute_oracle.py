"""H100 standalone CuTe oracle (v7.0 — TMA→WGMMA layout-pipeline truth-table).

Phase 0: smoke that <cute/tensor.hpp> + GMMA::Layout_MN_SW64_Atom compiles.

Phase 1a/1b/1c will be added incrementally in this same file once each gate
passes. The whole point of the CuTe path is to avoid hand-rolling WGMMA
descriptors — CuTe atoms are pre-validated against PTX descriptor math.

Modal image differs from the wgmma_tma_test path: this one clones CUTLASS
into /opt/cutlass and adds it to extra_cuda_cflags include path.
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
    # CUTLASS is header-only. Pin to a recent stable release tag for build
    # reproducibility. v3.6.0 is from late 2024 and supports SM90a fully.
    .run_commands(
        "git clone --depth 1 --branch v3.6.0 "
        "https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
    # Patch CUTLASS issue #1997 (fixed by upstream PR #2171, present in main
    # but not in v3.6.0): make_gmma_desc is CUTE_HOST_DEVICE but calls
    # cast_smem_ptr_to_uint which is CUTE_DEVICE only — NVCC strict mode
    # rejects this. Upstream fixed it by promoting the leaf to CUTE_HOST_DEVICE
    # (host-side it just casts pointer numerically; the actual SMEM access only
    # happens on device, so the host half is benign). We do the same patch.
    .run_commands(
        "perl -i -0777 -pe "
        "'s/CUTE_DEVICE\\s+uint32_t\\s+cast_smem_ptr_to_uint/"
        "CUTE_HOST_DEVICE\\nuint32_t\\ncast_smem_ptr_to_uint/g' "
        "/opt/cutlass/include/cute/arch/util.hpp"
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-cute-oracle-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_oracle(phase: int = 0) -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from torch.utils.cpp_extension import load

    print(f"\n{'=' * 72}", flush=True)
    print(f"  CuTe oracle — phase {phase}", flush=True)
    print(f"{'=' * 72}", flush=True)

    # JIT-load the cute_oracle extension with CUTLASS include path. We do
    # this here (not in cuda_ext/__init__.py) because the CUTLASS dependency
    # is microtest-only — we don't want to pull /opt/cutlass into the
    # production image just for this oracle.
    print("Compiling cute_oracle.cu with CUTLASS include path...", flush=True)
    ext = load(
        name="cute_oracle_ext",
        sources=["/repo/src/sparsespline_ffn/cuda_ext/cute_oracle.cu"],
        extra_include_paths=[
            "/opt/cutlass/include",
            "/opt/cutlass/tools/util/include",
        ],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-std=c++17", "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xptxas=-v", "-lineinfo",
            # CUTLASS 3.x requires this for some constexpr templates
            "-DCUTE_USE_PACKED_TUPLE=1",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )

    print("Compile + load OK. Running oracle...", flush=True)
    out: dict = {"phase": phase}

    if phase == 0:
        # Smoke: kernel writes the actual atom shape dimensions so we can
        # see what tile_to_shape will require in Phase 1. Pass criterion is
        # just "the kernel launched and the magic word is intact".
        # Pass dummy A/B since the dispatcher signature requires them.
        dummy_A = torch.zeros((64, 16), dtype=torch.float16, device="cuda")
        dummy_B = torch.zeros((16, 32), dtype=torch.float16, device="cuda")
        results = ext.cute_oracle(0, dummy_A, dummy_B)
        sentinel = results[0]
        torch.cuda.synchronize()
        s = [sentinel[i].item() for i in range(10)]
        magic = s[9] & 0xFFFFFFFF
        print(f"  GMMA::Layout_MN_SW64_Atom<half_t>  shape: ({s[0]}, {s[1]})  size={s[8]}",
              flush=True)
        print(f"  GMMA::Layout_K_SW64_Atom <half_t>  shape: ({s[2]}, {s[3]})",
              flush=True)
        print(f"  GMMA::Layout_MN_SW128_Atom<half_t> shape: ({s[4]}, {s[5]})",
              flush=True)
        print(f"  GMMA::Layout_K_SW128_Atom <half_t> shape: ({s[6]}, {s[7]})",
              flush=True)
        print(f"  magic: 0x{magic:08X}", flush=True)
        passed = (magic == 0xC0DECAFE) and all(d > 0 for d in s[:8])
        out["atom_MN_SW64"]  = (s[0], s[1])
        out["atom_K_SW64"]   = (s[2], s[3])
        out["atom_MN_SW128"] = (s[4], s[5])
        out["atom_K_SW128"]  = (s[6], s[7])
        out["sentinel_magic"] = f"0x{magic:08X}"
        out["passed"]         = passed
        print(f"  passed: {'YES' if passed else 'NO'}", flush=True)

    elif phase == 11:
        # Phase 1a: TMA-only SW64 round-trip parity.
        torch.manual_seed(0)
        B = (torch.randn(16, 32, dtype=torch.float16, device="cuda") * 0.5)
        B = B + torch.arange(B.numel(), dtype=torch.float16, device="cuda").reshape(B.shape) * 0.001
        B = B.contiguous()
        dummy_A = torch.zeros((64, 16), dtype=torch.float16, device="cuda")

        results = ext.cute_oracle(11, dummy_A, B)
        dump = results[0]
        torch.cuda.synchronize()

        # Multiset check: dump should contain the SAME values as B, possibly
        # in a different linear order (because SW64 swizzle scrambles
        # addresses). Sort both and compare.
        B_sorted    = torch.sort(B.flatten().float())[0]
        dump_sorted = torch.sort(dump.flatten().float())[0]
        multiset_diff = (B_sorted - dump_sorted).abs().max().item()

        # Sanity: check that dump != B linearly (proves swizzle DID happen)
        linear_diff = (dump.flatten().float() - B.flatten().float()).abs().max().item()
        # Sanity: check dump isn't all zero
        dump_max = dump.abs().max().item()

        passed = (multiset_diff < 1e-3) and (dump_max > 0.0)

        print(f"  B  : shape={tuple(B.shape)}  max_abs={B.abs().max().item():.4f}",
              flush=True)
        print(f"  dump  shape={tuple(dump.shape)}  max_abs={dump_max:.4f}",
              flush=True)
        print(f"  multiset_diff_max  = {multiset_diff:.6f}  (must be ~0)",
              flush=True)
        print(f"  linear_diff_max    = {linear_diff:.6f}  (>0 means swizzle was applied)",
              flush=True)

        out["B_max_abs"]       = B.abs().max().item()
        out["dump_max_abs"]    = dump_max
        out["multiset_diff"]   = multiset_diff
        out["linear_diff"]     = linear_diff
        out["passed"]          = passed
        print(f"  passed: {'YES' if passed else 'NO'}", flush=True)

    elif phase == 13:
        # Phase 1c: TMA→WGMMA fused parity.
        torch.manual_seed(0)
        A = (torch.randn(64, 16, dtype=torch.float16, device="cuda") * 0.5).contiguous()
        B = (torch.randn(16, 32, dtype=torch.float16, device="cuda") * 0.5).contiguous()

        # Reference: full-fp32 matmul on the host-side fp32-cast tensors.
        D_ref = (A.float() @ B.float())

        results = ext.cute_oracle(13, A, B)
        D = results[0]
        torch.cuda.synchronize()

        diff = (D - D_ref).abs()
        max_abs   = diff.max().item()
        mean_abs  = diff.mean().item()
        max_rel   = (diff.max() / (D_ref.abs().max() + 1e-9)).item()
        d_max     = D.abs().max().item()
        ref_max   = D_ref.abs().max().item()
        d_zero    = d_max < 1e-3

        passed = (not d_zero) and (max_abs < 1e-2)

        print(f"  D       : max_abs={d_max:.4f}", flush=True)
        print(f"  D_ref   : max_abs={ref_max:.4f}", flush=True)
        print(f"  err     : max_abs={max_abs:.4f}  max_rel={max_rel:.4f}  "
              f"mean_abs={mean_abs:.4f}", flush=True)
        if not passed:
            print(f"\n  D[:4, :4]:", flush=True)
            print(f"  {D[:4, :4].cpu().numpy()}", flush=True)
            print(f"\n  D_ref[:4, :4]:", flush=True)
            print(f"  {D_ref[:4, :4].cpu().numpy()}", flush=True)
        print(f"  passed: {'YES' if passed else 'NO'}", flush=True)

        out["D_max_abs"]   = d_max
        out["D_ref_max"]   = ref_max
        out["max_abs_err"] = max_abs
        out["max_rel_err"] = max_rel
        out["mean_abs"]    = mean_abs
        out["passed"]      = passed

    else:
        raise RuntimeError(
            f"Phase {phase} not implemented. Use 0 / 11 / 13."
        )

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(phase: int = 0):
    out = run_oracle.remote(phase)
    if not out.get("passed"):
        raise SystemExit(1)
