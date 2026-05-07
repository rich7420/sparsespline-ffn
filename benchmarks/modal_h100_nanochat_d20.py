"""nanochat d20 reference-recipe launcher.

Wraps nanochat's stock `scripts/base_train.py` — DOES NOT modify the
recipe.  Two Modal entry points:

  * `run_d20_8gpu`  — 8 × H100 SXM, torchrun --standalone --nproc_per_node=8
  * `run_d20_1gpu`  — 1 × H100 (Smoke A only)

Both forward all CLI args verbatim to base_train.py so flag changes
upstream (Karpathy's repo) automatically pick up.

Usage:
    modal run benchmarks/modal_h100_nanochat_d20.py::main \
        --mode smoke_a --run-tag mlp_d20_smoke_a

    modal run benchmarks/modal_h100_nanochat_d20.py::main \
        --mode smoke_b --run-tag mlp_d20_smoke_b

    modal run benchmarks/modal_h100_nanochat_d20.py::main \
        --mode full_mlp --run-tag mlp_d20_reference_seed0
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "numpy", "pytest", "pyarrow", "tokenizers",
        "tiktoken", "regex", "huggingface-hub", "ninja",
        "wandb", "datasets",
        # psutil — used by nanochat.report at end of base_train.py
        # to log peak memory.  Missing it caused a cosmetic post-train
        # crash (exit 1) on the V2 MLP run even though all 16,600 steps
        # and the final eval completed cleanly.
        "psutil",
    )
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        # dispatcher_runs/** holds the local `tee` log written during launch;
        # if not ignored, Modal sees the file change during image build and
        # aborts with "modified during build process".
        ignore=[".venv/**", ".git/**",
                "nanochat/.nanochat-runtime/**", "nanochat/.venv/**",
                "benchmark_runs/**", "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands(
        "cd /repo && pip install -e .",
    )
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-nanochat-d20", image=IMAGE)


# Map mode → arg list for `python -m scripts.base_train`.  Recipe fields
# below match nanochat#1 reference EXACTLY (Karpathy's d20 speedrun);
# only `--num-iterations`, `--save-every`, `--core-metric-max-per-task`
# differ between smokes and full.
def build_args(mode: str, run_tag: str, ffn_type: str = "mlp",
                resume_from: int = -1,
                c_lr: float = 0.02, c_weight_decay: float = 0.0,
                rlkv_h_ratio: float = 2.0, rlkv_r: int = 32,
                rlkv_l: int = 22,  # NOTE: lowercase 'l' for Modal CLI compat
                rlkv_fwd_kernel: str = "v11_cuda",
                # NOTE: wgmma_v5_cuda's dispatch table caps chunks_per_block ≤ 8,
                # which limits the kernel to N ∈ {1024, 2048, 4096} at NPARTS=4,
                # BN=128. Production d20 microbatch is N = device_batch_size ×
                # max_seq_len = 32 × 2048 = 65 536, far above v5's range.
                # Default to Triton bwd (autotunes for any N) until v5's
                # template table is extended to cover larger chunks_per_block
                # or NPARTS is made adaptive. v5 still works for microbench /
                # ablation at small N.
                rlkv_bwd_kernel: str = "triton",
                rlkv_diagnostics_every: int = 0,
                lr_schedule_num_iterations: int = 0,
                rlkv_grid_lo: float = -3.0,
                rlkv_grid_hi: float = 3.0,
                rlkv_lambda_warmup_steps: int = 0,
                rlkv_lambda_warmup_lo: float = 1.0) -> list[str]:
    # `--run=dummy` disables wandb (no API key on Modal).
    # `--model-tag=<run_tag>` controls the checkpoint subdirectory so
    # different runs / smokes don't collide in /data/nanochat/base_checkpoints.
    common = [
        "--depth=20",
        "--device-batch-size=32",
        "--total-batch-size=524288",
        "--max-seq-len=2048",
        "--eval-tokens=10485760",
        "--run=dummy",
        f"--model-tag={run_tag}",
    ]
    if ffn_type != "mlp":
        # Full RL-KV CLI passthrough.  See nanochat/scripts/base_train.py
        # parser for full schema.
        common += [
            "--ffn-type", ffn_type,
            "--rlkv-h-ratio", str(rlkv_h_ratio),
            "--rlkv-r", str(rlkv_r),
            "--rlkv-L", str(rlkv_l),  # base_train.py argparse stores this as args.rlkv_L
            "--rlkv-fwd-kernel", rlkv_fwd_kernel,
            "--rlkv-bwd-kernel", rlkv_bwd_kernel,
            "--rlkv-grid-lo", str(rlkv_grid_lo),
            "--rlkv-grid-hi", str(rlkv_grid_hi),
            "--rlkv-lambda-warmup-steps", str(rlkv_lambda_warmup_steps),
            "--rlkv-lambda-warmup-lo", str(rlkv_lambda_warmup_lo),
            "--c-lr", str(c_lr),
            "--c-weight-decay", str(c_weight_decay),
            # RL-KV at full d20 batch (device_batch=32 × 2048 = 65 536 tokens
            # per rank) OOMs at the cross-entropy stage on 80 GB H100 SXM.
            # Diagnosis: flash_spline_feature saves z [N, h] bf16 ≈ 320 MB
            # per layer × 20 layers = 6.4 GB of activation graph state that
            # MLP doesn't have. Halve device-batch so that activation memory
            # halves; total-batch is unchanged via grad-accum, so per-step
            # gradient and training trajectory are bit-identical to the full
            # batch case. argparse takes the LAST value, so any later
            # mode-specific override (e.g. smoke_a's batch-of-4) still wins.
            "--device-batch-size=16",
        ]

    if mode == "smoke_a":
        # 1 × H100, 10 iter, build / params / one step.
        # Override common's --device-batch-size=32 / --total-batch-size=524288
        # because the d20 model + AdamW + Muon + RL-KV spline activations
        # don't fit a single 80 GB H100 at full batch; smoke_a is just a
        # build sanity check, so a tiny micro batch is fine. argparse takes
        # the LAST value when a flag is repeated, so these overrides win.
        return common + [
            "--device-batch-size=4",
            "--total-batch-size=8192",        # 4 seqs × 2048 tok = 8 K → grad-accum 1
            "--num-iterations=10",
            "--eval-every=-1",
            "--core-metric-every=-1",
            "--save-every=-1",
        ]
    if mode == "smoke_b":
        # 8 × H100, 100 iter, eval + CORE + ckpt.
        # core-metric-max-per-task=50 matches the full-run setting (and
        # Karpathy's default); smaller values crash with "Sample larger
        # than population" because some CORE tasks have <10 examples.
        return common + [
            "--num-iterations=100",
            "--eval-every=50",
            "--core-metric-every=100",
            "--core-metric-max-per-task=50",
            "--save-every=50",
        ]
    if mode == "smoke_c":
        # 8 × H100, resume from step 50 → step 150
        return common + [
            "--num-iterations=150",
            "--eval-every=50",
            "--core-metric-every=-1",
            "--save-every=-1",
            "--resume-from-step", str(resume_from),
        ]
    if mode == "full_mlp":
        # MLP d20 paper run.  --target-param-data-ratio=20 auto-derives
        # ~16 600 iterations / 8.70 B tokens at our vocab=32K.
        return common + [
            "--target-param-data-ratio=20",
            "--eval-every=250",
            "--core-metric-every=2000",
            "--core-metric-max-per-task=50",
            "--save-every=2000",
        ]
    if mode == "full_rl_kv":
        # RL-KV d20 paper run — TOKEN BUDGET LOCKED to MLP d20 by passing
        # --num-iterations=16600 EXPLICITLY (not --target-param-data-ratio).
        # Reason: RL-KV's transformer_matrices is smaller than MLP's, so
        # auto-deriving from target-param-data-ratio=20 would give RL-KV
        # only ~13 000 steps / 6.8 B tokens, confounding the comparison.
        # See plan §18 "Token-budget lock for RL-KV vs MLP".
        full_rl_kv_args = common + [
            "--num-iterations=16600",
            "--eval-every=250",
            "--core-metric-every=2000",
            "--core-metric-max-per-task=50",
            "--save-every=2000",
        ]
        if resume_from > 0:
            # Resume from a previously saved ckpt step. Used after Modal
            # worker preemption — preempted Function is auto-restarted with
            # the SAME input by Modal, so passing --resume-from-step here
            # makes the auto-restart pick up from the last save_every ckpt
            # instead of restarting training from scratch (loses ~3 hr).
            full_rl_kv_args += ["--resume-from-step", str(resume_from)]
        return full_rl_kv_args
    if mode == "rlkv_pilot_1B":
        # 1 B-token mini sweep — same fixed step count for r32 and r64
        # grid fairness.  Do NOT use --target-param-data-ratio here.
        return common + [
            "--num-iterations=1900",
            "--eval-every=200",
            "--core-metric-every=1000",
            "--core-metric-max-per-task=50",
            "--save-every=-1",
        ]
    if mode == "rlkv_pilot_1B_diag":
        # Same as rlkv_pilot_1B but with architectural diagnostics enabled.
        # Reports y_delta_rms/y_base_rms, C_norm, bin_entropy etc. every eval.
        return common + [
            "--num-iterations=1900",
            "--eval-every=200",
            "--core-metric-every=1000",
            "--core-metric-max-per-task=50",
            "--save-every=-1",
            f"--rlkv-diagnostics-every={rlkv_diagnostics_every or 200}",
        ]
    if mode == "rlkv_pilot_2B_fullsched":
        # 2B finalist with full-d20 LR schedule prefix (per reviewer §5):
        # train 3800 steps but normalize warmup/warmdown over 16 600 — the
        # entire 3800-step run sits in flat-LR phase, isolating the question
        # "does more flat-LR time close the gap?".
        return common + [
            "--num-iterations=3800",
            f"--lr-schedule-num-iterations={lr_schedule_num_iterations or 16600}",
            "--eval-every=200",
            "--core-metric-every=2000",
            "--core-metric-max-per-task=50",
            "--save-every=-1",
            f"--rlkv-diagnostics-every={rlkv_diagnostics_every or 200}",
        ]
    if mode == "mlp_pilot_1B_db16":
        # CONTROLLED SPEED BASELINE for the RL-KV pilot grid.
        # Same microbatch config as the RL-KV pilots so that throughput +
        # quality comparisons isolate the architecture from the
        # device_batch / grad_accum knob.  Note that the MLP V2 reference
        # baseline uses device_batch=32 (best feasible), this is the
        # "matched microbatch" cell:
        #     MLP_V2 (device_batch=32, grad_accum=1) = "best feasible MLP"
        #     mlp_pilot_1B_db16 (16, 2)              = "matched microbatch"
        #     RL-KV pilots      (16, 2)              = "best feasible RL-KV"
        # Reviewer-fairness: this run pins down whether the RL-KV vs MLP V2
        # quality lead is due to architecture or the grad_accum difference.
        # Same eval / CORE / save schedule as rlkv_pilot_1B for direct
        # bpb-curve overlay.
        return common + [
            "--device-batch-size=16",
            "--num-iterations=1900",
            "--eval-every=200",
            "--core-metric-every=1000",
            "--core-metric-max-per-task=50",
            "--save-every=-1",
        ]
    if mode == "rlkv_pilot_2B":
        # 2 B-token gate pilot — same fixed step count for arch comparison.
        return common + [
            "--num-iterations=3800",
            "--eval-every=200",
            "--core-metric-every=2000",
            "--core-metric-max-per-task=50",
            "--save-every=-1",
        ]
    raise ValueError(f"unknown mode: {mode}")


def _run_base_train(num_gpus: int, args_list: list[str]) -> tuple[int, str, str]:
    """Invoke base_train.py with proper torchrun-vs-python dispatch."""
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": "/repo/nanochat:/repo/src",
        "NANOCHAT_BASE_DIR": "/data/nanochat",
        "OMP_NUM_THREADS": "1",
    }
    if num_gpus > 1:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m", "scripts.base_train", "--",
        ] + args_list
    else:
        cmd = ["python", "-m", "scripts.base_train"] + args_list
    print(f"[launcher] CMD: {' '.join(cmd)}", flush=True)
    print(f"[launcher] cwd=/repo/nanochat  num_gpus={num_gpus}", flush=True)
    proc = subprocess.run(cmd, cwd="/repo/nanochat", env=env,
                            capture_output=False, check=False)
    return proc.returncode, "", ""


@app.function(gpu="H100", timeout=2 * 3600,
                volumes={"/data": DATA_VOLUME})
def run_d20_1gpu(args_list: list[str]) -> dict:
    rc, _, _ = _run_base_train(1, args_list)
    return {"rc": rc, "num_gpus": 1}


@app.function(gpu="H100:8", timeout=24 * 3600,
                volumes={"/data": DATA_VOLUME})
def run_d20_8gpu(args_list: list[str]) -> dict:
    rc, _, _ = _run_base_train(8, args_list)
    return {"rc": rc, "num_gpus": 8}


@app.local_entrypoint()
def main(mode: str = "smoke_a", run_tag: str = "",
         ffn_type: str = "mlp",
         resume_from: int = -1,
         c_lr: float = 0.02, c_weight_decay: float = 0.0,
         rlkv_h_ratio: float = 2.0, rlkv_r: int = 32,
         rlkv_l: int = 22,
         rlkv_fwd_kernel: str = "v11_cuda",
         rlkv_bwd_kernel: str = "triton",
         rlkv_diagnostics_every: int = 0,
         lr_schedule_num_iterations: int = 0,
         rlkv_grid_lo: float = -3.0,
         rlkv_grid_hi: float = 3.0,
         rlkv_lambda_warmup_steps: int = 0,
         rlkv_lambda_warmup_lo: float = 1.0) -> None:
    """One Modal call per mode.

    mode ∈ {smoke_a, smoke_b, smoke_c, full_mlp, full_rl_kv,
              rlkv_pilot_1B, rlkv_pilot_1B_diag, rlkv_pilot_2B,
              rlkv_pilot_2B_fullsched}
    """
    if not run_tag:
        run_tag = f"{ffn_type}_d20_{mode}"
    args_list = build_args(
        mode, run_tag, ffn_type=ffn_type, resume_from=resume_from,
        c_lr=c_lr, c_weight_decay=c_weight_decay,
        rlkv_h_ratio=rlkv_h_ratio, rlkv_r=rlkv_r, rlkv_l=rlkv_l,
        rlkv_fwd_kernel=rlkv_fwd_kernel, rlkv_bwd_kernel=rlkv_bwd_kernel,
        rlkv_diagnostics_every=rlkv_diagnostics_every,
        lr_schedule_num_iterations=lr_schedule_num_iterations,
        rlkv_grid_lo=rlkv_grid_lo, rlkv_grid_hi=rlkv_grid_hi,
        rlkv_lambda_warmup_steps=rlkv_lambda_warmup_steps,
        rlkv_lambda_warmup_lo=rlkv_lambda_warmup_lo,
    )

    print(f"[launcher] mode={mode}  run_tag={run_tag}  ffn_type={ffn_type}",
          flush=True)
    print(f"[launcher] base_train.py args: {args_list}", flush=True)

    if mode == "smoke_a":
        # 1 × H100 — just verifies the model builds / one step works
        out = run_d20_1gpu.remote(args_list)
    else:
        # everything else uses 8 × H100
        out = run_d20_8gpu.remote(args_list)

    print(f"[launcher] result: {out}", flush=True)
