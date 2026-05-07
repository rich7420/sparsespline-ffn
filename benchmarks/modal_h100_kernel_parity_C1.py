"""Run the formal kernel parity harness on H100 (C1 deliverable).

This is the C1 wrapper for `tests/test_rlkv_kernel_parity.py` — the C0
local-shape harness already exists; this file dispatches the same pytest
file to a 1×H100 SXM container so the WGMMA-only / production-shape /
H100-CUDA-extension parametrisations actually execute.

Why C1 needs a dedicated wrapper:
  - The local 3080 auto-skips `wgmma_kernel`, `v10_cuda`, `v11_cuda`,
    `hopper_cuda`, and `wgmma_v5_cuda` parametrisations (sm_80 < sm_90a).
  - The local 3080 also lacks the 80 GB needed for production shapes
    (N=32768, h=2560 ≈ 56 GB activation budget).
  - On H100 those parametrisations un-skip and we get the same correctness
    coverage as the d20 production training stack actually used.

What this writes back to repo dir:
  - `dispatcher_runs/2026-05-05_kernel_parity_C1_h100.log` (raw pytest output)
  - `docs/_artifacts/kernel_parity_C1_h100_summary.md` (markdown table)

Reproducer:
    /home/rich-wsl/.local/bin/modal run \\
        benchmarks/modal_h100_kernel_parity_C1.py::main
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
        "numpy", "pytest", "pytest-timeout", "pyarrow", "tokenizers",
        "tiktoken", "regex", "huggingface-hub", "ninja",
    )
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**",
                "nanochat/.nanochat-runtime/**", "nanochat/.venv/**",
                "benchmark_runs/**", "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-kernel-parity-C1-h100", image=IMAGE)


@app.function(gpu="H100", timeout=3600)
def run_parity_h100(pytest_filter: str = "") -> dict:
    """Run pytest on the H100, STREAMING stdout/stderr line-by-line.

    Streaming output is critical — earlier C1 attempt with capture_output=True
    hung silently on a single test for >29 min before being killed. With
    streaming we see the failing/hanging test name in real time.
    """
    import os
    import re
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": "/repo/nanochat:/repo/src",
        "OMP_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",  # ensure pytest doesn't buffer per-test output
    }
    cmd = [
        sys.executable, "-u", "-m", "pytest",
        "/repo/tests/test_rlkv_kernel_parity.py",
        "-v", "--tb=short",
        "-s",                     # disable pytest output capturing (CRITICAL)
        "--timeout=300",          # per-test timeout
        "--color=no",
    ]
    if pytest_filter:
        cmd += ["-k", pytest_filter]
    print(f"[c1] CMD: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd, cwd="/repo", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    log_lines = []
    for line in proc.stdout:
        # Print line-by-line so Modal forwards it back to the local tee'd log.
        print(line, end="", flush=True)
        log_lines.append(line)
    rc = proc.wait()
    log = "".join(log_lines)

    # Parse pytest summary: "53 passed, 7 skipped in 234.5s"
    summary_re = re.compile(
        r"=+\s*((?:\d+\s+(?:passed|failed|skipped|error|warning)s?,?\s*)+)"
        r"in\s+([\d.]+)s")
    m = summary_re.search(log)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    duration = None
    if m:
        for chunk in m.group(1).split(","):
            chunk = chunk.strip()
            for k in counts:
                if chunk.endswith(k) or chunk.endswith(k + "s"):
                    counts[k] = int(chunk.split()[0])
        duration = float(m.group(2))
    return {
        "rc": rc,
        "counts": counts,
        "duration_s": duration,
        "log": log,
    }


@app.local_entrypoint()
def main(pytest_filter: str = "") -> None:
    """Dispatch parity pytest to H100 and write summary back to repo dir.

    Args:
        pytest_filter: optional pytest -k filter, e.g. "h100 or wgmma" to
            only run the H100-specific parametrisations.
    """
    import json
    import os
    from datetime import datetime

    repo_root = "/home/rich-wsl/sparsespline-ffn"
    log_path = os.path.join(
        repo_root, "dispatcher_runs",
        f"2026-05-05_kernel_parity_C1_h100.log")
    summary_path = os.path.join(
        repo_root, "docs", "_artifacts",
        f"kernel_parity_C1_h100_summary.md")

    print(f"[c1] dispatching to H100 (filter={pytest_filter or '<all>'}) ...",
          flush=True)
    result = run_parity_h100.remote(pytest_filter=pytest_filter)

    # 1. Raw pytest log
    with open(log_path, "w") as f:
        f.write(result["log"])
    print(f"[c1] raw log → {log_path}", flush=True)

    # 2. Markdown summary
    counts = result["counts"]
    duration = result.get("duration_s") or 0.0
    rc = result["rc"]
    verdict = "PASS" if rc == 0 and counts["failed"] == 0 else "FAIL"
    md = f"""# Kernel parity harness — C1 (H100) summary
**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Pytest filter:** `{pytest_filter or '(none — full file)'}`
**Verdict:** **{verdict}**

| Metric | Value |
|---|---:|
| Tests passed | {counts['passed']} |
| Tests failed | {counts['failed']} |
| Tests skipped | {counts['skipped']} |
| Tests errored | {counts['error']} |
| Total duration (s) | {duration:.1f} |
| Pytest exit code | {rc} |

Raw log: `dispatcher_runs/2026-05-05_kernel_parity_C1_h100.log`

## How to reproduce

```bash
/home/rich-wsl/.local/bin/modal run \\
    benchmarks/modal_h100_kernel_parity_C1.py::main
```

To run only the H100-specific parametrisations (skip the local-shape
subset that already passed on the 3080):

```bash
/home/rich-wsl/.local/bin/modal run \\
    benchmarks/modal_h100_kernel_parity_C1.py::main \\
    --pytest-filter "h100 or wgmma"
```
"""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(md)
    print(f"[c1] summary  → {summary_path}", flush=True)

    print(f"\n[c1] verdict = {verdict}  "
          f"({counts['passed']} passed, {counts['failed']} failed, "
          f"{counts['skipped']} skipped in {duration:.1f}s)", flush=True)
    if verdict == "FAIL":
        raise SystemExit(2)
