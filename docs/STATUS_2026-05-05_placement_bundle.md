# Status — placement / rank / Run A bundle frozen
**Date:** 2026-05-05

This bundle adds **only** dispatcher prep for the P0 sequential experiments
(placement sweep, rank sweep, clean Run A 200 M). No live training was kicked
off. The bundle is **frozen**: no further changes until the H100 dispatch is
complete and the results are in.

## Scope

- `nanochat/nanochat/gpt.py` — added `rlkv_placement_indices()` helper,
  `RLKV_PLACEMENT_TYPES` tuple, and four new placement variants
  (`early33`, `middle33`, `late20`, `late50`). Existing `mlp` / `rl_kv_b2`
  / `rl_kv_b2_late33` paths are unchanged; `late33` still produces the
  ceil(n/3) = 7 layers for d20 that the existing Run C training used.
- `nanochat/scripts/base_train.py` — `--ffn-type` choices extended; banner
  now prints the resolved layer-index list at startup.
- `benchmarks/modal_h100_placement_sweep.py` — 5-cell dispatcher (P0-Sequential-1).
- `benchmarks/modal_h100_rank_sweep.py` — 5-cell dispatcher (P0-Sequential-2).
- `benchmarks/modal_h100_runA_clean_200M.py` — 1-cell dispatcher (P0-Sequential-4).

All three Modal dispatchers default to dry-run; pass `--execute` to fire.

## H100 dispatch order (when chat-eval frees Modal)

```
.venv/bin/python benchmarks/modal_h100_placement_sweep.py --execute   # ~3 hr
.venv/bin/python benchmarks/modal_h100_runA_clean_200M.py --execute   # ~30 min
.venv/bin/python benchmarks/modal_h100_rank_sweep.py      --execute   # ~2.5 hr
```

Run-A clean is intentionally placed **between** the two sweeps: it is short
(30 min) and closes the largest confound on the negative control (the d20
Run A had Modal preempt + dataloader RNG reset). The placement sweep is
the headline experiment so it goes first; rank sweep is the second main
line of evidence (capacity / rank bottleneck).

## Test status

After the bundle landed I ran `.venv/bin/python -m pytest tests/ -x -q`. The
lightweight tests (`test_b2_spline.py` 5/5, `test_cli.py` 8/8,
`test_diagnostics.py` 13/13, `test_flash_spline_feature_autograd.py` partial)
all passed. The remaining 200 tests are CUDA-kernel-heavy and were
**deferred to the H100 parity suite** (P2-Parallel-5 / P0-Sequential-3) — none
of them exercise the placement-dispatch code path I edited, which is a pure
addition to `make_ffn()`'s name → layer-index map.

## Frozen — do not modify in this branch

- `nanochat/nanochat/gpt.py` make_ffn / rlkv_placement_indices
- `nanochat/scripts/base_train.py` --ffn-type choices + banner
- `benchmarks/modal_h100_placement_sweep.py`
- `benchmarks/modal_h100_rank_sweep.py`
- `benchmarks/modal_h100_runA_clean_200M.py`

## Soft recommendations for a *future* polish pass (not this branch)

1. Warn when a placement variant resolves to an empty layer set (small d).
2. Emit a header line in the dispatcher output containing seed, budget, step
   count, ffn_type, rank — so the dry-run is self-describing.
3. Add a `--force` / duplicate-tag guard in the dispatchers so an
   accidental `--execute` rerun cannot overwrite an existing checkpoint dir
   on the Modal volume.

None of these block firing the sweeps.

## Pointers

- Plan: `docs/PLAN_2026-05-04_neurips_experiment_queue.md`
- Companion analyses (already done): `RESULTS_2026-05-04_core_uncertainty.md`,
  `RESULTS_2026-05-04_core_trajectory.md`, `RESULTS_2026-05-04_cost_normalized.md`,
  `RESULTS_2026-05-04_failure_cases.md`.
