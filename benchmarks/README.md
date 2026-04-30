# Benchmarks

This directory is for small, reproducible benchmark entry points.  Keep large
experiment outputs, model checkpoints, and paper receipts out of git.

Recommended first benchmark categories:

- parameter count and active parameter count;
- forward/backward latency for `FullMixTuckerFFN` versus `MLPFFN`;
- activation memory under reference mode;
- future reference-vs-kernel numerical and speed comparisons.
