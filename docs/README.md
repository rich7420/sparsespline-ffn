# Documentation

Start here:

- [Architecture](ARCHITECTURE.md)
- [Theory](THEORY.md)
- [Migration from pal-kan](MIGRATION_FROM_PAL_KAN.md)

The library contract is intentionally small: the PyTorch reference module,
initialization helpers, and the transformer FFN factory.  Kernel-specific
documentation should be added only after a fused implementation exists.
