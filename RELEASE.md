# Release Checklist

## 0.1.x Baseline

1. Run `scripts/check_project.sh`.
2. Confirm `sparsespline_ffn.__version__` matches `pyproject.toml`.
3. Update `CHANGELOG.md`.
4. Confirm GitHub Actions is green on `main`.
5. Tag the release:

   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```

## Compatibility Policy

Until `1.0.0`, the project may change APIs while the FullMix-Tucker design and
kernel contract are still being validated.  The reference implementation,
config object, and `build_ffn` are the intended stable surface.
