# Contributing

ThirdEye treats evaluation methodology as production code.

Before submitting a change:

1. Add or update tests for schemas, grading, lineage, or report behavior.
2. Preserve the boundary between observational and controlled evidence.
3. Do not display a score without its protocol, scorer, lineage, and uncertainty.
4. Keep local mode usable without a hosted service.

Run:

```powershell
python -m pip install -e ".[dev,server]"
python -m pytest
ruff check thirdeye tests
```

