# M3 Python binding overhead baseline

- Date: 2026-07-27
- Host: linux/amd64, CPython 3.13.14, PyO3/NumPy 0.29
- Mode: release extension, 64 sequential worlds, 2,000 Python calls
- Command: `uv run python tools/benchmark_bindings.py`

```json
{"calls":2000,"microseconds_per_call":127.70318999992014,"seconds":0.2554063799998403,"world_steps_per_second":501162.10879336705,"worlds":64}
```

The measured call cost is approximately 1.995 microseconds per world-step at
batch size 64, including conversion of 64 canonical state rows into a contiguous
NumPy matrix. This is a local regression baseline, not a portability claim.
