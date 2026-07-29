# Evidence

## M20 run diagnosis

Run: `/home/rob/runs/vsss-m20-run-0001`

- Checkpoint 375: replay idle spin 3.7%, semantic success 47.0%, unresolved 59.
- Checkpoint 400: replay idle spin 2.3%.
- Checkpoint 425: replay idle spin 37.7%, semantic success 27.4%, unresolved 93.
- At checkpoint 425, shot holdout success was 0%, pass/receive 8.3%, and
  rotation/recovery 0%.
- Geometry shaping was negative on 2,782 of 2,999 replay transitions; the
  geometry potential did not reward the spin.
- `best-semantic.json` correctly retained checkpoint 375.

The failure mode exploited indifference: constant opposite wheel commands stop
paying action-delta cost, generic wheel effort is small, and stagnation is
delayed and team-shared.

## Warm-start smoke

- Source: M20 `iteration-000375.pt`, policy version 375.
- Initialization reset optimizer, policy version, RNG, and curriculum.
- CPU integration smoke: 4,096 steps and 35 matches.
- CUDA smoke: 2,048 steps and 16 matches.
- Deterministic idle-spin evaluation: 1.0% then 3.2%, both below the 8% gate.
- Training idle-spin telemetry: approximately 3.7% on the two smoke iterations.

## Gates

- Python: 212 passed.
- Rust physics: 14 correctness tests passed.
- Web: 7 passed.
- Lint, type checking, protocol compatibility, and build: passed.
