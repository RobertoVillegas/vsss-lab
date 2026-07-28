## 1. Physical behavior

- [x] 1.1 Add goal side/back collision walls and containment tests
- [x] 1.2 Add force-derived actuator slew and correctness test
- [x] 1.3 Add action-delta reward regularization
- [x] 1.4 Record commanded and applied wheel telemetry in replays

## 2. Accelerated training

- [x] 2.1 Add auto/CPU/CUDA device selection with explicit fallback
- [x] 2.2 Batch network inference and PPO data over configurable vector worlds
- [x] 2.3 Expose device and vector count through CLI and Just recipes
- [x] 2.4 Exercise a real CUDA bootstrap, rollout, update, and checkpoint

## 3. Observability and evidence

- [x] 3.1 Add Rich table and bottom progress bar with non-TTY fallback
- [x] 3.2 Document physical provenance and measured CPU/CUDA throughput
- [x] 3.3 Run focused gates and archive the validated OpenSpec
