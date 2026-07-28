## 1. Evaluation foundation

- [x] Review relevant 2024–2026 primary literature and record adopted and
      rejected directions
- [ ] Freeze an M13 promoted baseline from a completed multi-seed comparison
- [ ] Define train, frontier, failure-replay, and immutable holdout suites
- [ ] Implement paired multi-seed evaluation with confidence intervals
- [ ] Add constrained promotion and machine-readable decision artifacts

## 2. Adaptive scenarios and reward search

- [ ] Define the typed, validity-checked scenario parameter space
- [ ] Implement learning-progress allocation and deduplicated failure replay
- [ ] Add routine/frontier/failure/holdout mixture telemetry
- [ ] Add resumable Optuna study storage and lineage manifests
- [ ] Implement smoke, screen, and confirm fidelities with pruning
- [ ] Run a fixed-reward curriculum ablation before joint reward search
- [ ] Run a bounded multiobjective reward search with paired seeds

## 3. Policy and teacher ablations

- [ ] Add recurrent MAPPO behind an explicit experiment configuration
- [ ] Compare MLP and GRU policies under camera-realistic partial observability
- [ ] Implement a bounded exact-simulator planner for one atomic skill
- [ ] Compare scratch, verified imitation warm start, and MAPPO fine-tuning
- [ ] Keep KAN and learned world models deferred unless their entry gates pass

## 4. Population consolidation

- [ ] Retain a bounded behaviorally diverse checkpoint league
- [ ] Measure whether specialists outperform the best single policy
- [ ] Distill complementary specialists only after a positive league result
- [ ] Verify that the distilled policy preserves latency and terminal outcomes

## 5. Accelerator feasibility

- [ ] Profile physics, bindings, copies, reward, inference, and PPO update
- [ ] Remove avoidable host allocation and conversion in vector rollouts
- [ ] Benchmark device-resident reward, observation, and reset kernels
- [ ] Prototype GPU VSSS primitives only if physics dominates the profile
- [ ] Compare accelerator traces with authoritative Rapier contacts and goals
- [ ] Record an adopt-or-reject decision for Warp/Newton or custom Rust/CUDA

## 6. Delivery

- [ ] Document commands, study artifacts, dashboards, and rollback
- [ ] Add contract, determinism, resume, and failure-injection tests
- [ ] Record CPU/CUDA throughput and total compute for every promoted study
- [ ] Run doctor, build, test, lint, CUDA smoke, and strict OpenSpec validation
- [ ] Commit signed Conventional Commits and push
