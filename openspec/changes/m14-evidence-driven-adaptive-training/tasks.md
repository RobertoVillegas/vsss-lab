## 1. Evaluation foundation

- [x] Review relevant 2024–2026 primary literature and record adopted and
      rejected directions
- [x] Review Ballchasing, RLGym, RLGymPPO_CPP, and released Nexto deployment
      tooling
- [ ] Freeze an M13 promoted baseline from a completed multi-seed comparison
- [ ] Define train, frontier, failure-replay, and immutable holdout suites
- [ ] Implement paired multi-seed evaluation with confidence intervals
- [ ] Add constrained promotion and machine-readable decision artifacts

## 2. Replay analytics

- [ ] Version possession, pressure, positioning, movement, and coordination
      definitions
- [ ] Implement per-match, per-team, and per-robot derived analytics
- [ ] Detect shots, saves, clearances, passes, assists, interceptions, double
      commits, and last-defender failures
- [ ] Add timelines, heatmaps, comparison tables, filters, and tabular export
- [ ] Validate sampling tolerance and event attribution with golden replays
- [ ] Feed failure descriptors into curriculum allocation, never directly into
      reward without an anti-farming ablation

## 3. Adaptive scenarios and reward search

- [ ] Define the typed, validity-checked scenario parameter space
- [ ] Implement learning-progress allocation and deduplicated failure replay
- [ ] Add routine/frontier/failure/holdout mixture telemetry
- [ ] Add resumable Optuna study storage and lineage manifests
- [ ] Implement smoke, screen, and confirm fidelities with pruning
- [ ] Run a fixed-reward curriculum ablation before joint reward search
- [ ] Run a bounded multiobjective reward search with paired seeds

## 4. Policy and teacher ablations

- [ ] Add recurrent MAPPO behind an explicit experiment configuration
- [ ] Compare MLP and GRU policies under camera-realistic partial observability
- [ ] Add a batched entity-attention policy and expose attention telemetry
- [ ] Compare continuous wheels with one symmetric, physically meaningful
      action abstraction at matched control frequency
- [ ] Implement a bounded exact-simulator planner for one atomic skill
- [ ] Compare scratch, verified imitation warm start, and MAPPO fine-tuning
- [ ] Keep KAN and learned world models deferred unless their entry gates pass

## 5. Population consolidation

- [ ] Retain a bounded behaviorally diverse checkpoint league
- [ ] Add an Elo-like historical rating with paired-color confidence reporting
- [ ] Measure whether specialists outperform the best single policy
- [ ] Distill complementary specialists only after a positive league result
- [ ] Verify that the distilled policy preserves latency and terminal outcomes

## 6. Accelerator feasibility

- [ ] Profile physics, bindings, copies, reward, inference, and PPO update
- [ ] Remove avoidable host allocation and conversion in vector rollouts
- [ ] Benchmark device-resident reward, observation, and reset kernels
- [ ] Prototype GPU VSSS primitives only if physics dominates the profile
- [ ] Compare accelerator traces with authoritative Rapier contacts and goals
- [ ] Record an adopt-or-reject decision for Warp/Newton or custom Rust/CUDA

## 7. Delivery

- [ ] Record a current comparative evidence note before every major block
- [ ] Document commands, study artifacts, dashboards, and rollback
- [ ] Add contract, determinism, resume, and failure-injection tests
- [ ] Record CPU/CUDA throughput and total compute for every promoted study
- [ ] Run doctor, build, test, lint, CUDA smoke, and strict OpenSpec validation
- [ ] Commit signed Conventional Commits and push
