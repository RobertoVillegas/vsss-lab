## 1. Contract and baseline

- [x] Record the frozen M14 configuration, checkpoint/evaluation identity, and
      current per-family behavior
- [x] Define versioned skill scenario parameters, difficulty axes, outcomes,
      and reason codes
- [x] Add an ADR for scenario compilation and semantic termination boundaries
- [x] Preserve loading of legacy static M14 scenarios

## 2. Deterministic scenario families

- [x] Implement deterministic parameter-to-canonical-state compilation
- [x] Implement mirror-safe approach, interception, save/deflection, clearance,
      shot, and pass/receive families
- [x] Randomize bounded ball speed/angle, robot spawn/heading, target geometry,
      and opponent pressure
- [x] Reject non-finite, outside-field, overlapping, unreachable, and
      already-terminal initial states
- [x] Add property tests over seeds, mirrors, difficulty bounds, and neighbor
      world isolation

## 3. Semantic predicates and termination

- [x] Implement typed running, success, failure, and unresolved outcomes
- [x] Implement causal approach and interception predicates
- [x] Implement goal-bound save/deflection and clearance predicates
- [x] Implement shot and pass/receive contact-chain predicates
- [x] Add short confirmation windows and semantic early termination
- [ ] Add golden near-miss, opponent-touch, rebound, own-goal, timeout, and
      repeated-contact anti-farming traces

## 4. Curriculum and rewards

- [ ] Allocate semantic frontier, routine rehearsal, full matches, and
      deduplicated failures with immutable holdouts excluded from gradients
- [ ] Advance independent difficulty axes from rolling success and learning
      progress
- [x] Mirror allocation across controlled colors and avoid fixed player roles
- [x] Add bounded terminal skill outcome reward with separate attribution
- [ ] Compare predicates-only, terminal skill reward, and dense shaping arms
- [ ] Reject any shaping arm that farms events or fails full-match transfer

## 5. Evaluation and observability

- [ ] Implement paired multi-seed per-family evaluation with confidence
      intervals and time-to-resolution
- [ ] Record coverage, success/failure/unresolved, physical validity,
      trajectory quality, and resolved drills/s by difficulty
- [ ] Add scenario and semantic outcome contracts to metrics and replay
      artifacts
- [ ] Add terminal-dashboard scenario telemetry
- [ ] Add viewer filters, labels, timelines, and comparisons for family,
      difficulty, controlled color, and semantic outcome

## 6. Training decision

- [ ] Run short learnability probes against random and deterministic heuristic
      controls
- [ ] Run M14-static versus M15-semantic curriculum ablation at matched compute
- [ ] Run a full-match-heavy control to measure catastrophic specialization
- [ ] Evaluate the candidate against frozen M14, heuristic, historical league,
      and immutable skill holdouts with paired colors and at least five seeds
- [ ] Write a machine-readable promote/reject decision before authorizing a
      high-budget run
- [ ] Publish the exact recommended next-run command only if the entry gates
      pass

## 7. Delivery

- [ ] Document scenario authoring, predicate definitions, dashboards,
      evaluation commands, artifacts, limitations, and rollback
- [ ] Record CPU/CUDA frames/s, matches/s, resolved drills/s, and total compute
- [ ] Run doctor, build, test, lint, CUDA smoke, and strict OpenSpec validation
- [ ] Commit small signed Conventional Commits and push to main
