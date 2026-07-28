## 1. Reward contract

- [x] Add versioned spacing and defensive-coverage configuration
- [x] Implement scalar and vector environment reward parity
- [x] Add unit tests for congestion and defensive threat
- [x] End scoreless horizons with a bounded draw penalty
- [x] End stagnant-ball episodes early with a distinct bounded penalty
- [x] Record goal, draw, and stagnation termination counts in training metrics

## 2. Training delivery

- [x] Add a fresh coordinated MAPPO experiment
- [x] Make new automatic runs select the coordinated experiment
- [x] Validate one CUDA iteration and checkpoint
- [x] Diagnose the completed 50 million-step M12 experiment
- [x] Replace proximity shaping with directional ball and attacker signals
- [x] Add a policy exploration floor and heuristic-to-self-play curriculum
- [x] Preserve strict historical-checkpoint compatibility
- [x] Add terminal checkpoint ranking without replay generation
- [x] Add TensorBoard-compatible events and in-viewer training charts
- [x] Mix current, bounded historical, and heuristic opponents deterministically
- [x] Add a machine-readable paired-run comparison command
- [ ] Run a fresh 50 million-step M13 paired experiment
- [ ] Compare goals, terminal outcomes, clustering, exploration, and throughput
      against run 0002

## 3. Gates

- [x] Run build, test, lint, CUDA smoke, and OpenSpec validation
- [x] Record historical run diagnosis and checkpoint ranking
- [ ] Record the fresh M13 comparison and archive this change
