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
- [ ] Run the 20 million-step paired experiment
- [ ] Compare congestion, defense, goals, progress, and throughput to run 0001

## 3. Gates

- [x] Run build, test, lint, and OpenSpec validation
- [ ] Record the final comparison and archive this change
