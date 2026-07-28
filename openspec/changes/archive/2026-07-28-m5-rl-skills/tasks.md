## 1. Contracts and dependencies

- [x] 1.1 Specify M5 scope, curriculum, artifacts, and executable gates
- [x] 1.2 Record the PPO artifact and checkpoint ADR
- [x] 1.3 Lock isolated PyTorch and TorchRL training dependencies

## 2. Skill environment

- [x] 2.1 Implement native go-to-target task and C0–C5 reset distributions
- [x] 2.2 Add task, observation, reward, termination, and curriculum tests

## 3. PPO lifecycle

- [x] 3.1 Implement TensorDict rollout, GAE, clipped PPO, and deterministic evaluation
- [x] 3.2 Implement config loading, JSONL metrics, checkpoint/save, and exact resume
- [x] 3.3 Add CLI and local train/evaluate/smoke commands

## 4. Verification

- [x] 4.1 Add deterministic smoke, checkpoint equivalence, metrics, and threshold tests
- [x] 4.2 Run OpenSpec, doctor, lint, build, test, and M5 skill gates
- [x] 4.3 Record evidence and limitations, sign small commits, and push
