## Why

M4 provides deterministic scripted skills and replay evidence, but the platform
cannot yet train, resume, or evaluate a learned controller. M5 establishes the
smallest reproducible RL vertical slice before multi-agent algorithms.

## What Changes

- Add a deterministic single-robot `go-to-target` training task.
- Add a CPU-first PPO trainer with versioned configuration.
- Add curriculum stages C0–C5 with explicit promotion thresholds.
- Add portable checkpoint/resume and append-only JSONL metrics.
- Add deterministic evaluation seeds and an executable 95% success gate.
- Do not add IPPO, MAPPO, shared team policies, self-play, league play, Ray,
  distributed collection, or CUDA optimization.

## Capabilities

### New Capabilities

- `single-agent-ppo-training`: Reproducible PPO training, metrics, and checkpoint lifecycle.
- `rl-skill-curriculum`: Versioned go-to-target curriculum and deterministic evaluation gates.

### Modified Capabilities

None.

## Impact

Activates `python/vsss_train`, `experiments/skills`, and local checkpoint/report
commands. Adds locked PyTorch and TorchRL training dependencies without putting
either library in the Rust simulation hot loop.
