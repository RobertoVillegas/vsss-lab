## Context

The native simulator already exposes contiguous deterministic batches, while M4
provides a trusted geometric controller. M5 needs a learned-policy slice whose
artifacts remain understandable without introducing multi-agent complexity.

## Goals / Non-Goals

**Goals:** train one differential-drive robot toward a target, progress through
C0–C5, evaluate fixed seeds, resume exactly at update boundaries, and emit
machine-readable evidence.

**Non-Goals:** IPPO/MAPPO, team credit assignment, shared multi-agent policies,
self-play, leagues, Ray, distributed rollout, CUDA tuning, and production model
serving.

## Decisions

1. The task wraps the native batch simulator. Robot 0 is controlled, the target
   is represented by the stationary ball, and all other robots are disabled in
   restored episode snapshots.
2. Policy observations are robot-centric `[target_dx, target_dy, cos(theta),
   sin(theta), vx, vy, omega]`, normalized by field dimensions and limits. This
   avoids physical robot identity and match metadata.
3. The policy is a small Gaussian actor plus critic. It is initialized by
   distilling the trusted M4 geometric skill, then refined on-policy. Rollouts
   use TensorDict; GAE and clipped PPO optimization follow current TorchRL
   conventions while keeping the loss implementation explicit and testable.
4. C0–C5 progressively expand target distance, bearing, initial heading, and
   start position. Promotion requires the configured success rate on fixed
   evaluation seeds; C5's normative gate is at least 95%.
5. Checkpoints are versioned PyTorch dictionaries containing config fingerprint,
   model/optimizer state, update, frame count, curriculum stage, and all random
   generator states. Loading defaults to weights-only-safe deserialization.
6. Metrics are schema-versioned JSONL. Configuration is versioned TOML committed
   under `experiments/configs`; run data and checkpoints remain outside Git.
7. CPU is the correctness baseline. CUDA is selectable only when available and
   is not an M5 acceptance requirement.
8. One policy action is held for four 5 ms physics ticks, matching the canonical
   20 ms control period and avoiding redundant neural-network inference.

## Risks / Trade-offs

- **Sparse terminal success can train slowly** → use bounded distance-progress
  shaping while success remains the promotion signal.
- **Physics initialization may introduce collisions** → disable non-participating
  robots and sample inside conservative field margins.
- **A 95% gate can be flaky** → evaluate fixed seeds with deterministic policy
  means and record per-seed outcomes.
- **PyTorch artifacts are not a stable interchange format** → version metadata,
  load only trusted local checkpoints, and defer export formats.

## Migration Plan

Add an isolated dependency group and Python package without changing M1–M4
contracts. Rollback removes the training group, package, config, commands, and
OpenSpec capability; native simulation and scripted baselines remain unchanged.

## Open Questions

Exact curriculum sample counts and network sizes remain experiment parameters,
not protocol contracts.
