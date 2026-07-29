# M18: Controlled PPO ablation

## Why

Rocket League PPO implementations demonstrate that strong policies can emerge
from PPO when network capacity, normalization, rollout scale, event rewards,
state diversity, and simulator throughput are treated as measured design
variables. VSSS Lab needs paired evidence before adopting larger networks or
different reward weights.

## What changes

- Make actor/critic activation and LayerNorm explicit configuration.
- Add an anti-farming, contact-entry reward for useful ball impulse.
- Add a matched-step, paired-seed M18 screening runner.
- Compare width, normalization/activation, PPO reuse, and causal reward as
  separate arms.
- Persist parameter count, throughput, PPO diagnostics, terminal score, and
  per-family semantic outcomes.

## Non-goals

- No algorithm replacement.
- No action-parser decision in the architecture screen.
- No adoption based on training return alone.
- No direct dependency on RLBot or Rocket League packages.
