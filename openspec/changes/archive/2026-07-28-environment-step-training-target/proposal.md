## Why

Optimizer iterations and completed matches vary with vector width, rollout
length, goals, and timeout policy. Environment steps are the standard stable RL
training-budget unit.

## What Changes

- Add mutually exclusive completed-match and environment-step run targets.
- Drive terminal progress and final checkpointing from the selected target.
- Add headless and live 20M-step Just recipes.

## Capabilities

### Modified Capabilities

- `ippo-mappo-training`: explicit environment-step budgets and reporting.

## Impact

Existing iteration and match targets remain available. Step targets can
overshoot by at most one complete vector rollout.
