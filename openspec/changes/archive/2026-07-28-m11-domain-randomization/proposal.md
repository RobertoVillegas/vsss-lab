## Why

A policy that succeeds only at nominal simulator parameters is not credible for
sim-to-sim or future hardware transfer. The declared randomization ranges must
become seeded runtime behavior and an OOD promotion gate.

## What Changes

- Sample friction, restitution, and per-motor strength at reset.
- Inject bounded action latency, command drops, and observation noise.
- Preserve deterministic seeded episodes and expose realized parameters.
- Add an OOD suite comparing robust and nominal controllers.

## Capabilities

### New Capabilities

- `seeded-domain-randomization`: deterministic physics, actuator, transport, and
  observation perturbations.
- `ood-robustness-evaluation`: paired-seed non-regression and superiority gate.

## Impact

Adds a composable backend wrapper, experiment manifest, evaluator, tests,
evidence, and commands. Nominal behavior remains the default.
