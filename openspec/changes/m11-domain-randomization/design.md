## Context

Randomization is an environment concern, not policy identity or physics API.
Every episode must record what was sampled so results are reproducible and
debuggable.

## Decisions

1. Use one NumPy generator seeded at reset and fixed sampling order.
2. Rebuild the native backend with sampled friction/restitution; apply six
   independent motor multipliers before stepping.
3. Model latency with a bounded FIFO, drops by retaining the last delivered
   command, and noise only on continuous observable state fields.
4. Keep canonical ground truth internally for evaluation and replay; policies
   receive perturbed observations.
5. Compare controllers on paired OOD seeds and require both positive progress
   and a configured aggregate margin.

## Risks

An evaluator can reward a controller tailored to its faults. The committed suite
publishes ranges, seeds, horizon, metric, paired samples, and limitations. M12
will replace synthetic distributions with measured hardware distributions.

## Rollback

The wrapper is opt-in. Removing it restores nominal native construction without
changing policies, checkpoints, or canonical interchange.
