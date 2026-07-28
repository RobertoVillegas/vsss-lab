# M11 domain randomization and OOD evidence

## Outcome

The paired 20-seed OOD suite passes:

```json
{
  "nominal_progress": 0.1442173469453492,
  "robust_progress": 0.24405753971565264,
  "margin": 0.09984019277030343,
  "passed": true
}
```

Progress is reduction in robot-to-ball distance after 160 fixed simulation
ticks. Both controllers see the same initial state and realized domain on every
paired seed. The robust fixture has positive progress and exceeds the nominal
fixture by more than the required 0.03 m aggregate margin.

Reproduce and retain the full per-seed samples with:

```bash
just ood-evaluate reports/m11/ood.json
```

## Held-out distribution

Each episode records:

- friction in `[0.25, 0.90]`;
- restitution in `[0.10, 0.60]`;
- six independent motor multipliers in `[0.55, 1.25]`;
- action latency from zero through five steps;
- action-drop probability in `[0.05, 0.35]`;
- position-noise standard deviation in `[0.005, 0.045]` m;
- heading-noise standard deviation in `[0.01, 0.18]` rad.

One seeded generator and a fixed sampling order reproduce parameters, drops,
noise, and trajectory exactly. Tests run two identical seeded wrappers
tick-for-tick and require array equality.

## Boundary and limitations

Physics parameters rebuild the native world at reset. Motor variation scales
each actuator independently. A FIFO introduces latency, dropped commands retain
the last delivered safe command, and Gaussian noise touches policy observations
only. Canonical ground truth remains separate for replay, events, and scoring.

This is a deterministic engineering gate, not evidence that the heuristic
fixture is a trained robust policy or that the synthetic distribution matches
hardware. A later training experiment should apply the wrapper to MAPPO and
evaluate frozen checkpoints. M12 should estimate distributions from camera and
robot telemetry, add correlated/bias noise, packet bursts, battery effects, and
actuator dynamics.

Rollback selects the existing nominal `NativeBackend`; no policy API,
checkpoint, canonical state, or physics trait changes.
