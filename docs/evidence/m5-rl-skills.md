# M5 RL skills evidence

- Date: 2026-07-28
- Host: linux/amd64, Python 3.13.14
- Runtime: PyTorch 2.13.0+cu130, TorchRL 0.12.0
- Configuration: `experiments/configs/m5-go-to-target.toml`
- Seed: 7

## Result

The versioned CPU configuration trained 40,960 control frames in 30.55 seconds
with a 926,764 KiB peak RSS. The resulting trusted local checkpoint was 79 KiB.

Independent deterministic C5 evaluation used seeds 20,007 through 20,106:

```json
{"episodes":100,"mean_final_distance":0.0911070572095732,"passed":true,"success_rate":0.95,"successes":95,"threshold":0.95}
```

Evaluation took 41.12 seconds with an 813,644 KiB peak RSS. One action is held
for four native 5 ms ticks, matching the 20 ms control period.

## Reproduction

```sh
just train-skill
just evaluate-skill
```

## Known limitations

- The result meets the gate exactly; more seeds and confidence intervals belong
  to later evaluation milestones.
- The default short run promotes through C2. C3–C5 distributions and promotion
  logic are executable and contract-tested, while longer sweeps are intentionally
  not a local correctness gate.
- The checkpoint is a trusted-local PyTorch artifact, not a portable deployment
  format.
- CPU is the correctness baseline; CUDA throughput is not yet tuned.

## Rollback

Revert the M5 commits and remove the external checkpoint. M1–M4 contracts and
the native simulator remain unchanged.
