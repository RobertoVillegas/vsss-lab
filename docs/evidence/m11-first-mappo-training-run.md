# First sustained MAPPO training run

## Run

The first sustained local MAPPO run completed 50 rollout/optimization
iterations:

```bash
just league-run /home/rob/runs/vsss-mappo-main 50 5
```

Artifacts are retained outside Git under `/home/rob/runs/vsss-mappo-main`:

- 51 checkpoints from initialization through policy version 50;
- ten evaluation replays, captured every five iterations;
- one metric record per iteration;
- registry entries with checkpoint SHA-256 hashes.

The last checkpoint is:

```text
/home/rob/runs/vsss-mappo-main/checkpoints/iteration-0050.pt
sha256:7d39dc78851b0fb655a275658f6966c7b45b0d1164b56b544aa1019d3173bfb4
```

Iteration 50 collected 1,000 frames and reported:

```json
{
  "entropy": 1.8494648064176242,
  "policy_loss": 0.003922623795612405,
  "progress": 0.880082412606009,
  "return_total": 0.8800824126060085,
  "value_loss": 0.000013421835454607844
}
```

## Independent tournament

The frozen checkpoint was evaluated over five seeds with both side assignments:

```bash
just league-tournament \
  /home/rob/runs/vsss-mappo-main/checkpoints/iteration-0050.pt \
  /home/rob/runs/vsss-mappo-main/tournament \
  experiments/configs/m6-mappo.toml \
  5
```

It produced two wins, one draw, and seven losses against the dynamic heuristic,
with a provisional Elo of `951.57` versus `1048.43`. No infrastructure failures
occurred and all ten matches have replays.

## Interpretation

The platform can train and serialize a real shared-policy MAPPO model today.
This checkpoint is useful for inspection and pipeline testing, but it is not a
promotion candidate: on-rollout progress does not yet transfer reliably to the
side-switched heuristic tournament.

The next training gate is therefore:

1. inject M11 domain randomization into MAPPO rollouts;
2. train nominal and randomized checkpoints under equal budgets;
3. evaluate both on paired nominal, side-switched, historical, and OOD fixtures;
4. promote only if the randomized candidate improves OOD results without a
   required-fixture regression.
