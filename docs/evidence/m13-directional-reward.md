# M13 directional reward evidence

## Observed baseline

Run `/home/rob/runs/vsss-training-run-0002` completed 50,003,968 environment
steps, 57,278 matches, and 3,052 PPO iterations in 3:29:46. It sustained roughly
3,973 frames/s at completion on CUDA with 64 physics worlds.

The policy did not improve monotonically:

- the sampled goal-ending rate peaked near 52% around iterations 500–750 and
  ended near 42–44%;
- rolling progress fell from roughly +0.24 early to +0.05–0.09 late;
- entropy fell from about +1.48 to -4.34;
- the final actor standard deviations were about 0.031 and 0.024;
- sampled late replays showed near-saturated actions, frequent teammate
  proximity, and possession states that did not reliably become goals.

These are diagnostics, not a causal proof. They justify changing the reward
contract and requiring terminal evaluation before selecting a checkpoint.

## Thesis-guided reward

Julio De La Torre's *Aprendizaje por refuerzo en sistemas multiagentes aplicado
a robots móviles* (2024), pages 74–78, reports that a proximity reward can teach
a robot to remain beside the ball and that shared policies can worsen
multi-robot collisions. Its most useful ball term compares cosine alignment of
ball velocity with the opponent and own goals:

`tanh(cos(v_ball, opponent_goal - ball)) -
tanh(cos(v_ball, own_goal - ball))`

M13 implements that bounded direction term and a penalty-only alignment between
the dynamically closest attacker velocity and the ball. Both are divided by the
episode horizon. A full scoreless episode also accumulates a bounded time cost
of one unit. Goal rewards remain ±10 and dominate the shaping terms.

M13 deliberately does not copy the thesis algorithm wholesale. The thesis uses
separate actors with centralized critics in MATD3; VSSS Lab retains a shared
MAPPO actor and centralized critic, native batched Rapier worlds, and dynamic
roles.

## Historical checkpoint ranking

Seven representative checkpoints were replayed for ten terminal matches each:
five fixed seeds, both reflected starting sides, and the M12 config. The ranker
uses W-L balance, then goal difference, then mean progress.

| Rank | Iteration | W-D-L | GF-GA | Mean progress |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2750 | 5-5-0 | 5-0 | +0.3393 |
| 2 | 500 | 3-6-1 | 3-1 | +0.6442 |
| 3 | 3052 | 3-6-1 | 3-1 | +0.2967 |
| 4 | 1500 | 2-7-1 | 2-1 | +0.4641 |
| 5 | 1000 | 2-7-1 | 2-1 | +0.4171 |
| 6 | 2250 | 1-9-0 | 1-0 | -0.0461 |
| 7 | 750 | 1-8-1 | 1-1 | +0.5082 |

The machine-readable result is
`docs/evidence/m13-run-0002-ranking.json`. Five seeds are enough to demonstrate
that “latest” and “best” differ, but not enough for a promotion decision; use at
least ten seeds for that gate.

## Reproduction

```bash
just league-rank-checkpoints \
  /home/rob/runs/vsss-training-run-0002 \
  500,750,1000,1500,2250,2750,3052 \
  experiments/configs/m12-mappo-coordinated.toml \
  10 \
  reports/checkpoint-ranking.json

just league-live-steps 50000000 25 60 25 auto 64
```

The second command allocates a fresh run automatically and uses the M13 config.
Do not resume run 0002 into M13 because its reward fingerprint is intentionally
different.
