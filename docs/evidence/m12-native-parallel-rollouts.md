# M12.1 native parallel rollouts

## Result

Self-play no longer constructs one Python-owned native simulator per world. One
`VectorMarlMatchEnv` owns a contiguous native `BatchSimulator`; actions cross
Python/Rust once per control step and fixed policy actions can execute all four
physics repeats under one released-GIL call.

`PhysicsBatch` uses stable-order Rayon stepping at 32 or more worlds and remains
sequential below that threshold. Microbenchmarks showed that scheduling overhead
hurts 16 tiny Rapier worlds, while parallelism helps materially at 64 and 256.

## CUDA sweep

The short-rollout sweep used the same MAPPO model, CUDA device, action repeat,
and host while the user's original run remained active:

| Worlds | Frames/s |
|---:|---:|
| 16 | 1,731 |
| 64 | 5,172 |
| 256 | 7,074 |

A complete 3,000-step compatibility iteration at 64 worlds completed without
OOM at 4,195 frames/s. The production 256-step PPO rollout sustained about
4,080 frames/s and 0.25 updates/s.

Matches end on goals or after 1,500 control steps (30 simulated seconds), and
their state persists across PPO updates. At measured throughput, the
timeout-only lower bound is about 2.7 matches/s, or roughly 10.2 hours for
100,000 matches; goals shorten that estimate. The runner accepts a completed
match target and reports actual matches/s.

## Collision audit

Replay iteration 0040 contained real oriented-box penetration up to 26.7 mm and
4,955 recorded ball/robot overlaps, including ball centers inside a robot; these
were not canvas artifacts. The ball collider incorrectly treated 0.046 kg as
area density, making its effective mass about 0.000067 kg. Exact collider mass,
stiffer contact constraints, predictive contacts, robot CCD, and four solver
iterations reduce sustained contact to the committed tolerance. Regression
tests drive robots into each other and a robot into the ball for 1,000 fixed
steps.

## Viewer stability

The replay viewer now uses a monospace stack and OpenType tabular-number/slashed
zero features globally. Rapidly changing metrics, actor commands, wheel speeds,
timeline values, and sidebar details retain stable glyph widths.
