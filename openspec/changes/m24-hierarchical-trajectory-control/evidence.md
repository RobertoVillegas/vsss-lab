# Evidence

## Why this boundary

Replay 425 from M23 emitted non-zero commands but left the ball nearly
stationary for most of the capture. The cooperation phase also concentrated
practice on pass/receive before reliable solo reacquisition.

Su et al., *Toward Real-World Cooperative and Competitive Soccer with
Quadrupedal Robot Teams* (CoRL 2025), use a 5 Hz MAPPO strategy policy over
Walk, Dribble, and Kick while verified low-level skills execute at 50 Hz. Their
direct end-to-end baseline produces less accurate and more erratic ball
trajectories. M24 adopts the separation of concerns, not their embodiment or
weights: VSSS primitives remain deterministic differential-drive controllers
inside the exact Rapier simulator.

Public comparisons:

- [CoRL 2025 paper](https://raw.githubusercontent.com/mlresearch/v305/main/assets/su25a/su25a.pdf)
- [RLGym](https://rlgym.org/) for learning-oriented action parsers and
  high-throughput headless evaluation
- [simulation_vsss](https://github.com/juliodltv/simulation_vsss) and
  [pSim](https://juliodltv.github.io/pSim/) for VSSS geometry and simulator
  behavior

## Exact simulator benchmark

Recorded in `experiments/reports/m24/trajectory-primitives.json`:

| Trial | Contact | Time | Exit error | Max ball speed |
| --- | ---: | ---: | ---: | ---: |
| stationary center | yes | 1.04 s | 11.24° | 0.502 m/s |
| moving forward | yes | 2.22 s | 4.15° | 0.520 m/s |
| moving lateral | yes | 4.58 s | 30.18° | 0.254 m/s |

All trials remained physically valid, produced translational robot motion, and
exited in the requested half-plane. Moving-lateral reacquisition is the current
limiting fixture and must be watched in long-run captures.

## Learning smokes

- MAPPO primitive: two CPU iterations, eight worlds, 4,096 frames completed
  with finite optimization metrics and checkpoint/replay artifacts.
- IPPO primitive: the same two-iteration contract completed independently.
- Focused primitive, curriculum, semantic, league, and prediction tests pass.

The smokes establish operability, not algorithm superiority. The paired configs
freeze every field except `algorithm` and `policy_id`; multi-seed outcome
evaluation is required before preferring MAPPO or IPPO.

## Rollback

M23 remains unchanged and uses `action_parser = "continuous"`. Primitive
checkpoints are configuration-fingerprinted and cannot be silently interpreted
as direct wheel actions.
