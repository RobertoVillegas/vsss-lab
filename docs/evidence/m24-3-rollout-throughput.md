# M24.3: where rollout time went, and 2.8x from removing it

Measured with cProfile over three iterations of the live configuration: 64 worlds, 256
rollout steps, `action_repeat = 4`.

## The starting picture

24.7 seconds per iteration, 663 frames per second, and 20.9 hours for fifty million steps.
The GPU sat at 17 per cent on an RTX 3070, so the bottleneck was never the network.

Of 70.6 seconds across three iterations, `collect_self_play_trajectory` took 68.3 and the
PPO update took 2.5. Optimization was three and a half per cent of training. Inside the
rollout:

| cost | seconds | calls |
| --- | --- | --- |
| scripted opponent controller | 17.1 | 196 608 |
| scalar `np.clip` | 14.1 | 2 872 459 |
| `go_to_target` | 10.7 | 679 061 |
| `_strike_target` | 7.8 | 49 523 |
| role assignment | 6.2 | 98 458 |

## What the call counts said

The scripted opponent was called 196 608 times for three iterations, which is
`3 x 256 x 64 x 4`: once per world **per physics substep**. It was replanning at 200 Hz
while the learner acted at 50. The configured `control_period` is 0.02 s, so the opponent
was reacting four times faster than the control period allows. That was a fidelity defect
and a quarter of all rollout time at once.

Scalar `np.clip` at 2.9 million calls is numpy dispatch overhead on single floats. The
builtin comparison is identical for scalars.

`_strike_target` operated on two-element numpy arrays, where per-call overhead dominates
arithmetic.

## Changes

- the scripted opponent plans once per decision, and the substeps run inside the native
  loop instead of four round trips through Python;
- `go_to_target` clips with builtins;
- `_strike_target` is written on scalars with `math`.

## Result

| | before | after |
| --- | --- | --- |
| seconds per iteration | 24.7 | 8.76 |
| frames per second | 663 | 1870 |
| fifty million steps | 20.9 h | 7.4 h |

A factor of 2.82, with the full suite and the end-to-end smoke green.

## What is still hot

`assign_roles` at 6.2 seconds over 98 458 calls, which is twice per world per decision and
may be redundant; `build_team_observation` at 5.5 seconds, built per world in Python;
`VonMises.sample` at 2.6 seconds for its rejection sampler; and roughly 1.2 million scalar
`np.clip` calls left in the primitives. The remaining work is vectorizing per-world Python
loops across worlds, which is a larger change than these.

Note that the opponent now acts at the control period rather than the physics period, so it
is slightly less reactive than in earlier runs. Match numbers are not comparable across that
change.
