# Design

## Action contract

The policy emits a three-way categorical skill and three bounded continuous
parameters: direction `x`, direction `y`, and intensity. The executor
normalizes the direction through `atan2`, reflects it for the yellow team, and
maps intensity from `[-1, 1]` to `[0, 1]`.

The replay transport token is:

```text
[skill, direction_x, direction_y, intensity]
```

`skill` retains the established `-1 / 0 / 1` stop/navigate/strike encoding.
The two-component direction avoids the artificial discontinuity of a scalar
angle at ±π.

## Learning

MAPPO optimizes the sum of:

- categorical log probability for the skill;
- tanh-transformed Gaussian log probability for direction and intensity.

The PPO ratio, entropy term, KL estimate, and clipping fraction use this joint
log probability. Bootstrap distillation teaches continuous teacher headings
and full primitive authority rather than multiplying wheel magnitude twice.

## Execution

`navigate` creates a local look-ahead target along the requested heading.
`strike` keeps the causal moving-ball acquisition and drive-through planner,
now with an arbitrary exit direction. The existing differential-drive target
controller, actuator acceleration limit, and action-delta penalty provide
curved steering and temporal smoothing.

## Compatibility

`primitive` remains the legacy 17-action parser. M24.2 uses the new
`parametric_primitive` parser and a distinct policy identity, so old
checkpoints cannot be loaded accidentally.
