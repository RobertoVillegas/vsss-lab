# Circular Heading Parameterization

## Why

M24.2 made the heading continuous, but its precision is not. The heading is a
tanh-bounded pair decoded with `atan2`, and the decode discards the norm, so the
norm acts as a concentration parameter confined to the unit square instead of the
unit circle. Attainable precision therefore depends on the heading: measured at the
shipped initialization, a saturated axis-aligned mean gives a circular deviation of
24.6 degrees while a saturated diagonal mean gives 0.8 degrees. Both goals sit at
`y = 0`, so the worst case is exactly the shot at the goal.

The only sharpening mechanism is the deviation, which is a state-independent
parameter, so sharpening the headings that matter removes exploration everywhere
else. The reported entropy does not measure angular exploration at all, because the
norm can grow without moving the deviation, which leaves the entropy bonus with no
effect on direction.

Intensity is degenerate for a separate reason: its distillation target is an
unreachable `1.0`, and the first run requests a median of 0.988 with nothing below
0.50 over 9000 samples. The strike reachability model ignores the requested
intensity when it selects an intercept, so a low request produces a robot chasing
a point it cannot reach, and navigation cannot exceed 0.8 of authority whatever the
request. In practice M24.2 emits a skill and a heading at full authority.

See ADR 0014.

## Milestone and non-goals

Successor milestone to M24.2, gated on ADR 0014 being accepted. Non-goals:

- no change to the skill set, the observation, the network width, or PPO itself;
- no removal of the current parameterization, which stays as the ablation and
  rollback baseline;
- no reward change; that is a separate change and a separate ADR.

## What changes

- parameterize the heading on the circle: an unbounded mean direction and a
  state-dependent concentration, with sampling, log-probability, and entropy
  evaluated as circular quantities;
- transport the sampled angle normalized by π, keeping the bounded action
  contract intact;
- give intensity a reachable distillation target and let it enter the strike
  reachability model, so requested authority and selected intercept agree;
- record the heading contract in the checkpoint so a policy cannot load under a
  parameterization it was not trained with;
- report angular concentration next to the heading-change statistic, so precision
  and churn can be read together.

## Success criteria

- circular deviation of the executed heading is within a small tolerance of being
  equal for an axis-aligned and a diagonal request at the same concentration;
- the entropy the bonus acts on moves when angular exploration moves;
- a requested intensity below full authority produces an intercept the robot can
  actually reach;
- rollout and update recover the same log-probability, so the ratio starts at one;
- the frozen action table and legacy M24 and M24.2 checkpoints remain loadable
  under their own parsers.
