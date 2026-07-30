# ADR 0014: Circular heading parameterization

- Status: proposed
- Date: 2026-07-30

## Context

M24.2 replaced the eight canonical headings of ADR 0013 with a continuous
heading, expressed as a tanh-bounded pair interpreted as `(cos θ, sin θ)` and
decoded with `atan2`. The decode discards the vector's norm, so the norm acts as
a concentration parameter: a longer mean vector yields a tighter heading
distribution. Because `tanh` confines that vector to the unit square rather than
the unit circle, the attainable concentration depends on the heading itself.

Measured on the shipped policy, at the initialized `log_std = -0.5`, the circular
standard deviation of the decoded heading is 24.6 degrees for a saturated
axis-aligned mean and 0.8 degrees for a saturated diagonal mean — a factor of
about thirty. The axis-aligned heading is the one that matters, because both
goals lie at `y = 0`. A policy lined up behind the ball and requesting a strike
toward the goal therefore executes with roughly ±25 degrees of exit-direction
jitter, while a diagonal cross-field pass executes almost exactly.

The only sharpening mechanism available is a smaller deviation, but `log_std` is a
state-independent parameter, so sharpening the headings that matter would remove
exploration everywhere else at the same time. The reported entropy does not
measure angular exploration at all, since the norm can grow without moving
`log_std`, which makes the entropy bonus ineffective for direction.

The intensity channel is separately degenerate. The teacher's distillation target
is literally `1.0`, which is unreachable through `tanh` and drives the pre-tanh
mean far into saturation. The first M24.2 run requests intensity with a minimum of
0.931 and a median of 0.988 over 9000 samples, with nothing below 0.50. The strike
reachability model also assumes full authority when it selects an intercept, so a
low requested intensity produces a robot that chases an intercept it can never
reach. Navigation cannot exceed 0.8 of authority regardless of the request,
because its target is always a fixed 0.4 m ahead.

## Decision

Parameterize the heading as a circular distribution rather than as a bounded
Cartesian pair. The policy emits an unbounded two-vector whose `atan2` gives the
mean direction, and a separate state-dependent concentration through a positive
transform. Sampling, log-probability, and entropy are evaluated on the circle, so
angular precision is isotropic, exploration is state-dependent, the reported
entropy measures the quantity the bonus is meant to regularize, and the ±π
boundary remains continuous. The transported token carries the sampled angle
normalized by π, so it stays inside the existing bounded action contract and the
clip remains a no-op.

Intensity keeps its own bounded continuous parameter, with a distillation target
inside the reachable interval, and enters the strike reachability model so a
requested authority and the selected intercept agree.

The current parameterization stays available as the rollback and ablation
baseline under the existing `parametric_primitive` parser, and checkpoints record
which heading contract they were trained under.

## Alternatives considered

- **Normalize the Cartesian pair to unit length before decoding.** Removes the
  square only from the decode, not from the distribution of angles, so the
  anisotropy survives.
- **Emit the angle directly as a single bounded parameter.** Reintroduces the ±π
  discontinuity that M24.2 exists to remove.
- **Keep the pair and add a per-state deviation head.** Fixes state dependence but
  leaves precision heading-dependent, so the axis-aligned strike stays the worst
  case.

## Consequences

- Heading precision no longer depends on where the robot is aiming, which removes
  a systematic handicap on shots and passes toward the goal line.
- Exploration becomes state-dependent and legible: entropy and concentration
  describe angular behavior, so the existing bonus and telemetry act on it.
- Trajectories, log-probabilities, and entropy are not comparable with earlier
  M24.2 runs, so the milestone needs a fresh baseline rather than a continuation.
- A new distribution enters the sampling path and must be verified to reproduce
  its own log-probability between rollout and update, the invariant that keeps
  the PPO ratio meaningful.
