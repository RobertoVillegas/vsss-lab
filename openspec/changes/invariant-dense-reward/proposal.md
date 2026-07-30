# Invariant and Non-Farmable Dense Reward

## Why

M20 removed directional ball shaping on purpose, because positive horizontal ball
velocity could be earned on paths that miss the goal, and chose a goal-aperture
potential instead on the principle that a pose must not farm reward. That decision
stands and this change does not reinstate a raw progress term. What the M24.2 audit
found is that the accepted design does not hold in three places.

The potential is applied in the policy-invariant form, with its shaping discount
equal to the training discount, but the potential is not treated as zero at a
terminal state. The final transition therefore pays for ending an episode in a
favorable geometry — the pose-farming loophole the potential exists to close,
reappearing at the terminal.

The only dense term that can accumulate positively without bound is the
useful-contact impulse, and it is non-negative and triggered on the team's contact
edge. Sustained control earns nothing, a lateral pass earns nothing because only the
horizontal component counts, and no contact can be penalized. A robot oscillating at
the contact envelope boundary re-triggers the edge every decision and collects on
each re-entry, without ever advancing the ball.

The terminal pressures disagree with each other. The time penalty is divided by the
horizon, so a whole episode costs a twentieth of a draw, while stagnation costs twice
a draw. Keeping the ball jiggling enough to avoid the stagnation terminal, and never
attempting a goal, is cheaper than stagnating and safer than risking a concession.
The first M24.2 paired evaluation returned nine draws in ten matches; an untrained
policy explains that equally well, so this is an observation about what the reward
permits, not a diagnosis.

See ADR 0015.

## Milestone and non-goals

Maintenance of the accepted M20 reward design, gated on ADR 0015 being accepted.
Non-goals:

- no reinstated ball-progress or ball-direction coefficient;
- no hard-coded field zone and no reward for a pose;
- no change to the action space or to PPO; that is a separate change and ADR.

## What changes

- treat the geometry potential as zero at a terminal state, making the shaping term
  exactly policy-invariant;
- make the useful-contact term signed with respect to the attacking direction and
  attribute it to the controlled robot's contribution rather than to a team-level
  contact edge;
- express the episode time cost and the draw and stagnation terminals on one scale,
  so a stalemate is not the cheapest outcome available.

## Success criteria

- adding or removing the shaping term cannot change which policy is optimal, and a
  test demonstrates the terminal transition no longer pays for final geometry;
- oscillating at the contact envelope earns no positive return;
- moving the ball away from the attacking direction costs what moving it toward
  costs, at equal magnitude;
- the ordering of episode outcomes by return matches the ordering the terminals
  intend.
