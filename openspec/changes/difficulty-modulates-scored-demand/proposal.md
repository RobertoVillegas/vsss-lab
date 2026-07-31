# Difficulty Must Modulate the Scored Demand

## Why

The curriculum lowers difficulty for a family a policy is failing. That cannot work when
a family's difficulty axes do not move the quantity it is scored on.

Clearance is scored on the ball leaving the defensive third. Its generator chose the
ball's depth by a coin flip between two deep positions, independent of every difficulty
axis, so the easy end asked for the same 0.38 m of displacement as the hard end. Measured
at iteration 1100 of `vsss-m24-3-run-0003`: clearance scored 0 of 10 at every difficulty
band, while 38 of 40 trials touched the ball and timed out. Never succeeding once, the
family produced no reward, so nothing taught it, and difficulty reduction had no rung to
offer.

One family then blocked the milestone. `defense` advances only on `clearance >= 0.35`;
clearance crossed that four times in 43 evaluations and never twice in a row, so the
phase never advanced. The phases it gates are `cooperation`, which teaches passing, and
`rotation`. Both received zero allocation for the whole run and scored 0.00 throughout.
The policy learned to score directly and never learned to play, because play was never
allocated to it.

See ADR 0016.

## Milestone and non-goals

Maintenance of the M15 semantic curriculum, gated on ADR 0016. Non-goals:

- no change to any predicate or to what a skill means;
- no change to a reward coefficient, a phase gate, or the phase patience;
- no claim that families other than clearance were audited.

## What changes

- interpolate the clearance ball's starting depth across its difficulty axis, so the easy
  end is easy at the thing the drill scores;
- bump the generator revision, making holdouts of the two revisions explicitly
  incomparable rather than silently mixed.

## Success criteria

- clearance difficulty spans a solvable band and an unsolvable one for the current policy,
  rather than being uniformly unsolvable;
- the family becomes learnable from below without retraining, verified by re-evaluating an
  existing checkpoint;
- the evaluation record carries the generator revision that produced it.
