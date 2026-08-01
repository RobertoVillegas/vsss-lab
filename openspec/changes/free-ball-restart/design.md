# Design

## What the rule says and what the simulation did

Rule 15 gives an interval of ten seconds, a location, and a continuation. The simulation used
five seconds, no location, and a terminal worth minus one, which is twice a draw. Every part
of the resolution differed from the rule except the detection of the impasse itself.

## Restart rather than reset

A free ball is not an episode boundary, so it cannot go through `reset_state`, which zeroes
the clock, the scenario bookkeeping and the reward baselines. It reads the world's snapshot,
edits the ball and any robot inside the clearance, and restores that snapshot, leaving the
step count, score and episode identity untouched. The impasse anchor is re-seeded so the
clock restarts from the new position.

## Where the free ball does not apply

Rule 14 makes a goal-area impasse a goal kick, with a different placement and different robot
constraints. Implementing the free ball there would apply the wrong rule to that case, so the
impasse clock is reset and play continues unchanged. That leaves a known gap rather than a
silent substitution.

## Compatibility

No configuration key is removed: `stagnation_seconds` and `stagnation_penalty` stay loadable
because the checkpoint compatibility check rejects a stored key that no longer exists, and
they are documented as unread. Two keys are added and given neutral defaults in the legacy
map so prior checkpoints still load. The stagnation reward term stays in the decomposition at
a structural zero, so term accounting remains comparable across the change.

Episode length and return distributions shift, so the milestone needs a fresh baseline.

## Validation

A test parks a ball away from both goal areas with no robot near it, steps past the impasse
interval, and asserts the episode did not end, a free ball was counted, and the ball is on a
mark. An end-to-end smoke shows the stagnation terminal absent from a run's terminations and
its reward term at zero.
