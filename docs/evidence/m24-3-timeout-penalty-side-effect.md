# The timeout penalty doubled a bias it did not create

Run 0011 carries three fixes over run 0009 and diverges from it on the metric that matters.
Both start from the same distilled bootstrap at 0.552 strikes; by iteration 125, 0009 had
climbed to 0.760 and scored 0.2–0.6 goals per minute, and 0011 had fallen to 0.150 with
navigate at 0.848 and four consecutive evaluations scoring nothing.

## Measured, before touching anything

Which primitive resolves each foundation drill, driving one primitive on a loop with the rest
of the team idle, 20 seeds per cell at difficulty 0.1 (`tools/primitive_race.py`):

| family | primitive | resolved | success | steps |
| --- | --- | --- | --- | --- |
| approach | navigate | 1.00 | 1.00 | 19 |
| approach | strike | 1.00 | 1.00 | 19 |
| interception | navigate | 1.00 | **1.00** | **20** |
| interception | strike | 1.00 | **0.00** | **230** |
| shot | navigate | 0.55 | 0.55 | 115 |
| shot | strike | 0.55 | 0.55 | 125 |

There is no general bias toward navigate. `approach` is a tie at 19 steps and `shot` is a tie at
0.55. The whole effect is `interception`, where striking succeeds on none of twenty attempts and
burns the horizon, while blocking succeeds on all twenty in twenty steps.

## Why that is correct behaviour, and why it still broke the run

Interception is a blocking skill. `strike_target` selects a contact point *behind* the ball
relative to the requested exit heading, so against a ball travelling toward the defended goal
the robot must first get past it — goalward — before it can push back. Navigate drives straight
at the ball and stops it, which is what an interception is. The family is not misspecified.

What changed is the price of getting it wrong. Before, striking into an interception timed out
and paid nothing against navigate's `+1`, a gap of one. Charging the timeout makes it `-1`
against `+1`, a gap of two. The gradient against striking doubled on ten of the twenty-three
scenarios `foundation` allocates.

The bias was already there. Run 0009 lived with it because abstaining was free — and that same
freedom is what let 0009's predecessor collapse into running out the clock. The fix for one
pathology sharpened another.

## What this rules out

- It is not the behaviour gate: that only decides promotion and touches no reward.
- It is not the `rotation_recovery` ladder: `foundation` allocates it nothing, and the family is
  visibly working at 0.68–0.80 where 0009 sat at 0.00–0.10.
- It is not a general primitive preference, which the `approach` and `shot` ties rule out.

## What the probe missed, and why

`tools/probe_collapse.py` measured the timeout penalty from a randomly initialized policy and
found strikes *rising*, 0.283 to 0.695. From the distilled bootstrap the same change drives them
down. The starting point decides the sign, and the probe's limitation was written down when it
ran — it just was not treated as disqualifying. Any further probe of this reward has to start
from the bootstrap.
