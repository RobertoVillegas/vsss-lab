# The policy takes one shot and does not follow up

`shot` is the family that gates the foundation phase and the one the difficulty audit singles
out: every other family scores 1.00 at its easiest setting, `shot` scores about 0.50. This is
what the failures actually are.

## Measured

The run 0009 checkpoint at iteration 1500, 24 drills per level, mirrored across both colours,
other axes held at 0.10:

| difficulty | success | touches per attempt | median steps | outcomes |
| --- | --- | --- | --- | --- |
| 0.00 | 0.67 | 1.1 | 57 | 16 goals, 8 timeouts |
| 0.25 | 0.71 | 1.0 | 62 | 17 goals, 7 timeouts |
| 0.50 | 0.54 | 1.4 | 145 | 13 goals, 11 timeouts |
| 0.75 | **0.04** | 1.2 | 240 | 1 goal, **23 timeouts** |

Two things stand out.

**There are no failures, only timeouts.** Not one attempt at any level ends in a wrong touch, an
opponent touch or a concession. The policy never does the wrong thing; it runs out of time.

**It touches the ball about once.** At difficulty 0.75 the median attempt uses the entire
240-step horizon and still registers 1.2 touches. It had the time to re-approach and shoot
again, and did not. One strike per possession is the whole story: if that strike does not score,
the attempt is over.

Range is what defeats it. The ball starts 33 cm from the goal line at the easy end and 73 cm at
the hard one, and success falls from 0.67 to 0.04 across that.

## What this is not

It is not the collapse. The same checkpoint strikes on 80 per cent of decisions and holds its
stop fraction at 0.089; it wants the ball and goes for it. It is not a generator defect either —
unlike `rotation_recovery`, the easy end of `shot` is genuinely the easy end.

It is also not obviously a timeout-incentive problem. Run 0009 trained with an unresolved drill
costing what a failed one costs, and still averages 1.1 touches.

## Where the gradient is missing

A strike is paid for by the goal it produces, at `goal_coefficient = 10`, and by
`useful_touch_impulse` at 0.15 times the tanh of the ball's velocity change toward the goal on
the contact edge. A good strike is therefore worth about 0.11 against a goal's 10, roughly one
per cent.

From 73 cm the goal arrives on 4 per cent of attempts, so the expected return on shooting well
from range is 0.4 from the goal and a certain 0.11 from the impulse. There is very little
telling the policy that a *better* strike from distance is worth anything, and nothing at all
telling it that a second attempt is worth making.

## Candidates, not a conclusion

Three levers, none of them free, listed so the next change is a choice rather than a reflex:

- **Raise `useful_touch_impulse_coefficient`.** It is the only dense term that already rewards
  hitting the ball hard toward the goal, and it is invariant by construction under ADR 0015 —
  signed, and gated on the contact edge so it cannot be farmed by hovering. This is the smallest
  change that adds gradient exactly where it is missing.
- **Investigate re-engagement directly.** One touch per attempt may be a control problem rather
  than a reward one: the striker drives through the ball and may end up facing away with no
  cheap way back. That is measurable and has not been measured.
- **Do not reach for the score-aware reward (ADR 0020) here.** It changes what a goal is worth,
  and this is a problem of the policy not converting the chances it makes.
