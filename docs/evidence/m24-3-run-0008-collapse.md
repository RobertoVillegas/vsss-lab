# Run 0008: the policy was paid to give up

Run 0008 stopped going for the ball. This is what was measured, what caused it, and why nothing
noticed for eight hundred iterations.

## What happened

| iteration | `stop` | `strike` | mean intensity | goals per minute |
| --- | --- | --- | --- | --- |
| 1 | 0.04% | 55.6% | 0.87 | 0.6 |
| 369 | 0.55% | 42.0% | 0.74 | 0.4 |
| 921 | 2.8% | 18.6% | 0.60 | 0.2 |
| 1289 | 10.8% | 27.8% | 0.57 | 0.2 |
| 1848 | 32.0% | 11.0% | 0.40 | 0.0 |

Monotonic over 1800 iterations. Full-match evaluation first scored zero at iteration 700 and
was mostly zero from about 1000 onward — 28 of 74 evaluations scored nothing at all.

## The cause

A skill drill paid `+semantic_terminal_reward` for success and `-semantic_terminal_reward` for
failure. A drill that ran out of time paid **nothing**.

That makes abstaining better than attempting whenever the success rate is below one half:

| success rate | value of attempting | value of abstaining |
| --- | --- | --- |
| 0.75 | +0.5 | 0.0 |
| 0.50 | 0.0 | 0.0 |
| 0.30 | −0.4 | 0.0 |
| 0.25 | −0.5 | 0.0 |

The drill outcome mix shows the policy finding exactly that:

| iteration | success | failure | unresolved |
| --- | --- | --- | --- |
| 1 | 84% | 0% | 16% |
| 238 | 71% | 4% | 25% |
| 475 | 29% | **46%** | 25% |
| 712 | 17% | 21% | **62%** |
| 949 | 23% | 5% | **73%** |

Failures spike to 46 per cent at iteration 475, right after promotion into the harder families
at 426. Then failures fall away and unresolved climbs to 73 per cent. It did not learn to
succeed; it learned to run out the clock instead of failing, and running out the clock means not
engaging.

The final family rates say the same thing: `rotation_recovery` 0.25, `save_deflection` 0.30,
`shot` 0.50. Those are precisely the families where attempting had negative expected value.

## Why nothing noticed

The behaviour gate was `idle_spin_ratio <= 0.08`, and it passed on all seventy-four evaluations.
It did more than fail to catch the collapse: **the collapse improved its number**, from 0.079 at
iteration 25 to 0.005 at iteration 1848, because a stopped robot has no angular speed. The gate
watched for the previous pathology and a policy that does nothing passes it perfectly.

`mean_controlled_touches` did not catch it either, moving only 1.05 to 0.86. Drills place the
robot beside the ball, so contact happens whether or not the policy is trying. Full-match goals
per minute is the quantity that moved, tenfold.

## What was ruled out

The dense reward shaping was the first suspect and is not the cause, though it is weaker than
intended. `progress`, `attacker_alignment` and `ball_direction` are switched off deliberately —
M20 removed them as farmable and ADR 0015 states the absence is intended. Their replacement, the
goal-geometry potential, was measured to be almost entirely discount drift:

```text
predicted  c(γ−1)Φ = −0.000239     with c = 0.05, γ = 0.99, mean Φ = 0.477
observed in the run = −0.000220
```

Ninety-two per cent of that term is a standing charge proportional to how good the attacking
position is, and its informative part is the same order as the action-change penalty. That is
worth revisiting on its own, but it is not what drove the collapse: the drill terminal is two
orders of magnitude larger and points the wrong way.

## The fix

An unresolved drill now costs what a failed one costs. Attempting then weakly dominates
abstaining at every success rate, with equality only when success is impossible. Making the
timeout cost *more* than failure was considered and rejected: it would pay a policy to fail
quickly and deliberately, trading one perverse incentive for another.

The behaviour gate now also requires a stop-fraction ceiling and a goals-per-minute floor, so a
policy that has stopped playing cannot be promoted for not spinning.
