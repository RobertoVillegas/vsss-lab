# ADR 0021: A carry gradient the goal mouth can see

- Status: proposed
- Date: 2026-08-02

## Context

The policy scores from distance at the start of a play and does not build chances. Five measured
cycles located why, and the answer was not the reward: no primitive in the action set can finish
from 62 degrees or more off the shooting line, and matches present that angle three times in
four (`docs/evidence/m24-3-timeout-penalty-side-effect.md`). The shot drill only ever presented
chances from the line, because the line is the only place anything can finish from.

That leaves a gap the curriculum cannot close: nothing pays the team for *carrying the ball into
the position its primitives can convert*. M20 removed directional ball shaping and ADR 0015
affirmed the removal, on the stated ground that positive horizontal ball velocity is earnable on
paths that miss the goal. The objection is correct and this does not reinstate that term.

## Decision

Reward the change in a potential over the ball's position, where the potential is the **angular
width of the goal mouth subtended at the ball**, normalized against standing in front of it.

That shape answers M20's objection by construction rather than by tuning. It is maximal directly
in front of the goal and collapses at grazing incidence, so a ball in the corner is worth almost
nothing however close it is to the line. Measured over the field:

| ball at | potential |
| --- | --- |
| in front of the goal, on the line | 0.967 |
| centre, close (x=0.60) | 0.630 |
| centre, midfield (x=0.10) | 0.203 |
| touchline, mid height (x=0.45, y=0.60) | 0.097 |
| **corner beside the goal (x=0.70, y=0.62)** | **0.020** |

Carrying from the wing into the corner pays **−0.077**; carrying from midfield to in front of
the goal pays **+0.427**. The touchline drag is not merely unrewarded, it costs — which is the
requirement, because a carrier out there is blocked or stuck and a proximity-only reward would
teach exactly that route.

**The shaping is undiscounted.** Applied as `c·(γΦ' − Φ)` with γ at the training discount, the
standing charge `c·(γ−1)·Φ` measured **10.5 times** the part that pays for carrying — 1.859c
against 0.177c per episode over 1047 steps. Raising the coefficient does not improve that ratio;
both scale together. Applied as `c·(Φ' − Φ)` the charge is gone and the episode total is bounded
by `c`.

The terminal potential is zero, as ADR 0015 requires, so the last transition does not pay for
how the episode ended.

## Consequences

- This supersedes ADR 0015 on one point and only one: the shaping discount for this term is 1
  rather than the training discount. Policy invariance in the discounted objective is traded for
  a signal that is not 91 per cent a charge proportional to being well placed. The term remains
  bounded and non-farmable — it telescopes, so it cannot be collected twice for the same
  position, and the episode total cannot exceed `c` however the ball is moved.
- The existing `goal_geometry` term has the same defect, measured at 92 per cent drift, and is
  left alone here. Changing two shaping terms at once would make neither attributable.
- A goal must stay the dominant reward. With the carry bounded by `c` per episode and a goal
  paying `goal_coefficient`, the ratio is one number and it is the knob to ablate.
- The term is Python only for now. Porting it to `vsss-features` before its weight is settled
  would cost two changes for one.
- This does not fix finishing. The primitives still cannot convert an angled chance, and this
  change pays the team to stop creating them — which is a narrowing of the problem, not a
  solution to it.
