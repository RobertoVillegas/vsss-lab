# Finishing And Carry Scale

## Why

The team scores from distance at the start of a play and does not build chances. Five measured
cycles located the cause and it is the action set, not the reward
(`docs/evidence/m24-3-timeout-penalty-side-effect.md`):

| intent, shot drills sweeping the approach angle | on the line | 36° | 62° | 94° |
| --- | --- | --- | --- | --- |
| navigate at the ball | 0.80 | 0.30 | 0.00 | 0.00 |
| navigate at the goal, dribbling | 0.80 | 0.45 | 0.00 | 0.00 |
| strike | 0.55 | 0.05 | 0.00 | 0.00 |

Nothing finishes from 62 degrees or more, and matches present that angle three times in four.
Two things follow. `strike` is beaten by dribbling even on the shooting line, its own best case,
so it is not doing its job. And the action set has no way to finish an angled chance at all.

ADR 0021 added a carry gradient that pays for bringing the ball to a position the primitives can
convert, and left its weight unset. That weight and this gap are the same question — what a
chance is worth against what a goal is worth — so they are decided together.

## Milestone and non-goals

Finishing for the active milestone. Non-goals:

- dribbling is not being nerfed or removed. It is the better choice in several measured cases
  and the point is to give the set a way to finish from an angle, not to make one primitive win;
- no change to the observation, the network, or the curriculum;
- no reinstatement of directional ball shaping, which M20 removed and ADR 0015 affirmed;
- the goal-geometry shaping term keeps its 92-per-cent drift for now. Changing two shaping terms
  at once would make neither attributable.

## What changes

- a finishing primitive that can approach the ball from a chosen side while moving, so an angled
  chance is convertible at all. `strike` today selects a contact point behind the ball and gates
  the drive-through on a discrete alignment test, which from close range means backing away
  before it can line up;
- the carry gradient's coefficient and the goal coefficient are set together and ablated as one
  ratio, because a carry that is worth a fraction of a goal is the whole design;
- the difficulty audit is re-run over `shot`, since a primitive that can finish from an angle
  changes what the ladder measures.

## Impact

- `python/vsss_train/primitives.py`: the finishing primitive
- `python/vsss_train/config.py`: the two coefficients
- `crates/vsss-features/src/actions.rs`: the port, once the primitive is settled and not before
- `docs/evidence/`: the ablation over the ratio, and the audit after
- Any checkpoint trained before this cannot be compared on behaviour, only on outcome: the
  action set itself changes.
