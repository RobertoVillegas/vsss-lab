# Tasks

- [x] Measure why finishing fails, and record it: no primitive converts from 62 degrees or more,
      and dribbling beats striking on the shooting line, 0.80 against 0.55.
- [x] Add the carry gradient, off by default, with its shape measured over the field (ADR 0021).
- [x] Accept an ADR for the finishing primitive before implementing it (ADR 0022).
- [x] Implement the approach fix. The candidate mechanism turned out not to be the defect: the
      striker was not retreating, it was crawling.  tapers inside half a metre so a
      robot settles, and the acquisition point moves with the ball, so a striker six centimetres
      behind it advanced at twelve per cent of its speed. A  flag fixes the chase.
- [ ] The angle remains unsolved and needs a different change. Nothing converts from 60 degrees
      or more, before or after, on a bench that places the striker directly.
- [ ] Measure it with `tools/primitive_race.py angle` against both dribbling intents. Accept it
      only if it converts from an angle at a rate that is not zero and is not worse than
      dribbling on the line.
- [x] Confirm dribbling is unchanged: both driving intents score identically at every angle.
- [ ] Re-run `tools/audit_skill_difficulty.py` over `shot`. A primitive that can finish from an
      angle changes what the ladder measures, and `ball_angle` may become a usable axis where it
      was withdrawn in cycle 4 for being unreachable.
- [ ] Ablate the carry-to-goal ratio over at least three settings, reported against goals per
      minute and against how often the ball reaches a convertible position.
- [ ] Record the chosen ratio and the measurements behind it in `docs/evidence/`.
- [ ] Port the primitive and the carry potential to `vsss-features` behind equivalence tests,
      after the ablation has chosen and not before.
