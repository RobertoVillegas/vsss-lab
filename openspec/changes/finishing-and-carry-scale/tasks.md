# Tasks

- [x] Measure why finishing fails, and record it: no primitive converts from 62 degrees or more,
      and dribbling beats striking on the shooting line, 0.80 against 0.55.
- [x] Add the carry gradient, off by default, with its shape measured over the field (ADR 0021).
- [ ] Accept an ADR for the finishing primitive before implementing it.
- [ ] Implement the finishing primitive. The candidate is a continuous approach: blend the
      contact point behind the ball into a point through it as alignment improves, rather than
      switching on a discrete alignment test that makes a close robot retreat first.
- [ ] Measure it with `tools/primitive_race.py angle` against both dribbling intents. Accept it
      only if it converts from an angle at a rate that is not zero and is not worse than
      dribbling on the line.
- [ ] Confirm dribbling is unchanged: the same probe on the same drills must not move for the
      driving intents.
- [ ] Re-run `tools/audit_skill_difficulty.py` over `shot`. A primitive that can finish from an
      angle changes what the ladder measures, and `ball_angle` may become a usable axis where it
      was withdrawn in cycle 4 for being unreachable.
- [ ] Ablate the carry-to-goal ratio over at least three settings, reported against goals per
      minute and against how often the ball reaches a convertible position.
- [ ] Record the chosen ratio and the measurements behind it in `docs/evidence/`.
- [ ] Port the primitive and the carry potential to `vsss-features` behind equivalence tests,
      after the ablation has chosen and not before.
