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
- [x] Measure it. `tools/probe_finishing_angle.py` places the striker at a chosen angle instead
      of trusting a drill, because the generator only ever places it on the line. Scoring is
      unchanged within noise: 0.46 to 0.50 at 30 degrees, zero from 60 degrees either way.
- [x] Confirm dribbling is unchanged: both driving intents score identically at every angle.
- [ ] Re-run `tools/audit_skill_difficulty.py` over `shot`. A primitive that can finish from an
      angle changes what the ladder measures, and `ball_angle` may become a usable axis where it
      was withdrawn in cycle 4 for being unreachable.
- [ ] Ablate the carry-to-goal ratio over at least three settings, reported against goals per
      minute and against how often the ball reaches a convertible position.
- [ ] Record the chosen ratio and the measurements behind it in `docs/evidence/`.
- [x] Port the approach fix to `vsss-features`. Deferring it was the stated plan, and the
      equivalence test refused: it caught the divergence the moment Python moved, and a native
      path disagreeing with its reference is worse than porting a small settled change early.
- [ ] Port the carry potential, after the ablation has chosen its coefficient.
