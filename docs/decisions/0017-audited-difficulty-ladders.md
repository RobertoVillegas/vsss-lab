# ADR 0017: Audited difficulty ladders and a declared axis map

- Status: accepted
- Date: 2026-07-31

## Context

ADR 0016 fixed one family after measurement showed its difficulty axes did not move the
demand it is scored on. Finding the next by hand is not a method, so all seven families
were measured with a fixed probe swept across each axis.

The first sweep raised all five axes together and made every family look like a wall.
That was a defect in the method: the axes are declared independent and the curriculum
samples them independently, so a compound sweep compounds the demand and misrepresents
what a policy ever faces. Sweeping one axis at a time, with the rest held easy, produced a
matrix of seven families by five axes that named each defect directly.

What the matrix showed, before any repair:

- `target_width` and `opponent_pressure` were inert in all seven families under both a
  scripted and a trained probe. Two of five declared axes were decoration, so the
  curriculum believed it was exploring a space that did not exist.
- `save_deflection` could not be compiled at high angle: a fixed angular deflection
  eventually points the ball away from the goal, and the validator rejects a save that is
  not goal-bound.
- `shot` could not be compiled near the goal once range became its axis, because the
  defenders are parked there and the ball overlapped them.
- `pass_receive` deflected every launch at least 0.45 rad off the receiver, and its ball
  speed was the only reason the ball ever arrived. It scored zero at every band under both
  probes, for the entire life of two runs.
- `rotation_recovery` held the support's journey back into coverage constant, which is the
  rotation it scores, and was inverted under the scripted probe.
- `approach` was solved at every level by everything.
- `interception` and `save_deflection` ramped ball speed to 1.05 m/s, which nothing can
  intercept from behind, so every band above the middle was dead.

## Decision

A difficulty axis is declared for a family only when a capable probe measures a gradient
along it. `FAMILY_AXES` records that map, the curriculum advances only declared axes, and
`tools/audit_skill_difficulty` is the acceptance test: no axis may be invalid or inverted,
and every declared axis must read as a ramp under a capable probe.

Two probes are required and neither is sufficient alone. The scripted controller
establishes validity and detects an axis that runs backwards, but it cannot shoot or pass,
so its failures are not evidence of a defect. A trained checkpoint probes at the capability
the curriculum actually faces, which is where a ramp has to exist.

The six families above were repaired, the axis map was pruned to nine measured gradients
across seven families, and the generator revision was bumped.

## Consequences

- Both probes now report no invalid and no inverted axis. Under the capable probe every
  declared axis reads as a ramp except `pass_receive.ball_angle`, which is a steep but live
  cliff, and `rotation_recovery.spawn_distance`, which declines noisily.
- The scripted probe still reports cliffs, which is expected and not a defect: its
  competence threshold sits at the bottom of the ladder. Its value is validity and
  inversion, not ramp shape.
- A freshly bootstrapped policy now scores 0.50 on `pass_receive` and 0.65 on
  `rotation_recovery`, families that scored zero at initialization and after fourteen
  hundred iterations of two separate runs. The phases that teach play have a foothold for
  the first time.
- Holdouts and every prior evaluation belong to earlier generator revisions and are not
  comparable. The milestone needs a run from scratch, which is the point.
- Difficulty is now a four-dimensional space at most for any family, not five. That is a
  reduction in declared capability, honestly reflecting what was ever measurable.
