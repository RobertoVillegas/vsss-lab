# Audited Difficulty Ladders

## Why

ADR 0016 repaired one family whose difficulty axes did not move the demand it scores.
Measuring all seven the same way showed the defect was systemic: two of five declared axes
were inert everywhere, one family could not be compiled at high angle, one deflected every
pass off its receiver so no controller ever completed one, one held constant the very
rotation it scores, and two ramped ball speed past anything interceptable.

A curriculum that lowers difficulty for a failing family cannot help when the axis it
lowers changes nothing, and phase gates that read such a family read a step function. Both
runs of this milestone stalled in their second phase and never allocated a single drill of
the two families that teach play.

See ADR 0016 and ADR 0017.

## Milestone and non-goals

Maintenance of the M15 semantic curriculum. Non-goals:

- no change to a predicate or to what any skill means;
- no change to a reward coefficient, a phase gate, or the phase patience;
- no claim that a scripted controller's failure proves a defect, since it can neither shoot
  nor pass.

## What changes

- sweep difficulty one axis at a time in the audit, because the axes are independent and a
  compound sweep misrepresents the demand;
- declare per family which axes have a measured gradient, and advance only those;
- repair the six families the matrix indicted, so no cell is invalid or inverted;
- bump the generator revision, so holdouts of different revisions cannot be mixed.

## Success criteria

- no axis is invalid or inverted under either probe;
- every declared axis reads as a ramp under a capable probe;
- a freshly bootstrapped policy scores above zero on the families that teach play, which
  scored zero at initialization and after fourteen hundred iterations before this change;
- the audit runs as a tool, so the next regression is measured rather than discovered.
