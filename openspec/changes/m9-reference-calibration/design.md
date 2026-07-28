## Context

Julio's public simulator and thesis are the initial golden model, but published
metrics are not interchangeable with measurements from this project. Calibration
therefore compares isolated phenomena in canonical SI units and preserves source
and uncertainty beside every adopted value.

## Decisions

1. Keep source parameters and scenarios as reviewable JSON.
2. Compare differential-drive straight and center-turn motion to their analytic
   reference; compare passive ball decay to a committed reference envelope.
3. Report absolute position, heading, velocity, and stop-time errors independently.
4. A tolerance is a product decision, never silently fitted by the test.
5. Do not vendor assets whose reuse license is unclear; record URLs, hashes when
   locally verified, dimensions, and migration status instead.

## Rollback

Calibration is an observer/test layer. Removing it changes no runtime backend or
training contract.
