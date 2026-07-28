## Context

pSim's outer 1.70 m width includes two 0.10 m-deep goals, so it agrees with the
canonical 1.50 m playing length. Its renderer and Box2D geometry additionally
encode 70 mm corner clips and standard field markings that were absent locally.

## Decisions

1. Treat corner chamfers as reference-backend calibration, not a new serialized
   contract field, because the current VSSS field profile has one calibrated
   layout and existing replay/config schema compatibility must be preserved.
2. Derive all viewer coordinates from canonical field-centered SI coordinates.
3. Draw visual tags as presentation only; robot IDs and policy assignment remain
   independent.
4. Use public implementations as dimensional references and write original code.
   pSim is GPLv3, while `simulation_vsss` is MIT.

## Validation

- A physics regression sends the ball diagonally into a corner and asserts the
  chamfer limits its reach.
- Existing collision, goal, deterministic replay, TypeScript, and full local
  gates remain green.

## Rollback

Revert this change to restore rectangular collision and minimal field rendering.
