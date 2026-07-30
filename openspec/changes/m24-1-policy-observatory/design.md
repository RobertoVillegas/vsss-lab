# Design

## Replay contract

Replay schema v2 keeps three separate concepts:

- `policy_intents`: categorical decision and policy confidence;
- primitive target, phase, acquisition/exit geometry;
- `actions` and snapshot wheel speeds: commanded and physically applied motion.

Legacy replays remain readable because the new field is optional.

## Interaction

The right rail remains a compact six-actor roster. Selecting an actor expands
only that card and activates its field overlay. The timeline groups consecutive
equal primitives into segments, so long captures remain legible. Event markers
from reward-independent replay analytics seek to their corresponding frame.

## Training diagnostics

Primitive policies report normalized categorical entropy using `ln(17)` as the
maximum and a 17-bin action histogram. The UI groups bins into stop, navigate,
and strike while retaining direction-level inspection. Continuous and lattice
runs keep their compatible diagnostics.

## Layout

The left rail ends after replay selection, score, time, and policy matchup.
Vision remains rendered with safe defaults but its detailed controls are
removed from the primary simulation workflow.
