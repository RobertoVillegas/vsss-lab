## Why

The replay field was rendered as a plain rectangle and the reference physics
used square corners. The calibrated VSSS layout includes 70 mm clipped corners,
penalty areas, goal-area arcs, restart markers, and visual robot tags.

## What Changes

- Add physical 70 mm corner chamfers to the Rapier reference backend.
- Render the canonical 1.50 x 1.30 m playing surface plus 0.10 m goals with the
  calibrated markings found in the public Julio simulators.
- Add compact team and per-robot visual tags without changing physical identity.
- Keep rendering out of the training hot loop and retain Rapier as the canonical
  training backend.

## Capabilities

### Modified Capabilities

- `rapier-reference-physics`: clipped-corner collision geometry.
- `training-replay-web`: calibrated field markings and robot tags.
