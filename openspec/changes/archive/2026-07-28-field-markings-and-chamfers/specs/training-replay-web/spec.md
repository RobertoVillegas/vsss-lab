## ADDED Requirements

### Requirement: Calibrated field presentation

The replay viewer SHALL render the canonical playing surface, goals, clipped
corners, penalty areas, goal-area arcs, restart markers, and robot visual tags
from canonical field-centered coordinates.

#### Scenario: Inspect a captured match

- **WHEN** the viewer renders any canonical replay frame
- **THEN** the field markings and robot tags remain aligned with the physical
  entities at every viewport size
