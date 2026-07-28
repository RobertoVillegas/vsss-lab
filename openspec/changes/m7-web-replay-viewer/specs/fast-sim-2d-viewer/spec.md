## MODIFIED Requirements

### Requirement: Interactive replay controls

The viewer SHALL provide native or browser playback with pause, exact
single-frame stepping, seek, forward/backward skip, iteration selection when a
run contains multiple captures, and playback speed independently from
simulation execution.

#### Scenario: Step a paused replay

- **WHEN** a paused replay advances by one recorded frame
- **THEN** the viewer renders exactly the next recorded frame without advancing
  any physics backend
