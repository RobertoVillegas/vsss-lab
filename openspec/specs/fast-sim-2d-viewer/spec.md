# fast-sim-2d-viewer Specification

## Purpose
TBD - created by archiving change m4-live-visualization. Update Purpose after archive.
## Requirements
### Requirement: Shared live and replay scene

The 2D viewer SHALL render field geometry, robots, ball, score, and event
overlays from the same visual-frame model for live and replay sources.

#### Scenario: Compare live and replay tick

- **WHEN** the viewer receives the same tick from a live source and a replay
- **THEN** both produce the same exact-tick scene projection

### Requirement: Interactive replay controls

The viewer SHALL support pause, exact single-tick stepping, seek, and playback
speed independently from simulation execution.

#### Scenario: Step a paused replay

- **WHEN** a paused replay advances by one tick
- **THEN** the viewer renders exactly the next recorded frame without advancing
  any physics backend

### Requirement: Diagnostic overlays

The viewer SHALL be able to display robot identifiers, headings, velocity
vectors, action values, trajectories, event markers, reward values when
present, and live-frame drop counts.

#### Scenario: Inspect a goal tick

- **WHEN** a displayed frame contains a goal event
- **THEN** the scene identifies the event and exposes its exact recorded tick

### Requirement: Deterministic headless rendering

The viewer projection SHALL support a headless mode that generates a stable
artifact for fixed input, viewport, and viewer version.

#### Scenario: Render the same replay frame twice

- **WHEN** an identical frame and viewport are rendered twice in headless mode
- **THEN** the generated artifact checksums are identical

