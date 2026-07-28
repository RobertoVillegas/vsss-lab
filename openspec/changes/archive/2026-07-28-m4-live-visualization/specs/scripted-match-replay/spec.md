## MODIFIED Requirements

### Requirement: Inspectable replay

The replay SHALL contain versioned metadata, snapshots, actions, events, and
checksums; a headless inspector SHALL validate and summarize it; and the replay
SHALL adapt to the same visual-frame model consumed by live visualization.

#### Scenario: Inspect valid replay

- **WHEN** the inspector reads a completed replay
- **THEN** it reports ticks, score, goals, and final checksum

#### Scenario: Visualize valid replay

- **WHEN** the viewer opens a completed replay
- **THEN** every recorded tick can be decoded into a visual frame without
  executing a physics backend
