## ADDED Requirements

### Requirement: Backend lifecycle
The backend SHALL reset, step exactly one fixed tick, snapshot, restore, and
produce a deterministic checksum.

#### Scenario: Restore snapshot
- **WHEN** a world is stepped, restored, and replayed with identical actions
- **THEN** its state and checksum match the first execution

### Requirement: Basic batch
The batch backend SHALL hold independent worlds and reset one world without
changing any other world.

#### Scenario: Reset one world
- **WHEN** world one is reset in a two-world batch
- **THEN** world zero retains its tick and state
