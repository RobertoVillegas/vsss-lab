# Semantic skill curriculum delta

## ADDED Requirements

### Requirement: Roster-scaled skill acquisition

The curriculum MUST train skills with the smallest meaningful active roster and
MUST retain earlier rosters as rehearsal when advancing toward 3v3.

#### Scenario: Pass skill avoids premature 3v3 congestion

- **WHEN** a pass/receive training scenario is compiled
- **THEN** it uses two controlled robots
- **AND** it uses one opponent for routine practice or two for frontier practice
- **AND** inactive robots neither collide nor contribute to reward geometry

#### Scenario: Rotation requires team context

- **WHEN** a rotation/recovery scenario is compiled
- **THEN** all three controlled robots are active
- **AND** at least two opponents provide realistic pressure

### Requirement: Contextual contact incentives

The environment MUST distinguish incidental contact, productive challenges, and
sustained deadlocks.

#### Scenario: Brief challenge is not punished

- **WHEN** two robots contact for less than the configured grace interval
- **THEN** no sustained-contact penalty is emitted

#### Scenario: Ally deadlock is discouraged

- **WHEN** allies remain in contact beyond the grace interval without ball
  involvement
- **THEN** a bounded duration-sensitive penalty is emitted
- **AND** separating from the contact is recorded as an escape

#### Scenario: Defensive opponent block remains legal

- **WHEN** a robot contacts an opponent while the ball continues meaningful
  progress
- **THEN** the contact is measured
- **BUT** no opponent-deadlock penalty is emitted

### Requirement: Coordination-aware promotion

Checkpoint selection MUST expose pass, rotation, contact, and coverage gates
instead of relying only on aggregate semantic success.

#### Scenario: Aggregate improvement hides a coordination regression

- **WHEN** overall success improves but pass or rotation falls below its floor
- **THEN** the checkpoint is not promoted as best
- **AND** the failed gates are recorded in training telemetry
