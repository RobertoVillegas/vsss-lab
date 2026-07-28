# policy-league-registry Specification

## Purpose
TBD - created by archiving change m7-league-self-play. Update Purpose after archive.
## Requirements
### Requirement: Versioned policy registry
The system SHALL atomically persist schema-versioned policy entries containing
policy identity/version, category, status, checkpoint path/hash, algorithm,
rating, parent, creation time, and training iteration.

#### Scenario: Register immutable checkpoint
- **WHEN** a new checkpoint with a unique policy ID/version is registered
- **THEN** the registry records its SHA-256 and leaves prior entries unchanged

#### Scenario: Reject duplicate identity
- **WHEN** an existing policy ID/version is registered again
- **THEN** the registry fails without changing its file

### Requirement: Seeded weighted matchmaking
The league SHALL select opponents from configured eligible categories using an
explicit seed and SHALL record the realized selection.

#### Scenario: Repeat matchmaking
- **WHEN** registry, weights, exclusions, and seed are identical
- **THEN** the selected opponent is identical

### Requirement: Historical retention
Promoted and superseded main policies SHALL remain addressable as immutable
historical opponents.

#### Scenario: Promote a candidate
- **WHEN** a candidate becomes main
- **THEN** the previous main becomes historical and both checkpoints remain registered

