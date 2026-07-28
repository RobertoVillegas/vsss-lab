# canonical-interchange Specification

## Purpose
TBD - created by archiving change m1-canonical-spec. Update Purpose after archive.
## Requirements
### Requirement: Strict versioned JSON
Canonical roots SHALL round-trip through JSON without loss and SHALL reject
unknown fields or unsupported schema versions.

#### Scenario: Round-trip a golden state
- **WHEN** the M1 match-state fixture is deserialized, validated, serialized, and deserialized
- **THEN** the final value equals the initial value

#### Scenario: Reject an unknown field
- **WHEN** a canonical JSON object contains a field not defined by schema version 1
- **THEN** deserialization fails

### Requirement: Discoverable contract reflection
The crate SHALL expose deterministic type and field descriptors for canonical
root contracts.

#### Scenario: Inspect match state
- **WHEN** a consumer queries the reflection catalog for `MatchState`
- **THEN** it discovers the schema, tick, time, score, ball, robots, and events fields

### Requirement: Canonical team reflection
The system SHALL transform yellow-team perspective into canonical positive-x
orientation by rotating the field 180 degrees and swapping team-relative labels.

#### Scenario: Reflection is an involution
- **WHEN** any valid normalized match state is reflected twice
- **THEN** the original state is recovered

#### Scenario: Goals and scores follow teams
- **WHEN** a state with a blue goal event and unequal scores is reflected
- **THEN** blue/yellow scores, teams, and goal event bits are swapped
