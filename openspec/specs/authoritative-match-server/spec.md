# authoritative-match-server Specification

## Purpose
TBD - created by archiving change m8-external-match-server. Update Purpose after archive.
## Requirements
### Requirement: Authoritative fixed-tick match
The server SHALL exclusively own simulation state, control deadlines,
adjudication decisions, replay, and result.

#### Scenario: Miss a control deadline
- **WHEN** a controller action is absent or late at a control boundary
- **THEN** the configured safe fallback is applied and recorded deterministically

### Requirement: Isolated controller competition
Controllers SHALL run in independent processes on loopback or an internal-only
container network and SHALL not receive direct simulator access.

#### Scenario: Run heterogeneous controllers
- **WHEN** Rust and Python controllers negotiate compatible capabilities
- **THEN** both complete a match and receive the authoritative result

### Requirement: Auditable result
The server SHALL emit configuration, controller manifests, canonical states,
action decisions, outcome, and checksum in its match artifact.

#### Scenario: Inspect a completed match
- **WHEN** the artifact is replayed
- **THEN** every applied or rejected controller decision is attributable

