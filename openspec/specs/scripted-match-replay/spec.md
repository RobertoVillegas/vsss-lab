# scripted-match-replay Specification

## Purpose
TBD - created by archiving change m4-heuristic-baselines. Update Purpose after archive.
## Requirements
### Requirement: Reproducible 3v3 match
Identical config, kickoff, seed, controllers, and tick count SHALL produce the
same final checksum, score, and replay.

#### Scenario: Repeat scripted match
- **WHEN** the same 3v3 match is run twice
- **THEN** replay bytes and final summary are identical

### Requirement: Inspectable replay
The replay SHALL contain versioned metadata, snapshots, actions, events, and
checksums, and a headless viewer SHALL validate and summarize it.

#### Scenario: Inspect valid replay
- **WHEN** the viewer reads a completed replay
- **THEN** it reports ticks, score, goals, and final checksum

