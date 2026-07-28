# heuristic-team-control Specification

## Purpose
TBD - created by archiving change m4-heuristic-baselines. Update Purpose after archive.
## Requirements
### Requirement: Scripted skills
The system SHALL provide deterministic go-to-target, go-to-ball, and goalie
wheel commands bounded by the normalized action range.

#### Scenario: Target ahead
- **WHEN** a target is directly ahead of a robot
- **THEN** both wheels receive equal positive commands

### Requirement: Dynamic identity-free assignment
Role assignment SHALL depend on physical state and SHALL permute equivalently
when robot slots and identities are permuted.

#### Scenario: Permute team robots
- **WHEN** the same physical robots are reordered
- **THEN** assigned roles and actions reorder with them
