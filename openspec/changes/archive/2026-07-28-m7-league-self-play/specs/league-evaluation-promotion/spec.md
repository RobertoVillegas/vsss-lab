## ADDED Requirements

### Requirement: Deterministic tournament reports
The system SHALL evaluate scheduled policy pairs over fixed seeds and side
switches and record score, progress, outcome, duration, policy versions,
configuration hash, and infrastructure status.

#### Scenario: Repeat a tournament
- **WHEN** policies, simulator configuration, seeds, and schedule are unchanged
- **THEN** the canonical report is byte-identical

### Requirement: Elo rating
Successful tournament matches SHALL update ratings with standard logistic Elo;
draws SHALL score one half and infrastructure failures SHALL not update ratings.

#### Scenario: Update winner and loser
- **WHEN** a rated match has a winner
- **THEN** winner gain equals loser loss within numerical tolerance

### Requirement: Non-regression promotion
A candidate SHALL be promoted only after passing identity gates, unseen seeds,
main, historical, and heuristic fixtures with the configured aggregate margin
and no fixture regression.

#### Scenario: Reproduce promotion decision
- **WHEN** the same candidate and evaluation manifest are evaluated twice
- **THEN** both decisions and canonical reports are identical

#### Scenario: Reject regressive candidate
- **WHEN** any required fixture falls below its regression floor
- **THEN** the candidate remains unpromoted
