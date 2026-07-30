## MODIFIED Requirements

### Requirement: Paired skill and transfer evaluation

The system SHALL evaluate every candidate over paired colors, independent
seeds, immutable skill holdouts, full matches, heuristic play, the promoted
incumbent of its own lineage, and historical policies of that lineage before
authorizing a high-budget run. A comparison SHALL share one action parser with
both policies, because an action space is not portable across milestones.

#### Scenario: Skill success rises but match play regresses

- **GIVEN** a candidate with higher interception and pass drill success
- **WHEN** its paired full-match confidence floor regresses against the promoted
  incumbent of its lineage
- **THEN** promotion and the high-budget run are rejected

#### Scenario: Opponent belongs to a different action space

- **GIVEN** a candidate and an opponent whose action parsers differ
- **WHEN** a paired evaluation is requested for that pair
- **THEN** the evaluation is rejected rather than reinterpreting one policy's
  transport tokens under the other's parser

#### Scenario: Candidate clears entry gates

- **GIVEN** deterministic valid generation, complete difficulty coverage,
  learnability above controls, non-regressed full matches, and recorded
  throughput
- **WHEN** the promotion evaluator accepts the candidate
- **THEN** it writes a machine-readable decision and the exact high-budget run
  command may be published
