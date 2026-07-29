## ADDED Requirements

### Requirement: Phased semantic skill consolidation

The curriculum MUST focus training on one skill phase while retaining all skill
families in immutable evaluation.

#### Scenario: Future skill is not yet active

- **WHEN** the curriculum is in foundation
- **THEN** pass and rotation scenarios are excluded from training gradients
- **AND** pass and rotation remain present in holdout evaluation

#### Scenario: Phase promotion

- **WHEN** all current phase gates pass on two consecutive holdout evaluations
- **THEN** the curriculum advances exactly one phase
- **AND** preserves bounded rehearsal of earlier skills

### Requirement: Consolidated checkpoint selection

Checkpoint selection MUST prefer broader retained competence over an isolated
minimum-family improvement within the same phase.

#### Scenario: Weak-family gain degrades global outcomes

- **WHEN** two checkpoints pass the same number of gates in the same phase
- **AND** one has higher global success and fewer unresolved trials
- **THEN** that consolidated checkpoint ranks higher
