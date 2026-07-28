## ADDED Requirements

### Requirement: Identity-free agent observations
The system SHALL build team-canonical agent observations from relative state,
Deep Sets teammate/opponent pools, goals, and match context without physical IDs
or identity-bound role fields.

#### Scenario: Rename and reorder robots
- **WHEN** physical IDs and storage slots are permuted while geometry is preserved
- **THEN** observations follow the physical agents and contain no identity signal

### Requirement: Shared decentralized actor
The system SHALL apply one shared actor independently to each of three agent
observations and produce one two-wheel action per agent.

#### Scenario: Permute agent inputs
- **WHEN** the three agent observations are permuted
- **THEN** the three actions are permuted identically within numerical tolerance

#### Scenario: Execute without centralized state
- **WHEN** the trained actor performs inference
- **THEN** each action is computable from only that agent's observation and shared weights
