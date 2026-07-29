## ADDED Requirements

### Requirement: Operational semantic run entry
The system SHALL launch high-budget semantic training only through an explicit
M15 protocol with inspectable initialization provenance and periodic holdout
selection.

#### Scenario: Warm-start from a compatible M14 policy
- **GIVEN** a checkpoint with the same algorithm and actor architecture
- **WHEN** an M15 run initializes from it
- **THEN** actor and critic parameters transfer while optimizer, policy version,
  RNG, and curriculum state start fresh

#### Scenario: Warm-start architecture differs
- **GIVEN** a checkpoint whose hidden size, policy architecture, or action parser
  differs
- **WHEN** initialization is requested
- **THEN** training rejects it before creating a misleading lineage

#### Scenario: A specialist hides missing skills
- **GIVEN** a checkpoint that masters approach but has zero defensive success
- **WHEN** periodic holdout selection compares candidates
- **THEN** minimum family success is ranked before aggregate success

#### Scenario: Semantic launch is requested
- **GIVEN** the dedicated live semantic recipe
- **WHEN** a new run is allocated
- **THEN** it uses the M15 configuration, records policy provenance, evaluates
  immutable holdouts periodically, and exposes the replay viewer
