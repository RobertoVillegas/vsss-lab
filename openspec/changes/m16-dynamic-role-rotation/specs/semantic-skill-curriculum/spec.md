# Delta: semantic skill curriculum

## MODIFIED Requirements

### Requirement: Learning-progress difficulty curriculum

The system SHALL allocate semantic skills and independent bounded difficulty axes
using rolling success and learning progress, SHALL prioritize weak skill families,
and SHALL retain non-zero mastered-skill, failure, and full-match rehearsal.

#### Scenario: One skill collapses while others remain mastered

- **GIVEN** recent success for one family is materially below the others
- **WHEN** the teacher allocates non-failure atomic drills
- **THEN** the weak family receives greater expected allocation while every mastered
  family retains a non-zero rehearsal probability

#### Scenario: A weak skill remains at the introductory frontier

- **GIVEN** sustained failure at the minimum supported difficulty
- **WHEN** curriculum adaptation lowers an axis
- **THEN** the axis remains at the declared 0.05 training floor

### Requirement: Skill-specific causal outcomes

The system SHALL evaluate versioned skill predicates from initial drill context,
authoritative observed transitions, and the same dynamic role assignment supplied
to the policy.

#### Scenario: Three-player rotation completes

- **GIVEN** a failed attacker, an incoming challenger, and a coverage player
- **WHEN** the challenger assumes attack and neutralizes the threat, coverage advances
  to support, and the failed attacker recovers to coverage
- **THEN** rotation-recovery succeeds only after the complete handoff remains valid
  for the confirmation window

#### Scenario: Rotation leaves the goal uncovered

- **GIVEN** a nominal role handoff
- **WHEN** the team remains uncovered for more than ten percent of the drill horizon
- **THEN** the transition cannot satisfy rotation-recovery
