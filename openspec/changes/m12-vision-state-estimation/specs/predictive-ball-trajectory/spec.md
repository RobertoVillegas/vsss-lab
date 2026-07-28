## ADDED Requirements

### Requirement: Causal ball projection

The trajectory predictor SHALL derive future ball samples only from the selected
current measured/estimated state, its uncertainty, a pinned field profile, and a
configured prediction horizon.

#### Scenario: Future truth changes

- **WHEN** two replay variants share identical measurements through a decision
  tick but diverge afterward
- **THEN** their policy-visible estimates and predictions at that tick are
  identical

### Requirement: Calibrated field interaction

The collision-aware predictor SHALL account for damping, restitution, walls,
goals, and 70 mm corner chamfers without advancing or mutating the authoritative
match.

#### Scenario: Project toward a chamfer

- **WHEN** the estimated ball velocity intersects a clipped field corner
- **THEN** the projected path reflects from the chamfer according to the pinned
  model while canonical state remains unchanged

### Requirement: Interception query

The predictor SHALL report earliest valid intersection time and point for a
configured goalkeeper line or segment, including uncertainty and no-intersection
outcomes.

#### Scenario: Ball approaches allied goal

- **WHEN** the projected path crosses the goalkeeper movement segment within the
  horizon
- **THEN** the predictor returns the causal interception point, time, and
  uncertainty

### Requirement: Optional policy feature

Projected positions or interception values SHALL enter a policy only through a
versioned observation adapter and controlled ablation.

#### Scenario: Load an existing MAPPO checkpoint

- **WHEN** no predictive observation adapter is selected
- **THEN** the checkpoint receives the unchanged current-state observation schema
