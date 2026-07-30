# Hierarchical trajectory control

## ADDED Requirements

### Requirement: Primitive action semantics

The training system SHALL offer a versioned categorical primitive parser whose
outputs are stop, directional navigation, and directional strike.

#### Scenario: shared policy selects a strike

- **WHEN** either logical team selects the same canonical strike direction
- **THEN** both commands SHALL be geometrically equivalent under field reflection
- **AND** no physical robot identity SHALL own the primitive

### Requirement: Causal ball acquisition

Strike SHALL select a reachable acquisition point using only state available at
the current decision tick.

#### Scenario: moving ball

- **WHEN** the ball has non-zero velocity
- **THEN** the controller SHALL target a bounded future point rather than only
  the current ball position
- **AND** the chosen target SHALL be reproducible from the same state

### Requirement: Directed contact

Strike SHALL acquire behind the ball before driving through it.

#### Scenario: stationary ball

- **WHEN** a robot begins away from a stationary ball and requests a goal-facing
  strike
- **THEN** exact Rapier replay SHALL produce contact
- **AND** the ball SHALL leave contact with positive velocity in the requested
  half-plane

### Requirement: Experimental isolation

The repository SHALL retain direct-wheel MAPPO and provide paired MAPPO/IPPO
primitive configurations.

#### Scenario: algorithm comparison

- **WHEN** the paired short-run commands are executed with equal seeds, worlds,
  rollout length, and reward configuration
- **THEN** the only algorithmic difference SHALL be the critic observation
