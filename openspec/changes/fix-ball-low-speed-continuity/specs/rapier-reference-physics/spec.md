# Delta: Rapier reference physics

## ADDED Requirements

### Requirement: Continuous passive ball deceleration

The reference backend SHALL evolve an unobstructed moving ball continuously
under configured damping and SHALL NOT expose an engine sleep threshold as an
instantaneous stop at ordinary VSSS play speeds.

#### Scenario: Low-speed ball crosses the engine sleep window

- **GIVEN** an unobstructed ball moving at 0.2 m/s with zero angular velocity
- **WHEN** more than two seconds of fixed simulation time elapse
- **THEN** its speed remains positive, decreases monotonically, and follows the
  configured passive damping curve without a discontinuity

#### Scenario: Ball starts at physical rest

- **GIVEN** a stationary unobstructed ball
- **WHEN** the physics world advances without contacts or applied forces
- **THEN** its pose and velocity remain unchanged without numerical drift
