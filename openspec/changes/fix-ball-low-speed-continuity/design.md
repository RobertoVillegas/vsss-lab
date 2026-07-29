# Design

## Evidence from the failing replay

At replay index 1679 (`t=33.58 s`) the ball was at `(0.4369, 0.1738)` and
moving at 0.1935 m/s. At index 1680 it remained in open field, with the nearest
robot 0.204 m away and no event, but both velocity components and angular
velocity became exactly zero. The ball had remained below Rapier's 0.4 m/s
linear and 0.5 rad/s angular sleep thresholds for approximately two seconds.

## Reference comparison

pSim 0.2.4 uses Box2D with a 0.01 m/s sleep tolerance and applies its ball
friction force with `wake=True` on every step. Its moving ball therefore
decelerates continuously rather than sleeping at ordinary play speed.

Julio's Gazebo `simulation_vsss` uses a physical 3D sphere with ODE contact,
surface friction, and auto-disable enabled. Auto-disable is appropriate at
physical rest, but does not support Rapier's much larger 0.4 m/s threshold in
our metre-scale planar model.

## Decision

Set `can_sleep(false)` only on the Rapier ball body. A single active body per
world has negligible cost compared with six controlled robots and prevents
engine-specific sleep thresholds from becoming part of learned dynamics.
Configured linear/angular damping still reduces velocity, and match stagnation
logic supplies an explicit application-level terminal condition.
