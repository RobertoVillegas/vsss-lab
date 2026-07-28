## ADDED Requirements

### Requirement: Adaptive parallel world stepping
The native batch SHALL advance independent worlds in stable index order and
SHALL use parallel scheduling only above a measured beneficial threshold.

#### Scenario: Step sixty-four worlds
- **WHEN** one action set is supplied for each of 64 worlds
- **THEN** worlds advance in parallel and results retain input world order

### Requirement: Fused repeated stepping
The native binding SHALL accept a positive repeat count and advance fixed
actions without reacquiring the Python GIL between repeats.

#### Scenario: Repeat four fixed actions
- **WHEN** the binding receives a batch and repeat count four
- **THEN** its result equals four ordered single steps in every world
