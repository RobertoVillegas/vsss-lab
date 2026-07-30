## ADDED Requirements

### Requirement: continuous geometric intent

The learned policy SHALL select a semantic skill independently from its
continuous direction and intensity parameters.

#### Scenario: intermediate heading

- **WHEN** navigation requests a heading between canonical M24 directions
- **THEN** the executor SHALL preserve that heading without binning it.

#### Scenario: angle wrap

- **WHEN** a heading crosses from +π to -π
- **THEN** its vector representation SHALL remain continuous.

### Requirement: joint policy optimization

MAPPO SHALL include both categorical skill likelihood and bounded continuous
parameter likelihood in its PPO ratio.

#### Scenario: policy update

- **WHEN** an M24.2 trajectory is optimized
- **THEN** its PPO ratio SHALL use the joint skill and parameter log probability.

### Requirement: observable control quality

Captured replay intent SHALL expose requested angle and intensity. Training
metrics SHALL expose mean intensity and mean/p95 heading change.

#### Scenario: captured intent

- **WHEN** an M24.2 replay is captured
- **THEN** each learned actor intent SHALL report exact angle and intensity.

### Requirement: legacy isolation

Legacy `primitive` checkpoints and replays SHALL remain readable, while an
M24.2 policy SHALL use a distinct parser and policy identity.

#### Scenario: legacy replay

- **WHEN** the viewer opens a legacy `primitive` replay
- **THEN** it SHALL retain its original discrete direction interpretation.
