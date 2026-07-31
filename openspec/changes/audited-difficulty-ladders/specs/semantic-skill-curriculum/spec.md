## ADDED Requirements

### Requirement: Declared and audited difficulty axes

A family SHALL declare only the difficulty axes along which a capable probe measures a
gradient, and the curriculum SHALL advance only declared axes. Every declared axis SHALL be
compilable at every level, SHALL NOT be inverted, and SHALL read as a graded decline under a
capable probe. Difficulty SHALL be audited one axis at a time, because the axes are
independent and a compound sweep misrepresents the demand a policy faces.

#### Scenario: Axis that changes nothing

- **WHEN** a probe's success rate is unchanged across an axis
- **THEN** that axis is not declared for that family, rather than advanced by the curriculum

#### Scenario: Axis that runs backwards

- **WHEN** a probe succeeds more often at higher difficulty than at lower
- **THEN** the axis is reported as inverted and is not fit to declare

#### Scenario: Two probes disagree

- **GIVEN** a scripted controller that cannot perform a skill
- **WHEN** it fails every band of that family
- **THEN** the ladder is reported as untested rather than as a defect, and a capable probe
  decides its shape

#### Scenario: Every level compiles

- **WHEN** a declared axis is swept across its full range
- **THEN** every level produces a valid scenario
