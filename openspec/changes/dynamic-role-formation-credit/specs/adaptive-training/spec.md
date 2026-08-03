## ADDED Requirements

### Requirement: Dynamic-role formation credit

The trainer SHALL keep terminal outcomes shared while rewarding improvement in the formation of
active support and coverage responsibilities without binding either responsibility to identity.

#### Scenario: Robots rotate responsibilities

- **WHEN** a support or coverage responsibility moves to another physical robot
- **THEN** formation credit follows the responsibility and no robot ID appears in the reward

#### Scenario: Static formation

- **WHEN** support and coverage hold an unchanged formation
- **THEN** they cannot repeatedly collect positive formation reward

#### Scenario: Reduced roster

- **WHEN** a semantic scenario disables support or coverage
- **THEN** the inactive responsibility contributes neither geometry nor reward

### Requirement: Role-resolved action telemetry

Training metrics SHALL report primitive selection fractions separately for attacker, support,
and coverage.

#### Scenario: Aggregate actions hide collapse

- **WHEN** attacker and coverage select the same primitive at materially different rates
- **THEN** both rates are present in the run record

