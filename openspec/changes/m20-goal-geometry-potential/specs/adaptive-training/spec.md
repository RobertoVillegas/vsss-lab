## ADDED Requirements

### Requirement: Goal-aware potential shaping

Training MAY shape attacking geometry only through a bounded state potential
whose discounted transition reward cannot be farmed by preserving a pose.

#### Scenario: Attacker remains aligned and motionless

- **WHEN** the attacker stays behind the ball without changing the state
- **THEN** geometry shaping is non-positive
- **AND** it cannot outscore advancing the ball along a valid goal line

#### Scenario: Apparent forward shot misses the aperture

- **WHEN** the attacker-to-ball ray intersects the goal line outside the usable
  goal opening
- **THEN** the aperture component is zero
- **AND** no hard field-zone penalty is required

#### Scenario: Roles rotate

- **WHEN** tactical responsibility changes between teammates
- **THEN** geometry is evaluated for the current dynamic attacker
- **AND** no fixed robot identity receives a privileged reward

### Requirement: Oriented robot corner containment

The physical field MUST contain the full oriented robot footprint at each
clipped corner.

#### Scenario: Robot drives diagonally into a corner

- **WHEN** a 75 mm square robot applies sustained force toward a 70 mm chamfer
- **THEN** the complete robot support remains behind the inner diagonal face
- **AND** the authoritative state cannot place its body inside the clipped area
