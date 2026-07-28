# ood-robustness-evaluation Specification

## Requirements

### Requirement: Paired held-out evaluation
Robust and nominal policies SHALL be compared on identical held-out seeds,
initial states, perturbation distributions, and horizons.

#### Scenario: Evaluate robust promotion
- **WHEN** the robust policy has positive mean progress and exceeds the nominal
  policy by the configured margin
- **THEN** the OOD gate passes and records every paired score

### Requirement: No nominal-suite substitution
OOD ranges SHALL be versioned separately from nominal training defaults.

#### Scenario: Audit suite distribution
- **WHEN** an OOD report is inspected
- **THEN** all ranges and realized samples are present
