## ADDED Requirements

### Requirement: Stable-width live telemetry
The replay viewer SHALL use monospace tabular-number typography for rapidly
changing controls, metrics, timestamps, and actor telemetry.

#### Scenario: Numeric telemetry changes
- **WHEN** adjacent replay frames contain different digits
- **THEN** the surrounding labels and columns retain stable glyph widths
