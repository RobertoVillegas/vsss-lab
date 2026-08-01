## ADDED Requirements

### Requirement: The per-decision environment computation is native

Per-world environment computation SHALL execute natively and SHALL be invoked once per
decision for the whole batch, not once per world. Python dictionaries SHALL NOT appear in the
per-decision path.

#### Scenario: One call per decision

- **WHEN** a batch of worlds advances one decision
- **THEN** the observation, reward and detection work crosses the binding once for the batch

#### Scenario: Restart without a dictionary

- **WHEN** an impasse restarts play
- **THEN** the world is edited natively rather than by serializing and restoring a snapshot

### Requirement: A ported slice must agree with its reference

A slice SHALL NOT replace its Python implementation until a golden-equivalence test shows the
two agree on recorded states, within a tolerance small enough that no decision branch differs.
The Python implementation SHALL remain available as the reference until its slice is retired.

#### Scenario: Native and reference disagree

- **WHEN** the native and Python results differ beyond the stated tolerance on any recorded
  state
- **THEN** the slice is not accepted, regardless of the speedup it shows

#### Scenario: Difference that changes a branch

- **WHEN** a difference is within tolerance in value but flips a threshold comparison
- **THEN** it is treated as a disagreement rather than as rounding

#### Scenario: Reversibility

- **WHEN** a slice is in flight
- **THEN** the tree builds, the suite passes, and the Python path can still be selected
