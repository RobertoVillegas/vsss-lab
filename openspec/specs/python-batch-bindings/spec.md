# python-batch-bindings Specification

## Purpose
TBD - created by archiving change m3-python-env-api. Update Purpose after archive.
## Requirements
### Requirement: Contiguous native batch API
Python SHALL reset and step multiple worlds using contiguous NumPy arrays, with
actions shaped `[world, 6, 2]` and one stable flattened state row per world.

#### Scenario: Step two worlds
- **WHEN** a contiguous action tensor for two worlds is stepped
- **THEN** the result is a contiguous state matrix with two rows

### Requirement: Native lifecycle
Python SHALL snapshot and restore individual worlds without changing neighbors.

#### Scenario: Replay through Python
- **WHEN** a snapshot is restored and identical actions are replayed
- **THEN** the resulting state row is exactly equal

