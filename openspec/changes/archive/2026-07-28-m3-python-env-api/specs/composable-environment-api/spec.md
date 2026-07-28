## ADDED Requirements

### Requirement: Composable environment contracts
The Python layer SHALL define replaceable observation, action, reward, reset, and
termination protocols without importing them into the Rust hot loop.

#### Scenario: Replace observation builder
- **WHEN** a custom observation builder is supplied
- **THEN** reset and step use its output without backend changes

### Requirement: Standard adapters
The package SHALL provide PettingZoo Parallel, Gymnasium single-robot, and
Gymnasium centralized-team adapters with valid spaces and return signatures.

#### Scenario: Parallel API validation
- **WHEN** PettingZoo's parallel API test runs for random actions
- **THEN** the adapter satisfies the contract without warnings or shape errors
