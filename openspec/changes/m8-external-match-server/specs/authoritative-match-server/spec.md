## ADDED Requirements

### Requirement: Authoritative fixed-tick match
The server SHALL exclusively own match state, seeds, simulation clock,
controller assignments, action validation, events, replay, and result.

#### Scenario: Complete a heterogeneous match
- **WHEN** two compatible controllers remain healthy until termination
- **THEN** the server emits one reproducible result and checksum-valid canonical replay

### Requirement: Deterministic deadline adjudication
For each controller slot and control tick, the server SHALL accept at most one
valid on-time action and deterministically apply the configured fallback for a
missing or late action.

#### Scenario: Controller misses a deadline
- **WHEN** no valid action exists at the control deadline
- **THEN** the configured repeat-last-safe or zero action is applied and recorded

### Requirement: Controller lifecycle and isolation
The server SHALL negotiate capabilities, monitor heartbeat leases, and terminate
or forfeit controllers according to configured health policy without blocking
the other controller or simulation clock.

#### Scenario: Controller disconnects during play
- **WHEN** one controller's heartbeat lease expires
- **THEN** the match records the infrastructure event and resolves it by explicit policy

### Requirement: Assignment is not physical identity
The server SHALL assign ephemeral slots, match sides, and team colors without
encoding a permanent marker, robot role, or policy specialization.

#### Scenario: Switch sides at halftime
- **WHEN** teams switch field sides or identification colors
- **THEN** controller and logical robot identity remain stable while assignment metadata changes
