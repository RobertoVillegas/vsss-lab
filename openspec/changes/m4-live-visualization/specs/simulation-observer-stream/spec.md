## ADDED Requirements

### Requirement: Canonical visual frame

The observer layer SHALL expose a versioned frame containing a monotonic tick,
simulation time, canonical match snapshot, actions, event flags, and optional
reward and diagnostic values without adding renderer dependencies to
`vsss-spec`.

#### Scenario: Adapt a completed simulation tick

- **WHEN** match execution completes a tick with canonical state and actions
- **THEN** every configured observer receives equivalent frame semantics for
  that tick

### Requirement: Optional headless execution

The simulator SHALL execute without constructing frames or loading graphics,
networking, ROS, or viewer runtimes when no observer is configured.

#### Scenario: Train without an observer

- **WHEN** a match is executed through the no-observer path
- **THEN** its final canonical checksum equals the observed execution checksum
  for identical inputs

### Requirement: Non-blocking live delivery

A live observer SHALL use bounded delivery, retain the newest sampled frame,
and count dropped stale frames without applying backpressure to simulation.

#### Scenario: Viewer consumes slower than simulation

- **WHEN** more sampled frames are produced than the live observer can consume
- **THEN** simulation continues and the observer reports dropped frames before
  yielding the newest available frame

### Requirement: Lossless recording remains explicit

Lossless replay recording SHALL be a separate sink with explicit failure
reporting and SHALL NOT be inferred from enabling a lossy live viewer.

#### Scenario: Live viewer drops frames

- **WHEN** live delivery drops intermediate frames while replay recording is
  enabled
- **THEN** the replay retains every configured recorded tick
