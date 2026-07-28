## Why

M7 can evaluate policies only inside one Python process. M8 must let independently
implemented Rust and Python controllers compete under one authoritative clock,
with reproducible deadlines, failure handling, and replay evidence.

## What Changes

- Activate the Rust protocol and match-server crates.
- Define a versioned FlatBuffers envelope and compatibility gates.
- Add loopback ZeroMQ controller sessions with handshake, capabilities,
  observations, actions, heartbeat, events, results, and errors.
- Enforce controller slots, monotonic sequences, per-tick deadlines, bounded
  actions, and deterministic late-action policy.
- Add minimal Rust and Python controller SDKs and heterogeneous tournament smoke.
- Preserve logical/team/visual identity separation for future camera use without
  implementing vision in M8.

## Capabilities

### New Capabilities

- `external-match-protocol`: Versioned wire envelope, compatibility, validation,
  and controller lifecycle.
- `authoritative-match-server`: Deterministic match ownership, deadlines,
  failures, isolation, replay, and results.
- `controller-sdk`: Small Rust and Python interfaces for independent controllers.

### Modified Capabilities

None.

## Impact

Activates `crates/vsss-protocol`, `crates/vsss-match-server`, controller SDK
packages, schemas, protocol fixtures, containers, commands, tests, and M8
evidence. Adds locked FlatBuffers and ZeroMQ dependencies outside all simulator
and learner hot loops.
