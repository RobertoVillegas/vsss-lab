## Context

The simulator, environment, policies, league, and replay capture now work
in-process. M8 introduces a trust and timing boundary so heterogeneous
controllers can compete without sharing code or runtime. The canonical spec
crate remains free of networking and serialization dependencies.

Official VSSS operation assigns removable team colors and personal visual tags.
ADR-0010 separates those observable markers from logical identity. M8 transports
assigned controller slots and canonical observations; it does not transport
camera images or implement association.

## Goals / Non-Goals

**Goals:**

- One authoritative server owns state, clock, validation, replay, and result.
- Rust and Python controllers negotiate compatible capabilities and compete.
- Every accepted/rejected/late action is deterministic and auditable.
- Wire schemas evolve additively and have golden cross-language fixtures.
- Local/container execution is private and reproducible.

**Non-Goals:**

- Learner-to-simulator networking, public internet play, distributed scheduling,
  camera pixels, marker detection, ROS/Gazebo, encryption, or untrusted
  multi-tenant sandboxing.

## Decisions

1. Use FlatBuffers with a four-byte file identifier and one union envelope.
   Every message carries protocol version, match ID, controller slot, monotonic
   sequence, server tick, timestamp, deadline, and typed payload. Tables evolve
   only by appending fields with stable IDs/defaults. `flatc --conform` and
   previous-version golden buffers gate changes.
2. Use native `zeromq/zmq.rs` ROUTER sockets in the server and DEALER clients.
   Explicit identities permit independent clients without blocking the server
   on request/reply lockstep. In M8 endpoints bind to loopback or private
   container networks only.
3. Run an authoritative fixed-tick state machine. At each control boundary the
   server publishes observations, collects at most one action per active slot,
   applies validated on-time actions, and deterministically repeats the last
   safe action or zeros it according to match configuration.
4. Use server tick and monotonic sequence for ordering. Wall-clock timestamps
   are diagnostics and never decide simulation order. Duplicate, stale, future,
   malformed, wrong-slot, and out-of-range actions are rejected and recorded.
5. Handshake assigns an ephemeral controller slot and match-side/team color.
   Physical marker and tactical role are not controller identity. Side switching
   changes assignment metadata, not controller or policy identity.
6. SDKs expose typed callbacks (`on_reset`, `act`, `on_event`, `on_result`) and
   hide transport framing. The first SDKs are Rust and Python; controllers can
   be synchronous internally while the SDK handles heartbeat and deadlines.
7. A match artifact contains config, seeds, build/protocol metadata, controller
   manifests, all adjudication decisions, canonical replay, and signed-off
   result checksum. Failures never silently become valid match outcomes.

## Risks / Trade-offs

- Async scheduling can make deadline boundaries flaky → use an injected clock
  and deterministic fake-clock contract tests.
- FlatBuffers generated code is noisy → commit schemas and generated bindings,
  check regeneration cleanliness, and review schema deltas.
- Native messaging semantics could diverge across SDKs → cross-language golden
  fixtures and a Rust-server/Python-client smoke are blocking.
- A controller can hang or flood → bounded queues, message size limits,
  heartbeat lease, per-slot sequence checks, and process/container isolation.
- Protocol metadata may accidentally encode physical roles → gate side-switch
  and permutation tests from ADR-0010.

## Migration Plan

First activate schemas and offline codec tests, then an in-memory authoritative
state machine, loopback transport, SDKs, container smoke, and heterogeneous
tournament. Existing in-process evaluation remains the default and rollback
removes the server path without changing simulator, learner, or replay APIs.

## Open Questions

TLS/authentication, remote orchestration, FlatBuffers reflection at runtime,
multi-match scheduling, and camera payload transport remain future work.
