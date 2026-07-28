# M8 external match server evidence

## Result

M8 passes its product gate: independently compiled Rust and Python controllers
completed one authoritative match without sharing a process, simulator object,
or controller runtime. The server alone owned time, state, adjudication, replay,
and result.

## Reproduction

```bash
just external-tournament reports/m8/external-match.jsonl 20
just external-container-smoke
```

The local run produced 22 JSONL records (header, 20 ticks, result), accepted all
20 decisions from each controller, and ended `blue=2 yellow=1`. Its replay
SHA-256 was
`9b2ce7626a09bec8876f46cbacf2bce9b6a916c31530aa59124293532896fa58`.

The container run used an internal-only Compose network, no published ports,
read-only filesystems, all Linux capabilities dropped, and separate server,
Rust-controller, and Python-controller processes. It ended `blue=2 yellow=1`
with server artifact SHA-256
`0027cfd6372f7337b86715d74f689a7f09c4e1b0eff9e320d281541377dba77a`.
Compose exited zero and its containers and private networks were removed.

## Contracts exercised

- FlatBuffers v1 hello, capabilities, reset, observation, action, and result.
- ZeroMQ ROUTER/DEALER identities over loopback or a private container network.
- Explicit controller slot and blue/yellow assignment.
- Server-side monotonic sequence, tick, deadline, action-range, and lease checks.
- Deterministic safe-action fallback and auditable decision records.
- Canonical Rapier state and result checksum owned by the server.

Unit and integration tests additionally cover malformed buffers, wrong protocol
identifier, non-finite actions, stale/future/wrong-slot actions, fake-clock
deadlines, lease expiry, side switching, and Rust/Python interchange.

## Julio compatibility

Julio De La Torre's public simulator, marker textures, and thesis were reviewed
for dimensions, identity semantics, observations, wheel commands, and controller
assumptions. The public project does not expose a standalone M8 controller that
implements this versioned wire contract, so a named Roberto-versus-Julio match
cannot yet be executed honestly. The compatibility path is the Python SDK:
adapt Julio's policy callback to three wheel-command pairs while the SDK owns
transport and assignment. The generic heterogeneous tournament is the blocking
M8 gate; the named exhibition remains conditional on receipt of that controller
or checkpoint.

## Performance, limitations, and rollback

The 20-control-tick match completes in roughly its configured 0.4 seconds after
the one-time build. M8 intentionally supports one match and two trusted local
controllers. It does not provide public-network security, multi-tenant
sandboxing, camera transport, ROS, or distributed scheduling. Wall-clock fields
can differ between runs; simulation ordering and adjudication use tick and
sequence, while canonical state remains the replay authority.

Rollback is non-invasive: league evaluation keeps its in-process path, and the
external server, SDK executables, commands, and competition Compose profile can
be removed without changing the canonical simulator, policy, or replay APIs.
