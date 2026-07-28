# Accelerator feasibility decision — 2026-07-28

## Local profile

Command:

```text
just profile-m14 100 reports/m14/profile-local.json
```

RTX 3070, 64 worlds, CUDA actor:

- 25,600 environment frames in 2.299 seconds;
- 11,136 frames/s in the isolated rollout profile;
- within the isolated rollout, Rust/Rapier physics + reward + reset: 77.36%;
- CPU observation construction: 11.94%;
- CUDA inference: 6.66%;
- host-to-device: 3.56%;
- device-to-host: 0.47%;
- a separate eight-world/eight-step probe measured 0.221 seconds for rollout
  assembly and 0.338 seconds for one PPO update.

The separate shared-actor microbenchmark measured 42.86 µs for observation
construction and 75.16 µs for one three-agent actor call. These measurements
are local evidence, not cross-machine claims.

## Decision

- **Keep:** Rust/Rapier remains authoritative.
- **Adopt:** phase-level profiling as a repeatable gate.
- **Adopt:** an explicit trace comparison over planar states and adjudication
  events; alternate backends need exact event parity, bounded geometric error,
  and at least 1.5× end-to-end throughput.
- **Reject now:** a Torch CUDA kinematic primitive as production physics. It
  does not implement Rapier contact manifolds, chamfered walls, goal geometry,
  damping, or adjudication and therefore cannot pass trace parity.
- **Defer:** Warp/Newton or custom Rust/CUDA contact implementation. Physics is
  large enough to justify a future spike, but M14 has no parity-passing
  candidate, and a second unverified engine would corrupt training evidence.
- **Next optimization:** batch observation construction and reduce host object
  creation before paying the ownership cost of another physics backend.

## Falsifiable gate

A candidate that is ten times faster but disagrees on one contact/goal event
must be rejected. A parity-passing candidate below 1.5× end-to-end speedup must
also be rejected.

## Executed CUDA spike

`just m14-accelerator-spike experiments/reports/m14-accelerator-spike.json
cuda 64 256` exercised 64 worlds, including 16 forced goal-entry traces and 16
robot-ball contact traces. The device-resident microkernels measured roughly
2.31M observation, 0.61M reward, and 1.33M reset world-operations/s.

The deliberately minimal Torch CUDA kinematic prototype achieved 29,443
frames/s versus 127,986 frames/s for authoritative Rust/Rapier (`0.23×`), had
3,968 goal-event mismatches, maximum planar error `2.711 m`, and mean error
`0.0868 m`. It fails both correctness and speed. The rejection is therefore
measured rather than hypothetical, and the prototype is not selectable by
training.

The production vector bridge now reuses CPU action storage across decisions,
and the environment reuses normalized-action, delta, native-action, and event
buffers. These changes remove recurrent host allocations without changing the
authoritative backend. The same 100-step profile produced 11,247 frames/s after
the change versus 11,136 frames/s in the recorded pre-change profile (`+1.0%`);
the optimization is retained as non-regressing, not presented as a major
throughput breakthrough.

The matched CPU spike measured 133,462 authoritative Rapier frames/s and 91,659
prototype frames/s; the CUDA spike measured 127,986 and 29,443 respectively.
These are isolated physics/prototype rates, while the full CUDA rollout profile
measured 11,247 environment frames/s including observation, copies, inference,
and reward/reset. All executed M14 smoke studies record wall-clock compute in
their machine-readable arms or fidelities. No candidate cleared terminal
promotion, so there is no promoted-study compute claim to omit or inflate.
