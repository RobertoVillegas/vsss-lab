## Context

M0 intentionally exposes no domain model. Physics and protocol work will begin
after M1, so the canonical crate must first define a stable vocabulary without
depending on any backend. The contract must be readable by Rust, serializable
for fixtures and run metadata, and mechanically inspectable.

## Goals / Non-Goals

**Goals:**

- Make every physical scalar's SI unit explicit in the type system.
- Define the complete M1 state, action, event, geometry, and config model.
- Provide deterministic validation, reflection, JSON interchange, and symmetry.
- Freeze representative JSON shapes through golden fixtures.

**Non-Goals:**

- Physics behavior, action adaptation, replay execution, or reward semantics.
- A definitive cross-process wire protocol; FlatBuffers belongs to its own
  subsequent change.
- Python, ROS/Gazebo, PyTorch, networking, or visualization integration.

## Decisions

1. Use transparent `f32` newtypes for SI quantities. This prevents accidental
   unit mixing while retaining compact, FFI-friendly representation. A generic
   dimensional-analysis library was rejected because it expands the public API
   and compile surface before profiling demonstrates need.
2. Use plain structs/enums with `serde` derives and strict JSON objects. JSON is
   inspectable and suitable for golden fixtures and run configuration; it is not
   declared the performance wire format. Unknown fields are rejected so schema
   drift is visible.
3. Keep event flags as a documented `u32` newtype. This has a stable serialized
   representation and avoids coupling the contract to a macro crate.
4. Define yellow-team canonicalization as a 180-degree field rotation plus
   blue/yellow relabeling. Robot identities remain physical identifiers. Applying
   the transform twice must recover the original normalized state.
5. Implement reflection as static descriptors owned by `vsss-spec`. Rust runtime
   reflection and generated schemas were rejected because M1 needs a small,
   deterministic catalog, not code generation.
6. Semantic validation returns the first stable field-path error. Serialization
   checks shape; validation checks finiteness, positivity, ranges, uniqueness,
   schema version, and timing compatibility.

These choices are formalized in ADR-0001 and ADR-0002.

## Risks / Trade-offs

- [Floating-point reflection can preserve `-0.0`] → comparisons and fixtures use
  canonicalized normalized values where required.
- [Strict JSON rejects forward-added fields] → schema versions are explicit and
  migrations will be introduced before evolving fixtures.
- [Manual reflection can drift] → contract tests compare the catalog with public
  serialized roots and require updates alongside type changes.
- [Configuration is broader than the first physics backend] → validation remains
  backend-neutral and backend-specific tuning is deliberately minimal.

## Migration Plan

Replace the M0 placeholder with M1 modules, add locked dependencies, then land
ADRs, fixtures, and contract tests together. Downstream work starts only after
all gates pass. Rollback is a single revert because no released consumer exists;
the M0 placeholder can then be restored.

## Open Questions

- Binary wire representation and compatibility rules remain for the dedicated
  protocol change.
- Exact competition dimensions remain configurable until calibrated against the
  chosen ruleset and legacy simulator.
