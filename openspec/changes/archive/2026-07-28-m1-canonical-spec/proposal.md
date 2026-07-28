## Why

M0 provides executable scaffolding but no domain vocabulary. M1 must establish a
backend-independent, versioned canonical contract before physics, bindings, or
protocol implementations can safely depend on it.

## What Changes

- Add explicit SI unit types and canonical field/robot/ball geometry.
- Add versioned entities, actions, event flags, match configuration, and
  semantic validation.
- Add canonical blue/yellow reflection as a tested involution.
- Add static contract reflection and strict JSON serialization.
- Add golden fixtures and rules/symmetry contract tests.
- Add ADRs for unit and coordinate-system decisions.
- Replace the M0-only boundary in `AGENTS.md` with milestone-neutral rules.
- Do not add physics, RL, Python bindings, FlatBuffers, ROS/Gazebo, networking,
  replay execution, or vision.

## Capabilities

### New Capabilities

- `canonical-domain-model`: Explicit units, geometry, entities, actions, events,
  and match configuration with semantic validation.
- `canonical-interchange`: Versioned reflection, strict JSON serialization,
  field symmetry, and golden compatibility fixtures.

### Modified Capabilities

None.

## Impact

The public API of `crates/vsss-spec` changes from an M0 placeholder into the M1
contract. `serde` and `serde_json` become locked Rust dependencies. Downstream
milestones must consume these types instead of introducing competing domain
models.
