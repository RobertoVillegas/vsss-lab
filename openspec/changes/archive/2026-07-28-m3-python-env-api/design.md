## Context

Python training needs low-overhead bulk calls while public ecosystem adapters use
dictionaries. The dictionary conversion must remain outside the Rust hot loop.

## Goals / Non-Goals

**Goals:** NumPy batch arrays, GIL-detached Rust stepping, standard adapters,
component protocols, contract tests, and overhead evidence.

**Non-Goals:** rewards beyond zero/default events, policy code, TorchRL, rendering,
parallel Rust worlds, or stable public wheels.

## Decisions

1. Build a mixed package with maturin and a private `_native` module.
2. Native actions have shape `[world, 6, 2]`; state is contiguous `[world, 51]`.
3. Keep JSON only for construction/snapshots, not per-tick stepping.
4. PettingZoo/Gymnasium adapters wrap one native world and perform dict conversion.
5. Composition is defined with Python `Protocol` interfaces and simple defaults.

## Risks / Trade-offs

- Version-specific CPython wheels initially → automate release wheels later.
- Python adapters allocate dictionaries → benchmark native API separately.
- State layout can drift → publish indices and contract tests.

## Migration Plan

Add bindings without changing Rust backend contracts. Rollback removes the native
crate and Python adapter package.
