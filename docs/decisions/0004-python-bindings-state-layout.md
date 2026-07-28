# ADR-0004: Python bindings and flattened state layout

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

Python policies need contiguous batch data while ecosystem adapters require
object- and dictionary-oriented APIs.

## Decision

Use a private PyO3 0.29 extension built by maturin as `vsss_env._native`.
Actions are contiguous float32 `[world, 6, 2]`. State rows contain 77 float32
values in canonical field order: five match values, five ball values, eleven
values for each of six robots, then event flags. Rust stepping detaches from the
Python interpreter. Public PettingZoo/Gymnasium adapters perform conversions
outside the native hot loop.

## Consequences

Batch stepping has one Python call and contiguous output, but schema integers are
represented as float32 in the tensor view. JSON remains available for lifecycle
snapshots, not tick traffic.

## Alternatives considered

Returning nested dictionaries allocates per entity and couples the hot loop to
Python. A structured NumPy dtype complicates common tensor conversion. DLPack is
deferred until TorchRL integration.

## Validation and rollback

Shape, contiguity, replay, random environment, and overhead tests gate M3.
Rollback removes the extension and adapters without changing Rust physics.
