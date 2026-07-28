## Why

M2 is usable only from Rust. M3 must expose contiguous batch stepping to Python
and standard environment contracts before policies or baselines are added.

## What Changes

- Add a PyO3/maturin mixed package and NumPy batch API.
- Add composable Python protocols for observations, actions, reset, reward, and termination.
- Add PettingZoo Parallel and Gymnasium single-robot/team adapters.
- Add API/random-environment tests and measure Python binding overhead.
- Do not add learning algorithms, shaped rewards, heuristics, self-play, or rendering.

## Capabilities

### New Capabilities

- `python-batch-bindings`: Native contiguous reset/step/snapshot/restore.
- `composable-environment-api`: RLGym-like components and standard adapters.

### Modified Capabilities

None.

## Impact

Activates `vsss-python` and `python/vsss_env`; adds locked PyO3, NumPy, maturin,
Gymnasium, and PettingZoo dependencies.
