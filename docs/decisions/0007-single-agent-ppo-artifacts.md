# ADR-0007: Single-agent PPO and training artifacts

- Status: accepted
- Date: 2026-07-28
- Owners: Roberto Villegas

## Context

M5 needs reproducible learned skills before the platform introduces shared
multi-agent policies and self-play. Training state and evidence cross process
boundaries and therefore require an explicit contract.

## Decision

Use the native simulator for a robot-centric go-to-target task and a CPU-first
Gaussian actor/critic trained with clipped PPO and GAE. Rollouts use TensorDict,
while configs use versioned TOML, metrics use append-only versioned JSONL, and
trusted local checkpoints use versioned PyTorch dictionaries. A checkpoint
records model and optimizer state, update/frame/stage progress, config
fingerprint, and Python, NumPy, and PyTorch RNG state.

Curriculum stages C0–C5 broaden reset distributions monotonically. Promotion and
the final 95% gate use fixed seeds and deterministic mean actions. Checkpoints
live outside the repository under `/home/rob/checkpoints` by default.

## Consequences

Experiments are resumable and auditable without coupling PyTorch to the Rust
core. PyTorch checkpoints remain library-specific and must be treated as trusted
local inputs. Portable inference export and multi-agent TensorDict layouts are
deferred.

## Alternatives considered

Stable-Baselines3 would reduce implementation work but conflicts with the chosen
TorchRL direction. Starting directly with MAPPO would combine environment,
credit-assignment, and infrastructure risks before a single-agent gate exists.

## Validation and rollback

Task contracts, deterministic smoke training, checkpoint equivalence, metrics
schema, curriculum promotion, and the 95% evaluator gate validate M5. Rollback
removes the isolated training package and dependency group without changing the
simulator.
