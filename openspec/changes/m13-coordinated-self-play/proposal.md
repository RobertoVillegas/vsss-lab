## Why

The first 20 million-step MAPPO baseline shows genuine learning and physically
valid contacts, but frequent teammate congestion and weak defensive coverage.
The existing shared reward makes ball pursuit dense while defense is learned
mostly from sparse conceded goals.

## What Changes

- Add a small continuous teammate-congestion cost.
- Add potential-based defensive coverage progress activated by ball threat.
- Preserve permutation-safe dynamic roles and shared-policy symmetry.
- Start a fresh checkpoint lineage; do not resume the old reward fingerprint.
- Compare congestion, defense, goals, progress, and throughput against run 0001.

## PRD Milestone

M13 — coordinated self-play follow-up after the M12 physical/training diagnostic.
