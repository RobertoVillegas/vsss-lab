## Context

RL results need deterministic scripted baselines and artifacts for regression.
M4 consumes the M3 flattened state but does not alter native physics.

## Goals / Non-Goals

**Goals:** reusable skills, dynamic roles, identity permutation tests, 3v3
runner, JSONL replay, and headless inspection.

**Non-Goals:** optimal soccer, GUI rendering, reward shaping, policies, league,
or a definitive binary replay schema.

## Decisions

1. Controllers output normalized differential wheel commands.
2. Assignment minimizes geometric role costs every tick; physical ID is never a cost.
3. Teams canonicalize attack direction by coordinate sign.
4. Replays are M4 JSONL with header, canonical snapshots, actions, and checksums.
5. Viewer means deterministic CLI inspection in M4; visual GUI is deferred.

## Risks / Trade-offs

- Assignment can chatter → deterministic hysteresis is deferred but swaps are measured.
- JSONL is verbose → later replay milestone owns compact schema.
- Simple controllers may stalemate → gate requires reproducibility, not strength.

## Migration Plan

Pure Python additions wrap M3. Rollback removes packages and artifacts.
