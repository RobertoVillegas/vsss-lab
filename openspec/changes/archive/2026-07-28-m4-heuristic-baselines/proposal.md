## Why

M3 exposes standard environments but has no competent scripted opponents or
end-to-end match artifact. M4 establishes reproducible baselines before RL.

## What Changes

- Add differential-drive go-to-target, go-to-ball, and goalie controllers.
- Add state-dependent dynamic role assignment without fixed identity roles.
- Add deterministic 3v3 match runner and versioned JSONL replay.
- Add a headless replay inspector/viewer with checksum and summary.
- Do not add learning, rewards, rendering GUI, league, or external protocols.

## Capabilities

### New Capabilities

- `heuristic-team-control`: Permutation-equivariant skills and dynamic roles.
- `scripted-match-replay`: Reproducible 3v3 execution and inspectable replay.

### Modified Capabilities

None.

## Impact

Activates `python/vsss_baselines`, `python/vsss_eval`, and `tools/replay_viewer`.
