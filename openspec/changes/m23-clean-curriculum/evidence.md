# Evidence

Recorded on 2026-07-30:

- `just doctor`: green.
- Focused MAPPO, league, and semantic suites: 84 passed.
- `just lint`: Rust formatting/Clippy, Ruff, mypy, and TypeScript green.
- `just build`: bindings, protocols, Rust workspace, Python, and web green.
- `just test`: 216 Python tests, 7 web tests, and all Rust workspace tests green.
- Two-iteration CPU smoke with one semantic evaluation per checkpoint:
  1,024 environment steps, six completed episodes, two checkpoints, and one
  replay. No `initialization.json` was created, confirming random initialization.
- Smoke metrics independently recorded `episode_kinds`, `match_outcomes`,
  `goal_events`, `total_full_matches`, `total_match_outcomes`, and
  `total_goal_events`.
- Both smoke evaluations remained in foundation because their skill gates were
  not met; a false match gate was recorded but did not terminate or promote the
  phase.

## Rollback

Use the M22 launcher and an explicit M21 checkpoint. M23 changes no checkpoint,
replay, physics-state, or environment-observation schema.
