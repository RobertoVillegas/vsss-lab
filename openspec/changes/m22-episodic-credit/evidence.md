# Evidence

Recorded on 2026-07-29:

- `just doctor`: green.
- Focused MAPPO, league, role, and semantic suites: 87 passed.
- `just lint`: green across Rust, Ruff, mypy, and TypeScript.
- `just build`: Rust workspace, Python bytecode, bindings, protocols, and web build green.
- `just test`: 215 Python, 7 web, and all Rust workspace tests green. The
  loopback ZeroMQ test was run outside the filesystem sandbox because socket
  creation is intentionally denied inside it.
- Two-iteration M22 CPU smoke: 1,024 environment steps, six completed episodes,
  checkpoints and replay emitted. Canonical metrics contained independent
  `completed_episode_return`, `match_outcomes`, `terminations`, and fragment
  `return_total` fields.

## Rollback

Select the M21 configuration and launcher. M22 changes no replay, physics, or
checkpoint serialization contract; M21 checkpoints remain valid teachers for
warm initialization.
