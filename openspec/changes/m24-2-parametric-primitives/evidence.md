# Evidence

## Contract

- M24.2 uses `parametric_primitive`; legacy M24 retains `primitive`.
- The transport token contains skill, direction vector, and intensity.
- The policy ID and run-directory prefix are distinct from discrete M24.

## Tests

- 66 focused Python tests pass, including two new continuous-heading and
  intensity contract tests.
- Python mypy passes over the repository's configured 103-source scope.
- Ruff check and formatting pass over all 411 Python files.
- All nine Replay Studio component tests, TypeScript typecheck, and production
  Vite build pass.
- Rust tests reached the loopback ZeroMQ integration test; all preceding tests
  passed and that test was blocked by the managed sandbox's socket policy.

## Smoke runs

Two end-to-end CPU runs completed rollout, joint MAPPO optimization,
checkpointing, metrics, and replay capture:

- 2 worlds / 512 environment steps with the minimal bootstrap;
- 2 worlds / 512 environment steps with 128 bootstrap samples and three
  distillation epochs.

The second run produced positive progress (`+0.421`), two completed matches,
and exact replay headings including `+42.7°`, `+25.1°`, and `+44.4°`.
Recorded mean requested intensity was `95.9%`. The replay identifies its parser
as `parametric_primitive` and includes angle, intensity, phase, target, and
exit vector per learned actor.

The full aggregate `just test` retry could not resolve a PyPI dependency after
the managed execution environment rejected network escalation for quota, not
because of a test assertion or implementation failure.
