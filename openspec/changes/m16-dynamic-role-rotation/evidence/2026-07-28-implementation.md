# M16 implementation evidence — 2026-07-28

## Behavioral contracts

- `tests/test_dynamic_roles.py` proves unique identity-free roles, goalkeeper
  responsibility transfer, marginal-change hysteresis and emergency rotation.
- Semantic scenario/predicate tests cover mirrored `rotation_recovery` drills.
- Legacy actors explicitly ignore the five new role context values; `role_mlp`
  consumes all nine context values and checkpoints declare the architecture.

## Training smoke

One CPU iteration with two worlds and the M16 configuration completed:

- 512 environment steps;
- checkpoint and replay written;
- 56 holdouts evaluated: seven families, two colors and four difficulty bands;
- role switches and uncovered ratio present in metric telemetry.

Command:

```text
uv run --group train python -m vsss_league.cli run \
  --config experiments/configs/m16-mappo-rotational.toml \
  --match-config tests/golden/m1_match_config.json \
  --match-state tests/golden/m1_match_state.json \
  --iterations 1 --capture-every 1 --capture-seconds 1 \
  --checkpoint-every 1 --device cpu --num-envs 2 \
  --semantic-eval-every 1 --semantic-eval-seeds 1
```

## Gates

- `just test`: 187 Python tests, seven web tests and all Rust workspace tests passed.
- `just lint`: rustfmt, Clippy, Ruff, mypy and TypeScript passed.
- `just build`: protocol, Rust, Python compileall and Vite production build passed.
- Focused role/league/replay suite: 87 tests passed.

## Timeout-reset correction

The first long M16 run exposed that the vector environment returned one NumPy
array as both `done` and `terminated`. Marking a semantic timeout as truncated
therefore also cancelled its reset, and the cached timeout was counted on every
remaining rollout step. The environment now returns independent arrays.

A regression test runs two semantic worlds for 256 decisions with the skill
horizon at 250 and proves that each world can emit at most one unresolved result
before reset. A two-iteration CLI smoke completed 1,024 steps without repeated
cached outcomes. Semantic evaluation artifacts now retain both the compatible
coarse bands and exact difficulty levels such as `0.10`, `0.25`, `0.40`, and
`0.65`.

Post-correction gates:

- `just test`: 188 Python tests plus Rust and web suites passed.
- `just lint`: rustfmt, Clippy, Ruff, mypy, and TypeScript passed.
- `just build`: protocol, Rust, Python, and Vite production build passed.
