# Evidence

## The gate was unreachable, proved analytically

`go_to_target` (`python/vsss_baselines/controllers.py`) spends at most
`TURN_AUTHORITY = 0.08` of the normalized wheel limit on turning. The detector
measured `|right − left| / 2`, which equals `|turn| ≤ 0.08` for those wheels, and
the configured threshold is `idle_spin_turn_threshold = 0.13`. Clipping and the
parametric intensity scale can only shrink the differential further, so no state
and no token could raise the flag under `primitive` or `parametric_primitive`.

The pathology stayed reachable: at a heading error of 90 degrees or more,
`forward = max(0, cos(error)) = 0` while `turn = ±0.08`, which is a robot rotating
in place without driving. That satisfies the drive, speed, and ball-distance
conditions and failed only the turn condition.

The existing detector test passed because it injects synthetic wheels of
`(−0.8, 0.8)` — unreachable through either skill parser — with a threshold of
`0.25` rather than the configured value.

## Measured on the M24.2 run in flight

The first paired evaluation of `vsss-m24-2-run-0001` (iteration 25, 280 attempts)
reported `idle_spin_ratio` of exactly `0.0` and `behavior_gate_passed` true, with
`semantic_max_idle_spin_ratio = 0.08`. Nothing about that zero was informative.

Replay intent from the same iteration, over 9000 samples: skills split
`navigate 6074 / strike 2926 / stop 0`, intensity minimum `0.931`, median `0.988`,
maximum `1.000`, with no sample below `0.50`, and 853 distinct headings at
0.1-degree resolution.

## After the change, same configuration

An end-to-end parametric smoke run reports `idle_spin_ratio = 0.0885` against the
`0.08` ceiling, so `behavior_gate_passed` is false: the gate discriminates for the
first time. The run record now carries `resolved_drills_per_second = 6.05` over
`36.5` seconds, `mean_controlled_touches = 0.75`, and
`physical_validity_rate = 1.0`.

## Gates

- 239 Python tests pass, including a detector test that asserts the hardest
  skill-parser turn-in-place raises the flag with the configured threshold and does
  not raise it under the previous unnormalized comparison, and an environment test
  asserting a learned opponent's wheels equal one parse of its token.
- `mise run lint` green across `cargo fmt`, `cargo clippy`, Ruff over 421 files,
  mypy over 103 sources, and the web typecheck.
- `openspec validate --strict` accepts this change.
