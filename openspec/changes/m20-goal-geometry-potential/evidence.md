# Evidence

## M19 replay 0200 diagnosis

Source:
`/home/rob/runs/vsss-m19-run-0001/replays/iteration-000200.jsonl`

- At simulation time `57.74 s`, yellow R4 had center
  `(-0.712519, 0.612477)` and orientation `-3.141436 rad`.
- Its complete square support penetrated the nominal chamfer by approximately
  `0.069996 m`; this was authoritative physics state, not viewer-only overlap.
- The replay contained unattended attacking windows at `24.46–27.64 s` and
  `28.40–39.52 s`, using `|ball.x| > 0.48 m` and nearest enabled robot farther
  than `0.18 m`. The latter persisted `11.12 s` and reached `0.232876 m`.

## Regression checks

- The ball still deflects from the 70 mm clipped corner.
- A robot driven diagonally into the same corner remains behind the inner face
  when its rotated square support is included.
- An unchanged aligned-behind-ball state produces negative geometry shaping.
- Advancing both attacker and ball along the same valid line produces positive
  geometry shaping.

## Gates

- Rust workspace tests: passed, including 14 Rapier correctness cases.
- Python tests: 209 passed.
- Web tests: 7 passed.
- Lint, type checking, build, and protocol compatibility: passed.
- CUDA rollout smoke: 2 iterations, 4,096 environment steps, 25 matches;
  checkpoints and replays completed with per-team geometry telemetry.
