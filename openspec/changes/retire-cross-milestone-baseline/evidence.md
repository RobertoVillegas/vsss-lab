# Evidence

## The tooling could not run

`tools/m15_candidate_probe.py` loaded `vsss-training-run-0004/checkpoints/
iteration-000425.pt` and `vsss-training-run-0003/checkpoints/iteration-001450.pt`
by absolute path. Neither run directory exists under `/home/rob/runs`, which
holds only `vsss-m18` through `vsss-m24-2` runs and `vsss-semantic-run-0003`
through `-0006`. The probe raised on checkpoint load before reaching any
comparison, and `justfile` was its only reference.

## The comparison was silently wrong before it was impossible

`evaluate_policy_pair_scorecard` constructed `MarlMatchEnv` without an action
parser, so both teams ran the default continuous parser. For a parametric
candidate that meant width-4 transport tokens against a width-2 environment.
The probe also cast its actor to `SharedActor`, hiding the mismatch from type
checking.

## No live gate referenced the retired baseline

`experiments/configs/m24-2-mappo-parametric.toml` gates promotion on
`semantic_promotion_floors`, `semantic_max_idle_spin_ratio = 0.08`,
`semantic_min_match_win_rate = 0.20`, and `semantic_max_match_draw_rate = 0.70`.
`python/vsss_league/cli.py` computes eligibility from those three checks and
scores full matches against the heuristic only.

## Recorded outcome of the retired probe

`docs/evidence/m15-bounded-probes.md` records `semantic-shared@50` at 0 of 60
holdouts and 0 wins, 8 draws, 2 losses against frozen `directional-shared@425`,
with decision `reject_large_run`. Both M15 changes are archived.

## Gates after this change

- `tests/test_league.py` scores a one-lineage parametric pair and asserts a
  mixed-parser pair is rejected;
- `tests/test_marl.py` asserts the environment names the offending side and the
  width its parser expects;
- 237 Python tests pass, `mise run lint` is green across Rust, Ruff, mypy, and
  the web typecheck, and `openspec validate --strict` accepts this change.
