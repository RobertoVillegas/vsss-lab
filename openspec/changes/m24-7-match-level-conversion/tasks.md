# M24.7 tasks

Iteration order follows agent-brief.md; one change per cycle. Terminal goal: draw <=0.70
sustained over >=8 consecutive semantic evals with win>=0.15, full 50M-step run.

## Cycle 1 — H1: early-stop truncated a still-improving run
- [x] Plan: copy m24-6-mappo-role-formation.toml -> m24-7-h1 config with new fingerprint
      (policy_id m24.7-h1-mappo-shared-v1, seed 247) and semantic_regression_patience=10000
      (effectively disable the holdout early-stop). Everything else identical to M24.6.
- [x] Provision: uv sync --group train --locked; uv run maturin develop --release. Done (Rust 1.97.1, native .so rebuilt).
- [x] Smoke run (25 it / 409,600 steps) from the H1 config. Done: vsss-m24-7-run-0001,
      clean exit (stopped:false, 25 it, 734 matches), curriculum allocation valid.
- [x] Full run 50M steps from the H1 config: vsss-m24-7-run-0002 finished (final it 3052,
      clean exit). Final W/D/L win 0.1003 / draw 0.8118 / loss 0.0880. 122 evals; gate
      passed on 9 evals, max consecutive streak = 2, SUSTAINED_8 = no.
- [x] H1 verdict: **FALSIFIED**. Full 50M budget did not convert draws to wins.

## Cycle 2 - H2: approach ladder over-demand is not the binding constraint
- [ ] Implement semantic_axis_caps in MarlConfig + record() clamp (Python-only, new fingerprint).
- [ ] Config m24-7-h2-approach-cap.toml (policy_id m24.7-h2-..., new seed).
- [ ] Provision + smoke (allocation_valid) + full run; verdict -> ADR/promote or H3.
