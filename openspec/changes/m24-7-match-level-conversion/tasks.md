# M24.7 tasks

Iteration order follows agent-brief.md; one change per cycle. Terminal goal: draw <=0.70
sustained over >=8 consecutive semantic evals with win>=0.15, full 50M-step run.

## Cycle 1 — H1: early-stop truncated a still-improving run
- [x] Plan: copy m24-6-mappo-role-formation.toml -> m24-7-h1 config with new fingerprint
      (policy_id m24.7-h1-mappo-shared-v1, seed 247) and semantic_regression_patience=10000
      (effectively disable the holdout early-stop). Everything else identical to M24.6.
- [ ] Provision: uv sync --group train --locked; uv run maturin develop --release.
- [ ] Smoke run (25 it / 409,600 steps) from the H1 config; must exit cleanly with
      allocation_valid: true.
- [ ] Full run 50M steps (~3,052 it) from the H1 config in parallel to /home/rob/runs/;
      watcher records gate at each semantic eval.
- [ ] Falsification: if final draw <=0.70 sustained >=8 evals -> SUCCESS (ADR + promote).
      If draw still >0.70 -> H1 falsified (early-stop was not the binding constraint);
      record in evidence.md, commit, move to H2.
