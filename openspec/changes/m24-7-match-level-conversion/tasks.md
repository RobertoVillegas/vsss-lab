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
- [ ] Full run 50M steps (~3,052 it) from the H1 config: vsss-m24-7-run-0002 RUNNING
      (pid 191113); watcher /tmp/opencode/watch-m24-7-run1.sh (pid 191158) polls every 60s.
- [ ] Falsification: if final draw <=0.70 sustained >=8 evals -> SUCCESS (ADR + promote).
      If draw still >0.70 -> H1 falsified (early-stop was not the binding constraint);
      record in evidence.md, commit, move to H2.
