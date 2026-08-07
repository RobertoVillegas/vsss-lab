# Evidence — M24.7 run

Milestone: convert the M24.6 skill layer into full-match wins.
Gate (Terminal Goal): each semantic eval's `full_match_evaluation` draw_rate <= 0.70 AND
win_rate >= 0.15, sustained over >= 8 consecutive evals, in a full 50M-step run.
Baseline M24.6 (`vsss-m24-6-run-0002`): win 0.089 / draw 0.834 / loss 0.077, semantic
peak 0.768, phase defense->integration, early-stopped at iter 2,900.

## Run ledger

### Run 0001 — H1 smoke
- Config: experiments/configs/m24-7-h1-mappo-role-formation.toml
- 25 iterations / 409,600 steps, clean exit (stopped:false, 734 matches), curriculum
  allocation valid. (Smoke only.)

### Run 0002 — H1 full (FINISHED, falsified)
- Config: `experiments/configs/m24-7-h1-mappo-role-formation.toml` (seed 247,
  policy_id m24.7-h1-mappo-shared-v1)
- Change: `semantic_regression_patience` 12 -> 10000 (holdout early-stop disabled) so the
  gate is read at full 3,052-iteration budget instead of a mid-integration stop.
- Run dir: `/home/rob/runs/vsss-m24-7-run-0002`, 50M steps, final iteration 3052, clean exit.
- **Final full-match cumulative: win 3286 / draw 26599 / loss 2882**
  (win 0.1003 / draw 0.8118 / loss 0.0880 over 32,767 matches).
- 122 semantic evals (6 matches each); gate (win>=0.15 AND draw<=0.70 per eval)
  passed on only 9 evals, **max consecutive streak = 2** (at it-350), trailing = 0.
  `SUSTAINED_8 = no` -> **gate NOT met**.
- Trajectory: cumulative draw stayed ~0.75-0.84 throughout the full 3,052-iteration budget;
  per-eval full-match scorecards were draw-dominated (draw 0.667-1.000, win 0.000-0.167)
  in every phase, including late integration (it-2750..3050 draw 0.833-1.000).

## Hypothesis verdicts

- H1 — early-stop truncated the run: **FALSIFIED** (run 0002). Running the full 50M-step
  budget improved win modestly (0.089 -> 0.100) and draw slightly (0.834 -> 0.812) versus
  M24.6 but the gate (draw<=0.70 AND win>=0.15 over >=8 consecutive evals) was never
  reached; late-phase training did not convert draws to wins.
- H2 — approach ladder over-demand: not started.
- H3 — skill-against deficit: not started.
- H4 — role dynamics deadlock possession: not started.
