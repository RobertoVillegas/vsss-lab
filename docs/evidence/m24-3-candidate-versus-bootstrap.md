# M24.3: the trained policy loses to its own bootstrap

Measured with `evaluate_policy_pair_scorecard` on run `vsss-m24-3-run-0004`, paired colors,
five seeds, full-length matches. The opponent is that run's own `iteration-0000`, the policy
distilled from the scripted teacher before any gradient step.

| candidate | W-D-L | goals for | goals against | beats bootstrap |
| --- | --- | --- | --- | --- |
| iteration 200 | 2-6-2 | 2 | 2 | no |
| iteration 600 | 0-8-2 | 0 | 2 | no |
| iteration 1000 | 1-6-3 | 1 | 3 | no |

Training does not merely fail to improve match play; it moves below its starting point and
keeps going. Over the same range the semantic drills improved — shot reached 0.80 and
interception 0.50 — so the drills and the match diverge.

Nothing measured this before. The adaptive-training spec requires a paired terminal bound
against the promoted baseline, the league registry holds that baseline as the run's `main`
entry, and no evaluation compared them. The heuristic scorecard that does run compares the
candidate with a scripted controller, which cannot show a policy regressing relative to
where it started.

Recorded per evaluation as `semantic_evaluation.incumbent_evaluation`. It is recorded rather
than gated: a bound nobody has watched yet should not start rejecting checkpoints on its
first run.
