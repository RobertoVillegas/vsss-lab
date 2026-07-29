# M15 operational probes

Date: 2026-07-28

The active M14 run was inspected at checkpoint 700. On 60 paired semantic
holdouts it achieved 100% approach, 60% shot, and 0% interception,
save/deflection, clearance, and pass/receive. Its latest 100-update mean progress
was -0.413, so continuing it to 50 million steps was rejected.

## Matched M15 probes

Clean and warm-start arms each ran 10 CUDA iterations, 64 worlds, 256 rollout
steps, 163,840 environment steps, and immutable three-seed paired-color
evaluation at iterations 5 and 10.

| Arm | mean progress | matches | final holdout success | final exploration |
| --- | ---: | ---: | ---: | --- |
| M14 policy warm start | -0.171 | 134 | 8/36 (22.2%) | log std -1.05 / -1.14 |
| clean teacher start | +0.110 | 233 | 6/36 (16.7%) | log std -0.50 / -0.50 |

Both arms retained easy approach and shot behavior during the bounded probe,
while difficult defensive and passing holdouts remained unsolved. The clean arm
was selected as the default because it produced positive progress, more resolved
matches, and substantially more exploration without relying on inherited M14
bias. Warm start remains available only for explicit ablation.

Easy training cells already reported approximately 91% interception, 88%
save/deflection, 46% clearance, 100% approach and shot, and 0% pass/receive by
iteration 10. This is evidence that the curriculum begins below the immutable
0.65-difficulty holdouts and exposes pass/receive as the immediate frontier.

The next long run is therefore measurable rather than blindly authorized:
periodic holdouts write an append-only evaluation history and balanced
`best-semantic.json`; the last checkpoint is not assumed to be the best.
