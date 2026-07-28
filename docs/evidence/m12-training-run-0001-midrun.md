# M12 training run 0001 mid-run diagnostic

Date: 2026-07-28

The active 20 million environment-step baseline was inspected near iteration
235, approximately 3.85 million steps (19%). CUDA remained selected. Mean return
improved from -0.968 over iterations 1–25 to +0.152 over iterations 211–235;
mean progress moved from +0.105 to +0.128 and entropy declined from 1.778 to
1.121.

Captured replays confirm frequent robot clustering. The minimum robot-center
separation remains approximately 0.075 m, matching the robot body width, so the
observed overlap is sustained physical contact rather than collider penetration
or a viewer-coordinate defect. Congestion varies substantially between captures
and has not declined monotonically.

The current shared reward emphasizes ball progress, closest-agent approach, goal
events, and action smoothness. It has no explicit defensive coverage or teammate
congestion term. Sparse goals and a permutation-safe shared policy therefore
make early all-agent pursuit plausible. The run remains valid as a baseline and
continues unchanged; modifying rewards mid-run would invalidate its checkpoint
lineage. A later paired ablation should test congestion and defensive-coverage
terms with identical seeds and budgets.
