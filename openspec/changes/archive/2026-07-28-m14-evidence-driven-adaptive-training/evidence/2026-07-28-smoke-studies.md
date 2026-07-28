# M14 executable smoke studies — 2026-07-28

These are integration-scale studies, not promotion evidence. Their budgets are
deliberately too small to establish policy quality.

## Fixed-reward curriculum ablation

Command:

```text
just m14-curriculum-ablation \
  experiments/reports/m14-curriculum-smoke.json cuda 3
```

Both arms used seeds 600014–600016, three real MAPPO updates per seed, exact
Rapier rollouts, and paired-color terminal evaluation:

| arm | W-D-L | terminal score | mean return | action saturation | compute |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform | 0-6-0 | 0.500 | -0.1642 | 0.00463 | 1.825 s |
| adaptive | 0-6-0 | 0.500 | +1.9017 | 0.00521 | 1.010 s |

Decision: **no terminal advantage**. The shaped-return increase cannot promote
the curriculum and is evidence for retaining the terminal gate.

## Multi-fidelity bounded search

Command:

```text
just m14-study 2 experiments/reports/m14-study-smoke cuda
```

Two trials completed smoke/screen/confirm with respectively 1/3/5 declared
seeds. SQLite retained the study; JSONL retained six fidelity lineage records.
Both confirmation terminal scores were 0.500 (all draws). Trial 0 had lower
coordination failure (0.00260); trial 1 used slightly less confirmation compute
(1.889 s versus 1.919 s). Both remain Pareto candidates only because the smoke
budget did not resolve terminal skill.

Decision: the runner, persistence, seed escalation, objectives, and pruning
contracts pass. Neither reward vector is promotable, and no checkpoint lineage
is changed.
