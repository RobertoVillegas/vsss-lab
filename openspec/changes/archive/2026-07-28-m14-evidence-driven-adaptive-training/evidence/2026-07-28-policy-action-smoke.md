# Policy memory and action-parser smoke — 2026-07-28

These paired CUDA smokes prove executable ablation lineages, not superiority.
Each arm used three learner seeds, three real MAPPO updates per seed, 15%
entity/ball dropout plus Gaussian observation noise where applicable, and
paired-color terminal evaluation.

## MLP versus GRU

Command:

```text
just m14-policy-ablation experiments/reports/m14-policy-smoke.json cuda 3
```

| architecture | parameters | W-D-L | score | mean return | compute |
| --- | ---: | --- | ---: | ---: | ---: |
| MLP | 53,508 | 0-6-0 | 0.500 | -0.1678 | 1.894 s |
| GRU | 152,580 | 0-6-0 | 0.500 | -0.1692 | 1.154 s |

The GRU state is isolated by world/agent, reset only for completed worlds,
stored with rollout transitions for PPO, and its architecture is checkpointed.
The smoke resolves no terminal advantage, so MLP remains default.

## Continuous versus symmetric lattice

Command:

```text
just m14-action-ablation experiments/reports/m14-action-smoke.json cuda 3
```

Both arms used `action_repeat = 4`.

| parser | W-D-L | score | mean return | compute |
| --- | --- | ---: | ---: | ---: |
| continuous wheels | 0-6-0 | 0.500 | -0.1663 | 2.380 s |
| 9-action symmetric lattice | 0-6-0 | 0.500 | -0.1751 | 1.714 s |

The lattice uses a categorical PPO likelihood rather than pretending quantized
continuous actions came from a Gaussian. It starts a distinct checkpoint
fingerprint. No terminal advantage was resolved, so continuous wheels remain
default.
