## Decisions

1. Count one control decision in one world as one environment step.
2. Keep match count, matches/s, PPO updates, and frames/s as secondary metrics.
3. Stop only after a complete rollout so PPO never optimizes a partial batch.
4. Checkpoint the policy on the rollout that reaches or exceeds the target.
5. Use 20M steps as the first production learning-curve budget.
