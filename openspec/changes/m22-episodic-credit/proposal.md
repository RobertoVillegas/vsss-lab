# M22: episodic credit and outcome gates

M21 exposed a conservative local optimum and a rollout-boundary error: matches
persisted between learner iterations, but GAE treated every iteration as a
terminal truncation. M22 preserves value credit across continuing rollouts,
reports completed episodes independently from fragments, and prevents semantic
checkpoint selection from treating draws as sufficient evidence.

The milestone also strengthens emergency coverage assignment and samples more
near-goal clearance and loose-ball finishing situations.

## Non-goals

- Physical-robot transfer or M12 hardware integration.
- Replacing MAPPO or changing network width.
- Reward-function search or population-based tuning.
