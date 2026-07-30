# M23: clean curriculum from scratch

M22 validated rollout bootstrapping but exposed a circular promotion rule:
foundation required full-match wins before the curriculum taught defense,
cooperation, rotation, and integrated play. Its per-iteration match telemetry
also lost goal identity during the configured post-goal grace period.

M23 trains a new policy from random initialization. Skill and behavior gates
advance teaching phases; match outcomes remain mandatory for final promotion.
Early stopping is deferred until integration, and match, drill, and goal
telemetry are reported independently.

## Non-goals

- Warm-starting from any M21 or M22 checkpoint.
- Introducing another reward term without controlled evidence.
- Physical-robot or camera deployment.
