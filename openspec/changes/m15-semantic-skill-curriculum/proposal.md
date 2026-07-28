## Why

M14 established valid scenario resets, moving-ball frontier examples,
learning-progress allocation, failure replay, and immutable holdouts. Its
scenario suite is nevertheless structural rather than a complete soccer
curriculum: several skills still start from a static ball, robot placement is
mostly inherited from one base state, and curriculum success is currently
reduced to whether blue scores.

That label is wrong for defense. A robot can correctly intercept, stop, or
deflect an incoming ball without scoring during the short drill. Conversely, a
contact can be useless or harmful even though it is easy to farm. The next
large run therefore needs skill-specific physical predicates, early
termination, controlled difficulty, mirrored sides, and separate outcome
telemetry before additional reward or model complexity.

## What Changes

- Replace hand-authored position-only patches with typed parameterized scenario
  families for approach, interception, save/deflection, clearance, shot, and
  pass/receive.
- Initialize ball position and planar velocity, robot poses and headings, and
  bounded opponent pressure from deterministic seeds.
- Mirror every trainable family across colors and field sides while preserving
  canonical physical validity.
- Define versioned success, failure, and unresolved predicates per skill using
  authoritative contacts, ball trajectory, zones, possession chains, and goals.
- Terminate drills immediately after a terminal success/failure or a bounded
  unresolved timeout.
- Advance difficulty from learning progress and rolling success bands by
  varying speed, angle, distance, target width, opponent pressure, and
  observation disturbance.
- Retain full matches and routine rehearsal so atomic drills do not replace
  emergent team play.
- Keep skill outcome rewards bounded and separately attributed; require an
  anti-farming comparison before any new shaping term enters the default
  full-match reward.
- Expose scenario identity, skill phase, difficulty, terminal predicate, and
  outcome in metrics, replays, the terminal dashboard, and the web viewer.
- Add paired multi-seed skill evaluations and block a new long-run default
  until the curriculum passes validity, coverage, learning, and regression
  gates.

## PRD Milestone

M15 — semantic skill curriculum, immediately after M14 evidence-driven adaptive
training and before another high-budget training run.

## Explicit Non-goals

- No physical camera, robot, ROS, or sim-to-real work; M12 remains deferred.
- No second physics backend and no learned world model.
- No reward for raw touches, possession time, passes, or saves without a
  causal physical predicate and anti-farming evidence.
- No hard-coded permanent player identities or fixed tactical roles.
- No replacement of full 3v3 matches with isolated drills.
- No online mutation of scenario semantics or reward definitions inside an
  active checkpoint lineage.
- No promotion from a single scenario, seed, color, shaped return, or training
  success rate.
