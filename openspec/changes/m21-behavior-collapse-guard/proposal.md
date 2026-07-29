# M21: Behavior collapse guard

## Why

M20 checkpoint 425 developed deterministic turn-in-place behavior despite
negative geometry shaping. The existing regularizers penalized wheel magnitude
and action changes but did not distinguish useful orientation from a sustained
remote spin. Semantic checkpoint ranking also lacked a behavioral eligibility
gate.

## What changes

- Penalize sustained turn-in-place commands only when the robot remains slow
  and outside the ball-control envelope.
- Preserve a grace period for legitimate orientation corrections.
- Measure idle-spin agent time and decision ratio during training and immutable
  semantic evaluation.
- Block phase promotion and best-checkpoint selection when deterministic idle
  spin exceeds a configured ceiling.
- Warm-start a clean optimizer and curriculum from M20 checkpoint 375.

## Non-goals

- No penalty for ordinary curved driving, turning near the ball, or brief
  orientation changes.
- No resume from M20 checkpoint 425 or reuse of its optimizer state.
