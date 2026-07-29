# M20: Goal-aware geometry potential

## Why

M19 can reward positive horizontal ball velocity even when the projected path
misses the goal and dies in a corner. A direct reward for standing behind the
ball would create a second loophole: an attacker could preserve the pose
instead of completing the play.

## What changes

- Replace directional shaping in the M20 profile with a goal-aperture-aware
  state potential.
- Use the transient attacker role; no robot identity owns the reward.
- Reward only discounted change in potential, so a static aligned pose cannot
  farm reward.
- Keep goals, semantic outcomes, and useful contact impulses authoritative.
- Add geometry telemetry primitives and regression tests for corner lines and
  camping behind the ball.
- Replace the zero-area corner barrier with a finite diagonal collider that
  contains oriented robot bodies as well as the ball.

## Non-goals

- No hard-coded good or bad field zones.
- No blanket penalty for visiting corners or standing ahead of the ball.
- No PPO change and no mutation of M19 runs already in progress.
