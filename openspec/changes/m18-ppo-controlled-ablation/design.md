# Design

## Source-derived hypotheses

- `128_tanh`: M17 control.
- `256_tanh`: width-only comparison.
- `256_tanh_ln`: isolates normalization.
- `256_relu_ln`: Rocket League-style activation plus normalization.
- `256_relu_ln_2epoch`: tests lower PPO sample reuse at equal collected steps.
- `256_relu_ln_impulse`: adds a causal, one-shot useful-ball-impulse reward
  while reducing continuous directional shaping.

Every arm uses identical environment steps, semantic roster curriculum, seeds,
physics, opponent mode, and holdouts. Wall-clock time is reported but never
substitutes for matched experience.

## Useful impulse

The reward fires only when a controlled robot enters ball contact. It measures
the positive change in ball velocity projected toward the opponent goal; the
reward contribution is bounded with `tanh`. Persistent overlap cannot farm it.

## Selection

The report ranks promotion-gate pass count first, semantic success second,
terminal score third, unresolved rate fourth, and throughput fifth. A smoke
screen selects a candidate for a longer confirmation; it does not silently
replace the default production configuration.

## Action parser

Continuous versus lattice remains a separate paired experiment. The current
lattice actor lacks M17 role context, so including it here would confound action
space with policy architecture.
