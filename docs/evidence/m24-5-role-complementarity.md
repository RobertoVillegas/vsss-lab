# M24.5 per-role gradient plus pooled complementarity

M24.4 (ADR 0026) split the pooled formation potential into per-role additive terms for
attribution and gradient flow. The citation audit of arXiv:2506.09434 (Thm 3.3 and the football
experiments) surfaced a tension that ADR 0026's addendum now records: a linear outer aggregation
over the two responsibilities is heterogeneity-neutral, while the pooled geometric mean is the
Schur-concave outer bottleneck that makes a division of labour strictly superior. M24.5 shapes
both levels.

## Experimental design

| arm | role_formation | support / coverage | hysteresis | purpose |
| --- | ---: | ---: | ---: | --- |
| M24.3-role | 0.10 pooled | 0 / 0 | 0.18 / 0.20 | pooled only, churned assignment |
| M24.4 | 0 | 0.15 / 0.15 | 0.30 / 0.30 | per-role only (running) |
| M24.5 | 0.10 | 0.15 / 0.15 | 0.30 / 0.30 | per-role + bottleneck |

The pair M24.4 vs M24.5 isolates the outer-aggregator effect: identical per-role terms,
identical hysteresis, the only delta is the pooled bottleneck. The pair M24.3-role vs M24.5
isolates the per-role-plus-hysteresis effect under an active bottleneck.

## Why 0.10 for the pooled term

The pooled geometric mean multiplies the per-role contributions, so when both responsibilities
are healthy it behaves like a third shaping term at roughly half the per-role scale; its
discounted-return bound is 0.10, the same one per cent of a goal ADR 0024 set and run-0002
tolerated without drowning carry (5) or goal (10). The per-role terms stay at 0.15 each so the
responsibility that drifts still receives the larger, better-attributed gradient.

## Scale measured before training

From the same 6,000 role decisions in `iteration-002000.jsonl` used for the M24.4 measurement
(mean support potential 0.262, mean coverage potential 0.166), the pooled potential is the
geometric mean `sqrt(support * coverage)`, mean 0.208. Its per-decision absolute reward at
coefficient 0.10 sits at roughly 0.0004, inside the per-role band already measured
(0.0005-0.0007) and far below carry and goal.

## Acceptance signals

- Against M24.4 at matched iterations: role strike fractions separate further (support and
  coverage fall while attacker stays highest), and the pooled potential rises, indicating the
  two responsibilities improve together rather than one at the other's expense.
- Full-match floors hold: goals per minute >= 0.2, draw rate <= 0.70.
- The heterogeneity-gain tool (see `tools/heterogeneity_gain.py`) reports a positive gap between
  the role-conditioned policy and its role-ablated counterpart; a gap near zero in both arms
  means the role machinery is dead weight regardless of shaping.

## Rollback

Configuration-only: set `role_formation_coefficient = 0.0` to reproduce M24.4 exactly, or all
four ADR 0026 knobs to their neutrals for the pre-ADR reward.
