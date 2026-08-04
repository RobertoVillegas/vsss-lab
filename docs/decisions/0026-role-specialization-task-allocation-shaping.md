# ADR 0026: Role-specialization shaping and stable assignment

- Status: accepted
- Date: 2026-08-03
- Owners: Roberto Villegas

## Context

Run `vsss-m24-3-role-run-0002` made the "everyone is a striker" collapse measurable.
At the best semantic checkpoint support selected `strike` on 60 per cent of decisions and
coverage on 62 per cent, against attacker 72 per cent. The three strike fractions are barely
separated, so the role-conditioned actor has not specialized. Two causes stand out in the
metrics:

- The formation shaping is nearly inaudible. `role_formation` contributes about -0.0002 per
  decision while `goal_scored` contributes 0.0018 to 0.0043. Its discounted-return bound was
  deliberately kept at one per cent of a goal by ADR 0024.
- The assignment churns. Rotation telemetry shows 500 to 800 role switches per iteration. A
  one-hot role that flips every few decisions cannot give a conditioned actor a stable
  specialization target, and the formation potential hops between robots as the targets move.

Recent cooperative-MARL results agree with this reading. Amir, Bettini and Prorok
(arXiv:2506.09434) prove that whether heterogeneity pays is decided by reward
curvature: a task-allocation structure whose inner aggregator is Schur-convex and whose outer
aggregator is Schur-concave makes a division of labour strictly superior to a uniform one.
Bettini, Kortvelesy and Prorok (arXiv:2412.16244) show in 2v2 and 5v5 soccer that constraining
behavioural diversity produces complementary roles such as passing and goalkeeping. A role-based
MAPPO study (Stanford CS224R 2025) reports the same end-of-training collapse to homogeneity and
calls for regulating the role signal; the quadruped soccer pipeline (CoRL 2025,
arXiv:2505.13834) trains skills then a MAPPO/FSP strategy layer that yields dynamic role
allocation. None of these methods lets a robot own a role; ours already assigns identity-free
transient responsibilities.

## Decision

Give each non-attacking responsibility its own dense task-allocation credit and make the
assignment stable enough to learn against.

### Per-role formation potentials

Replace the single combined support/coverage potential with two potentials, each the geometric
mean of `exp(-distance / 0.25 m)` over the robots currently assigned that role, at its own
target:

- `support_formation` at the existing support target (the pass lane behind the ball);
- `coverage_formation` at the existing coverage target (the space between ball and own goal).

Each is rewarded as terminal-zeroed discounted potential shaping exactly like the combined
term. The geometric mean is the smooth Schur-concave bottleneck ADR 0024 already uses: within a
role it keeps every active robot accountable, and because the two terms are additive rather than
one pooled mean, the team can no longer trade coverage away while support improves. `0` disables
either term. The legacy `role_formation_coefficient` remains the combined potential for
backward-compatible checkpoints.

### Stable assignment

Make the role hysteresis an explicit training knob instead of a fixed pair of constants:
`role_switch_penalty` and `role_emergency_margin`, threaded through the native assigner and the
Python reference. Raising them keeps a robot on its responsibility across marginal geometry
changes, giving the conditioned actor a stable one-hot signal while the emergency override still
rotates when staying put is clearly wrong. The fresh run raises both from `0.18`/`0.20` to
`0.30`/`0.30`.

## Consequences

- The fresh run is not return-comparable with any earlier run: the reward and the assignment
  dynamics both change, so it must start with new optimizer, RNG, registry, and curriculum
  state. Checkpoints written before this ADR remain loadable only when every new knob sits at
  its neutral default.
- Role identity remains transient and identity-free; the shaping follows the assignment.
- Rollback is configuration-only: set the per-role coefficients to `0` and the hysteresis knobs
  to `0.18`/`0.20`. No checkpoint or artifact is deleted.

## Addendum (2026-08-03): citation audit and the outer-aggregator tension

### Citation audit

All four load-bearing citations were re-verified against arXiv on 2026-08-03:

- arXiv:2506.09434 exists (Amir, Bettini, Prorok; v4 2026-03-01) and its abstract matches the
  claim above. The "ICLR 2026" venue originally written in this ADR is **not confirmed**: the
  arXiv page lists no venue. The theorems, quoted from the paper itself:
  - Thm 3.1: strictly Schur-**convex** inner task aggregators plus a coordinate-wise increasing
    outer aggregator imply ΔR > 0 (heterogeneity strictly helps), unless the optimal homogeneous
    allocation is trivial.
  - Thm 3.2: Schur-**concave** inner aggregators imply ΔR = 0.
  - Thm 3.3: a strictly Schur-**convex** outer aggregator (under a constant-sum task-score
    assumption specific to that theorem) implies ΔR = 0 — so for the outer level the relationship
    reverses: a Schur-**concave** outer aggregator is what favours heterogeneity.
  - Experiments confirm "concave outer operators and convex inner operators benefit
    heterogeneous teams", and the prediction transfers to their football MARL environment.
- arXiv:2412.16244 exists and its abstract confirms role emergence improving team outcomes in
  cooperative team play.
- arXiv:2505.13834 exists and its CoRL 2025 venue is confirmed in the arXiv comments.
- arXiv:2605.12388 exists (event-triggered behavioural diversity; identity-free behaviour
  instantiated in response to events). No venue listed. It is the closest published design to
  our transient-role machinery and motivates event-triggered reassignment as a later milestone.
- A "role-based mean field MARL" (RMFQ) reference from the original search was not found and
  must not be cited. arXiv:2603.15661 exists but is off-topic (LLM agent security).

### The outer-aggregator tension, and the M24.5 correction

The per-role split in this ADR made the *outer* aggregation over support and coverage **linear**
(weighted sum). By the theorem above, a linear outer aggregator is both Schur-convex and
Schur-concave, hence heterogeneity-neutral: it removes the incentive for a division of labour
that the pooled geometric mean (a Schur-concave outer bottleneck) provided. The split remains
justified for attribution and gradient flow — the pooled term's product structure pays nothing
for improving one responsibility while the other sits near zero — but it should not have
replaced the bottleneck outright.

M24.5 therefore shapes both levels: the per-role terms keep gradient and attribution alive, and
the legacy pooled `role_formation` term is re-enabled at a moderate coefficient to restore the
Schur-concave outer bottleneck. M24.4 (per-role only) serves as the comparison arm, giving the
attribution this ADR's original bundle lacked.
