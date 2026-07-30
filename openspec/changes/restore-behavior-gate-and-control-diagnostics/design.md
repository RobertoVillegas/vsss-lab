# Design

## Why the gate was unreachable

`go_to_target` returns `clip(forward ∓ turn)` with `turn = TURN_AUTHORITY ·
clamp(error / (π/2))` and `TURN_AUTHORITY = 0.08`. The detector measures
`|right − left| / 2`, which for those wheels equals `|turn| ≤ 0.08`, and clipping
only shrinks it. The parametric parser scales the result by an intensity in
`(0, 1]`, shrinking it further. The configured threshold is `0.13`. It was
authored for `m21-mappo-anti-spin.toml`, which sets `action_parser =
"continuous"`, where the policy writes wheels directly and the differential
reaches `1.0`. The value was copied into the skill-parser configurations, where
the same number means something unattainable.

## Decision: normalize by attainable authority, not by parser branch

The threshold is reinterpreted as a fraction of the turn authority the parser can
request. `parser_turn_authority` returns `1.0` for wheel-space parsers and
`TURN_AUTHORITY` for the two skill parsers, and the detector divides by it. The
reported turn intensity becomes the same dimensionless quantity for every action
space, so the configured threshold, the reward coefficient, and the promotion
ceiling all keep the meaning they had under `continuous`.

The alternative — a second threshold per parser — was rejected because it leaves
two numbers that must be retuned together and does not make the reported telemetry
comparable across milestones.

The behavior gate's environment previously hardcoded the thresholds, so the
configured values could not reach it. It now receives them from the run.

## Decision: one decision, one opponent parse

The single environment re-parsed the opponent token inside the `action_repeat`
loop, giving a learned opponent a closed loop at the physics rate while the
learner ran open-loop at the decision rate. The parse moves out of the loop, next
to the learner's. The scripted controller keeps re-planning per substep, which is
what the vector environment already does for it, because a scripted baseline is
not a policy being evaluated.

## Consequences for a run

The idle-spin penalty was contributing exactly zero under both skill parsers and
becomes live, bounded by `idle_spin_coefficient` per step. The behavior ceiling
becomes falsifiable, and a policy that has not yet learned to drive will fail it —
which is the gate working, not a regression. Because the same flag feeds phase
eligibility, phase advancement is now guarded as originally specified. These take
effect for the next run; a run already in flight holds the previous code.

Reported action statistics will shift once absent robots stop contributing, so
values are not directly comparable with earlier runs of the same configuration.
The heading statistic loses its spurious right angles for the same reason.

## Validation, compatibility, rollback

Validation: a detector test drives the hardest turn-in-place a skill parser can
produce and asserts the flag is raised with the configured threshold and is
*not* raised under the previous unnormalized comparison; an environment test
asserts a learned opponent's wheels equal a single parse of its token against the
pre-step state; the existing wheel-space detector test is unchanged, pinning
`continuous` behavior. Plus the full suite, `mise run lint`, OpenSpec strict
validation, and an end-to-end parametric smoke run whose behavior gate now
reports a non-zero ratio.

Compatibility: no configuration key, checkpoint format, replay schema, or metrics
key is removed; four throughput fields are added to the semantic evaluation
record. Checkpoints remain loadable.

Rollback: `git revert`. The gate returns to being vacuously true, which is the
state this change exists to end.
