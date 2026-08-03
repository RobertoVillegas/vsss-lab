# ADR 0024: Balanced role formation and phase eligibility

- Status: accepted
- Date: 2026-08-03
- Owners: Roberto Villegas

## Context

The first ADR-0023 run reached 10.65 million environment steps but did not preserve healthy role
separation. At iteration 650 support and coverage selected `strike` on 58.4 and 55.2 per cent of
decisions, while the global `stop` fraction was 18.6 per cent. The best semantic checkpoint
remained iteration 125, before useful role separation emerged.

Replay geometry exposed an aggregation defect. Relative to run 0013, coverage distance improved
from 0.784 m to 0.562 m while support distance regressed from 0.379 m to 0.548 m. The arithmetic
mean formation potential nevertheless improved from 0.214 to 0.230 because one responsibility
could compensate for the other.

The run also remained in `foundation` for 650 iterations. At iterations 175 and 200 every phase
skill floor passed and motion was healthy, but a zero-goal sample from ten short paired matches
reset the phase streak. That withheld the defensive teaching later evaluations showed was weak:
clearance 0.275 and save/deflection 0.25.

## Decision

Use the geometric mean of the active support and coverage contributions. It stays in `[0, 1]`,
is permutation invariant, equals the single active contribution on a reduced roster, and makes
either abandoned responsibility a bottleneck. Keep goals and every named reward shared by the
team. Reduce the formation coefficient from `0.20` to `0.10` for the fresh-run fingerprint.

Separate phase motion eligibility from final behavior promotion. Idle spin and excessive `STOP`
still reset the teaching-phase streak. Goal throughput remains mandatory for checkpoint
promotion, but no longer prevents the curriculum from teaching later skills. Skill floors and
all final promotion gates remain unchanged.

## Consequences and rollback

The next run must start with new optimizer, RNG, registry, and curriculum state. Its behavior is
not return-comparable with the ADR-0023 run. Roll back by restoring the arithmetic mean, setting
the coefficient to `0.20`, and passing the full behavior verdict to phase advancement. The old
run and checkpoints remain untouched.
