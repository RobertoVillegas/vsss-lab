# ADR 0023: Dynamic-role formation shaping

- Status: accepted
- Date: 2026-08-02
- Owners: Roberto Villegas

## Context

Run `vsss-m24-3-run-0013` made the credit-assignment defect measurable. At iteration 2000 the
attacker selected `strike` on 87.5 per cent of replay decisions, but support selected it on 55.1
per cent and coverage on 48.2 per cent. The three actors receive the same team reward. A goal,
useful contact, and ball carry therefore reinforce every action the team happened to take, even
when support or coverage abandoned its responsibility.

Paying only the scorer would make the competition for the ball stronger. Assigning rewards to
robot IDs would violate ADR 0008 and would turn physical identity into a tactical role. The
terminal outcome must remain shared and the responsibility must remain transient.

## Decision

Keep goals, concessions, semantic outcomes, carry, and match results team-level. Add a bounded
formation potential over the robots currently assigned to the active `support` and `coverage`
responsibilities:

```text
formation reward = c · (gamma · Phi(next) - Phi(current))
```

`Phi` is the mean of `exp(-distance / 0.25 m)` for the active support and coverage robots at
their existing dynamic-role targets. Inactive roster slots do not contribute. The potential is
zero on a terminal transition.

The assignment remains the existing identity-free minimum-cost permutation with hysteresis.
The reward never refers to a marker, robot ID, or array slot as an owner of a role. When robots
rotate, the responsibility and its shaping move together.

Record primitive choice by dynamic role in every training metric. Aggregate action fractions
are insufficient because a healthy attacker-heavy `strike` rate and a coverage collapse can
have the same total.

## Consequences

- Winning remains a shared objective; no robot is rewarded for stealing a final touch.
- A static formation cannot farm the term. Terminal zeroing preserves the accepted potential
  boundary from ADR 0015.
- The term is a coordination prior, not a hard action mask: coverage may still challenge and an
  emergency assignment may rotate immediately.
- Returns and behaviour are not comparable with runs before this ADR. Run 0013 remains the
  rollback baseline and the coefficient can be set to zero for a configuration-only rollback.

