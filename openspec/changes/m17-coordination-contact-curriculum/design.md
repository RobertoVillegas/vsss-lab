# Design

## Roster ladder

Semantic tasks use the smallest roster that expresses the skill:

| Skill | Routine | Frontier |
| --- | --- | --- |
| approach, shot | 1v0 | 1v1 |
| interception, save | 1v1 | 2v1 |
| clearance | 1v1 | 2v2 |
| pass/receive | 2v1 | 2v2 |
| rotation/recovery | 3v2 | 3v3 |

Disabled canonical robots remain observable with `enabled=0`, receive masked
actions, and do not participate in distances, congestion, or role costs.
Routine rehearsal remains sampleable after harder rosters unlock.

## Contextual contact

Contact accounting uses conservative body-distance thresholds and consecutive
decision steps. A grace interval permits incidental challenges. Thereafter:

- sustained ally contact is penalized when the ball is not being contacted;
- sustained opponent contact is penalized only when ball progress is stagnant;
- separating motion clears the streak and emits an escape;
- goal grace never accumulates contact penalties.

The reward is team-level and identity-free. It cannot assign a permanent keeper
or attacker.

## Static obstacles

A stationary same-size body is useful as a deterministic contact-escape
holdout, but not as routine training. Real opponents yield, turn, and contest
the ball; optimizing against an immovable obstacle would teach lateral
clearance artifacts and over-penalize necessary goal-mouth contact.

## Promotion

Overall mean success is insufficient. Promotion requires non-regression floors
for pass/receive and rotation/recovery, plus bounded ally deadlock and uncovered
coverage rates. The latest checkpoint may continue training, while
`best-semantic.json` selects the checkpoint satisfying these gates.

## Rollback

All behavior is configuration-gated. Setting contact coefficients to zero and
using the legacy roster mode restores the M16 reward and scenario distribution.
