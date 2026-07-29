# Design

## Phase ladder

1. `foundation`: approach, shot, interception; 5% full matches.
2. `defense`: interception, save/deflection, clearance; 10% full matches.
3. `cooperation`: pass/receive; 10% full matches.
4. `rotation`: rotation/recovery; 15% full matches.
5. `integration`: all skills; configured 20% full matches.

Non-integration phases reserve 20% of skill selection for earlier-phase
rehearsal. Failure replay is restricted to the current focus so a stale hard
case cannot silently reopen a future objective.

## Promotion

Paired immutable holdouts continue to evaluate every family. A phase advances
only after its gates pass on two consecutive evaluations. Phase and streak are
checkpointed with curriculum state and exposed in telemetry.

## Rewards

Terminal skill and match outcomes remain authoritative. Ball-direction and
defensive shaping are reduced from 1.0 to 0.25, alignment is disabled, and time
pressure falls from 1.0 to 0.25. A bounded contact-entry impulse of 0.25 rewards
only ball acceleration toward the opponent goal.

## Checkpoint selection

Selection ranks completed phase evidence, full promotion eligibility, number of
passed gates, global semantic success, fewer unresolved trials, and only then
minimum-family success. This prevents a tiny weak-family gain from replacing a
more consolidated checkpoint.
