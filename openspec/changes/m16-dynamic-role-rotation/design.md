# Design

## Assignment

For each team state, evaluate all six permutations of attacker, support and coverage.
Costs use projected time-to-ball, attack angle, support-lane distance, goal-side
position and distance to the own goal. A switching cost supplies hysteresis; a
clearly safer assignment may override it immediately.

No robot ID, marker ID or array slot is an input to the cost. Roles are transient
responsibilities and remain reproducible from the observable match state.

## Policy contract

`role_mlp` retains a shared actor and centralized MAPPO critic. Each actor observation
adds a role one-hot vector, a recent-change marker and a team-uncovered marker.
Legacy actors explicitly slice the original four context values so existing
checkpoints keep their architecture.

## Learning and evaluation

`rotation_recovery` starts with a failed attacker beyond the play and an incoming
replacement. Success requires the replacement to touch and neutralize the threat
while the former attacker recovers goal-side.

Curriculum axis advancement uses a persistent per-family update counter instead of
the bounded outcome-window length. Holdouts evaluate difficulty bands 0.10, 0.25,
0.40 and 0.65. After eight warmup evaluations, four consecutive regressions stop the
run while preserving `best-semantic.json`.
