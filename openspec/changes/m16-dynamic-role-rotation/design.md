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

`rotation_recovery` starts after a failed attack with an incoming challenger, the
former attacker beginning its exit, and the former coverage player ready to advance.
Success requires the challenger to touch and neutralize the threat, the former
coverage player to become support, and the former attacker to become coverage. The
predicate reads the same stateful role assignment supplied to the policy and rejects
transitions that leave the team uncovered for more than ten percent of the drill.

Curriculum axis advancement uses a persistent per-family update counter instead of
the bounded outcome-window length. Holdouts evaluate difficulty bands 0.10, 0.25,
0.40 and 0.65. After eight warmup evaluations, four consecutive regressions stop the
run while preserving `best-semantic.json`.

## M16.1 corrective curriculum

The first M16 runs exposed two measurement defects. The rotation predicate could pass
without a three-player handoff, and round-robin allocation spent equal capacity on
mastered and failing families. M16.1:

- carries stateful robot-to-role assignments into training and holdout predicates;
- measures the complete attacker/support/coverage handoff;
- allocates 80% of non-failure skill sampling by squared weakness and 20% uniformly,
  preserving rehearsal of mastered skills;
- holds every difficulty axis at a 0.05 floor; and
- lengthens semantic selection warmup/patience to 12/6 evaluations to reduce noisy
  max-min stops while retaining the same safety rule.
