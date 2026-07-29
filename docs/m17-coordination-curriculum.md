# M17 coordination curriculum

M17 addresses the failure mode observed in `vsss-semantic-run-0006`: isolated
approach and shooting converged, while passing regressed, rotation remained at
zero, coverage degraded, and sustained robot contact increased after the best
checkpoint.

## Roster progression

Training does not start every skill in a congested 3v3 state:

| Skill | Routine roster | Frontier roster |
| --- | --- | --- |
| approach, shot | 1v0 | 1v1 |
| interception, save/deflection | 1v1 | 2v1 |
| clearance | 1v1 | 2v2 |
| pass/receive | 2v1 | 2v2 |
| rotation/recovery | 3v2 | 3v3 |

Inactive canonical robots are visible as inactive, receive masked actions, and
are excluded from distance, congestion, defense, role, and PPO-loss geometry.
This keeps one shared three-agent policy while teaching the smallest meaningful
coordination problem first.

Static robot-sized obstacles are not used as routine opponents. They are useful
for an isolated escape diagnostic, but they cannot yield, turn, challenge, or
recover like an opponent and would encourage obstacle-specific steering.

## Contact shaping

The environment measures contact at policy decision time. The first 0.5 seconds
are a grace interval. Beyond it:

- ally contact is penalized only without productive ball involvement;
- opponent contact is penalized only when neither contact nor ball movement is
  productive;
- ending a sustained contact records an escape;
- all penalties are bounded and team-level.

Terminal telemetry reports ally/opponent contact-seconds, new deadlocks,
escapes, completed rotation trials, uncovered coverage, and roster allocation.

## Selection and opponent population

Semantic selection records explicit promotion gates for pass/receive,
rotation/recovery, clearance, interception, and save/deflection. A higher
aggregate score cannot silently replace a coordination-capable checkpoint with
one that regressed on these families.

The heuristic-only bootstrap is 100 iterations. Subsequent iterations sample
self-play, recent historical checkpoints, and the heuristic; cumulative mode
counts are persisted in each metric record.

## Run

Start clean because M17 changes scenario semantics and reward geometry:

```bash
just league-live-semantic 50000000 25 60 25 auto 64
```

Do not resume or warm-start an M16 run when measuring the effect of this
curriculum.
