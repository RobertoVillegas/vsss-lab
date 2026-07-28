# Population consolidation evidence — 2026-07-28

## Prior art reviewed

Nexto's public lineage demonstrates the value of self-play and historical
opponents, while modern league training systems retain past policies to reduce
cycling. Elo remains useful for matchmaking but collapses distinct behavioral
specialties into one scalar.

## Decisions

- **Adopt:** historical Elo for auditable pairwise updates and paired-color
  reports.
- **Adapt:** retain a bounded archive through greedy quality-diversity over
  reward-independent possession, pressure, congestion, and action-jerk
  descriptors, seeded by the best-rated policy.
- **Adopt:** distillation only when a specialist mixture beats the best single
  policy and its lower confidence delta clears the configured floor.
- **Reject:** retaining every checkpoint, rating-only diversity, and automatic
  distillation after training.

## Falsifiable gate

Near-duplicate high-rated policies cannot crowd out behaviorally distinct
specialists in a bounded archive. A positive point estimate with unresolved
confidence must block distillation.

## Executed consolidation decision

The frozen M13 terminal comparison (`0-9-1`, ten color-paired games over five
independent seeds) is retained as the incumbent reference. It is deliberately
not relabelled a successful policy: “promoted baseline” here means the frozen
registry incumbent that future candidates must beat.

The bounded archive and paired-color evaluator found no confidence-resolved
specialist advantage at this fidelity. The consolidation decision is
`specialist_advantage_not_resolved`; therefore distillation was not attempted.
This is the intended conditional outcome, not missing work. The executable
gate now requires a positive paired result before training a student and,
whenever it does train one, requires both non-regressed terminal confidence and
no inference-latency increase before acceptance.
