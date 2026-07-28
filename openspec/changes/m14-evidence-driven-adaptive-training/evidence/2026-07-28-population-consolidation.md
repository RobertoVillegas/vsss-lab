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
