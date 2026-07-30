# Tasks

- [x] Normalize idle-spin turn intensity by the parser's attainable turn authority.
- [x] Give the paired behavior gate the run's configured idle-spin thresholds.
- [x] Parse a learned opponent once per decision in the single environment.
- [x] Restrict action diagnostics to agents on the field.
- [x] Exclude episode boundaries and undirected vectors from the heading statistic.
- [x] Report the opponent kind from the selector instead of inferring it from a key.
- [x] Persist paired evaluation throughput, validity, and touches in the run record.
- [x] Render exploration deviation and skill mix for a hybrid policy.
- [x] Add detector, opponent-parse, and parser-authority tests.
- [x] Complete local gates: full Python suite, `mise run lint`, OpenSpec strict validation,
      and an end-to-end parametric smoke run.
- [ ] Re-baseline the M24.2 behavior ceiling once a run reports a non-zero idle-spin ratio,
      since no prior run measured one.
