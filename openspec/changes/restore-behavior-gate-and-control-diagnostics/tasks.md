# Tasks

- [x] Judge idle spin on measured angular speed, with its threshold in physical units.
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
- [ ] Re-baseline the behavior ceiling once a run reports a trend under the physical
      detector. A fresh policy sits at roughly 0.09 against a 0.08 ceiling, so the ceiling
      is plausible but still unvalidated by a trained policy.

## Corrected during implementation

The first attempt normalized the commanded wheel differential by the parser's attainable
turn authority. That preserved reachability but not meaning: under a skill parser the
normalized quantity reduces to the heading error as a fraction of ninety degrees, so the
configured threshold came to mean "aiming 11.7 degrees off". Three paired evaluations of
`vsss-m24-3-run-0001` sat flat at about 1.8 times the ceiling, which is the signature of a
detector measuring the wrong quantity. Detection moved to measured angular speed, which is
parser-independent by construction rather than by rescaling.
