# Tasks

- [x] Accept ADR 0014 before implementation begins.
- [x] Add the circular heading contract: unbounded mean direction, per-state concentration.
- [x] Sample, score, and measure entropy as circular quantities in rollout and update.
- [x] Transport the sampled angle normalized by π inside the existing bounded contract.
- [x] Assert rollout and update recover one log-probability, so the ratio starts at one.
- [x] Move the teacher intensity target inside the reachable interval.
- [x] Let requested authority inform strike intercept selection.
- [x] Reject a checkpoint loaded under a different heading contract.
- [x] Report angular concentration beside the heading-change statistic.
- [x] Add the isotropy, log-probability equality, intercept authority, wrap, entropy, and
      loader rejection tests.
- [x] Keep the previous parameterization selectable and green as the ablation baseline.
- [x] Complete local gates and an end-to-end smoke run reaching a paired evaluation.
- [ ] Record a fresh baseline from a full run; earlier M24.2 numbers are not comparable.
- [ ] Decide the unreachable-intercept fallback: when no candidate is reachable at any
      authority the executor still selects the furthest prediction, so a very slow request
      can chase a point it cannot reach. Needs trajectory evidence, not a guess.
- [ ] Revisit navigation authority: its target is a fixed 0.4 m ahead, so `forward`
      saturates at 0.8 and the top fifth of authority is unreachable for navigate.

## Narrowed during implementation

- Wrap continuity is asserted on the requested heading and its target, not on the
  executed wheels: reversing by half a turn is a genuine differential-drive tie that
  `go_to_target` breaks by the sign of the heading error.
- Authority changes intercept selection only where full authority would have committed
  early; the unreachable fallback is unchanged and left as an open task.
