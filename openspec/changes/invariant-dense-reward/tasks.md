# Tasks

- [x] Accept ADR 0015 before implementation begins.
- [x] Treat the geometry potential as zero at a terminal state in both environments.
- [x] Sign the useful-contact impulse, keeping the contact edge so it stays an impulse.
- [x] Add the terminal-invariance, wrong-way, and envelope-oscillation tests.
- [x] Keep the existing anti-farming tests green.
- [x] Complete local gates.
- [ ] Record a fresh baseline; earlier returns are not comparable.
- [ ] Revisit the time cost once a run reports returns under the corrected terms, which
      is the open question the coefficient rescale was dropped in favor of.

## Dropped during implementation

- Removing the contact edge: it would pay on every contact step, which is the dense
  ball-advancement reward M20 removed.
- Per-robot impulse attribution: the state exposes team contact, not ownership, and the
  sign already closes the farm.
- Rescaling the time, draw, and stagnation coefficients: unjustified without a run under
  the corrected terms.
