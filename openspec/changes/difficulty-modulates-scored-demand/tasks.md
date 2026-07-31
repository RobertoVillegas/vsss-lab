# Tasks

- [x] Accept ADR 0016 before implementation begins.
- [x] Interpolate the clearance ball depth across its difficulty axis.
- [x] Bump the generator revision so holdouts across revisions cannot be mixed.
- [x] Add a test asserting the easy end asks for less displacement than the hard end.
- [x] Verify without retraining that the family becomes learnable from below.
- [x] Complete local gates: full Python suite, `mise run lint`, OpenSpec strict validation.
- [ ] Audit the remaining families for the same defect. This change fixes the one case
      measurement exposed and does not claim the others were checked.
- [ ] Record a fresh baseline; evaluations on the previous revision are not comparable.

## Rejected on measurement, not on taste

- Raising the useful-contact coefficient: the term is linear near zero, so no coefficient
  makes a 0.05 m/s touch a signal without making a 1.5 m/s strike pay several times a goal.
- Letting the strike drive through harder: sweeping the drive-through offset from 0.28 to
  1.00 m moved clearance from 0.00 to 0.03. The executor was not the binding constraint.
- Lowering the phase patience: the phase would have advanced with the policy still unable
  to clear.
