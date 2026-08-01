# Tasks

- [x] Accept ADR 0019 before implementation begins.
- [x] Record the profile that motivates the work, including the physics share.
- [x] Add the crate that will hold the ported computation, depending on `vsss-spec` only.
- [x] Slice 1: observations. Native batch call, golden-equivalence test, throughput reported
      at 76x on the ported work.
- [ ] Check whether role assignment is genuinely called twice per world per decision before
      porting it; a redundant call is cheaper to delete than to port.
- [ ] Slice 2: role assignment.
- [ ] Slice 3: reward terms, asserted term by term against the recorded decomposition.
- [ ] Slice 4: contact, deadlock and idle-spin detection.
- [ ] Slice 5: the free-ball restart, which today serializes a snapshot dictionary inside the
      hot loop.
- [ ] Slice 6: the action executor, whose scalar geometry runs once per robot per decision.
- [ ] Retire each Python reference only after its slice has been green for a full run.
- [ ] Re-measure the floor after every slice; the thirteen-times estimate is an extrapolation
      and should be replaced by measurement as it is approached.
