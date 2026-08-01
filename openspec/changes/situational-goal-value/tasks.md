# Tasks

- [ ] Accept ADR 0020 before implementation begins.
- [x] Record the measurement that motivates the work: the score and clock channels are constants
      across the training distribution, and the score channel has never carried a lead.
- [ ] Measure `λ_for` and `λ_against` from symmetric self-play, not from a matchup the policy
      dominates. Record the sample size and the confidence interval, and write both rates into
      `docs/evidence/`.
- [ ] Implement the win-probability model against the measured rates, with a table of
      `ΔP` by lead and time remaining in the evidence so the reward is readable without running
      it.
- [ ] Replace the flat goal term with `W · ΔP + g · (±1)`, keeping both weights in config.
- [ ] Assert the invariance the design claims: the win-probability term summed over an episode
      equals `P(end) − P(start)` to float tolerance, on recorded episodes.
- [ ] Randomize the starting lead and clock in the scenario generator, under a new generator
      revision.
- [ ] Verify by measurement that both context channels now carry variance; the audit that found
      them dead is the one that has to come back clean.
- [ ] Declare the situation as a difficulty axis and run `tools/audit_skill_difficulty.py` over
      it. Accept the axis only if the audit reports a ramp.
- [ ] Regenerate the immutable holdouts and record the revision boundary, so no comparison
      spans two distributions silently.
- [ ] Ablate `g` over at least three settings. Report goals for per minute and the conceded rate
      in the last thirty seconds of a one-goal lead — the second only moves if the behaviour this
      change is about actually appeared.
- [ ] Port the reward term to `vsss-features` behind an equivalence test, once the Python
      reference is settled. Not before: porting a term that is still being tuned costs two
      changes for one.
