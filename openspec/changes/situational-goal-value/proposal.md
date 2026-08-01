# Situational Goal Value

## Why

The policy cannot play to the scoreline, and the reason is measurable rather than architectural.
The match situation is already in the observation — context channel 0 is time remaining, channel
1 is the score difference — and over 96 000 observations of the live configuration both are
constants. The score channel is exactly zero in 99.79 per cent of them and its maximum over the
whole sample is zero: the policy has never seen a lead. Every semantic scenario resets to
`score_blue=0, score_yellow=0, simulation_time=0.0`, and a thirty-second episode moves the clock
across five per cent of a match.

Two of nine context channels are therefore dead inputs, and a goal pays `goal_coefficient = 10.0`
whether it is the eleventh of a rout or the equalizer with ten seconds left.

See ADR 0020.

## Milestone and non-goals

Reward and scenario fidelity for the active milestone. Non-goals:

- episodes do not become halves. A match reaches the policy as a sequence of play fragments
  separated by referee stoppages it cannot hear, which is the shape training already has;
- no change to the action space, the network, or the observation layout. The two channels this
  change makes useful already exist;
- no term that pays for defending while ahead. That is a hand-authored strategy and ADR 0015
  exists because the policy farms them;
- the goal rates are measured once and frozen, not tracked during a run.

## What changes

- a goal is rewarded by `W · ΔP(win | lead, time remaining) + g · (±1)` rather than a flat
  coefficient, with `P(win)` from a Skellam model of two Poisson scoring processes;
- the scenario generator randomizes the starting lead and clock, so the two context channels
  carry variance and the situation becomes trainable;
- the two scoring rates are measured from symmetric self-play, recorded as evidence, and fixed
  as configuration;
- the semantic curriculum gains the situation as a difficulty axis, audited by the existing
  tool like every other axis;
- immutable holdouts are regenerated, because the scenario distribution has changed and reusing
  them would compare across two distributions without saying so.

## Impact

- `python/vsss_train/marl_env.py`: the goal term, and the win-probability model behind it
- `python/vsss_train/semantic_scenarios.py`: the situation axis and a generator revision
- `python/vsss_train/config.py`: `W`, `g`, and the two frozen rates
- `crates/vsss-features`: the reward term, once its Python reference has an equivalence test
- `docs/evidence/`: the measured rates, and the ablation over `g`
- Behaviour before and after is not comparable, because the network can now condition on
  something it could not see before.
