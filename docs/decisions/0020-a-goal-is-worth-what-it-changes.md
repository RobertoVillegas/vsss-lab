# ADR 0020: A goal is worth what it changes

- Status: proposed
- Date: 2026-08-01

## Context

The policy is asked to play well, and it does so identically whether it leads by ten or trails
by two. It cannot do otherwise, and the reason is not the one that was first assumed.

The observation already carries the match situation: context channel 0 is time remaining and
channel 1 is the score difference over ten. Measured over 96 000 observations of the live
configuration, those two channels are constants:

| channel | observed range | standard deviation |
| --- | --- | --- |
| time remaining | 0.9500 – 1.0000 | 0.014 |
| score difference | −0.1000 – 0.0000 | 0.005 |

The score channel is exactly zero in 99.79 per cent of observations, and its maximum over the
whole sample is zero: **the policy has never seen a lead.** Every semantic scenario resets to
`score_blue=0, score_yellow=0, simulation_time=0.0`, and a thirty-second episode moves the clock
by five per cent of a match. Two of the nine context channels are dead inputs, and a network
trained on a constant learns to ignore it.

The first diagnosis was that this is a credit-horizon problem — γ = 0.99 at 50 Hz gives a
two-second horizon against a three-hundred-second half, so the value function cannot represent
protecting a lead. That is true and it is not the binding constraint. The binding constraint is
that the state distribution contains no scorelines to be aware of.

The horizon argument also assumed the match is one long episode. In deployment it is not. The
referee stops play, an operator repositions the robots and resumes; the policy cannot hear the
referee, so a stoppage reaches it as an externally imposed state transition. A match is a
sequence of play fragments, which is the shape training already has. Fragments do not need to
become halves.

What does need to change is that a goal pays `goal_coefficient = 10.0` in every situation. The
eleventh goal of a rout pays what the equalizer with ten seconds left pays.

## Decision

A goal is rewarded by how much it changes the probability of winning, plus a small flat term for
goal difference:

```
goal reward = W · ΔP(win | lead, time remaining) + g · (±1)
```

and the scenario generator randomizes the lead and the clock, so those two channels carry
variance and the situation is something the policy can be trained on.

`P(win)` treats goals as two Poisson processes, so the goal difference at the end is Skellam
distributed and the probability has a closed form in the lead, the time remaining, and one rate
per side. In symmetric league play at half a goal per minute a side:

| from a lead of | 15 s | 30 s | 60 s | 120 s | 300 s |
| --- | --- | --- | --- | --- | --- |
| value of scoring, +1 | 0.052 | 0.088 | 0.129 | 0.154 | 0.141 |
| cost of conceding, +1 | −0.445 | −0.401 | −0.337 | −0.262 | −0.174 |
| value of scoring, 0–0 | 0.445 | 0.401 | 0.337 | 0.262 | 0.174 |

One goal up with fifteen seconds left, conceding costs eight and a half times what scoring
gains, so the team defends. One goal up with five minutes left the two are nearly equal, so it
keeps playing. Neither behaviour is written down; both fall out of the arithmetic.

The flat term exists because the win-probability term does not. From three goals up every entry
in the table is within 0.003 of zero, so `W · ΔP` alone would tell a team leading 3–0 that
nothing it does matters. Goal difference decides the group table, and `g` is the weight on
caring about it.

The two rates are measured once from symmetric self-play and then frozen. Re-estimating them
during a run would make the reward non-stationary and would break the invariance argument below.

## Consequences

- The reward becomes situational without any situational rule being written. The alternative —
  a term that pays for defending while ahead — is a hand-authored strategy, and ADR 0015 exists
  because the policy farms those: score once, park, collect.
- `ΔP` telescopes. Summed over an episode it is `P(win at the end) − P(win at the start)`,
  bounded by one, so this is potential-based shaping on win probability and satisfies ADR 0015
  by construction rather than by argument.
- The credit-horizon problem largely dissolves rather than being solved. `ΔP` is paid at the
  moment of the goal, so nothing has to propagate across a match; γ = 0.99 credits the approach
  that produced it, exactly as the goal-geometry potential already does. Protecting a lead is
  learned from states seconds before a concession, not from a value function that can see four
  minutes ahead.
- Two dead observation channels become live. This is a change to what the network can condition
  on, so a policy trained before and after is not comparable on behaviour, only on outcome.
- The rates are a modelling choice with a number attached. If they are wrong the reward is
  miscalibrated in a way that is legible — the table above is the whole model — rather than
  hidden in a coefficient.
- Randomized scorelines change the scenario distribution, so the immutable holdouts of the
  semantic curriculum must be regenerated or explicitly exempted. Reusing them unchanged would
  silently compare across two different distributions.
