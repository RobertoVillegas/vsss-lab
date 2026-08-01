# The policy has never seen a lead

Two of the nine context channels the policy observes are the match situation: channel 0 is time
remaining, channel 1 is the score difference over ten. This measures what they actually carry.

## Method

The live configuration (`experiments/configs/m24-3-mappo-circular.toml`), 64 worlds, 1500
decisions, random actions, worlds reset on termination as training does. Every observation's
context channels 0 and 1 are collected — 96 000 samples.

Random actions rather than a trained policy on purpose: this measures the *scenario
distribution*, which is what the network is trained over, not what one policy happens to reach.

## What they carry

| channel | min | max | standard deviation |
| --- | --- | --- | --- |
| time remaining | 0.9500 | 1.0000 | 0.014 |
| score difference | −0.1000 | **0.0000** | 0.005 |

The score channel takes exactly two values across 96 000 samples, 0.0 and −0.1, and is 0.0 in
99.79 per cent of them. Its maximum is zero: **no observation in the sample carries a lead.**

The clock spans five per cent of a match, so the endgame — the region where the situation
changes what a team should do — is never observed either.

## Why

`semantic_scenarios.py` resets every scenario with
`state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)`, and the
episode horizon is 1500 decisions of 20 ms, thirty seconds against a three-hundred-second half.
A goal inside those thirty seconds is uncommon and can only move the channel to ±0.1.

A network trained on a constant input receives no gradient distinguishing values of it. These
two channels cannot be the reason for any behaviour the policy has.

## What a situational reward would be worth

Under a Skellam model of two Poisson scoring processes, `P(win)` has a closed form in the lead
and the time remaining. In symmetric league play at half a goal per minute a side:

| from a lead of | 15 s | 30 s | 60 s | 120 s | 300 s |
| --- | --- | --- | --- | --- | --- |
| value of scoring, +3 | 0.000 | 0.001 | 0.005 | 0.018 | 0.052 |
| value of scoring, +1 | 0.052 | 0.088 | 0.129 | 0.154 | 0.141 |
| value of scoring, 0–0 | 0.445 | 0.401 | 0.337 | 0.262 | 0.174 |
| cost of conceding, +1 | −0.445 | −0.401 | −0.337 | −0.262 | −0.174 |
| cost of conceding, 0–0 | −0.445 | −0.401 | −0.337 | −0.262 | −0.174 |

Today every one of those cells pays `goal_coefficient = 10.0`.

One goal up with fifteen seconds left, conceding costs 8.5 times what scoring gains. With five
minutes left the two are within twenty per cent of each other. The first says defend, the second
says keep playing, and neither is written anywhere — both follow from the rates.

From three up, every cell is within 0.003 of zero, which is why a win-probability term alone
would tell a team leading 3–0 that nothing matters. Goal difference decides the group table
(rule 8.1), so a flat term has to stay.

## Rates that must not be used

`m24-3-rules-fidelity.md` measured 0.473 goals per minute for and 0.024 against, over 330 paired
matches. Those come from a matchup the policy dominates. Under them:

| from a lead of | 15 s | 60 s | 300 s |
| --- | --- | --- | --- |
| value of scoring, +1 | 0.003 | 0.007 | 0.006 |
| cost of conceding, +1 | −0.445 | −0.315 | −0.060 |

A one-goal lead is already 0.99 to win, so there is almost nothing left to shape in exactly the
region the change is aimed at. The rates have to come from symmetric self-play.

## The ten-goal rule

The LARC 2025 rulebook does contain a ten-goal limit, and it is not a mercy rule. Rule 8.3
caps the scoreline recorded in a walkover — when an opponent does not appear, the present team
plays three minutes alone and the result is recorded "respeitando-se o limite de 10 gols de
diferença". A contested match runs its full two periods of five minutes (rule 7) whatever the
score, and the winner is decided on goal difference (rule 8.1).
