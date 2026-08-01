# Design

## Why the horizon was the wrong diagnosis

The first reading of this problem was that γ = 0.99 at 50 Hz gives a two-second credit horizon
against a three-hundred-second half, so the value function cannot represent protecting a lead
for four minutes. The arithmetic is right — γ raised to a half is 3.4e-66 — and it is not what
blocks this.

Two things dissolve it. The first is that the policy has never observed a lead, so no horizon
would help. The second is that `ΔP(win)` is paid at the instant of the goal. Nothing has to
propagate across a match: a two-second horizon credits the approach that produced the goal, and
a concession seconds away is what teaches defensive positioning, exactly as the goal-geometry
potential already works over the same interval.

Long episodes would still be needed to learn a *plan* that spans a half. They are not needed to
learn a policy that reads the situation it is handed, which is what a fragment is.

## The model

Goals arrive as two independent Poisson processes with rates `λ_for` and `λ_against` per minute.
Over `t` minutes remaining, the goal difference added from here is Skellam distributed, and with
a current lead `L`:

```
P(win) = P(Skellam(λ_for·t, λ_against·t) > −L) + ½ · P(Skellam(...) = −L)
```

The half-weight on the tie is a modelling choice, not a rule: in group play a draw is neither a
win nor a loss, and splitting it keeps `P` continuous as the lead crosses zero. In knockout play
the rules resolve a draw by extra time and penalties (rule 8.2), which this does not model.

The whole model is two numbers. That is deliberate: a miscalibrated reward should be legible as
a wrong rate rather than buried in a coefficient.

## Why the rates are frozen

Re-estimating `λ` from the run in progress would be the obvious thing and it is wrong twice
over. The reward would become non-stationary, so the policy would be chasing a target that moves
because it improved. And `ΔP` telescoping to `P(end) − P(start)` — which is what makes this
potential-based and non-farmable under ADR 0015 — only holds if `P` is a fixed function of
state. A `P` that drifts is not a potential.

They are measured once from symmetric self-play, written into evidence, and set in config.

## Why the flat term exists

From three goals up, every entry in the value table is within 0.003 of zero. Under `W · ΔP`
alone, a team leading 3–0 would be told that nothing it does matters, which is the opposite of
the intent. Goal difference decides the group table (rule 8.1), so `g` carries it.

`g` is also the knob that decides park-the-bus against keep-attacking, and it is one number.
The ablation is `g ∈ {0, small, large}` against goals-for-per-minute and the conceded rate in
the last thirty seconds of a one-goal lead — a measure that only moves if the behaviour this
change is about actually appeared.

## Rates measured against a lopsided opponent are not usable

The rates on hand — 0.473 for and 0.024 against, from 330 paired matches — come from a matchup
the policy dominates. Under them a one-goal lead is already 0.99 to win and there is no gradient
left at `+1`, which is precisely the region this change is meant to shape. They are recorded in
`m24-3-rules-fidelity.md` as a throughput and fidelity measurement and must not be reused here.

Symmetric self-play is the right sample because it is the distribution the league trains
against.

## Randomizing the situation

The lead and the clock become a difficulty axis of the semantic curriculum, generated like the
others and audited by `tools/audit_skill_difficulty.py`, so an axis that turns out to be inert
or inverted is caught rather than assumed. ADR 0017 exists because two of five axes were inert
everywhere and nothing noticed.

An axis over the situation has an ordering that is not obvious: is a two-goal deficit harder
than a one-goal lead? The audit answers it by measurement rather than by declaration, and the
answer is what the ladder is built from.

## Holdouts

The scenario distribution changes, so the immutable holdouts no longer sample the same space.
Regenerating them breaks comparability with earlier runs, which is the honest outcome, and
keeping them would compare two distributions while reporting one number. They are regenerated
under a new generator revision.

## What this does not give

The policy still cannot hear the referee. A stoppage arrives as an imposed state transition, and
nothing here changes that. What changes is that the state it is handed after the stoppage now
carries a situation it was trained to read.
