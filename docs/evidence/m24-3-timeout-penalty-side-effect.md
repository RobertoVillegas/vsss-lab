# The timeout penalty doubled a bias it did not create

Run 0011 carries three fixes over run 0009 and diverges from it on the metric that matters.
Both start from the same distilled bootstrap at 0.552 strikes; by iteration 125, 0009 had
climbed to 0.760 and scored 0.2–0.6 goals per minute, and 0011 had fallen to 0.150 with
navigate at 0.848 and four consecutive evaluations scoring nothing.

## Measured, before touching anything

Which primitive resolves each foundation drill, driving one primitive on a loop with the rest
of the team idle, 20 seeds per cell at difficulty 0.1 (`tools/primitive_race.py`):

| family | primitive | resolved | success | steps |
| --- | --- | --- | --- | --- |
| approach | navigate | 1.00 | 1.00 | 19 |
| approach | strike | 1.00 | 1.00 | 19 |
| interception | navigate | 1.00 | **1.00** | **20** |
| interception | strike | 1.00 | **0.00** | **230** |
| shot | navigate | 0.55 | 0.55 | 115 |
| shot | strike | 0.55 | 0.55 | 125 |

There is no general bias toward navigate. `approach` is a tie at 19 steps and `shot` is a tie at
0.55. The whole effect is `interception`, where striking succeeds on none of twenty attempts and
burns the horizon, while blocking succeeds on all twenty in twenty steps.

## Why that is correct behaviour, and why it still broke the run

Interception is a blocking skill. `strike_target` selects a contact point *behind* the ball
relative to the requested exit heading, so against a ball travelling toward the defended goal
the robot must first get past it — goalward — before it can push back. Navigate drives straight
at the ball and stops it, which is what an interception is. The family is not misspecified.

What changed is the price of getting it wrong. Before, striking into an interception timed out
and paid nothing against navigate's `+1`, a gap of one. Charging the timeout makes it `-1`
against `+1`, a gap of two. The gradient against striking doubled on ten of the twenty-three
scenarios `foundation` allocates.

The bias was already there. Run 0009 lived with it because abstaining was free — and that same
freedom is what let 0009's predecessor collapse into running out the clock. The fix for one
pathology sharpened another.

## What this rules out

- It is not the behaviour gate: that only decides promotion and touches no reward.
- It is not the `rotation_recovery` ladder: `foundation` allocates it nothing, and the family is
  visibly working at 0.68–0.80 where 0009 sat at 0.00–0.10.
- It is not a general primitive preference, which the `approach` and `shot` ties rule out.

## What the probe missed, and why

`tools/probe_collapse.py` measured the timeout penalty from a randomly initialized policy and
found strikes *rising*, 0.283 to 0.695. From the distilled bootstrap the same change drives them
down. The starting point decides the sign, and the probe's limitation was written down when it
ran — it just was not treated as disqualifying. Any further probe of this reward has to start
from the bootstrap.

## Cycle 2: the policy conditions on drills, and defaults to navigate in matches

The cycle-1 reading was that a correct per-family signal had become a global bias because the
network could not tell the families apart. Measured, that is refuted, and the truth is more
specific.

What each checkpoint asks for on the opening state of a drill, 24 seeds per family
(`tools/probe_conditioning.py`):

| checkpoint | interception | shot | approach |
| --- | --- | --- | --- |
| 0009, iteration 1500 | strike 1.00 | strike 1.00 | strike 1.00 |
| 0011, iteration 175 | **navigate 1.00** | **strike 1.00** | navigate 0.42 / strike 0.58 |

The run that looked broken is the one selecting correctly: it blocks in interception, where
striking succeeds on none of twenty attempts, and strikes in shot. The run that looked healthy
strikes everywhere, including where striking cannot work. Falling aggregate `strike_fraction` is
therefore not a collapse — the drill mix is dominated by families where navigate is right.

That leaves the scoring gap unexplained, so the same question was put to a full match
(`tools/probe_match_tokens.py`), 200 decisions across 32 worlds, split by where the ball is:

| checkpoint | ball in the attacking third | midfield |
| --- | --- | --- |
| 0009, iteration 1500 | strike 0.82 | strike 0.87 |
| 0011, iteration 175 | **navigate 0.95** | **navigate 1.00** |

In a match the 0011 policy navigates almost exclusively, including in the attacking third where
a goal requires a strike. It learned to recognize *drill configurations* — a ball placed near
the goal with the striker behind it — not the underlying situation of having a chance to shoot.
Run 0009 scored because striking indiscriminately happens to work in a match, not because it
understood anything.

The drill terminal is now the sharpest signal in the reward: `±1` on every drill, against goals
that arrive rarely. Thirty-five per cent of episodes are full matches, and they are losing the
argument.

## Cycle 3: the drills are not outvoting the matches — dead end

Cycle 2 ended on the reading that the drill terminal had become the sharpest signal in the
reward and that full matches, at thirty-five per cent of episodes, were losing the argument. If
that were true the fix would be balance, and it is not true.

Reward accumulated per finished episode, environment terms only, 1200 decisions across the live
configuration (`tools/probe_reward_balance.py`):

| episode kind | episodes | mean | mean magnitude | steps |
| --- | --- | --- | --- | --- |
| drill | 28 | −6.89 | 10.39 | 381 |
| full match | 6 | −6.93 | 10.09 | 733 |

The two carry the same reward to within one per cent. The drill terminal adds `±1` on top of a
magnitude near ten — a tenth, not an order of magnitude. Whatever is teaching the policy to
navigate through matches, it is not that drills shout louder.

This also rules out the obvious rebalancing fixes: lowering the drill terminal, or raising
`semantic_full_match_fraction`, would each move a quantity that is already balanced.

What remains is generalization rather than magnitude. The policy strikes on drill opening states
and navigates in matches; if both kinds of experience carry equal reward, then what differs is
what the states *look like*. The shot drill places the ball near the goal with the striker
already behind it, which is a configuration a match rarely presents.
