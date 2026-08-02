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

## Cycle 4: the shot drill taught a shape, not a situation

Cycles 2 and 3 left generalization as the only surviving explanation. Three quantities describe
a finishing chance for the robot nearest the ball; measured across 120 shot-drill opening states
and 2148 match states with the ball in the attacking third (`tools/probe_geometry_overlap.py`):

| quantity | drill p10/p50/p90 | match p10/p50/p90 |
| --- | --- | --- |
| robot to ball, m | 0.10 / 0.18 / 0.26 | 0.06 / 0.19 / 0.37 |
| ball to goal, m | 0.33 / 0.53 / 0.73 | 0.35 / 0.59 / 0.79 |
| **off the shooting line, degrees** | **1.4 / 7.0 / 13.9** | **22.2 / 100.4 / 144.9** |

Both distances overlap almost exactly. The angle does not overlap at all: the drill's ninetieth
percentile sits below the match's tenth. Within 45 degrees of the shooting line the drill is at
1.00 and a match at 0.26.

`_place_behind` parked the striker directly behind the ball with ±3.5 cm of lateral jitter, so
every finishing chance the drill ever presented started on the line. The policy learned to
strike from a shape that play produces a quarter of the time, and navigated through everything
else. That is the same class of defect as the two ladders already repaired, and the third
instance of it.

The approach angle is now what `ball_angle` carries for `shot`, spanning 8 to 120 degrees, and
the axis is declared in `FAMILY_AXES`. The drill's angle spread becomes 5.6 / 62.5 / 125.0
against the match's 22.2 / 100.4 / 144.9 — overlapping where it did not before.

A side effect worth recording: `spawn_distance` for `shot` improves from
`0.50 1.00 0.50 0.50 0.12`, non-monotonic and starting at half, to `1.00 0.75 0.75 0.75 0.50`.
The old placement was producing awkward starts at short range that the range axis was being
blamed for.

### The new axis is a cliff, and that is the finding

Audited against the 0009 checkpoint, `ball_angle` for `shot` reads `0.50 0.00 0.00 0.00 0.00` —
solvable at the easy end and impossible from level 0.25, which is 36 degrees off the line. The
policy cannot finish from an angle at all.

The range was left wide rather than compressed to make the ladder read as a ramp. Matches
present 45 degrees or worse three times in four, so a drill that stops short of that would teach
the same shape in a narrower band. The cliff is where capability ends, it is now measurable, and
the curriculum can advance through it as the policy improves — which it could not do while the
demand was absent.

## Cycle 5: no primitive can finish from an angle, and cycle 4 is withdrawn

Cycle 4 widened the shot drill's approach angle to match what play presents, and audited the new
axis as a cliff. The question left open was whether that cliff is the policy's limit or the
primitive's. It is neither — it belongs to the whole action set.

Driving one intent on a loop through shot drills while sweeping `ball_angle`, 20 seeds per cell,
scoring rate (`tools/primitive_race.py angle`):

| intent | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| --- | --- | --- | --- | --- | --- |
| navigate at the ball | 0.80 | 0.30 | **0.00** | 0.00 | 0.00 |
| navigate at the goal, dribbling | 0.80 | 0.45 | **0.00** | 0.00 | 0.00 |
| strike | 0.55 | 0.05 | **0.00** | 0.00 | 0.00 |

Nothing scores from level 0.5, which is 62 degrees off the shooting line. Stop, navigate and
strike all reach zero together. A second reading worth keeping: even on the line, dribbling at
the goal (0.80) beats striking (0.55). The strike primitive is not the best finisher in its own
best case.

### Why cycle 4 was withdrawn

The drill's narrow angle was not a defect measured against play — it was matched to what the
action space can do. Widening it teaches a demand no primitive can satisfy, and the audit says
so: `shot`'s `spawn_distance` went from a ramp to `beyond-reference` and its `ball_speed` from a
ramp to `inverted`. Narrowing the widened range to 40 degrees did not recover them either.

The change is reverted and the generator revision returns to `m24.3-ladders-3`. Both `shot` axes
audit as ramps again. What survives is the knowledge: the drill-to-match geometry gap measured in
cycle 4 is real, and it cannot be closed from the curriculum side.

A bug of mine is recorded with it. The widened placement added a lateral offset to a rotated
position, which shortens the radius as the bearing turns, and the scenario validator caught a
robot overlapping the ball. The validator earned its keep.

### What this leaves

The chain is now closed and it does not end at the reward:

1. Charging the drill timeout doubled an existing penalty on striking into interceptions, where
   striking genuinely cannot work.
2. The policy responded correctly per drill, and navigated through matches.
3. Reward magnitude is balanced between drills and matches, so it is not being outvoted.
4. The shot drill only ever presented chances from the shooting line, which is a quarter of what
   play presents.
5. And that is because the shooting line is the only place any primitive can finish from.

The action set, not the reward and not the curriculum, is what caps finishing.

## Run 0012: the carry gradient at a tenth of a goal does not carry

ADR 0021's term was enabled at `ball_progress_coefficient = 1.0` against a goal paying 10, so a
full carry is worth a tenth of scoring. The claim to test is not that it scores — finishing is
capped by the action set — but that it brings the ball to where finishing is possible. Measured
on that, over 300 decisions across 32 worlds (`tools/probe_ball_position.py`):

| checkpoint | mean Φ | share Φ > 0.5 | share Φ > 0.3 |
| --- | --- | --- | --- |
| iteration 0, the distilled bootstrap | 0.154 | 0.015 | 0.058 |
| iteration 400, trained with the carry | 0.157 | **0.007** | **0.025** |

Four hundred iterations of a dense per-step signal left the mean potential flat and **halved**
the time the ball spends in a convertible position. The term did not merely fail to help; the
quantity it exists to raise went down.

Two things are worth separating. Even at the bootstrap the ball is in front of the goal 1.5 per
cent of the time, so the underlying occupancy is very low and the measurement is on a small
number. And the term is outvoted: over the last hundred iterations `goal_mouth` accumulated
−0.023 against `goal_conceded`'s −0.446, a factor of **19**. The loudest gradient in the reward
is still "do not concede", and the safe way to satisfy it is to keep the ball away from our own
goal, which is not the same as taking it to theirs.

The run also crossed the behaviour gate's stop-fraction ceiling at iteration 400 (0.161 against
0.15) and was stopped there rather than spending six more hours.

### What this does and does not settle

It settles one of the three ablation settings the change asks for, and it settles it negatively.
It does not settle the design: a coefficient that is nineteen times quieter than the conceding
term has not been given a fair test of whether carrying can be taught, only of whether it can be
taught at this volume.

The next setting has to be chosen against that ratio rather than against the goal coefficient,
which is the mistake this one made. A carry worth a tenth of a goal sounded conservative and is
in fact inaudible.

## Run 0013: the carry works at five, and the coefficient was the whole story

Same term, same shape, coefficient raised from 1.0 to 5.0 — chosen against the conceding term
rather than against the goal, which is what run 0012 got wrong. At iteration 400:

| | strike | stop | shot | goals per minute |
| --- | --- | --- | --- | --- |
| 0011, no carry | 0.215 | 0.003 | 0.45 | 0.0 |
| 0012, carry at 1.0 | 0.322 | 0.161 | 0.25 | 0.0 – 0.2 |
| **0013, carry at 5.0** | **0.744** | **0.003** | **0.45** | **0.4** |

And on the term's own claim, the occupancy it exists to raise:

| checkpoint | mean Φ | share Φ > 0.5 | share Φ > 0.3 |
| --- | --- | --- | --- |
| the distilled bootstrap | 0.154 | 0.015 | 0.058 |
| iteration 400 at 1.0 | 0.157 | 0.007 | 0.025 |
| **iteration 400 at 5.0** | **0.174** | **0.046** | **0.102** |

Three times the bootstrap's time in a convertible position, against half of it at the lower
setting. The design was never the problem and neither was the shape; the coefficient was, and
specifically the habit of scaling a new term against the goal instead of against whatever
currently dominates the gradient.

This is the first setting of the ablation that works, not the end of it. A third point is still
owed, and it should bracket 5.0 from above.

### What now blocks promotion

`idle_spin_ratio` sits at 0.09 – 0.16 against its 0.08 ceiling and fails every evaluation. That
is not the pathology the gate was built for: this policy strikes on three quarters of decisions,
holds its stop fraction at 0.003, and scores 0.4 to 0.6 goals per minute — run 0009 spun at
0.005 – 0.04 and scored less.

The ceiling is not being raised. A gate that blocks is doing its job until the behaviour behind
the number is measured, and lowering a threshold because it is inconvenient is how the reward
that paid a policy to give up survived eight hundred iterations. What the flagged robots are
actually doing is the next thing to measure.
