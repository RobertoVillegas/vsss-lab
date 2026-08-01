# M24.3: simulation fidelity against the LARC VSSS 2025 rules

Read from the official rulebook (IEEE Very Small Size Soccer, Série A, LARC 2025) and
compared with `tests/golden/m1_match_config.json` and the training loop.

## The physical model is faithful

| rule | rulebook | configuration |
| --- | --- | --- |
| field | 150 × 130 cm | 1.5 × 1.3 m |
| goal | 40 cm wide, 10 cm deep, beyond the field | 0.40 / 0.10 |
| ball | 42.7 mm diameter, 46 g | radius 0.0215 m, 0.046 kg |
| robot | 7.5 cm cube | 0.075 × 0.075 m |
| corner chamfer | 7 × 7 cm isosceles | `CORNER_CHAMFER = 0.07` |
| match | two halves of 5 minutes | `match_duration = 600` s |

One discrepancy: the walls are 2.5 cm thick in the rules and 2.0 cm in the configuration.
Robot mass is 0.5 kg, which the rules do not constrain.

## The evaluation window is a tenth of a half

A paired match in `evaluate_checkpoint_scorecard` runs one horizon: 1500 decisions of 20 ms,
so **30 seconds**. A regulation half is 300 seconds.

Measured over 330 paired matches of the run in flight, 9900 seconds of play:

- 78 goals for, 4 against, 75 per cent draws
- 0.473 goals per minute for, 0.024 against
- projected over a regulation half: **2.4 – 0.1**; over a full match: **4.7 – 0.2**

The probability of at least one goal inside 30 seconds is about 0.24, which over 300 seconds
becomes 0.93. The draw rate is therefore a property of the window, not of the play, and a
policy that reads as drawing three matches in four would win a regulation match comfortably
against the same opponent.

This has a direct consequence: `semantic_max_match_draw_rate = 0.70` asks for fewer draws
than the window structurally produces, so the match gate rejects promotion for a reason that
has nothing to do with the policy. `goals_for_per_minute` is now recorded beside it, because
a rate is invariant to the window and a scoreline is made of rates.

## Rules of play that are not modelled

The physics are faithful; the refereeing is not. In rule order:

- **Free ball, rule 15.** An impasse of **10 seconds outside both goal areas** is resolved by
  repositioning the ball at the quadrant's free-ball mark and **continuing**. The simulation
  ends the episode after **5 seconds** of the ball moving less than 2 cm and charges a
  penalty. Both the window and the resolution differ, and the observed effect is large: four
  of six episodes in one capture ended at exactly 5.0 s.
- **Goal kick, rule 14.** An impasse inside the goal area for 10 seconds, or attacking with
  more than one robot inside the opponent's goal area, is a goal kick. Not modelled.
- **Defensive penalty, rule 9.3.** Defending with more than one robot inside your own goal
  area is a penalty. Not modelled; the simulation shapes this with a coverage reward instead.
- **Retention and covering, rules 9.1 and 17.** Only the goalkeeper may hold the ball, and no
  robot may cover more than 30 per cent of it. The longest captured episode of run 0003 was a
  robot in contact with the ball for 1462 consecutive ticks, 29 seconds, which under these
  rules would be retention against us. The simulation may be rewarding an illegal behaviour.
- **Goalkeeper, rule 3.2.3.** The keeper is whichever robot has spent longest inside its own
  goal area, with the counter reset on leaving, and it must clear an uncontested ball within
  10 seconds. The role assigner picks a goalie geometrically and has no dwell counter and no
  clearance obligation.

## What follows

The gap that matters most for current training is the stagnation terminal: the simulation
teaches that a stalled ball ends the game at a loss, where the rules restart play at a neutral
mark. That is a change to the episode contract and belongs in its own proposal.
