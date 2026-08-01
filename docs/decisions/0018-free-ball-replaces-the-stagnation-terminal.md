# ADR 0018: A free ball replaces the stagnation terminal

- Status: accepted
- Date: 2026-07-31

## Context

Rule 15 of the LARC VSSS rulebook resolves an impasse of ten seconds away from both goal
areas by placing the ball on the free-ball mark of the quadrant where it happened, moving
any robot within twenty centimetres to its own half, and **continuing play**. Rule 14
resolves an impasse inside a goal area with a goal kick. Neither rule ends the game.

The simulation ended the episode after five seconds of the ball moving less than two
centimetres and charged a penalty of 1.0, twice the draw penalty. Two things were wrong at
once: the window was half the rule's, and a restart was modelled as a loss.

The effect was measured, not assumed. In one captured iteration four of six episodes ended
at exactly 5.0 seconds with the ball never having moved at all, and stagnation was the
dominant terminal across training at seven to twenty-four occurrences per iteration against
one to five goals. The policy was being taught that a stalled ball ends the game badly, in a
game where it does not end at all.

## Decision

An impasse away from both goal areas restarts play at the quadrant's free-ball mark. The
ball is placed there at rest, any robot inside the clearance is moved to its own half, and
the episode continues with its clock and score intact. The window is the rule's ten seconds.

Inside a goal area the rules call for a goal kick, which is not modelled. There the impasse
clock is reset and play continues without repositioning, so the free ball does not silently
implement a rule it is not.

The `stagnation_seconds` and `stagnation_penalty` keys stay in the configuration, unread,
because the checkpoint compatibility check rejects a stored key that no longer exists.

The rulebook figure fixes the free-ball marks at 37.5 cm from each end, which the
implementation uses exactly. The lateral placement is read as the quadrant centre; that part
is inference from the figure rather than a stated dimension, and is recorded as such.

## Consequences

- An episode now ends on a goal or on the horizon, never on a stalled ball. Episode length
  distribution and returns shift, so this milestone needs a fresh baseline.
- The stagnation reward term is structurally zero and is kept in the decomposition so the
  accounting stays comparable.
- Free balls are counted per world, so the impasse rate remains observable even though it no
  longer terminates anything.
- A policy can no longer be punished for a stalemate it did not cause, and equally can no
  longer end a losing position by stalling.
- Four rules of play remain unmodelled: goal kick, defensive penalty, ball retention, and the
  dwell-counter definition of the goalkeeper. They are recorded in the change's tasks rather
  than implied by this one.
