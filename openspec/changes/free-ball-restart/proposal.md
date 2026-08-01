# Free Ball Restart

## Why

Rule 15 of the LARC VSSS rulebook resolves a ten-second impasse away from both goal areas by
placing the ball on the quadrant's free-ball mark and continuing. The simulation ended the
episode after five seconds of a near-static ball and charged a penalty twice the size of a
draw. The window was half the rule's and a restart was modelled as a loss.

Measured: in one captured iteration four of six episodes ended at exactly 5.0 seconds with
the ball never moving, and stagnation was the dominant terminal across training, seven to
twenty-four per iteration against one to five goals.

See ADR 0018 and `docs/evidence/m24-3-rules-fidelity.md`.

## Milestone and non-goals

Rules fidelity for the active milestone. Non-goals:

- no goal kick, defensive penalty, ball retention, or goalkeeper dwell counter; those rules
  stay unmodelled and are listed as tasks rather than implied;
- no change to reward coefficients beyond the stagnation term becoming structurally zero;
- no change to the action space or the curriculum.

## What changes

- an impasse away from both goal areas restarts play at the quadrant's free-ball mark, with
  robots inside the clearance moved to their own half, at the rule's ten seconds;
- inside a goal area the impasse clock resets without repositioning, because that case is a
  goal kick and is not modelled;
- free balls are counted per world so the impasse rate stays observable;
- the wall thickness matches the rulebook's 2.5 cm.

## Success criteria

- an impasse no longer ends an episode, and the ball is found on a free-ball mark afterwards;
- the stagnation terminal no longer appears in a run's terminations;
- the impasse rate remains readable through the free-ball count.
