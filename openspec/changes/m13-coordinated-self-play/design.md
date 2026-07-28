## Decisions

The dense ball signal is `tanh(cos(v_ball, enemy_goal - ball)) -
tanh(cos(v_ball, own_goal - ball))`, scaled by the episode horizon. It rewards
useful ball direction rather than merely being near the ball. Below a movement
threshold it is zero.

The closest teammate is selected dynamically as attacker. Its velocity-to-ball
alignment is a penalty-only signal, also horizon-scaled, so standing still or
moving away is costly without giving a fixed robot a permanent role. This
follows the shared-policy collision warning and reward equations in Julio De La
Torre's thesis while retaining this project's centralized MAPPO critic.

A `-1 / horizon` time term makes a scoreless full episode cost one reward unit.
Wheel effort and action delta remain small regularizers, not primary objectives.
Congestion uses the mean squared spacing violation across the three teammate
pairs; its lower coefficient prevents avoidance from dominating play.

Defensive coverage tracks the nearest teammate to a point inside the own goal
mouth aligned with the ball. Reward is the improvement in distance between
successive decisions, scaled continuously by threat as the ball enters the
defending half. This creates a dense defensive signal without assigning a fixed
robot identity or rewarding stationary camping.

All reward and exploration fields are included in the checkpoint fingerprint.
The M13 config uses a larger actor, higher entropy coefficient, and a minimum
`log_std` of -2.0. The optimizer clamps that floor after every update. The first
250 iterations use the deterministic dynamic heuristic as opponent; subsequent
iterations use a frozen copy of the current learner.

Old checkpoints may omit newly introduced fields only when the selected config
supplies their exact neutral legacy defaults. This preserves M12 inspection
without allowing an incompatible M13 resume.

Checkpoint ranking runs terminal matches from both reflected starting sides
against a fixed heuristic. Ordering is W-L balance, goal difference, then mean
field progress. It intentionally produces no replay files.

## Validation

- Unit tests cover scalar/vector reward parity, directionality, bounded costs,
  exploration clamping, legacy loading, and terminal scorecards.
- A CUDA smoke run must produce finite loss, return, and a checkpoint.
- The completed 50M run is ranked as historical evidence; a fresh M13 run is
  required for an identical-budget comparison.
