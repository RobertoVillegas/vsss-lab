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
The M13 actor samples a Gaussian latent action and transforms it through `tanh`;
PPO evaluates the corresponding bounded density with its Jacobian correction.
The optimizer clamps `log_std` to the configured `[-2.0, -0.2]` interval after
every update, and the transformed-policy entropy coefficient is 0.001. This
prevents the entropy bonus from driving most wheel commands into clipping.

The first 1,000 iterations use the deterministic dynamic heuristic as opponent.
Subsequent iterations sample 35% frozen current learner, 30% uniformly from the
latest 16 eligible historical checkpoints, and 35% dynamic heuristic. Selection
uses a local seeded RNG; inference-only historical actor loading cannot mutate
trainer RNG state.

Old checkpoints may omit newly introduced fields only when the selected config
supplies their exact neutral legacy defaults. This preserves M12 inspection
without allowing an incompatible M13 resume.

Checkpoint ranking runs terminal matches from both reflected starting sides
against a fixed heuristic. Ordering is W-L balance, goal difference, then mean
field progress. It intentionally produces no replay files.

Canonical `metrics.jsonl` remains the source of truth for run telemetry.
TensorBoard events are a derived scalar sink closed and flushed with the
trainer. The replay server exposes a bounded, evenly sampled history and the web
viewer renders synchronized curves without loading replay bodies. TensorBoard's
own server remains optional and separate from the replay server.

The paired-run comparison consumes canonical JSONL metrics, the final
checkpoint's exploration parameters, and evenly sampled replay frames. It
reports goal/terminal rates, rolling learning signals, throughput, and a
teammate-spacing clustering proxy in a machine-readable artifact.

## Validation

- Unit tests cover scalar/vector reward parity, directionality, bounded costs,
  transformed action densities, both exploration clamps, legacy loading, and
  terminal scorecards.
- A CUDA smoke run must produce finite loss, return, and a checkpoint.
- The completed 50M run is ranked as historical evidence; a fresh M13 run is
  required for an identical-budget comparison.
