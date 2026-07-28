## Decisions

Congestion uses the mean squared spacing violation across the three teammate
pairs. Its coefficient is deliberately small so agents cannot earn more by
avoiding play than by advancing or scoring.

Defensive coverage tracks the nearest teammate to a point inside the own goal
mouth aligned with the ball. Reward is the improvement in distance between
successive decisions, scaled continuously by threat as the ball enters the
defending half. This creates a dense defensive signal without assigning a fixed
robot identity or rewarding stationary camping.

Both terms are configuration fields included in the checkpoint fingerprint.
Changing them requires a new run rather than resuming an incompatible policy.

## Validation

- Unit tests cover congestion ordering and threat scaling.
- A CUDA smoke run must produce finite loss, return, and a checkpoint.
- Run 0001 remains the fixed baseline for a later identical-budget comparison.
