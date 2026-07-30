# Design

## The measurement that motivates the change

Circular standard deviation of the decoded heading, 200 000 samples per cell,
reproduced independently of the audit that first reported it:

| latent mean | deviation at `log_std = -0.5` | at `log_std = -1.8` |
| --- | --- | --- |
| (0, 0) | 210° | 197° |
| (1, 0) | 42.6° | 11.9° |
| (3, 0) saturated axis-aligned | 24.6° | 9.1° |
| (3, 3) saturated diagonal | 0.8° | 0.1° |

`tanh` bounds the mean vector to the square, so the reachable radius along an axis
is 1 while along a diagonal it is √2, and concentration grows with radius. The
worst-precision direction is the one aimed at the goal.

## Circular parameterization

The heading head emits an unbounded two-vector; its `atan2` is the mean direction,
which needs no bound because the circle has none. A second head produces
concentration through a positive transform, per state. Sampling draws from a
circular distribution around the mean direction, and its log-probability and
entropy are the circular ones, so:

- precision is set by concentration alone and is the same in every direction;
- concentration is state-dependent, so a policy can be sharp on a shot and broad
  while searching, which one shared deviation cannot express;
- the entropy the bonus multiplies is the angular entropy, so the bonus acts on
  the quantity it names.

The transported token carries the sampled angle divided by π. It stays in `[-1, 1]`,
so the existing clip and the width-4 contract are untouched, and the stored value is
the sampled variable itself rather than a squashed latent — the update re-scores it
directly with no change of variables, which removes the saturation-amplification
exposure of the tanh path.

## Intensity

Two separate defects: the teacher target `1.0` is outside the reachable interval and
drags the mean into saturation, and the strike reachability model assumes full
authority when selecting an intercept. The target moves inside the interval, and the
requested authority enters the arrival estimate so intercept selection and execution
agree. Navigation's fixed 0.4 m target, which caps authority at 0.8, is recorded as a
known limit of the executor rather than changed here.

## Compatibility and rollback

The current parameterization remains selectable, and the checkpoint records which
contract it was trained under so the loader rejects a mismatch instead of
reinterpreting weights. Legacy M24 and M24.2 checkpoints keep loading under their own
parsers. Rollback is configuration-level: select the previous contract.

## Validation

- The isotropy measurement above, as a test with a tolerance, for both
  parameterizations — the current one is expected to fail it and the new one to pass.
- A rollout-versus-update log-probability equality test for the circular
  distribution, the invariant that keeps the PPO ratio meaningful. No such test
  exists for the parametric path today, which is why the tanh path's exactness had
  to be established by inspection.
- An intercept-reachability test at reduced authority.
- Loader rejection across parameterizations.
- Trajectory and behavior benchmarks, plus one end-to-end run reaching a paired
  evaluation.

Absolute returns, entropy, and heading statistics are not comparable with earlier
M24.2 runs, so acceptance needs a fresh baseline rather than a continuation of the
run in flight.
