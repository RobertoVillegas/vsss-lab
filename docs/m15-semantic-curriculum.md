# M15 semantic skill curriculum

M15 teaches causal soccer skills in short, deterministic drills while retaining
full matches for transfer. It does not replace match evaluation, alter Rapier
physics, or authorize a large training run by itself.

## Scenario contract

`SkillScenarioParameters` records the family, seed, controlled color, five
normalized difficulty axes, semantic horizon, generator revision, and whether
the cell is an immutable holdout. The compiler produces a canonical six-robot
snapshot and rejects non-finite, out-of-field, overlapping, or initially
terminal states.

| Family | Initial condition | Success |
| --- | --- | --- |
| `approach` | stationary ball and offset robot | controlled robot contact |
| `interception` | moving goal-bound ball | contact followed by a confirmed safe trajectory |
| `save_deflection` | late goal-bound threat | confirmed post-contact removal of the threat |
| `clearance` | ball moving inside the defensive zone | controlled contact and exit from danger |
| `shot` | attacking approach with ball motion | controlled contact followed by the correct goal |
| `pass_receive` | ball launched by a declared passer | ordered uncontested reception in the target corridor at bounded speed |

Blue and yellow instances use mirrored geometry. The controlled robot rotates
with the seed, so a skill is not permanently assigned to one player identity.

## Outcomes, reward, and termination

Every drill is `running`, `success`, `failure`, or `unresolved` with a reason
code. Contacts count only on entry; persistent overlap cannot farm touches.
Interceptions and saves require a confirmation window so a rebound toward goal
does not count. Resolution terminates and resets only that vector world.

Success and failure add a bounded terminal skill reward separately from base
match reward. Timeout is unresolved, not inferred from shaped return. Full
matches retain normal goal grace, draw, and stagnation behavior.

## Allocation and difficulty

The scheduler mixes routine rehearsal, frontier cells, deduplicated failures,
and a configured full-match floor. Ball speed, ball angle, spawn distance,
target width, and opponent pressure advance independently from rolling success
and learning progress.

Immutable paired-color holdouts use separate seeds. Their transitions may
produce evaluation metrics but training feedback rejects them.

## Inspection

`metrics.jsonl` records allocation, family success, independent difficulty,
semantic outcome counts, and completed trial descriptors. Each descriptor
includes family, color, difficulty, parameter/state hashes, reason, steps, and
contact counts. Rich displays live outcomes; TensorBoard receives
`curriculum/*`, `skill_success/*`, and `skill_outcome/*`.

The web Training Metrics view contains outcome and per-family charts plus a
timeline filterable by family, controlled color, outcome, and difficulty band.
Atomic-drill video capture is not implemented yet; current video captures remain
full matches while trial metadata is directly inspectable.

## Evaluation

Run the immutable five-seed paired-color controls:

```bash
just m15-evaluate-control experiments/reports/m15/random.json random 5
just m15-evaluate-control experiments/reports/m15/heuristic.json heuristic 5
```

Evaluate a compatible checkpoint:

```bash
just m15-evaluate /path/to/checkpoint.pt \
  experiments/reports/m15/candidate.json 5 \
  experiments/configs/m15-mappo-semantic.toml
```

Reports contain every trial, Wilson confidence intervals, time-to-resolution,
and resolved drills/s. Promotion also requires paired full matches against
frozen M14, heuristic, and historical policies. Training return cannot
authorize a large run.

## Operational training

The default high-budget protocol starts clean rather than inheriting the M14
policy:

```bash
just league-live-semantic 50000000 25 60 25 auto 64
```

It uses `m15-mappo-semantic.toml`, evaluates paired immutable holdouts every 25
iterations, and writes `semantic-evaluations.jsonl` plus
`best-semantic.json`. Best selection compares minimum per-family success before
macro success and unresolved count, so mastered approach or shooting cannot
hide zero defense or passing.

The tuned defaults use a `1e-4` fine-tuning learning rate, `0.003` entropy
coefficient, `-1.2` log-standard-deviation floor, `0.5` bounded semantic
terminal reward, and 20% full-match floor. These are protocol parameters, not
universal target metrics; holdout skill success and paired match transfer remain
the actual objectives.

A compatible earlier policy can be tested explicitly with
`league-semantic-warm-steps-at`. It transfers actor and critic only; optimizer,
policy version, RNG, and curriculum restart, with source digest and reset
boundary in `initialization.json`.

## Resume and rollback

Semantic runs atomically persist `semantic-curriculum.json`. `--resume` requires
it and restores difficulty histories and deduplicated failures; incompatible
generator revisions are rejected.

Rollback selects `experiments/configs/m14-mappo-adaptive.toml`. M13 and M14
checkpoints remain loadable because M15 does not change actor tensor shapes.
Never resume an M15 run with an M14 configuration or vice versa.
