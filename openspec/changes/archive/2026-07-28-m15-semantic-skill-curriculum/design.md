## Context

The authoritative flat state already contains ball position/velocity and six
robot poses, twists, wheel states, teams, and match events. `reset_state`
restores any validated snapshot into an isolated Rapier world. M14 selects
routine, frontier, and failure scenarios, but its outcome callback currently
records `success = bool(events & BLUE_GOAL)`. That provides no useful teaching
signal for a save, interception, clearance, or received pass.

The existing M14 suite proves moving-ball initialization works:
`frontier-interception-right` uses `vx=-0.3`,
`frontier-defense-left` uses `vx=-0.2`, and an immutable mixed holdout uses
`vy=0.2`. M15 generalizes those examples into deterministic scenario families
with explicit semantics.

## Goals

1. Generate physically valid, diverse, reproducible atomic soccer situations.
2. Measure whether the requested skill was actually completed.
3. Increase useful terminal experiences per wall-clock second through early
   termination.
4. Preserve transfer to full matches and prevent diagnostic-event farming.
5. Make every reset and outcome inspectable and replayable.

## Decision 1: separate scenario parameters from canonical state

Each family uses a typed `SkillScenarioParameters` value containing:

- family and semantic version;
- controlled team and mirrored side;
- ball origin, speed, heading, and optional spin;
- controlled robot spawn region and heading interval;
- teammate/opponent activation and bounded spawn regions;
- target goal, target zone, or receiving corridor;
- difficulty dimensions and deterministic seed;
- horizon and predicate thresholds.

A compiler converts parameters into a canonical snapshot, followed by the
existing overlap, finite-value, and field-boundary validation. The generated
snapshot and source parameters are both hashed into the artifact. Runtime
training receives only the canonical state and typed drill context; the physics
backend remains unaware of curriculum policy.

## Decision 2: use causal physical predicates

Predicates consume the initial context and the authoritative transition/event
stream. They return `running`, `success`, `failure`, or `unresolved` plus
versioned reason codes.

| Family | Success | Failure | Important diagnostics |
| --- | --- | --- | --- |
| Approach | controlled contact with bounded impact and improved facing | timeout or moving away persistently | time-to-contact, heading error, wheel jerk |
| Interception | controlled touch before the predicted threat crossing, followed by a non-threatening trajectory | conceded goal or threat crosses before touch | interception time, post-touch miss distance |
| Save/deflection | an initially goal-bound ball becomes stopped or no longer intersects the full-ball goal aperture | full-ball goal | speed removed, angular deflection, clearance distance |
| Clearance | controlled touch moves the ball out of the defensive danger zone without immediate re-entry | conceded goal or timeout in danger | exit time, territorial displacement |
| Shot | full-ball goal; optionally a separate nonterminal on-target diagnostic | miss, own goal, or timeout | shot speed, aperture margin |
| Pass/receive | distinct allied contact chain reaches the receiving corridor without opponent touch and produces controlled reception | interception, out-of-corridor timeout, or own goal | pass progress, arrival speed, receiver control |

An isolated touch is never sufficient for interception, save, clearance, or
pass success. Trajectory tests use present authoritative position and velocity
after the contact; they do not reveal future simulator state to the policy.

## Decision 3: terminate drills at semantic resolution

The environment evaluates predicates after every physical control decision.
Success and failure terminate immediately after any required short confirmation
window. Timeout becomes `unresolved`, not an implicit failure, unless the
family explicitly declares it a failure.

Early termination reduces dead time and prevents post-success behavior from
changing attribution. Full matches retain their existing goal grace period and
termination rules.

## Decision 4: progress through bounded difficulty axes

Each family declares ordered difficulty axes rather than one opaque level:

- ball speed and approach angle;
- initial robot-to-intercept distance;
- distance to goal line;
- target corridor/aperture width;
- wall/rebound involvement;
- active opponents and their pressure;
- observation latency, noise, and dropout only after exact-state mastery.

The teacher targets scenarios whose recent success is between 15% and 85% and
prioritizes absolute learning progress. Mastered buckets remain in bounded
rehearsal. Failed replay descriptors may allocate an equivalent family and
difficulty bucket, but cannot modify its predicate or reward.

Difficulty changes start a versioned scenario lineage. Immutable holdouts cover
interpolation and extrapolation in speed, angle, spawn, and color.

## Decision 5: mix drills with full matches

M15 introduces an explicit allocation contract:

- 50% semantic skill frontier;
- 20% routine/mastery rehearsal;
- 20% full 3v3 matches sampled from the existing league;
- 10% deduplicated failure rehearsal.

Immutable holdouts run only during evaluation and never contribute gradients.
The exact percentages remain configuration values and require telemetry; the
default above must be compared against M14 and a full-match-heavy control.

Team membership is canonical but physical robot identity does not select a
permanent attacker, defender, goalkeeper, passer, or receiver policy.

## Decision 6: keep outcome signals distinct from default reward

Every drill emits:

- base MAPPO reward components;
- bounded skill outcome reward;
- semantic terminal outcome and reason;
- reward-independent diagnostics.

Goal and concession remain strongest. Skill outcome rewards are normalized so
that accumulating intermediate events cannot exceed the terminal objective.
Before promotion, an ablation compares:

1. M14 reward with semantic resets and predicates only;
2. predicates plus terminal skill outcome reward;
3. any proposed dense skill shaping.

Dense shaping is rejected if it improves drill metrics without improving paired
terminal full-match outcomes, or if event counts rise while causal quality
falls.

## Decision 7: evaluate coverage, learning, and transfer separately

The skill evaluator uses paired colors and independent seeds. It reports
per-family success/failure/unresolved rates, Wilson or bootstrap intervals,
time-to-resolution, post-contact trajectory quality, physical invalidity, and
throughput.

Curriculum acceptance requires:

- deterministic regeneration and mirror invariance;
- zero invalid or initially overlapping states;
- all declared difficulty cells sampled;
- improvement over random and deterministic heuristic controls on learnable
  families;
- no regression beyond configured confidence floors in full-match evaluation;
- no material throughput regression after accounting for earlier termination.

Training return cannot satisfy these gates.

## Data and artifact contracts

Scenario artifacts include schema version, family, difficulty vector, seed,
controlled team, canonical-state hash, parameter hash, parent lineage, and
generator revision.

Replay headers and terminal records include scenario identity and semantic
outcome. Metrics aggregate family/difficulty allocation and outcome counts.
The viewer can filter by family, difficulty, success/failure/unresolved, and
terminal reason and can jump to the resolving event.

## Compatibility

Existing M14 suites continue to load as version-1 static scenarios. They are
treated as legacy states without semantic outcome predicates and cannot be
selected by an M15 default configuration until migrated. Existing checkpoints
remain loadable because scenario semantics do not alter actor tensor shapes.

The M13 and M14 configurations remain immutable comparison controls. M15 uses a
new configuration and policy lineage.

## Validation

- Property tests cover deterministic generation, mirroring, bounds, and
  non-overlap over thousands of seeds.
- Golden traces cover each predicate's success, failure, unresolved, and
  adversarial near-miss cases.
- Vector tests prove one world's semantic termination/reset cannot affect
  another world.
- Reward tests prove raw touches and repeated contacts cannot farm terminal
  skill reward.
- Replay/metrics contract tests cover partial writes, resume, and attribution.
- CPU/CUDA smoke studies record frames/s, matches/s, resolved drills/s, and
  total compute.
- Promotion uses paired multi-seed full-match and immutable skill holdouts.

## Rollback

Disable the M15 scenario suite and select the immutable M14 configuration.
Scenario artifacts and outcome records are additive; rollback does not rewrite
checkpoints or canonical replays. A failed M15 candidate is rejected through
the existing promotion artifact, leaving the registry incumbent unchanged.
