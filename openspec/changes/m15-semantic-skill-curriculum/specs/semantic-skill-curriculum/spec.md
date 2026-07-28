## ADDED Requirements

### Requirement: Deterministic semantic scenario generation
The system SHALL compile versioned typed skill parameters into physically valid
canonical initial states deterministically, including ball motion, robot
placement, target geometry, controlled color, difficulty, and seed.

#### Scenario: Regenerate a moving interception drill
- **GIVEN** the same interception parameters, generator revision, and seed
- **WHEN** the scenario is compiled more than once
- **THEN** the canonical snapshots and parameter hashes are byte-identical

#### Scenario: Mirror a scenario across team colors
- **GIVEN** a valid blue-controlled scenario
- **WHEN** it is mirrored for yellow control
- **THEN** positions, velocities, headings, teams, target goal, and predicates
  are transformed consistently without binding a skill to a robot identity

#### Scenario: Generated state starts in invalid contact
- **GIVEN** parameters that place two bodies in overlap or outside the field
- **WHEN** compilation validates the canonical snapshot
- **THEN** the state is rejected before it can enter a rollout

### Requirement: Skill-specific causal outcomes
The system SHALL evaluate versioned skill predicates from initial drill context
and authoritative observed transitions as running, success, failure, or
unresolved with an inspectable reason code.

#### Scenario: Useful interception
- **GIVEN** a ball whose current trajectory intersects the controlled team's
  full-ball goal aperture
- **WHEN** an allied robot touches it before crossing and the confirmed
  post-touch trajectory no longer intersects that aperture
- **THEN** the drill terminates as interception success

#### Scenario: Farmed touch does not save
- **GIVEN** a goal-bound ball
- **WHEN** an allied robot touches it but the confirmed trajectory still enters
  the controlled goal
- **THEN** the touch is diagnostic only and the drill terminates as failure

#### Scenario: Controlled pass reception
- **GIVEN** distinct allied passer and receiver contacts with no intervening
  opponent touch
- **WHEN** the ball reaches the declared receiving corridor at bounded arrival
  speed and the receiver controls it
- **THEN** the drill terminates as pass/receive success

#### Scenario: Drill times out without resolution
- **GIVEN** no declared success or failure before the semantic horizon
- **WHEN** the horizon expires
- **THEN** the drill terminates as unresolved rather than being relabelled from
  shaped reward

### Requirement: Semantic early termination
The system SHALL stop an atomic drill after its predicate resolves and SHALL
preserve existing full-match goal grace and termination behavior.

#### Scenario: Save resolves before the maximum horizon
- **GIVEN** an initially goal-bound ball
- **WHEN** it is stopped outside the danger zone for the configured
  confirmation window
- **THEN** the drill ends immediately and subsequent actions cannot alter its
  attribution

#### Scenario: One vector world resolves
- **GIVEN** multiple independent vector worlds
- **WHEN** one world's skill predicate resolves
- **THEN** only that world resets and neighboring physics, memory, scenario,
  and outcome state remain unchanged

### Requirement: Learning-progress difficulty curriculum
The system SHALL allocate semantic skills and independent bounded difficulty
axes using rolling success and learning progress while retaining routine,
failure, and full-match rehearsal.

#### Scenario: Central slow interceptions are mastered
- **GIVEN** a bucket whose paired rolling success exceeds its mastery threshold
- **WHEN** the teacher allocates new frontier scenarios
- **THEN** it increases one declared difficulty axis or selects another
  learnable bucket while retaining bounded mastery rehearsal

#### Scenario: Immutable holdout is evaluated
- **GIVEN** a holdout speed/angle/color cell
- **WHEN** holdout evaluation runs
- **THEN** its transitions contribute metrics but never gradients, reward
  search, or curriculum mutation

#### Scenario: Atomic drills dominate allocation
- **GIVEN** a configuration whose observed full-match share falls below its
  declared floor
- **WHEN** allocation is validated
- **THEN** training is rejected or rebalanced before collecting gradients

### Requirement: Anti-farming skill rewards
The system SHALL attribute any skill reward separately from base reward,
bound it below decisive terminal objectives, and reject it unless an ablation
improves paired full-match outcomes without degrading causal event quality.

#### Scenario: Repeated harmless contacts increase
- **GIVEN** a candidate that produces more touches or pass-like contacts
- **WHEN** post-contact threat, reception control, and terminal outcomes do not
  improve
- **THEN** the shaping arm is rejected and cannot become the default reward

#### Scenario: Predicates improve training without extra shaping
- **GIVEN** semantic resets and early termination with the M14 reward
- **WHEN** they improve learnability and full-match transfer as much as a dense
  shaping arm
- **THEN** the simpler predicates-only arm is preferred

### Requirement: Paired skill and transfer evaluation
The system SHALL evaluate every candidate over paired colors, independent
seeds, immutable skill holdouts, full matches, heuristic play, the promoted
baseline, and historical policies before authorizing a high-budget run.

#### Scenario: Skill success rises but match play regresses
- **GIVEN** a candidate with higher interception and pass drill success
- **WHEN** its paired full-match confidence floor regresses against M14
- **THEN** promotion and the high-budget run are rejected

#### Scenario: Candidate clears entry gates
- **GIVEN** deterministic valid generation, complete difficulty coverage,
  learnability above controls, non-regressed full matches, and recorded
  throughput
- **WHEN** the promotion evaluator accepts the candidate
- **THEN** it writes a machine-readable decision and the exact high-budget run
  command may be published

### Requirement: Inspectable semantic artifacts
The system SHALL record scenario family, version, parameter and state hashes,
difficulty, seed, controlled color, predicate state, terminal reason, and
outcome in metrics and replay artifacts.

#### Scenario: Inspect a failed save in the viewer
- **GIVEN** a captured save/deflection drill
- **WHEN** the user selects its failed semantic outcome
- **THEN** the viewer identifies the initial threat, controlled robot,
  resolving contact or goal, difficulty, and terminal reason on the timeline

#### Scenario: Resume an interrupted semantic run
- **GIVEN** a checkpointed run with curriculum histories and deduplicated
  failures
- **WHEN** training resumes with the same configuration
- **THEN** scenario allocation, difficulty state, lineage, and outcome counters
  continue without reclassifying completed drills
