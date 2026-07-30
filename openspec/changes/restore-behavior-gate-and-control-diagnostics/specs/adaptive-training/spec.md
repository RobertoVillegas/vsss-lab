## ADDED Requirements

### Requirement: Parser-independent behavior detection

Behavior detection SHALL remain reachable for every supported action parser. Turn
intensity SHALL be reported as a fraction of the turn authority the active parser
can request, and behavior thresholds SHALL come from the run configuration rather
than from a copy held by the evaluator.

#### Scenario: Turn-in-place through a skill parser

- **GIVEN** a policy whose wheels are produced by a geometric controller
- **WHEN** it requests the largest turn-in-place that controller can execute while
  remaining slow and remote from the ball
- **THEN** the idle-spin flag is raised
- **AND** the reported idle-spin ratio can exceed the configured ceiling

#### Scenario: Threshold retuned in configuration

- **WHEN** a run changes an idle-spin threshold in its configuration
- **THEN** the behavior gate of its paired evaluation observes the new value

#### Scenario: Wheel-space policy is unaffected

- **GIVEN** a policy that emits wheel commands directly
- **WHEN** the same configured threshold is applied
- **THEN** detection behaves as it did before turn intensity was normalized

### Requirement: One decision executes one opponent parse

A learned opponent SHALL have its action parsed once per decision, as the learner
does, so that training, paired evaluation, and replay execute one token
identically. A scripted controller MAY re-plan per physics substep.

#### Scenario: Learned opponent in a paired match

- **WHEN** a checkpoint acts as the opponent for one decision
- **THEN** its wheel commands are the single parse of its token against the state
  observed at that decision

#### Scenario: Recorded opponent intent

- **WHEN** a replay records the opponent's intent for a decision
- **THEN** that record describes the command executed for the whole decision

### Requirement: Diagnostics describe the acting policy

Reported control diagnostics SHALL exclude agents absent from the field, SHALL not
compare headings across an episode boundary or for an undirected vector, and SHALL
expose exploration deviation and skill mix for a hybrid policy. The paired
evaluation record SHALL carry its resolved throughput.

#### Scenario: Roster smaller than the team

- **GIVEN** a scenario with fewer controlled robots than roster slots
- **WHEN** action statistics are reported
- **THEN** slots without a robot on the field contribute nothing to them

#### Scenario: Heading change across a reset

- **WHEN** a decision ends its episode
- **THEN** no heading change is reported between it and the next decision

#### Scenario: Hybrid policy inspection

- **WHEN** a run trains a policy with both categorical and continuous parameters
- **THEN** its exploration deviation and its skill mix are both available for
  inspection
- **AND** its paired evaluation record reports resolved drills per second
