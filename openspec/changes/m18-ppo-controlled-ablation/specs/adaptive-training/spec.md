## ADDED Requirements

### Requirement: Matched-compute PPO architecture comparison

The experiment system MUST compare policy variants with paired seeds and equal
environment steps while recording parameter count, throughput, PPO diagnostics,
terminal outcomes, and semantic-family outcomes.

#### Scenario: Compare width without hidden confounders

- **WHEN** 128 and 256 hidden-unit arms are evaluated
- **THEN** activation, normalization, rewards, seeds, rollouts, and evaluation
  holdouts remain identical

#### Scenario: Screen a Rocket League-inspired stack

- **WHEN** ReLU and LayerNorm are evaluated
- **THEN** a width-only and normalization-only arm remain in the report
- **AND** the combined arm is not credited solely to width

### Requirement: Causal useful-touch reward

The environment MUST optionally reward a controlled contact that increases ball
velocity toward the opponent goal without rewarding persistent overlap.

#### Scenario: Persistent ball overlap

- **WHEN** the same robot remains in ball contact across decisions
- **THEN** useful-touch reward is emitted only on contact entry

#### Scenario: Touch sends ball toward own goal

- **WHEN** contact increases velocity toward the controlled team's own goal
- **THEN** useful-touch reward is zero
