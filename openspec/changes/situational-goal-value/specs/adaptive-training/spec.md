## ADDED Requirements

### Requirement: A goal is worth what it changes

The reward for a goal SHALL depend on the match situation it occurs in, and SHALL be composed of
a term proportional to the change in win probability and a flat term for goal difference. A goal
SHALL NOT be worth the same amount in every situation.

#### Scenario: Protecting a narrow lead late

- **GIVEN** a one-goal lead with fifteen seconds of match time remaining
- **WHEN** the team concedes
- **THEN** the penalty is larger than the reward the same team would receive for scoring in
  that situation, by the ratio the win-probability model gives

#### Scenario: A lead early in the match

- **GIVEN** a one-goal lead with five minutes remaining
- **WHEN** scoring and conceding are compared
- **THEN** their magnitudes are close, so the situation does not by itself favour defending

#### Scenario: A rout

- **GIVEN** a lead large enough that the win probability is saturated
- **WHEN** the team scores again
- **THEN** the win-probability term contributes approximately nothing and the flat term
  continues to reward goal difference

### Requirement: The win-probability model is fixed and legible

The win probability SHALL be a fixed function of the lead and the time remaining, parameterized
by scoring rates that are measured once from symmetric self-play, recorded as evidence, and held
constant for the run. The rates SHALL NOT be re-estimated from the run they are used in.

#### Scenario: Rates drift during a run

- **WHEN** the policy's scoring rate changes as it improves
- **THEN** the reward's rates are unchanged, so the reward remains a fixed function of state

#### Scenario: Reward invariance

- **WHEN** the win-probability term is summed over an episode
- **THEN** it telescopes to the change in win probability across the episode, so it is bounded
  and cannot be farmed by repetition

### Requirement: Rates from an unbalanced matchup are not used

Scoring rates measured against an opponent the policy dominates SHALL NOT parameterize the
model, because the win probability saturates and leaves no gradient in the region the reward is
meant to shape.

#### Scenario: Reusing an existing measurement

- **WHEN** rates measured at 0.473 for and 0.024 against are proposed
- **THEN** they are rejected, because under them a one-goal lead already wins with probability
  0.99
