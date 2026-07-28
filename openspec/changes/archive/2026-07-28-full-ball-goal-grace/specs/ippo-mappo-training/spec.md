## ADDED Requirements

### Requirement: Post-goal closure
The match environment SHALL continue for the configured one-second goal pause
after a goal event before reporting match termination.

#### Scenario: Goal at 50 Hz control
- **WHEN** a goal event occurs with a one-second configured pause
- **THEN** the event is rewarded once and termination occurs after 50 control
  frames
