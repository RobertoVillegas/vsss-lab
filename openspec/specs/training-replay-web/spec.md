# training-replay-web Specification

## Purpose
TBD - created by archiving change m7-web-replay-viewer. Update Purpose after archive.
## Requirements
### Requirement: Run replay discovery

The local web viewer SHALL poll the configured run for canonical completed
replays, checkpoints, and complete metric records without exposing arbitrary
filesystem paths.

#### Scenario: Open a captured run

- **WHEN** the viewer server starts for a run containing iteration replay files
- **THEN** the browser lists those iterations in numeric order

#### Scenario: Trainer publishes a new capture

- **WHEN** a newer canonical replay appears while live-follow is enabled
- **THEN** the browser selects it within one polling interval and begins looped playback

### Requirement: Browser playback

The local web viewer SHALL render canonical snapshot geometry; support
iteration selection, play/pause, loop, speed, exact stepping, skipping, and
seeking; and distinguish recorded simulation time from inspection speed.

#### Scenario: Seek a paused iteration

- **WHEN** a developer selects a timeline position while playback is paused
- **THEN** the canvas renders the exact recorded frame at that position without advancing physics

#### Scenario: Inspect at recorded real-time speed

- **WHEN** the operator selects 1× playback
- **THEN** frames advance according to the replay's canonical control period

#### Scenario: Inspect a historical capture

- **WHEN** the operator manually selects an earlier iteration
- **THEN** live-follow pauses until explicitly resumed

### Requirement: Private local serving
The viewer server SHALL bind to loopback by default and SHALL constrain replay
reads to the configured run's replay directory.

#### Scenario: Request an unknown replay
- **WHEN** a client requests a filename that was not discovered in the run
- **THEN** the server rejects the request without reading another filesystem path

### Requirement: Actor control telemetry

The viewer SHALL display the recorded wheel commands, normalized intensity,
derived linear speed, turn rate, and direction for every actor at the selected
frame.

#### Scenario: Scrub control decisions

- **WHEN** the operator seeks to another frame
- **THEN** all actor telemetry updates from that exact frame's recorded actions

### Requirement: Capture result classification

The viewer SHALL classify completed captures by score, win/loss/draw from the
blue policy perspective, and whether any goal occurred.

#### Scenario: Find informative captures

- **WHEN** the operator filters for wins, losses, draws, or captures with goals
- **THEN** the iteration picker contains only matching completed captures

### Requirement: Stable-width live telemetry
The replay viewer SHALL use monospace tabular-number typography for rapidly
changing controls, metrics, timestamps, and actor telemetry.

#### Scenario: Numeric telemetry changes
- **WHEN** adjacent replay frames contain different digits
- **THEN** the surrounding labels and columns retain stable glyph widths

