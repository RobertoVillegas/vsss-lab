## ADDED Requirements

### Requirement: Run replay discovery
The local web viewer SHALL discover captured iteration replays from the
configured run directory and expose only valid replay filenames.

#### Scenario: Open a captured run
- **WHEN** the viewer server starts for a run containing iteration replay files
- **THEN** the browser lists those iterations in numeric order

### Requirement: Browser playback
The local web viewer SHALL render canonical snapshot geometry and support
iteration selection, play/pause, speed selection, exact frame stepping,
forward/backward skipping, and timeline seeking.

#### Scenario: Seek a paused iteration
- **WHEN** a developer selects a timeline position while playback is paused
- **THEN** the canvas renders the exact recorded frame at that position without advancing physics

### Requirement: Private local serving
The viewer server SHALL bind to loopback by default and SHALL constrain replay
reads to the configured run's replay directory.

#### Scenario: Request an unknown replay
- **WHEN** a client requests a filename that was not discovered in the run
- **THEN** the server rejects the request without reading another filesystem path
