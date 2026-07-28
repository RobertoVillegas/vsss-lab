## Why

Sustained runs need human-readable progress and a safe stop path. The browser is
an optional observer and must not become a training dependency.

## What Changes

- Print iteration return, progress, throughput, ETA, and checkpoint events.
- Convert SIGINT/SIGTERM into a graceful stop after the current iteration.
- Persist and register the last completed iteration before exiting.
- Keep trainer-only and viewer-only commands and add combined convenience commands.

## Capabilities

### Modified Capabilities

- `self-play-run-capture`: Add observable progress and graceful interruption.

## Impact

Only local orchestration and documentation change. The web server remains an
independent read-only process.
