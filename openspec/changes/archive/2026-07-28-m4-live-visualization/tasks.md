## 1. Contracts and documentation

- [x] 1.1 Record observer delivery and renderer separation in ADR-0006
- [x] 1.2 Add M4.1 scope, architecture, non-goals, and gate to the PRD
- [x] 1.3 Specify visual frames, sinks, viewer behavior, and replay adaptation
- [x] 1.4 Survey VSSS, RLGym, RocketSim, and detached viewer architectures

## 2. Observer foundation

- [x] 2.1 Add a backend-neutral visual-frame adapter outside `vsss-spec`
- [x] 2.2 Implement null, lossless replay, bounded latest-frame, and metrics sinks
- [x] 2.3 Integrate optional sinks into scripted match execution
- [x] 2.4 Add contract tests for monotonic frames, drop accounting, lossless
  recording, and checksum equivalence

## 3. Shared 2D projection

- [x] 3.1 Decode existing replay ticks into the shared visual-frame model
- [x] 3.2 Project exact frames into deterministic field and entity primitives
- [x] 3.3 Add deterministic headless rendering and golden artifact tests
- [x] 3.4 Add live-source integration tests using a deliberately slow consumer

## 4. Interactive viewer

- [x] 4.1 Add a leaf Bevy viewer crate without changing core crate dependencies
- [x] 4.2 Implement native replay pause, seek, exact step, and speed controls
- [x] 4.3 Add identifiers, headings, velocities, actions, trajectories, rewards,
  events, and dropped-frame overlays
- [x] 4.4 Add a bounded live connection and reuse the replay scene projection
- [x] 4.5 Verify headless parser/playback tests and document the WASM/client path

## 5. Evidence and delivery

- [x] 5.1 Benchmark watched and unwatched execution and record observer overhead
- [x] 5.2 Run `openspec validate --strict`, `just doctor`, `just lint`,
  `just build`, and `just test`
- [x] 5.3 Record artifacts, known limitations, compatibility, and rollback
- [x] 5.4 Create small signed Conventional Commits and merge without requiring a
  long-lived pull request
