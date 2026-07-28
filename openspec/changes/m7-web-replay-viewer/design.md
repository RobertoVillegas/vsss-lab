## Context

M7 writes immutable viewer-compatible JSONL files under `<run>/replays`.
The Bevy viewer provides native playback and the Python renderer provides stable
SVG output, but neither provides run-wide navigation. The intended environment
is WSL2/Linux with the browser on Windows.

## Goals / Non-Goals

**Goals:**

- Inspect every captured iteration through one private local URL.
- Keep replay parsing and playback independent from simulation and training.
- Preserve exact recorded-frame navigation and make keyboard operation useful.
- Build and test the frontend reproducibly with repository-pinned tooling.

**Non-Goals:**

- Live rollout streaming, replay mutation, training control, authentication,
  public hosting, 3D rendering, or replacing the native viewer.

## Decisions

1. Use React, TypeScript, and Vite 8 for a single feature-oriented screen.
   TanStack Router/Start are deferred because there is one route and no
   server-side or authentication boundary.
2. Use the Python standard library for the run API and production static
   serving. It validates filenames and binds to `127.0.0.1` by default. This
   avoids a second application runtime and keeps the server beside existing
   replay tooling.
3. Fetch one JSONL replay on iteration selection and parse it in the browser.
   Current captures are small enough for responsive local use; streaming and
   indexing are deferred until measured runs require them.
4. Render from canonical snapshot values on Canvas. Playback only selects
   recorded frames and never advances physics. Canvas resizes for HiDPI while
   preserving SI-space field proportions.
5. Keep Vite development proxying separate from production serving. The normal
   `just league-web` path builds assets and exposes one same-origin loopback URL.
6. Use a purpose-built frame transport rather than Remotion Player, Rive, or
   Video.js. Remotion targets renderable React video compositions, Rive consumes
   authored `.riv` state machines, and Video.js controls media elements. Their
   playback ergonomics inform this UI, while canonical JSONL remains the direct
   source and Remotion remains a future option for encoded video export.

## Risks / Trade-offs

- Large captures may pause during parsing → show loading/error state and add
  streaming/indexing only after measurement.
- Browser and native drawings can drift cosmetically → test geometry mapping
  and treat JSONL snapshots, not pixels, as the shared contract.
- A bound service could expose run artifacts → default and documented command
  use loopback only; file resolution is constrained to discovered replay names.

## Migration Plan

Additive only. Build the web assets, launch the loopback server, and retain all
existing commands. Rollback removes the web directory, Python server, and its
recipes without touching replay or run artifacts.

## Open Questions

Video/GIF export and live tailing can be considered after interactive replay
inspection is exercised on longer training runs.
