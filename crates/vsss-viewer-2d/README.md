# VSSS 2D viewer

Native Bevy 0.19 replay viewer for the fast simulator. It is a leaf workspace
crate: physics, canonical contracts, Python bindings, and training do not depend
on it.

Generate a replay and launch the viewer on a graphical client:

```bash
just match-scripted 1000 reports/m4-scripted.jsonl
cargo run -p vsss-viewer-2d -- reports/m4-scripted.jsonl
```

Controls:

- `Space`: play or pause;
- left/right arrows: exact previous/next recorded tick and pause;
- `Home`/`End`: seek to first/last frame;
- `=`/`-`: double/halve playback speed between 0.25x and 16x.

The scene shows field geometry, robots, headings, velocity vectors, wheel-action
indicators, ball velocity, recent ball trajectory, score, events, tick, and
simulation time. The existing SVG renderer remains the deterministic headless
path.

For a native live view, start the listener before the match producer:

```bash
just replay-view-live 127.0.0.1:42042
just match-live 10000 reports/m4-live.jsonl 127.0.0.1:42042
```

Live packets are sampled, compressed below the local MTU, sequenced, and lossy.
The title reports gaps detected by the receiver and local producer send errors.
The full JSONL replay remains the authoritative lossless record.

This devbox is headless, so window execution belongs on a graphical client. The
same crate passes:

```bash
just viewer-wasm-check
```

Browser replay rendering therefore compiles today. UDP live sockets are native
only; a later WebSocket adapter can feed the same `TickRecord` on the web
without changing physics or scene projection.
