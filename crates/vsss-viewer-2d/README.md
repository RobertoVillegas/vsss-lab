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

This devbox is headless, so native window execution belongs on a client with a
display. A WASM target is planned after live transport is stable; the viewer
crate intentionally contains no physics and can reuse the same playback state
and scene systems there.
