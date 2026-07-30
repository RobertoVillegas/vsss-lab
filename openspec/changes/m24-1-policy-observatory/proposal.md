# M24.1 — Policy Observatory

## Problem

The Replay Studio exposes applied wheel speeds, roles, physics, and camera
layers, but M24 policies choose categorical soccer primitives. A developer
cannot currently tell whether a bad trajectory came from a poor policy choice
or from primitive execution. The left sidebar also dedicates most of its space
to vision controls that are not useful during simulation training.

## Change

- Record the policy's categorical primitive, confidence, alternatives, target,
  phase, and exit direction independently from actuator commands.
- Replace the unused lower-left vision panel with a compact run/replay summary.
- Make actor cards selectable and reveal strategic and actuator detail for the
  selected robot.
- Overlay the selected primitive target and requested exit direction on the
  field.
- Add a multi-lane primitive timeline with clickable event markers.
- Replace Gaussian log-standard-deviation charts for primitive policies with
  categorical entropy and primitive-use telemetry.
- Expose curriculum phase, allocation, rehearsal, roster, and difficulty
  progression in the training dashboard.
- Self-host Geist Mono through the existing Bun/Vite dependency boundary.

## Non-goals

- Changing rewards or stopping training automatically from viewer warnings.
- Making camera-estimation layers policy-visible.
- Reconstructing primitive intent for historical v1 replays that did not record
  it.
