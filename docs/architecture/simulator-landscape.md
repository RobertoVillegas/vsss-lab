# VSSS and fast-simulation landscape

Date: 2026-07-27

## Decision summary

The survey reinforces ADR-0006: VSSS Lab should keep a standalone fast physics
engine and expose canonical state to optional visual consumers. A viewer is not
a second physics backend. ROS/Gazebo or Webots remain useful as independent
higher-fidelity calibration targets.

The closest implementation precedent is RocketSim plus RLViser: a fast
standalone simulator paired with a lightweight Rust/Bevy visualizer that listens
for state packets. VSS-SDK independently demonstrates the same simulator/viewer
split in the VSSS domain.

## Compared systems

| System | Physics and execution | Visualization boundary | What VSSS Lab should reuse |
|---|---|---|---|
| [simulation_vsss](https://github.com/juliodltv/simulation_vsss) | ROS Noetic and Gazebo, 3D differential robots, ball, camera, 3v3 | Gazebo GUI and ROS camera/topics | Dimensions, assets, launch scenarios, camera placement, joystick/manual test cases and M9/M10 calibration evidence |
| [Classic TraveSim](https://github.com/ThundeRatz/travesim) | ROS/Gazebo with differential-drive or direct wheel control | Gazebo and ROS topics | Control-mode equivalence scenarios and motor-controller calibration; not its ROS 1 runtime |
| [Current TraveSim](https://github.com/futebol-mini/travesim) | Webots, 3v3/5v5, physical robot and motor model | Webots GUI; VSSProto network boundary | VSSProto compatibility research, published physical parameters and external-team/replacer topology |
| [VSS-Simulator](https://github.com/VSS-SDK/VSS-Simulator) + [VSS-Viewer](https://github.com/VSS-SDK/VSS-Viewer) | Standalone Bullet simulator and referee | Separate viewer consumes state from simulator, vision and strategies through VSS-Core | The domain-proven separation of authoritative state producer and interchangeable visual consumer |
| [rSim](https://github.com/robocin/rsim) + [rSoccer](https://github.com/robocin/rSoccer) | ODE-based VSS/SSL simulation with direct `step`, `reset`, `get_state`; Gymnasium environments | Gymnasium `render_mode="human"` at the environment layer | Benchmark tasks, state/action comparison fixtures and environment API behavior; avoid inheriting old flattened state layouts |
| [FIRASim](https://github.com/fira-simurosot/FIRASim) / [grSim](https://github.com/RoboCup-SSL/grSim) | Competition-oriented robot-soccer simulation derived from grSim | Integrated client plus network messages | Protocol and referee interoperability fixtures, not the hot-loop architecture |
| [VSSS-RL paper](https://arxiv.org/abs/2003.11102) | Purpose-built VSSS training environment and sim-to-real experiments | Evaluation-oriented visualization | Skill benchmarks, transfer/randomization assumptions and comparison experiments |
| [RocketSim](https://github.com/ZealanL/RocketSim) | Standalone high-throughput Bullet-derived C++ simulation; accuracy is sufficient for feedback-driven ML rather than exact long-horizon replication | No required renderer in the simulation core | Throughput-first engine, consistent feedback, explicit accuracy envelope and backend-independent state |
| [RLGym](https://rlgym.org/Getting%20Started/overview/) | Replaceable transition engine, mutators, actions, observations, rewards and termination | Renderer is an optional configuration object | Keep rendering optional and invoked only for watched evaluation |
| [RLViser](https://github.com/VirxEC/rlviser) | Does not own physics | Lightweight Rust/Bevy process listens for UDP state packets; supports pause, speed, cameras and state editing | Strong precedent for the proposed Bevy leaf viewer and lossy live transport |

## Architectural findings

### State snapshots are required in addition to events

Goals, collisions, resets, and touches are useful annotations, but they cannot
animate continuous motion. A visual consumer needs sampled poses and velocities
plus discrete events. The authoritative replay must remain lossless; a live
viewer may keep only the newest snapshot.

### One frame model should serve live and replay

VSS-SDK lets its viewer consume simulator and vision states, while RLViser
accepts packets independently of the producer language. VSSS Lab should go one
step further: decode both replay records and live delivery into the same
`VisualFrame` before scene projection.

### Bevy remains a justified choice, not yet a core dependency

RLViser is a working Rust/Bevy precedent for a high-speed soccer simulator
viewer. VSSS Lab should keep the renderer in a leaf crate and prove frame
semantics, headless deterministic projection, and backpressure behavior before
adding the full Bevy dependency.

### Transport should be replaceable

UDP is appropriate for lossy local live state and has direct precedent in
RLViser. WebSocket is friendlier for a remote browser client. Replay files are
better for exact seeking. The frame contract must not choose among them; an
adapter benchmark should choose the first live transport.

### High fidelity remains a separate validation problem

The Gazebo and Webots VSSS projects contain valuable camera, controller, motor,
geometry, and collision behavior. They are calibration sources and later
backends, not renderers for Rapier. This avoids synchronizing two physical
worlds merely to display one of them.

## Resulting implementation order

1. Canonical visual-frame adapter and bounded observer sinks.
2. Deterministic replay-to-SVG projection as a graphics smoke test.
3. Lossy local live adapter with drop accounting.
4. Bevy native viewer reusing the exact projection model.
5. Browser/WASM or Rerun adapter after measurement.
6. Gazebo/Webots comparison scenarios during M9/M10.

## Legal and compatibility notes

Review licenses per artifact before importing code or assets. Prefer behavioral
fixtures and independently authored adapters. In particular, VSS-Viewer is
GPL-3.0 while its architecture can be studied without copying implementation;
the VSS-Simulator, TraveSim, simulation_vsss, RocketSim and RLViser repositories
advertise permissive licenses, but individual assets and transitive components
still require provenance review.
