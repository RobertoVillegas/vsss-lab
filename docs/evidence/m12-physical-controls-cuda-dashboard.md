# M12 physical controls, CUDA, and terminal observability

## Outcome

The reference simulator now encloses robots at goal side/back walls, applies a
force-derived wheel-speed slew limit, penalizes abrupt normalized action
changes, and exposes commanded versus applied wheels in replay telemetry.

Training supports `auto`, `cpu`, and `cuda` devices plus configurable vector
worlds. `auto` prefers CUDA and warns visibly if it falls back. The Rich terminal
dashboard keeps its progress bar below a table of current and rolling metrics.

## Physical provenance

Julio de la Torre's public VSSS model informed the geometry and actuator
interpretation:

- field collision model: <https://github.com/juliodltv/simulation_vsss/blob/main/models/vss_field/model.sdf>
- robot dimensions, 25 mm wheel radius, 68 rad/s wheel limit, effort, damping,
  and friction: <https://github.com/juliodltv/simulation_vsss/blob/main/urdf/robot.urdf>
- Gazebo wheel contact parameters: <https://github.com/juliodltv/simulation_vsss/blob/main/urdf/robot.gazebo>

The implementation does not copy meshes or textures. It uses canonical SI
parameters already recorded by the M9 calibration work.

## Measurements

On the WSL2 reference host with an RTX 3070, a full one-iteration run with 16
worlds produced 48,000 agent frames:

| Device | Worlds | Throughput |
|---|---:|---:|
| CUDA | 16 | 1,832 frames/s |
| CPU | 16 | 2,748 frames/s |

The model is small and Rapier stepping remains CPU/sequential, so CUDA kernel
launches and transfers are not yet amortized. CUDA is therefore the requested
default when available, while the dashboard exposes the selected device and
measured throughput so users can make an informed override.

The actuator test limits one 5 ms physics step to 0.8 rad/s, or at most 3.2
rad/s during a 20 ms control interval. A captured post-change replay showed no
robot outside the field/goal enclosure; the previous run exhibited 40–50 rad/s
command discontinuities and escaped robots.

The M9 straight/turn golden references now include the deterministic actuator
ramp: 0.47 m after one second at a 20 rad/s target and -2.358713 rad after
0.5 seconds at opposing 10 rad/s targets. Passive-ball damping is unchanged.

## Verification

- six Rapier correctness tests, including actuator slew and goal containment
- 22 focused Python league/MARL/web tests
- Ruff and strict mypy
- TypeScript build
- real CUDA bootstrap, rollout, PPO update, checkpoint, and clean exit
