# RL wheel-action scale and playback clock

## Finding

The original RL adapter clipped policy outputs to `[-1, 1]` and sent those
values directly to a backend whose wheel command unit is radians per second.
The resulting maximum linear speed was only `0.025 m/s`:

`0.025 m wheel radius × 1 rad/s = 0.025 m/s`.

Iteration 50 from the pre-fix run confirmed the symptom. Its actors emitted
large normalized actions (`0.51` to `0.91` mean absolute action) but travelled
only `0.013` to `0.023 m/s` on average. Those checkpoints do not represent a
physically useful controller.

The corrected boundary maps normalized policy actions to the configured
physical wheel limit. The reference limit is now `30 rad/s`, producing a
nominal straight-line maximum of `0.75 m/s`. A post-fix 60-second smoke capture
measured `0.56` to `0.62 m/s` average path speed for the blue actors and a 2–0
score.

## External comparison

Julio de la Torre's Gazebo simulator uses the same `25 mm` wheel radius. Its
joystick controller normally clips wheel commands to `30 rad/s`, while its URDF
declares a `68 rad/s` joint limit and `0.073 Nm` effort limit. The Gazebo world
sets `real_time_factor` to `0.5`; that changes wall-clock execution speed, not
the simulated time represented by the physics steps.

Sources:

- [Julio's robot wheel geometry and limits](https://github.com/juliodltv/simulation_vsss/blob/main/urdf/robot.urdf)
- [Julio's differential-drive command conversion](https://github.com/juliodltv/simulation_vsss/blob/main/scripts/joystick.py)
- [Julio's Gazebo real-time factor](https://github.com/juliodltv/simulation_vsss/blob/main/worlds/vss_field_camera.world)
- [Julio's velocity-control PID](https://github.com/juliodltv/simulation_vsss/blob/main/config/ros_control_config.yaml)

## Time semantics

The canonical reference step is `5 ms`. One policy action repeats for four
physics steps, so observations and replay frames occur every `20 ms` or `50 Hz`.
At viewer speed 1×, 3,000 captured frames therefore take 60 seconds. Training
may compute those 60 simulated seconds faster than 60 wall-clock seconds
without changing physical units.

The current Rapier backend applies commanded body velocity directly. Its
steady-state kinematics are dimensionally consistent, but motor acceleration,
wheel slip, latency, camera noise, and closed-loop hardware control still
require calibration before deployment. The Gazebo validation backend and
seeded domain randomization remain required sim-to-real gates.

## Training consequences

- Retrain all policies created before this correction.
- Use 60-second (`3,000` decision frame) rollouts for M6 MAPPO/IPPO.
- Keep checkpoints sparse and training resumable for sustained runs.
- Treat fast-sim throughput as a compute optimization; never change `dt` to
  make a policy appear faster.
