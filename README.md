# VSSS Lab

A modular platform for high-throughput Very Small Size Soccer simulation,
multi-agent reinforcement learning, self-play, and reproducible competition.

The implemented M0–M13 path includes canonical contracts, deterministic Rapier
physics, Python batch environments, scripted and RL baselines, MAPPO, league
self-play, replay viewers, an external FlatBuffers/ZeroMQ match server, reference
calibration, an opt-in ROS 2/Gazebo validation backend, and seeded domain
randomization. M12 adds the calibrated physical field and vision diagnostics;
M13 adds directional rewards, an exploration floor, staged self-play, and
terminal checkpoint ranking. Hardware integration remains future work.

## What the project is

VSSS Lab is an Apache-2.0 research and engineering platform for training a team
of three differential-drive robots to play Very Small Size Soccer. It combines:

- a deterministic 2D Rapier physics engine written in Rust;
- native batched simulation exposed to Python through PyO3;
- Gymnasium and PettingZoo-compatible environment adapters;
- scripted controllers, PPO skills, shared-policy IPPO/MAPPO, and league tools;
- a centralized critic for training and decentralized team observations at run
  time;
- fast headless training, optional CUDA learning, checkpoints, tournaments,
  replays, and a browser-based inspection studio;
- a ROS 2/Gazebo validation path plus camera estimation and causal ball
  prediction for eventual sim-to-real work.

The simulator runs virtual time faster than real time. Physics remains on native
CPU workers because Rapier advances many small independent worlds; PyTorch uses
CUDA for neural inference and optimization. The replay viewer is an observer
and never slows or changes the training environment.

## Current status

M0 through M11 are implemented. M12 and M13 are usable but not yet closed:

| Milestone | Implemented | Still required to close it |
| --- | --- | --- |
| M12 · vision and hardware | calibrated synthetic camera, CPU ball filter and robot EKF, association confidence, causal trajectory/interception prediction, ROS camera ingestion, viewer layers and CPU profiling hooks | record policy-visible estimates at every decision, run the predictive-feature ablation, publish hidden-truth accuracy/latency thresholds, benchmark a recorded camera, and produce hardware-in-the-loop safety evidence |
| M13 · coordinated learning | directional reward, dynamic attacker, time/effort/congestion/defense regularization, exploration floor, heuristic-to-self-play curriculum, terminal checkpoint ranking, TensorBoard and in-viewer charts | complete a fresh 50M-step run, compare it against run 0002 across multiple seeds, promote the best checkpoint, and archive the milestone |

The initial platform Definition of Done is substantially implemented, but the
project does not yet claim a validated physical-robot policy. The remaining
critical path is measurement and evaluation rather than another physics rewrite:

1. finish and compare the fresh M13 training run;
2. complete the M12 predictive-observation ablation and camera evidence;
3. validate a selected policy in Gazebo using the same contracts;
4. add hardware-in-the-loop safety limits and test with the real overhead
   camera and robots;
5. publish reproducible multi-seed algorithm baselines and a promoted model.

## Quick start

```bash
git clone git@github.com:RobertoVillegas/vsss-lab.git
cd vsss-lab
just doctor
just bootstrap
just build
just test
```

`mise` pins Python, Rust, uv, Ruff, and just. `uv.lock` is required so mise can source
the uv-managed `.venv` using `python.uv_venv_auto = "source"`.

## Containers

```bash
just container-cpu
just cuda-smoke
just ros-gazebo-smoke
```

Docker Desktop with its WSL2 backend is the only supported daemon on devbox-gpu.
Container bases are pinned by digest and run as a non-root user.

## Validation milestones

```bash
# Independent Rust and Python controllers
just external-tournament reports/m8/external-match.jsonl 50
just external-container-smoke

# Reference physics and backend portability
just calibrate-reference reports/m9/calibration.json
just backend-bridge-smoke

# Paired held-out robustness suite
just ood-evaluate reports/m11/ood.json
```

The ROS/Gazebo profile is a higher-fidelity validation target, not the training
hot loop. Evidence and limitations for each milestone live under
`docs/evidence/` and `docs/calibration/`.

## Persistent paths

Active repositories, builds, data, runs, checkpoints, and replays live respectively
under `/home/rob/src`, `/home/rob/work`, `/home/rob/data`, `/home/rob/runs`,
`/home/rob/checkpoints`, and `/home/rob/replays`. Never place active workloads in
`/mnt/c`, `/mnt/d`, or `/mnt/g`.

See `AGENTS.md`, `CONTRIBUTING.md`, and `platform/manifest.json` before changing the
workspace.

## Inspect a training run

`league-run` remains headless so rendering never slows the learner. After it
captures iterations, launch the run-wide browser viewer:

```bash
just league-web /home/rob/runs/vsss-first
```

Open <http://127.0.0.1:8765> in Windows. The viewer polls the run every two
seconds, follows and loops the latest completed capture, and reports the latest
checkpoint and metric. Selecting an older iteration pauses live-follow until
`Follow latest` is pressed. It can also play/pause, seek, step one recorded
frame, or skip 100 frames in either direction.

The `TRAINING METRICS` tab provides synchronized Recharts curves for return,
progress, policy/value loss, entropy, throughput, match terminations, and actor
exploration. The server returns a bounded, evenly sampled metric history, so a
long run does not load every replay or an unbounded time series.

New runs also write TensorBoard-compatible events under
`RUN_DIR/tensorboard/`. TensorBoard is optional: the built-in graphs are the
default experience and require only the replay viewer.

```bash
# Built-in replay plus training charts
just league-web /home/rob/runs/vsss-training-run-0003

# Optional full TensorBoard UI
just league-tensorboard /home/rob/runs/vsss-training-run-0003

# Start both read-only observers together
just league-observe /home/rob/runs/vsss-training-run-0003
```

TensorBoard runs separately on <http://127.0.0.1:6006>. Keeping it outside the
viewer avoids iframe, port, and lifecycle coupling while preserving its richer
plugin ecosystem. `metrics.jsonl` remains the canonical machine-readable run
record; TensorBoard events are derived observability.

Playback at 1× follows the recorded simulation clock (20 ms control periods,
50 Hz in the reference config). The 4× default is an inspection convenience and
does not change training or policy inference speed. Slow robot motion at 1× is a
property of the captured policy/actions, not a slowed simulator.

Long runs can be resumed without repeating bootstrap. The iteration count is
additional work; checkpoints and 60-second captures can be spaced independently:

```bash
just league-live-steps 20000000 25 60 25 auto 64
```

New runs use `experiments/configs/m13-mappo-directional.toml`. Its dense signal
rewards ball velocity toward the opponent goal and the dynamically selected
attacker moving toward the ball. It no longer rewards mere proximity, which can
teach a robot to camp beside the ball. A bounded time cost encourages terminal
goals, while small wheel-effort, action-change, congestion, and defensive terms
regularize play. The first 250 PPO iterations face the dynamic heuristic before
switching to self-play. Policy standard deviation is clamped to a configured
floor so a long run cannot silently collapse all exploration.

This is a new reward fingerprint: start a fresh automatic run rather than
resuming an M12 checkpoint. Historical M12 checkpoints remain loadable with
their original config and can be ranked by actual terminal results:

```bash
just league-rank-checkpoints \
  /home/rob/runs/vsss-training-run-0002 \
  500,750,1000,1500,2250,2750,3052 \
  experiments/configs/m12-mappo-coordinated.toml \
  10 \
  reports/checkpoint-ranking.json
```

The ranker plays every checkpoint from reflected field setups against the same
heuristic and orders win/loss balance, goal difference, then mean progress. Do
not select a deployment policy solely because it is the latest checkpoint.

The trainer reports return, progress, throughput, ETA, and checkpoint writes.
`Ctrl+C` requests a clean stop: the current iteration finishes and the latest
policy is checkpointed before exit. Training and viewing are independent:

```bash
# Terminal 1: training only
just league-resume /home/rob/runs/vsss-long 2500 25 60 25 auto 64

# Terminal 2: optional read-only viewer
just league-web /home/rob/runs/vsss-long

# Or target environment steps and launch the viewer together
just league-live-steps 20000000 25 60 25 auto 64
```

The trailing values select `device` and vectorized environment count. `auto`
selects CUDA when available and prints an explicit CPU fallback warning
otherwise. The expanded command always displays the named CLI flags
`--device` and `--num-envs`; use `cpu` deliberately for the current small-network
baseline when maximum measured throughput matters.

The CUDA/Rapier default is 64 worlds. A match ends on a goal or after 30
simulated seconds; PPO updates consume 256 control steps and persistent matches
span updates. Independent Rapier worlds are stepped inside one native batch and
use adaptive Rayon parallelism at 32 or more worlds.

Environment steps are the primary training-budget unit. The 20M preset is about
1,221 PPO updates, while the dashboard also reports completed matches and
matches/s for sporting interpretation.

Training uses Rich on an interactive terminal: a stable progress bar remains at
the bottom while current/rolling return, progress, PPO losses, device, vector
worlds, frames/s, and the latest checkpoint remain tabulated above it. Logs and
warnings are emitted above the live display. Redirected output automatically
uses one aligned text row per completed iteration.

Fast simulation intentionally computes virtual time faster than wall time while
preserving the 5 ms physics step and 20 ms control period. A 60-second replay is
always 60 simulated seconds even when the host produces it in a few seconds.

See `docs/calibration/m11-wheel-action-scale.md` and
`docs/evidence/m13-directional-reward.md` before comparing old checkpoints or
planning a physical-robot deployment.

The native WSLg viewer remains available for a single iteration:

```bash
just league-view /home/rob/runs/vsss-first 0010
```

## Tooling

The repository is intentionally reproducible and uses one selected tool per
job:

| Tool | Role |
| --- | --- |
| `mise` | pins Python, Rust, uv, Ruff, just and project tasks |
| `uv` | resolves the Python environments and locked training dependencies |
| Cargo | builds/tests the Rust workspace and PyO3 extension |
| `maturin` | installs the mixed Rust/Python package into `.venv` |
| Bun | installs and builds the React/Vite replay viewer |
| `just` | exposes stable human-facing workflows |
| Docker Compose | validates CPU, CUDA and ROS/Gazebo profiles |
| PyTorch/TorchRL | policy, critic, tensor trajectories and CUDA optimization |
| Rich | stable terminal progress and current/rolling metrics |
| TensorBoard/Recharts | optional full telemetry UI and built-in run charts |

Normal validation is:

```bash
just doctor
just build
just test
just lint
just container-cpu
just cuda-smoke
```

Dependencies are pinned in `uv.lock`, `Cargo.lock` and `bun.lock`. Container
bases are digest-pinned. Active runs and builds belong on Linux-native storage;
placing them under `/mnt/c`, `/mnt/d` or `/mnt/g` causes avoidable I/O and
filesystem-notification overhead.

## Architecture and implementation

```text
experiment TOML
      │
      ▼
Python league/trainer ─────► PyTorch MAPPO actor + centralized critic (CUDA)
      │                                  │
      │ batched actions                  │ checkpoints
      ▼                                  ▼
PyO3 batch boundary ◄────── Rust/Rapier deterministic worlds
      │
      ├──► metrics.jsonl ───► built-in Recharts dashboard
      ├──► TensorBoard events ─► optional TensorBoard UI
      ├──► sampled replay events ─► browser/native replay viewers
      └──► registry/checkpoints ─► tournament and promotion gates

ROS 2/Gazebo/camera adapters consume the same canonical state/action contracts
as a slower validation plane; they do not enter the training hot loop.
```

The Rust workspace owns units, canonical state, deterministic fixed-step
physics, collision/goal rules, snapshots, batch stepping, FlatBuffers protocol,
the external match server and native viewer. Python owns environment composition,
observations, rewards, learning, league orchestration, evaluation, vision and
research workflows. The browser reads recorded observer data; it never calls
the policy or physics engine.

Important boundaries:

- simulation truth, camera measurements, estimated state, and future prediction
  are distinct versioned records;
- a visual robot marker identifies a physical player but never assigns a fixed
  tactical policy role;
- renderer and telemetry failures cannot alter physics or learning;
- experiment values are fingerprinted into checkpoints, so incompatible reward
  configurations cannot be resumed silently;
- the fast simulator and ROS/Gazebo implement the same state/action concepts
  rather than becoming two independent training backends.

## Physics, field and controls

The reference field, goals, chamfered corners, robot and ball dimensions come
from the canonical match configuration and were calibrated against Julio De La
Torre's VSSS simulator. Rapier uses a 5 ms physics step; policies act every
20 ms by default. Differential-drive wheel commands pass through acceleration
limits, damping/friction and collision response. Tests block sustained
robot/robot overlap, robot/ball engulfment, false goals from partial line
crossing, and robots escaping through goal geometry.

A goal is valid only when the ball crosses the goal plane according to the
canonical radius-aware rule. The replay includes a short post-goal grace period
for visual closure, but the learning episode has a terminal goal outcome.
Scoreless horizons and stagnant-ball states are separate terminal reasons.

Fast training intentionally advances simulated time faster than wall time.
Playback at 1× represents the recorded physical clock, so one simulated second
has the same motion scale as one real second even if training generated it much
faster.

## Observations, actions and roles

Each robot receives an agent-centric observation containing normalized local
state and relative ball/teammate/opponent information. It emits continuous
left/right wheel commands. The shared actor is permutation-safe: physical ID,
team marker and array slot do not permanently mean attacker, defender or
goalkeeper.

Roles are state-dependent. M13 chooses the closest teammate only for its
attacker-alignment reward at the current step; defensive coverage similarly
uses the best-positioned teammate. This produces dynamic role pressure without
training three identity-specific policies. The centralized critic can use team
context during learning, while execution remains decentralized.

M12 adds an optional policy-visible perception record: timestamped camera
detections, association confidence, Kalman/EKF estimated state, covariance,
staleness and collision-aware ball projection. Prediction uses only the present
estimate and field physics; future simulator truth is explicitly excluded and
covered by a mutation test.

## Algorithm and curriculum

The main M13 learner is shared-policy MAPPO:

- three decentralized actor decisions with shared parameters;
- one centralized value function during training;
- Gaussian continuous actions passed through `tanh`;
- generalized advantage estimation, clipped PPO updates, entropy bonus and
  gradient clipping;
- a configurable minimum actor `log_std` clamped after optimizer steps to avoid
  silent exploration collapse;
- persistent vector worlds whose matches span PPO rollout boundaries;
- complete actor, critic, optimizer, random-state and policy-version
  checkpoints.

The first 250 PPO iterations play against the deterministic dynamic heuristic.
Later iterations play against a frozen copy of the current learner for the
duration of each update. Historical registry sampling and dedicated exploiters
remain a known league improvement.

New automatic runs use:

```text
experiments/configs/m13-mappo-directional.toml
```

The config selects 64 worlds, a 128-unit actor, 256 control decisions per
rollout, a 1,500-decision match horizon, CUDA when available, and the M13 reward
fingerprint. Override device/world count through the trailing `just` arguments;
`auto` reports an explicit CPU fallback.

## Reward design

M13's primary dense signal is bounded ball-direction alignment:

```text
tanh(cos(ball velocity, opponent goal - ball))
- tanh(cos(ball velocity, own goal - ball))
```

The dynamically closest robot receives a penalty-only velocity alignment toward
the ball. Goals remain the dominant ±10 sparse objective. A full scoreless
horizon accumulates one bounded time penalty; action change, wheel effort,
teammate congestion, and defensive coverage are smaller regularizers.

Mere proximity to the ball is deliberately not rewarded. The completed 50M M12
run showed that proximity/raw progress could produce camping, clustering,
saturated controls and possession that did not terminate in goals. This design
adapts the bounded directional equations and shared-policy collision warning
from Julio De La Torre's 2024 thesis while retaining MAPPO and dynamic roles
rather than copying its fixed-role MATD3 setup.

Reward changes are part of the checkpoint fingerprint. Start a new run when the
reward contract changes; do not force an incompatible resume.

## Metrics, replays and checkpoint selection

Each run contains:

| Artifact | Purpose |
| --- | --- |
| `metrics.jsonl` | canonical per-iteration result, losses, terminal counts, cumulative work, throughput and exploration |
| `tensorboard/` | derived TensorBoard event files for new runs |
| `checkpoints/` | resumable policy/critic/optimizer snapshots |
| `replays/` | sampled, labeled match captures loaded only when selected |
| `registry.json` | versioned policy lineage and league metadata |
| `viewer.log` / `tensorboard.log` | observer-process diagnostics |

Do not deploy the latest checkpoint automatically. Select candidates with fixed
seeds, reflected sides, terminal goals, historical opponents, and the scripted
heuristic:

```bash
just league-rank-checkpoints \
  /home/rob/runs/vsss-training-run-0002 \
  500,750,1000,1500,2250,2750,3052 \
  experiments/configs/m12-mappo-coordinated.toml \
  10 \
  reports/checkpoint-ranking.json
```

In the first historical sample, iteration 2750 produced a 5-5-0 W-D-L record
against the heuristic while the final iteration 3052 produced 3-6-1. Checkpoint
recency and training return are therefore insufficient promotion criteria. A
real promotion also needs multiple seeds, permutation, OOD and Gazebo gates.

## Inspirations and public references

The project reuses principles, measurements and public interfaces—not opaque
third-party model weights:

- [Julio De La Torre's simulation_vsss](https://github.com/juliodltv/simulation_vsss)
  and thesis: VSSS geometry/assets, differential-drive reference behavior,
  camera markers, Kalman/EKF estimation, reward pitfalls, directional shaping
  and MATD3 comparison.
- [pSim](https://juliodltv.github.io/pSim/usage/): compact VSSS simulation API
  and scenario ergonomics used as a reference, not a second hot-loop backend.
- [RocketSim](https://github.com/ZealanL/RocketSim): evidence that a specialized
  standalone headless physics engine can generate experience much faster than
  a rendered game.
- [RLGym](https://rlgym.org/Getting%20Started/quickstart/): separable
  observations/actions/rewards/resets, accelerated PPO collection and early
  termination of unproductive episodes.
- [RLBot v5](https://wiki.rlbot.org/v5/framework/architecture/): match
  orchestration, language-neutral server/client boundaries, live packets and
  controller inputs. RLBot is deployment infrastructure, not an RL trainer.
- [Necto/Nexto](https://github.com/Rolv-Arild/Necto): distributed fast
  self-play, general reward shaping, live graphs, population training and
  optional replay pretraining.
- [PettingZoo](https://pettingzoo.farama.org/): public simultaneous-action MARL
  semantics.
- [TorchRL](https://docs.pytorch.org/rl/): tensor-native trajectories and
  multi-agent learning components.
- [BenchMARL](https://github.com/facebookresearch/BenchMARL): reproducible
  PyTorch/TorchRL comparisons across algorithms, models and seeds.
- [HARL](https://github.com/PKU-MARL/HARL): reference HAPPO/HATRPO and
  heterogeneous-agent implementations.

Rocket League model weights are not directly reusable: cars have boost, aerial
motion, different contacts, observations and actions. Useful transfers are
evaluation discipline, league diversity, reward restraint, replay pretraining,
timeouts and distributed experience collection. RLBot's supplied ball
prediction is analogous to VSSS Lab's causal physics projection, not a learned
peek at future simulator state.

## MARL research direction

There is no universal “SOTA” algorithm across cooperative continuous-control
games; environment and evaluation protocol can reverse rankings. The
recommended sequence for this project is:

1. finish shared MAPPO as the trusted, multi-seed M13 baseline;
2. add recurrent MAPPO/GRU for camera delay, occlusion and partial
   observability;
3. add MASAC as an off-policy continuous-control baseline to measure sample
   efficiency;
4. benchmark Deep Sets/GNN or a multi-agent transformer only after MLP/GRU
   baselines are stable;
5. test HAPPO/HARL as an ablation rather than a default, because VSSS uses
   homogeneous robots and deliberately shares actor weights;
6. expand the league with historical sampling and fixed exploiters before
   scaling network size or distributed infrastructure.

BenchMARL is the preferred external comparison harness because it already
standardizes MAPPO, IPPO, MASAC, MADDPG, QMIX, multiple model families and
multi-seed reporting. Integrating its evaluation format is more valuable now
than replacing the readable in-repo learner.

## Repository map

| Path | Purpose |
| --- | --- |
| `crates/` | Rust contracts, Rapier physics, batching, protocol, match server and native viewer |
| `python/vsss_env/` | PyO3-facing environment and public adapters |
| `python/vsss_train/` | PPO/MAPPO models, trajectory schema and optimizer |
| `python/vsss_league/` | self-play, registry, tournaments, promotion, telemetry and run CLI |
| `python/vsss_vision/` | camera ingestion, association, filters and prediction |
| `web/replay-viewer/` | React/Vite replay and training-metrics studio |
| `experiments/configs/` | versioned, fingerprinted experiment definitions |
| `tests/golden/` | canonical field and match fixtures |
| `containers/` | CPU, CUDA and ROS/Gazebo development profiles |
| `docs/` | ADRs, calibration reports, evidence and the full product PRD |

## Known limitations

- the current curriculum does not yet sample the full historical policy
  registry during rollout collection;
- the 50M M13 comparison has not been run, so its reward changes are tested but
  not yet empirically promoted;
- physics collection remains CPU-bound even when PyTorch reports CUDA;
- runs created before this telemetry integration cannot reconstruct historical
  actor exploration or wall-clock throughput;
- the camera path has deterministic fixtures and ROS ingestion but no completed
  physical-camera accuracy envelope or robot hardware safety gate;
- no Rocket League/third-party model is bundled, and no result here should be
  described as universal MARL SOTA.
