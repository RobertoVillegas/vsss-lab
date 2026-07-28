# VSSS Lab

VSSS Lab is an open research platform for training, evaluating, and visualizing
multi-agent reinforcement-learning policies for Very Small Size Soccer.

It combines a fast deterministic simulator, GPU-accelerated learning, experiment
tracking, replay inspection, and robotics integration boundaries in one
reproducible workspace. The goal is not only to obtain policies that score, but
to study coordination, defense, passing, robustness, and the path from
simulation to physical robots.

## What it is

Very Small Size Soccer is a compact robot-soccer problem with continuous
control, collisions, partial information, adversarial play, and cooperation
between teammates. Those properties make it a useful test bed for multi-agent
reinforcement learning (MARL).

VSSS Lab provides:

- a headless 2D physics engine designed for high-throughput training;
- a shared-policy MAPPO training pipeline with centralized value estimation;
- parallel environments and CUDA-backed neural-network optimization;
- replay capture and a browser-based match inspector;
- live metrics, checkpoint ranking, and experiment comparison;
- camera-state estimation and prediction boundaries for sim-to-real work;
- ROS 2 and Gazebo integration points for robotics validation.

The simulator and viewer are deliberately separate. Training can run as fast as
the machine allows without rendering, while recorded state and action events can
be inspected later—or followed live—without changing simulation semantics.

## Architecture

```text
                         ┌──────────────────────────────┐
                         │  Experiment configuration    │
                         │  seeds · rewards · curriculum│
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────▼─────────────────────────┐
              │                Python learning layer              │
              │  MAPPO · rollout batching · checkpoints · metrics│
              │             PyTorch / TorchRL / CUDA              │
              └──────────────┬───────────────────┬────────────────┘
                             │ PyO3 batches      │ artifacts
              ┌──────────────▼────────────┐      ▼
              │   Rust simulation core    │  runs/<experiment>/
              │ Rapier2D · rules · rewards│  metrics · replays
              │ fixed-step · Rayon worlds │  checkpoints · config
              └──────────────┬────────────┘      │
                             │ state/events      │ HTTP
              ┌──────────────▼────────────┐      ▼
              │ Robotics integration      │  Web replay viewer
              │ camera · estimator · ROS 2│  playback · charts
              │ prediction · Gazebo       │  filters · actions
              └───────────────────────────┘
```

The main boundaries are:

1. **Simulation:** Rust owns authoritative physics, field geometry, collision
   resolution, match rules, observations, and reward signals.
2. **Learning:** Python owns policies, critics, rollout assembly, optimization,
   curricula, evaluation, and checkpoint lifecycle.
3. **Observability:** structured artifacts are the contract between training and
   the web viewer, charts, TensorBoard, and offline analysis.
4. **Robotics:** estimated camera state enters through an explicit adapter rather
   than being coupled to the training simulator.

## Technology and tooling

| Area | Technology | Role |
| --- | --- | --- |
| Physics | Rust, Rapier2D, Rayon | deterministic simulation and parallel worlds |
| Native bindings | PyO3, maturin | batched Rust/Python interface |
| Learning | Python, PyTorch, TorchRL | MAPPO rollout and optimization |
| Acceleration | CUDA | policy inference and gradient updates |
| Environment management | mise, uv | pinned tools and Python dependencies |
| Rust workspace | Cargo | native builds, tests, formatting, linting |
| Web viewer | React, TypeScript, Vite, Bun | replay and live-training inspection |
| Metrics | Recharts, TensorBoard, Rich | browser charts and terminal telemetry |
| Task runner | just | stable developer and experiment commands |
| Containers | Docker Compose | CPU, CUDA, and integration smoke environments |
| Robotics | ROS 2, Gazebo | integration and hardware-facing validation |

`mise` selects tool versions, `uv` owns Python resolution, Cargo owns the Rust
workspace, and Bun owns the viewer lockfile. The repository does not rely on
GitHub Actions; validation is explicit and local.

## Simulation and robotics model

The simulation core uses a fixed physics step of 5 ms and a 20 ms control
period. Separating those clocks gives collision resolution enough temporal
resolution while keeping policy decisions aligned with realistic robot control.
Training runs in virtual time, so it can execute much faster than a real match.

The field includes playable boundaries, goals, chamfered corners, robot and ball
dimensions, damping, friction, differential-drive controls, goal detection only
after the complete ball crosses the line, and a short post-goal closure period.
Robots and the ball are prevented from occupying physically impossible
overlapping states.

For camera-driven operation, the platform defines a causal state-estimation
path: detections are filtered into position and velocity estimates, then
projected forward to compensate for observation and actuation latency. Prediction
visualization represents the estimated ball trajectory; it is an inspection aid,
not privileged future information supplied to the policy.

## Learning system

The default learner is multi-agent proximal policy optimization (MAPPO):

- one policy is shared by the three allied robots;
- each actor receives its local, role-aware observation;
- a centralized critic uses joint training information;
- actions are continuous left/right wheel commands;
- rollout collection runs many simulation worlds in parallel;
- policy inference and optimization use CUDA when available;
- deterministic physics remains CPU-based and parallelized with Rayon.

CUDA accelerates the neural part of the workload. It does not automatically move
Rapier physics onto the GPU, so throughput depends on both CPU simulation and GPU
batch efficiency.

Training begins with a heuristic bootstrap and then transitions to self-play.
The opponent population mixes the current policy, recent historical
checkpoints, and a heuristic opponent. This reduces overfitting to a single
mirror opponent and makes regressions easier to detect.

## Observations, actions, and roles

Observations describe the ball, goals, teammates, opponents, robot motion, and
role context in normalized field coordinates. Policies act through bounded
differential wheel commands rather than teleportation or direct velocity
assignment.

Roles provide a coordination prior—such as pressure, support, and defense—without
hard-coding a complete strategy. The replay viewer exposes each actor's current
action, wheel intensity, speed, and role so learned behavior can be inspected at
the same timestamp as the match.

## Reward design

The reward is intentionally decomposed instead of relying only on goals:

- goals and concessions provide the strongest terminal signal;
- directional ball progress rewards movement toward the opponent goal and
  penalizes danger toward the own goal;
- defense rewards useful coverage and intervention;
- spacing and congestion terms discourage persistent clustering;
- pass-like teammate transitions provide a small cooperation signal;
- inactivity, unnecessary wheel effort, and abrupt control changes are
  regularized;
- episode duration is bounded so deadlocks cannot dominate data collection.

Directional progress uses the ball velocity relative to each goal:

```text
tanh(cos(ball_velocity, opponent_goal - ball))
  - tanh(cos(ball_velocity, own_goal - ball))
```

Goals remain substantially more valuable than shaping terms. Passing and
defensive rewards are evidence that useful play occurred, not substitutes for
winning. This prevents policies from farming safe intermediate behavior while
avoiding decisive play.

## Observability and evaluation

Every run is self-contained:

```text
run/
├── checkpoints/     policy and optimizer snapshots
├── replays/         lazily loaded captured matches
├── metrics.jsonl    append-only training metrics
├── run.json         status and latest-artifact pointers
└── configuration    reproducibility metadata
```

The terminal dashboard reports environment steps, completed matches,
frames/second, matches/second, return, losses, entropy, and checkpoint status.
The browser viewer adds:

- play, pause, seek, rewind, skip, loop, and speed controls;
- lazy replay discovery and loading;
- filters for goals, wins, losses, draws, passes, and other events;
- synchronized robot actions and state;
- ball trajectory inspection;
- live polling for newly captured iterations and checkpoints;
- in-project metric charts backed by the same run artifacts.

TensorBoard remains available as a specialized optional view. Checkpoints should
be selected through seeded evaluation against a population—not by choosing the
largest training return or simply taking the last file.

## Evidence-driven experimentation

Training and model selection are separate concerns. VSSS Lab evaluates
candidates on paired seeds from both team colors and uses terminal match
outcomes with confidence intervals. Shaped return, possession, touches, Elo, and
checkpoint recency remain diagnostics; none can promote a policy alone.

The adaptive-training layer provides:

- immutable holdouts plus routine, frontier, and deduplicated failure scenarios;
- validity-checked scenario mutation and learning-progress allocation;
- persistent multiobjective Optuna studies with smoke, screen, and confirmation
  fidelities;
- bounded reward and PPO search parameters with commit, seed, parent, compute,
  and pruning lineage;
- feed-forward, isolated-GRU, entity-attention, and symmetric wheel-action
  ablation contracts;
- exact-simulator CEM demonstrations for atomic skills only;
- behaviorally diverse historical policy retention and confidence-gated
  distillation;
- reward-independent replay events, possession, pressure, positioning,
  coordination, timelines, heatmaps, JSON, and CSV export;
- phase-level CPU/CUDA profiling and a trace-parity gate for any future
  accelerator backend.

Rapier remains the authoritative physics engine. A faster alternate backend is
adoptable only if it preserves contact and goal traces and improves end-to-end
throughput by a material margin.

## Inspirations and references

VSSS Lab is informed by public simulators, robot-soccer tooling, and MARL
research. These projects are references and comparison points, not code or model
weight dependencies:

- [Julio de la Torre's simulation_vsss](https://github.com/juliodltv/simulation_vsss)
  and [pSim documentation](https://juliodltv.github.io/pSim/) for VSSS field,
  control, visualization, and reinforcement-learning ideas;
- [RocketSim](https://github.com/ZealanL/RocketSim) and
  [RLGym](https://rlgym.org/) for high-throughput headless simulation and
  learning-oriented environment design;
- [RLBot v5](https://wiki.rlbot.org/v5/) for bot interfaces, match orchestration,
  and reproducible evaluation;
- [Necto](https://github.com/Rolv-Arild/Necto) for population-based
  Rocket League learning patterns;
- [PettingZoo](https://pettingzoo.farama.org/) for multi-agent environment
  conventions;
- [TorchRL](https://docs.pytorch.org/rl/) and
  [BenchMARL](https://github.com/facebookresearch/BenchMARL) for current MARL
  abstractions and reproducible baselines;
- [HARL](https://github.com/PKU-MARL/HARL) for heterogeneous-agent algorithm
  comparisons.

Rocket League and VSSS differ significantly in dynamics, dimensionality, action
space, and embodiment. Their training architecture and evaluation practices can
transfer; their learned weights and task-specific rewards generally cannot.

## Research direction

The present architecture supports controlled comparisons beyond shared-policy
MAPPO, including recurrent policies, heterogeneous-agent learners,
counterfactual credit assignment, prioritized opponent sampling, league
exploiters, and additional centralized-training/decentralized-execution
baselines.

Claims of improvement should be based on multiple seeds and fixed evaluation
populations. Useful measures include win/draw/loss rate, goal differential,
touches, useful passes, defensive interventions, congestion, inactivity,
smoothness, robustness to latency/noise, and wall-clock sample throughput.

## Getting started

The supported development environment is Linux:

```bash
git clone https://github.com/RobertoVillegas/vsss-lab.git
cd vsss-lab

just doctor
just bootstrap
just build
just test
```

For CUDA training, verify that the NVIDIA driver is available to Linux before
starting a long run:

```bash
just cuda-smoke
```

Useful evaluation and profiling commands:

```bash
# Paired-color, five-seed terminal evaluation.
just league-tournament \
  /path/to/checkpoint.pt \
  reports/tournament \
  experiments/configs/m13-mappo-directional.toml \
  5

# Reward-independent replay JSON and team CSV.
just replay-analyze \
  /path/to/replay.jsonl \
  reports/replay-analytics.json \
  reports/replay-teams.csv

# CPU/GPU rollout phase profile.
just profile-m14 200 reports/m14/profile.json
```

The complete adaptive-study command matrix, artifact locations, interruption
semantics, and rollback procedure are in
[the M14 experiment runbook](docs/m14-experiment-runbook.md).

## Train and inspect a run

Start a 50-million-environment-step training run with automatic run naming,
replay capture every 25 learner iterations, 60-second captures, checkpoints every
25 iterations, automatic CUDA selection, and 64 parallel worlds:

```bash
just league-live-steps 50000000 25 60 25 auto 64
```

The command allocates a directory such as
`~/runs/vsss-training-run-0001`, starts the private viewer at
`http://127.0.0.1:8765`, and runs training in the foreground. CUDA is selected
when available; otherwise the command reports that it is using CPU.

Training persists independently of replay consumption. The viewer only reads
artifacts and can be started later:

```bash
just league-web ~/runs/vsss-training-run-0001
```

Optional TensorBoard:

```bash
just league-tensorboard ~/runs/vsss-training-run-0001
```

To expose both observability views together:

```bash
just league-observe ~/runs/vsss-training-run-0001
```

Use Linux-native paths for active runs. Keeping high-frequency training artifacts
under `/home/...` rather than Windows-mounted paths such as `/mnt/c`, `/mnt/d`,
or `/mnt/g` avoids WSL filesystem overhead.

## Resume and evaluate

Resume an interrupted run:

```bash
just league-resume \
  ~/runs/vsss-training-run-0001 \
  2500 25 60 25 auto 64
```

Rank checkpoints with repeated seeded matches:

```bash
just league-rank-checkpoints \
  ~/runs/vsss-training-run-0001 \
  24 experiments/configs/m12-mappo-coordinated.toml \
  11,23,37 \
  ~/runs/vsss-training-run-0001/checkpoint-ranking.json
```

Compare two completed runs:

```bash
just league-compare-runs \
  ~/runs/baseline \
  ~/runs/candidate \
  ~/runs/comparison.json
```

Graceful interruption preserves the latest checkpoint and run metadata. A resumed
run restores policy, optimizer, counters, and population state.

## Containers and local validation

CPU container gate:

```bash
just container-cpu
```

CUDA smoke gate:

```bash
just cuda-smoke
```

ROS 2 and Gazebo integration smoke gate:

```bash
just ros-gazebo-smoke
```

Normal repository validation:

```bash
just doctor
just build
just test
just lint
```

## Repository map

```text
crates/                 Rust simulation, physics, rules, and native bindings
python/                 learning, league orchestration, evaluation, and adapters
experiments/configs/    versioned experiment configurations
web/replay-viewer/      React replay and metrics application
tools/                  run allocation, replay serving, and developer utilities
tests/                  golden states and cross-layer integration coverage
containers/             CPU, CUDA, and robotics container definitions
docs/                   product, design, research, and evidence documents
compose.yaml            local container orchestration
Justfile                supported developer and experiment commands
mise.toml               pinned toolchain and environment tasks
pyproject.toml          Python project and dependency groups
Cargo.toml              Rust workspace
```

## Scope and limitations

VSSS Lab is a research platform, not a certified competition robot stack.
Physical deployment still requires team-specific camera calibration,
communications, motor control, safety constraints, and validation against real
hardware.

Physics throughput remains partly CPU-bound even during CUDA training. Simulation
parameters approximate VSSS dynamics but must be system-identified for a
particular robot. Population sampling currently favors a recent historical
window rather than a full prioritized league. Finally, reported results are
meaningful only with their configuration, seeds, evaluation opponents, hardware,
and software revision.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
