# VSSS Lab

A modular platform for high-throughput Very Small Size Soccer simulation,
multi-agent reinforcement learning, self-play, and reproducible competition.

The implemented M0–M11 path includes canonical contracts, deterministic Rapier
physics, Python batch environments, scripted and RL baselines, MAPPO, league
self-play, replay viewers, an external FlatBuffers/ZeroMQ match server, reference
calibration, an opt-in ROS 2/Gazebo validation backend, and seeded domain
randomization. Vision and hardware integration remain M12 work.

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

Playback at 1× follows the recorded simulation clock (20 ms control periods,
50 Hz in the reference config). The 4× default is an inspection convenience and
does not change training or policy inference speed. Slow robot motion at 1× is a
property of the captured policy/actions, not a slowed simulator.

Long runs can be resumed without repeating bootstrap. The iteration count is
additional work; checkpoints and 60-second captures can be spaced independently:

```bash
just league-resume /home/rob/runs/vsss-long 10000 100 60 100
```

The trainer reports return, progress, throughput, ETA, and checkpoint writes.
`Ctrl+C` requests a clean stop: the current iteration finishes and the latest
policy is checkpointed before exit. Training and viewing are independent:

```bash
# Terminal 1: training only
just league-resume /home/rob/runs/vsss-long 10000 100 60 100

# Terminal 2: optional read-only viewer
just league-web /home/rob/runs/vsss-long

# Or launch a new run and its viewer together
just league-live /home/rob/runs/vsss-live 10000 100 60 100
```

Fast simulation intentionally computes virtual time faster than wall time while
preserving the 5 ms physics step and 20 ms control period. A 60-second replay is
always 60 simulated seconds even when the host produces it in a few seconds.

See `docs/calibration/m11-wheel-action-scale.md` before comparing old
checkpoints or planning a physical-robot deployment.

The native WSLg viewer remains available for a single iteration:

```bash
just league-view /home/rob/runs/vsss-first 0010
```
