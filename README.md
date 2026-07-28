# VSSS Lab

A modular platform for high-throughput Very Small Size Soccer simulation,
multi-agent reinforcement learning, self-play, and reproducible competition.

M0 establishes the reproducible monorepo only. It intentionally contains no physics,
training loop, ROS, Gazebo, ZeroMQ, or definitive FlatBuffers schemas.

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
```

Docker Desktop with its WSL2 backend is the only supported daemon on devbox-gpu.
Container bases are pinned by digest and run as a non-root user.

## Persistent paths

Active repositories, builds, data, runs, checkpoints, and replays live respectively
under `/home/rob/src`, `/home/rob/work`, `/home/rob/data`, `/home/rob/runs`,
`/home/rob/checkpoints`, and `/home/rob/replays`. Never place active workloads in
`/mnt/c`, `/mnt/d`, or `/mnt/g`.

See `AGENTS.md`, `CONTRIBUTING.md`, and `platform/manifest.json` before changing the
workspace.