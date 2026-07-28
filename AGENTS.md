# Agent instructions

## Required workflow

1. Read the issue, this file, and relevant ADRs.
2. Run `just doctor` before edits.
3. Work only in the requested milestone and scope.
4. Keep commits small, signed, and reversible.
5. Run `just lint`, `just build`, and `just test` before handoff.
6. Record benchmarks when performance may change.
7. Include artifacts, evidence, known limitations, and rollback in the PR.

## Architecture boundaries

- `vsss-spec` cannot depend on physics, PyTorch, ROS, or Python.
- ROS, ZeroMQ, Docker, Ray, and Python dictionaries never enter the simulation hot loop.
- Physical robot identity must not imply a dedicated policy or fixed tactical role.
- Contract changes require an ADR and contract tests.
- Dependencies belong to locked project environments or digest-pinned containers.
- Do not install CUDA, PyTorch, OpenCV-CUDA, ROS, or Gazebo globally.
- Active artifacts must remain in Linux-native storage, never `/mnt/*`.

## M0 boundary

M0 provides scaffolding and executable gates only. Do not implement Rapier, RL,
self-play, ROS/Gazebo, remote protocols, or vision in this milestone.