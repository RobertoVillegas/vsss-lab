# Training and evaluation tooling status

Which experiment tools inform current work, and which answered a question for a
milestone that is now closed. A deprecated tool is kept for reproduction of its
recorded evidence, not deleted, because its report is cited by an archived
change. None of them is wired into a live gate.

## Current

| Tool | Recipe | Purpose |
| --- | --- | --- |
| `tools/evaluate_m15.py` | `just m15-evaluate`, `just m15-evaluate-control` | Score any config's immutable skill holdouts with a policy, random, or scripted control. Parser-aware. |
| `tools/rank_checkpoints.py` | — | Rank a run's checkpoints by terminal match outcome. Parser-aware. |
| `tools/benchmark_m15.py` | `just m15-benchmark` | Semantic rollout and optimization throughput. Pinned to the M15 config so successive measurements stay comparable. |
| `tools/m14_accelerator_spike.py` | `just m14-accelerator-spike` | Device kernel benchmark. Independent of policy architecture. |
| `tools/benchmark_m24_trajectories.py` | `just m24-trajectory-benchmark` | Trajectory quality for the current control stack. |
| `python/vsss_train/marl_cli.py` | `just marl-prepare`, `just marl-evaluate` | Teacher distillation and progress-versus-random check. Parser-aware. |

## Deprecated

Each is pinned to the configuration of a closed milestone and does not evaluate
the M24.2 parametric action space. Point one at a current config and it will
either measure the wrong policy class or be rejected by the environment's action
shape check.

| Tool | Milestone | Recorded outcome |
| --- | --- | --- |
| `tools/m14_study.py` | M14 | `docs/evidence`, M14 reward and PPO search |
| `tools/m14_curriculum_ablation.py` | M14 | uniform versus adaptive curriculum |
| `tools/m14_policy_ablation.py` | M14 | MLP versus GRU under partial observability |
| `tools/m14_action_ablation.py` | M14 | continuous versus symmetric wheel lattice |
| `tools/m14_teacher_ablation.py` | M14 | scratch versus imitation-seeded MAPPO |
| `tools/profile_m14.py` | M13/M14 | rollout boundary profile; builds a `SharedActor` directly |
| `tools/m15_ablation.py` | M15 | curriculum and reward arms at matched compute |
| `tools/m18_ppo_ablation.py` | M18 | PPO architecture and causal-reward screening |

## Retired

`tools/m15_candidate_probe.py` was deleted in
`openspec/changes/retire-cross-milestone-baseline`. It compared a candidate with
frozen policies from run directories that no longer exist, across action spaces
that cannot share a match. The comparison the project wants instead is a paired
terminal scorecard against the promoted incumbent of the candidate's own
lineage, which is a deferred task in that change.
