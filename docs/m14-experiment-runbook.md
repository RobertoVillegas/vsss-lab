# M14 experiment runbook

M14 keeps training, evaluation, model selection, and visualization separate.
Generated reports live under ignored `experiments/reports/`; durable conclusions
belong in the versioned evidence notes.

## Before allocating compute

```bash
just doctor
just cuda-smoke
```

The default long-run command now selects
`experiments/configs/m14-mappo-adaptive.toml`, the typed M14 scenario suite,
CUDA when available, and 64 Rapier worlds:

```bash
just league-live-steps 50000000 25 60 25 auto 64
```

The terminal dashboard is authoritative for progress and throughput. The
browser at `http://127.0.0.1:8765` is read-only: it polls checkpoints, metrics,
and captured replays without owning the trainer.

## Studies and artifacts

```bash
just m14-curriculum-ablation experiments/reports/m14-curriculum.json cuda 3
just m14-study 20 experiments/reports/m14-study cuda
just m14-policy-ablation experiments/reports/m14-policy.json cuda 3
just m14-action-ablation experiments/reports/m14-action.json cuda 3
just m14-teacher-ablation experiments/reports/m14-teacher.json cuda 3
just m14-accelerator-spike experiments/reports/m14-accelerator.json cuda 64 256
```

Optuna storage and lineage are resumable. Smoke, screen, and confirm results
must retain their configuration hash, revision, seeds, fidelity, compute time,
raw paired outcomes, pruning reason, and parent trial. A study result is not a
promotion until the constrained paired evaluator writes an accepted decision.

For a completed training run:

```bash
just league-web ~/runs/vsss-training-run-0001
just league-tensorboard ~/runs/vsss-training-run-0001
just league-compare-runs ~/runs/baseline ~/runs/candidate ~/runs/comparison.json
```

Replay analytics are calculated lazily in the viewer. The same versioned data
can be exported without the web client:

```bash
just replay-analyze \
  ~/runs/vsss-training-run-0001/replays/iteration-000025.jsonl \
  experiments/reports/replay.json \
  experiments/reports/replay-teams.csv
```

## Interruption, resume, and rollback

One `Ctrl+C` requests a graceful checkpointed stop; a second press within two
seconds forces termination. Resume with the exact original experiment config:

```bash
just league-resume \
  ~/runs/vsss-training-run-0001 \
  2500 25 60 25 auto 64 \
  experiments/configs/m14-mappo-adaptive.toml
```

Rollback never rewrites a run. Keep the registry incumbent, reject the candidate
decision artifact, and start a new lineage from the last accepted checkpoint.
If an M14 feature regresses behavior, select its explicit baseline configuration
(MLP, continuous wheels, fixed curriculum, or no imitation) in a new run.
Rapier remains authoritative; the rejected CUDA kinematic spike is deliberately
not exposed as a backend flag.
