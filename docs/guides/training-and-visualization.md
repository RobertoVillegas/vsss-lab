# Training runs and visualization

M7 captures an immutable checkpoint and evaluation replay after selected real
rollout/optimize iterations. The viewer is outside the training hot loop.

## Start a run

```sh
just league-run /home/rob/runs/vsss-first 10 1 60 100 auto 64
```

Arguments are, in order, `run_dir`, `iterations`, `capture_every`,
`capture_seconds`, `checkpoint_every`, `device`, and `num_envs`. The expanded
Python command prints the corresponding named flags. One M7 iteration is one
vectorized native C8 trajectory followed by one MAPPO optimization update; it
is the meaningful unit to inspect rather than a supervised-learning epoch.

The current default is shared-policy MAPPO: three decentralized actors share
weights while a centralized critic is used during training. PPO/IPPO remain
available for controlled ablations.

`device=auto` selects CUDA when available and otherwise continues on CPU with a
visible warning. Physics remains native Rapier on CPU; CUDA batches actor,
critic, distillation, and PPO tensor work. On the July 2026 RTX 3070 reference
host, one 16-world/48,000-frame iteration measured about 1,832 frames/s on CUDA
and 2,748 frames/s on CPU. The small network does not yet amortize CUDA transfer
and launch costs, so CUDA support is functional rather than a throughput claim.
Parallel native physics stepping is the next performance optimization.

That optimization landed in M12.1: a single native batch now owns all worlds,
releases the GIL, and schedules batches of at least 32 worlds through Rayon.
On the same host, a full 64-world CUDA iteration sustained about 4,195 frames/s
while another 16-world training process was active. A controlled short-rollout
sweep measured approximately 1,731, 5,172, and 7,074 frames/s at 16, 64, and
256 worlds respectively. Sixty-four is the default balance between throughput,
GPU batch size, memory, and policy-update frequency.

Episodes last at most 1,500 control steps (30 simulated seconds), while PPO
updates consume 256 steps. Vector worlds persist across updates. Prefer a
completed-match target when planning a run:

```sh
just league-live-matches /home/rob/runs/vsss-100k 100000 25 60 25 auto 64
```

The terminal reports completed matches and matches/s. An iteration is an
optimizer update, not a match.

Artifacts:

```text
/home/rob/runs/vsss-first/
  registry.json
  metrics.jsonl
  checkpoints/iteration-0000.pt
  checkpoints/iteration-0001.pt
  replays/iteration-0001.jsonl
```

## Inspect behavior

On the headless devbox, render a static SVG:

```sh
just league-render /home/rob/runs/vsss-first 0001
```

Run the interactive Bevy viewer only on a machine/session with a graphical
display:

```sh
just league-view /home/rob/runs/vsss-first 0001
```

Compare iterations by changing the zero-padded identifier (`0001`, `0005`,
`0010`). Replays contain exact snapshots, actions, events, policy versions, and
checksums; playback does not rerun physics.

## Tournament

```sh
just league-tournament \
  /home/rob/runs/vsss-first/checkpoints/iteration-0010.pt \
  /home/rob/runs/vsss-first/tournament
```

This writes a canonical report plus side-switched evaluation replays. Promotion
is a separate explicit action driven by a fixture manifest; training never
silently replaces the current main policy.

Do not infer promotion quality from rollout return alone. Run the side-switched
tournament and inspect its report; the first 50-iteration reference run is
documented in `docs/evidence/m11-first-mappo-training-run.md`.
