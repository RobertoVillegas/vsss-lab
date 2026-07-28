# Training runs and visualization

M7 captures an immutable checkpoint and evaluation replay after selected real
rollout/optimize iterations. The viewer is outside the training hot loop.

## Start a run

```sh
just league-run /home/rob/runs/vsss-first 10 1
```

Arguments are `run_dir`, `iterations`, and `capture_every`. One M7 iteration is
one native C8 trajectory followed by one MAPPO optimization update; it is the
meaningful unit to inspect rather than a supervised-learning epoch.

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
