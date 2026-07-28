# M7 league and self-play evidence

- Date: 2026-07-28
- Host: linux/amd64, Python 3.13.14
- Configuration: `experiments/configs/m6-mappo.toml`
- Seed: 7

## Real training and capture

A three-iteration run performed three native C8 rollout/optimize cycles and
advanced the learner from policy version 0 to 3:

```json
{"iterations":3,"latest_version":3,"viewer_replays":"<run>/replays"}
```

The run took 8.09 seconds with 915,868 KiB peak RSS and produced:

- four immutable checkpoints (`0000` through `0003`);
- three metrics records;
- three 1,000-frame learned-policy replays;
- an atomic registry linking every checkpoint hash and parent.

One run occupied 7.6 MiB. The iteration-0003 replay checksum was
`ab1b5df9c9a2dc8e7c6035ed626b41a1be1d3b3412f34fca4e8d0776a2ae7c32`
and passed the existing replay checksum inspector. The same replay rendered to
SVG through the existing viewer pipeline.

## Tournament and non-regression

A side-switched four-match tournament over two seeds took 4.25 seconds. The
three-update candidate lost all four fixtures against the M4 heuristic and its
Elo moved from 1000 to 944.20.

This is useful negative evidence: a few self-play PPO iterations can regress
against the heuristic. M7 preserves the original main, registers the new
checkpoint only as a candidate, and the promotion contract rejects it rather
than silently replacing main.

The canonical tournament report, Elo conservation, fixture floors, identity
gate, and promotion decision are covered by deterministic tests.

## Reproduction

```sh
just league-run /home/rob/runs/vsss-first 10 1
just league-inspect /home/rob/runs/vsss-first
just league-render /home/rob/runs/vsss-first 0001
just league-view /home/rob/runs/vsss-first 0001
```

The interactive Bevy command requires a graphical session. On the headless
devbox use `league-render`; open the resulting SVG on the client device.

## Known limitations

- M7 workers are synchronous and local.
- The default smoke-scale tournament is evidence of mechanics and reproducible
  rejection, not a statistically powered policy comparison.
- Elo is descriptive; fixture non-regression remains the promotion authority.
- Replays are lossless JSONL and about 2 MiB per 1,000 captured control frames.
- Automatic checkpoint pruning is intentionally absent because historical
  retention is part of the milestone contract.

## Rollback

Revert M7 commits and remove the selected run directory. Checkpoint/replay
deletion is explicit because active artifacts live outside Git.
