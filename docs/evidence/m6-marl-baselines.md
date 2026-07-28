# M6 MARL baseline evidence

- Date: 2026-07-28
- Host: linux/amd64, Python 3.13.14
- Runtime: PyTorch 2.13.0+cu130, TorchRL 0.12.0
- Configuration: `experiments/configs/m6-mappo.toml`
- Seed: 7

## Functional result

The shared actor was distilled from 2,048 deterministic M4 teacher states in
4.40 seconds. Its final distillation loss was `0.12785093486309052`; the
algorithm-safe MAPPO checkpoint was 164 KiB.

Against random actions on the same 20 C8 seeds and 1,000-step horizons:

```json
{"margin":0.6629508006558076,"passed":true,"policy_progress":0.6589359857763132,"random_progress":-0.004014814879494388,"seeds":20}
```

Evaluation took 16.35 seconds. Actor and both critics also pass exact agent and
entity permutation tests. IPPO and MAPPO each execute finite parameter updates,
and stale policy versions are rejected before mutation.

## Readability and performance pass

`just benchmark-marl 5000` measured:

```json
{"actor_parameters":14468,"agents_per_call":3,"iterations":5000,"observation_us":33.536291000200436,"shared_actor_us":85.37335639994126}
```

The implementation keeps one batched actor call for all three agents, holds each
action across the four physics ticks in the 20 ms control period, and
minibatches complete agent groups so MAPPO never loses centralized context.
Named `TeamBatch` fields and `select_batch`/`stack_team_batches` helpers replace
opaque reshape/index expressions.

## Reproduction

```sh
just prepare-marl
just evaluate-marl
just m6-smoke
just benchmark-marl
```

## Known limitations

- The competence gate measures coordinated approach/ball progress, not match
  win rate; historical score-based promotion belongs to M7.
- M6 proves both loss paths and artifacts but does not claim long-horizon IPPO
  versus MAPPO sample-efficiency results.
- The distilled M4 initialization is a bootstrap, not evidence that tactical
  roles emerged from reward alone.
- Deep Sets intentionally discards some pairwise structure; attention and GNN
  comparisons remain later experiments.

## Rollback

Revert the M6 commits and delete external M6 checkpoints. M1–M5 simulation,
viewer, scripted baselines, and single-agent PPO remain unchanged.
