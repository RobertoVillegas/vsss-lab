# The native hot loop, measured against what it predicted

ADR 0019 asked for its own thirteen-times estimate to be replaced by measurement as it was
approached. It has been, and the estimate did not hold. This records what was measured, and why
the prediction was wrong, because the reason is more useful than the number.

## Method

Same machine, same configuration (`experiments/configs/m24-3-mappo-circular.toml`, 64 worlds,
256 rollout steps), and the same compiled extension in both cases. The before case runs the
Python tree at `b7b86a9`, the commit before the first slice, against the current binary — the
old Python calls no method the old binary did not have, so the only variable is the Python.

Four iterations, the median of the last three, discarding the first to warm allocation.

## What changed

| | before | after | |
| --- | --- | --- | --- |
| environment step, 96 decisions × 64 worlds | 3.47 s | 0.23 s | 15.1x |
| full training iteration | 8.60 s | 4.03 s | 2.13x |

The environment did what the ADR expected. The iteration did not, and the gap between the two
rows is the whole finding.

## Why the prediction was wrong

The ADR measured one environment step and found the physics at 1.2 per cent against the Python
above it at 98.8 per cent. It then treated everything above the physics as movable and projected
the whole iteration down to 0.68 seconds. That step is where the reasoning failed: the
environment is not the iteration, and the parts of the rollout that are not the environment were
never in that split.

Profiling an iteration now, the 4.03 seconds divide roughly as:

| stage | seconds | share |
| --- | --- | --- |
| rollout collection | 3.99 | 87 |
| — environment step | 0.77 | 17 |
| — von Mises rejection sampling | 0.71 | 15 |
| — distribution construction and validation | 0.45 | 10 |
| — tensor creation | 0.23 | 5 |
| PPO update | 0.59 | 13 |

The ADR also stated that the learner would not move because PPO and the network were three per
cent of the time. That was true when it was written and is not true now: the denominator shrank
by a factor of eleven under the environment, so the learner side is around a quarter of an
iteration and the sampling machinery inside the rollout is another quarter. The conclusion the
ADR drew from the three per cent — that porting the learner would trade PyTorch's ecosystem for
nothing — still holds, because the remaining cost is in PyTorch's own sampling and distribution
code rather than in code this project wrote.

## What is worth doing next

The largest single item is von Mises rejection sampling, which the circular heading
parameterization requires (ADR 0014) and which is PyTorch's implementation, not ours. It rejects
until it accepts, so its cost varies with concentration; a high-concentration policy samples
faster than an exploratory one.

Distribution construction is next, and most of it is argument validation that cannot fail here:
concentration is a softplus with a floor and scale is a softplus, so the invariants PyTorch
checks are enforced where the parameters are produced. Turning validation off for training
measured 3.75 s to 3.23 s on this configuration, a further 14 per cent.

Neither is an environment concern, so neither belongs to this change.
