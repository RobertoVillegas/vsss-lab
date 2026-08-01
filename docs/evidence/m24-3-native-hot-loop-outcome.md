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

## Whole-trajectory equivalence

Each slice has its own equivalence test, but those compare one function at a time. Driving both
environments with identical actions compares the composition, which is what a run actually
depends on. 200 decisions across 32 worlds, same seeds, same action stream:

| | difference |
| --- | --- |
| per-decision rewards, terminals, ball position, observation context | at most 1.2e-7 |
| decisions agreeing within 1e-6 | 200 of 200 |

Run twice. The first pass used the configuration as written and looked cleaner than it was:
seven reward terms carry a zero coefficient there, so they contribute zero to both
implementations and agree without either being exercised. The second pass forced every reward
coefficient to one, which brought the attacker alignment, ball direction, progress, action
delta, wheel effort, useful touch, goal geometry and defensive coverage terms into the
comparison. All agreed exactly or to 5e-8.

Four terms are still not covered by this: the two deadlock penalties, idle spin and teammate
congestion never fire under random play in 200 decisions. Those are covered by the per-slice
tests instead, which force the conditions deliberately — the contact test widens the contact
distance to one metre so streaks reach the deadlock branch, and the idle-spin test asserts that
flags actually fired rather than comparing all-false against all-false.

## A cost that only appears at scale

Measuring throughput against world count found a defect the per-decision profile could not.
`collect_self_play_trajectory` rebuilt every world's observation in Python whenever *any* world
reset:

```text
if reset_occurred:
    next_observation = stack_team_batches([
        build_team_observation(state, ...) for world, state in enumerate(environment.states)
    ])
```

Resets are rare — 0.2 per cent of world-decisions — but with 512 worlds the chance that *some*
world resets is not. It fired on 95 of 256 decisions and cost a quarter of the iteration: the
price of one reset was a full-batch rebuild. At 64 worlds it fires less often and costs eight
times less per firing, which is why it was invisible until the batch grew.

Replaced by a native rebuild over the assignments already in force. Rebuilding must not
re-assign: the untouched worlds have not moved, and running the hysteretic assigner a second
time on the same state gives a different assignment, not a cheaper one.

Throughput by configuration, before and after, measured under contention with a live run so
both columns are lower bounds:

| worlds × steps | before | after | |
| --- | --- | --- | --- |
| 64 × 256 | 3 202 | 4 038 | +26% |
| 128 × 256 | 5 422 | 6 116 | +13% |
| 256 × 256 | 6 622 | 8 708 | +32% |
| 512 × 256 | 7 731 | 11 107 | +44% |
| 512 × 128 | 8 922 | 10 968 | +23% |

Scaling worlds is sublinear — eight times the worlds buys 2.75 times the throughput — because
what remains is per decision rather than per world: 256 sequential policy forwards, 256 rounds
of von Mises sampling, 256 distribution constructions.

One incidental result: 512 × 256 now beats 512 × 128. The shorter rollout was only ahead because
it triggered the reset rebuild less often. That matters because 128 steps at 50 Hz is 2.56
seconds of experience against a two-second credit horizon, so GAE would truncate inside the
horizon — the faster-looking option was also the riskier one, and it is no longer faster.

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
