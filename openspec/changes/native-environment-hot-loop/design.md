# Design

## Where the time is

Over one iteration, 64 worlds, 256 decisions:

| stage | seconds |
| --- | --- |
| `_native.step_repeated` for the whole batch | 0.06 |
| the Python layer in `env.step` | 4.95 |
| policy forward on the GPU | 0.32 |
| PPO update | 0.30 |

The physics is 1.2 per cent of `env.step`. The Python layer runs about nine loops over the
world list per decision — observations, roles, ball direction, attacker alignment, goal
geometry, congestion, contact, deadlocks, idle spin — each doing small numpy work per world.
At 64 worlds that is several hundred Python calls per decision.

## Why a new crate

`vsss-spec` is the canonical domain model and must not acquire training concerns; that
boundary predates this work. `vsss-physics-rapier` is a backend behind `vsss-physics-api`, and
observations are not physics. The ported logic therefore lands in its own crate depending on
`vsss-spec`, consumed by `vsss-python` and, in time, by anything else that needs the same
features without paying for Python.

## Batch at the boundary, not per world

The current bindings already expose the batch: `step_repeated` advances every world in one
call and rayon parallelizes above a threshold of thirty-two worlds. The ported calls follow
that shape. Crossing the binding once per decision instead of once per world is most of the
win; the rest is that the arithmetic stops being numpy dispatch on two-element arrays.

## Equivalence before speed

The failure mode of a port like this is not a crash, it is a quiet semantic drift: a reward
term that rounds differently, an observation channel in a different order, a threshold that
flips. Each slice therefore ships with a test that runs both implementations over recorded
states and compares them, and the tolerance is set below the smallest gap that any comparison
in the code depends on. A slice that is faster and disagrees is a regression.

Keeping the Python implementation until the slice is retired also means the migration can stop
between any two slices with a working tree, which matters because this competes for attention
with the training work itself.
