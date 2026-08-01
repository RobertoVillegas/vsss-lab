# ADR 0019: The environment hot loop belongs in Rust

- Status: accepted
- Date: 2026-08-01

## Context

`AGENTS.md` already states the boundary: ROS, ZeroMQ, Docker, Ray, and Python dictionaries
never enter the simulation hot loop. The physics has honoured it since M2 and lives in
`vsss-physics-rapier`, batched across worlds by `vsss-batch` with rayon. What grew back into
Python is everything the physics does not do: observations, role assignment, reward terms,
contact and deadlock detection, idle-spin flags, and the free-ball restart, all computed with
a Python loop over worlds on every decision.

The cost was measured, not assumed. Over one iteration of the live configuration, 64 worlds
and 256 decisions:

| stage | seconds | share |
| --- | --- | --- |
| native physics alone | 0.06 | 1.2 per cent of `env.step` |
| the Python layer above it | 4.95 | 98.8 per cent of `env.step` |
| policy forward on the GPU | 0.32 | |

The simulation is not what costs. A profile of the same configuration put ninety-seven per
cent of an iteration in the rollout and three in the PPO update, with the GPU at seventeen per
cent. A first pass of pure-Python fixes — planning the scripted opponent once per decision
instead of once per physics substep, replacing scalar `np.clip`, and rewriting `_strike_target`
on scalars — already returned 2.8x. What remains above the physics is per-world Python.

If the whole layer moved down, the floor is the physics plus the policy forward plus the
update: about 0.68 seconds against today's 8.76, so roughly thirteen times. Fifty million
steps would take half an hour rather than a working day, which changes what can be asked of a
run: ablations instead of one attempt.

## Decision

Move the per-world environment computation into Rust, behind the existing PyO3 batch surface,
one slice at a time. Each slice is complete only when a golden-equivalence test shows the
native path agrees with the Python one on recorded states, and the Python implementation stays
in the tree as the reference until its slice is retired.

The learner does not move. PPO and the network are three per cent of the time and sit on
PyTorch's ecosystem; porting them would trade the ecosystem for three per cent. Rust has
maturing tensor crates, and none of that is relevant to where this project's time goes.

Slices, in the order their cost and their independence suggest:

1. observations, the most self-contained: state in, tensors out;
2. role assignment, which is called twice per world per decision and should be checked for
   redundancy before it is ported;
3. reward terms, which the decomposition already names individually, so equivalence can be
   asserted term by term;
4. contact, deadlock and idle-spin detection;
5. the free-ball restart, which today edits a JSON snapshot dictionary inside the hot loop and
   is the clearest violation of the stated boundary.

## Consequences

- The hot loop stops being Python, which is the architecture the repository already declared.
- Equivalence tests, not benchmarks, decide when a slice is done. A port that is faster and
  subtly different is a regression, and floating-point differences that change a branch are
  the failure mode to watch.
- Every slice is reversible while its Python reference remains, so the migration can stop at
  any point with a working tree.
- Throughput gains change what a baseline means: runs before and after a slice are comparable
  in behaviour only if its equivalence test passes, which is precisely why the test gates it.
- A new crate holds the ported logic. `vsss-spec` stays free of it, since the canonical domain
  model must not acquire training concerns.
