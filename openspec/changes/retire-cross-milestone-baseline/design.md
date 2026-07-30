# Design

## Why the comparison cannot be built

`MarlMatchEnv` and `VectorMarlMatchEnv` hold a single `action_parser` and apply
it to both teams, so one match cannot host a width-2 wheel policy and a width-4
parametric policy at once. Making the parser per-team would put two action
spaces inside one episode, which contradicts the M24.2 requirement that a
parametric policy carry a distinct parser and policy identity. The checkpoint
loader already refuses opponents whose config fingerprint differs, so the league
treats a cross-lineage opponent as an error rather than a fixture.

## What the retired comparison was

The frozen "M14" reference was the M13 `directional-shared@425` continuous-wheel
policy, frozen so that shaped-return improvements could not hide a transfer
regression. Its recorded result was a rejection: `semantic-shared@50` scored 0
wins, 8 draws, and 2 losses against it, with zero holdout successes. The
comparison answered a question about M13-era reward shaping, not about semantic
skill acquisition, and M14 never promoted a replacement checkpoint.

## Decision

Retire the cross-milestone term and keep the in-lineage one. The paired policy
scorecard survives with an explicit `action_parser`, because the adaptive
training spec already requires a paired terminal bound against the promoted
baseline, and the registry plus checkpoint loader already restrict that pairing
to one fingerprint. Implementing that gate is deferred to its own task so this
change alters no threshold or cadence.

## Validation, compatibility, rollback

Validation is the paired scorecard test for one lineage plus the rejection of a
mixed pairing, the environment-boundary action shape tests, and OpenSpec strict
validation. Compatibility is unaffected: no promotion threshold, config key,
checkpoint format, or replay schema changes, and archived M14 and M15 evidence
stays as dated record. Rollback is `git revert` of this change; restoring the
deleted probe additionally requires two checkpoints that no longer exist, which
is itself the argument for retiring it.
