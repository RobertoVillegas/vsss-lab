# Retire the Cross-Milestone Baseline Comparison

## Why

The semantic curriculum spec still requires paired regression against M14, a
milestone whose policies used continuous wheel actions and a different network,
reward basis, and curriculum. M24.2 replaced the action space with a parametric
parser and declared legacy isolation, so no environment can host both action
spaces in one match. The comparison is therefore unimplementable, and it was
never implemented: the live promotion decision reads semantic holdout floors, an
idle-spin behavior gate, and a heuristic match gate.

Its tooling has already decayed. `tools/m15_candidate_probe.py` loaded two
checkpoints by absolute path from runs that no longer exist on disk, and scored a
candidate against them through a paired evaluator that silently used the default
continuous parser for both teams. The M15 evidence records that probe's own
outcome as a rejection, and both M15 changes are archived.

The gate the project actually wants already exists in the adaptive training
spec: a paired terminal comparison against the promoted incumbent of the same
lineage. That is a same-parser, same-fingerprint comparison the league registry
and checkpoint loader already support.

## Milestone and non-goals

This is maintenance for the active M24.2 milestone. Non-goals:

- no cross-parser matchmaking between action spaces;
- no new promotion gate, threshold, or evaluation cadence in this change;
- no rewrite of archived M14 or M15 evidence, which stays as dated record.

## What changes

- retire the requirement to regress candidates against M14, replacing it with
  the promoted in-lineage incumbent;
- delete the orphaned M15 candidate probe and its recipe;
- keep the paired policy scorecard as the substrate for the incumbent
  comparison, bound to one action parser and covered by tests.

## Success criteria

- no live spec sentence requires a comparison that cannot be constructed;
- no recipe invokes a tool that cannot run;
- a paired scorecard across two action spaces fails loudly instead of scoring
  wheel commands as skill tokens;
- the deferred incumbent gate is recorded as an explicit follow-up task.
