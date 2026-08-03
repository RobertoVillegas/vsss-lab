# Dynamic Role Formation Credit

## Why

Run 0013 learned an attacker but not sufficient role separation: support selected `strike` on
55.1 per cent of the final replay and coverage on 48.2 per cent. The terminal reward is correctly
shared, but no dense term says that the rest of the team improved its transient formation.

## Milestone and non-goals

This is an M24.3 corrective change based on the completed run. Non-goals:

- no reward tied to robot identity or last touch;
- no fixed goalkeeper, attacker, or dedicated policy;
- no action mask that prevents an emergency challenge;
- no lowering of semantic, match, or behaviour promotion gates;
- no change to the carry coefficient selected by run 0013.

## What changes

- Add bounded potential shaping for active support and coverage responsibilities.
- Expose primitive selection fractions by dynamic role in training metrics.
- Start the next run from a fresh reward fingerprint; run 0013 checkpoints are evaluation and
  distillation inputs, not resumable optimizer state under the new reward.

