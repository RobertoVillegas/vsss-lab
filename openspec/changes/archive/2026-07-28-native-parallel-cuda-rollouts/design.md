## Decisions

1. Keep Rapier as the deterministic CPU physics backend and parallelize
   independent worlds rather than introducing a second physics model.
2. Preserve stable result order by collecting parallel results by indexed world.
3. Run fewer than 32 worlds sequentially because measured Rayon scheduling cost
   exceeds the work saved for tiny batches.
4. Fuse action repeats only when opponent actions remain fixed; the heuristic
   controller continues recomputing between physics steps.
5. Default to 64 worlds. 256 raises raw throughput further but increases memory
   and reduces policy-update frequency too aggressively for the baseline.
6. Apply monospace and tabular-number features at the viewer root so every
   descendant inherits stable glyph metrics.

## Evidence

Measurements and frame-budget interpretation are recorded in
`docs/evidence/m12-native-parallel-rollouts.md`.
