# Evidence

## Run 0006 diagnosis

- Best semantic checkpoint: iteration 350 at 51.8% aggregate success.
- Approach and shot reached 100%; pass/receive fell to 8.3%;
  rotation/recovery remained at 0%.
- Conservative replay analysis at 8.2 cm found iteration 500 accumulated
  44.0 ally-contact seconds and 59.8 opponent-contact seconds, including
  individual contacts lasting more than 19 seconds.
- Defensive uncovered ratio rose from 1.8% at iteration 25 to 31.4% at
  iteration 500.
- All 500 iteration records reported `heuristic-dynamic`, explained by the
  previous 1,000-iteration heuristic-only bootstrap.

## Behavioral tests

- Brief ally contact emits no penalty during the configured grace period.
- Contact becomes a deadlock only after grace.
- Productive opponent challenges remain measured but unpenalized.
- Separation after sustained contact records an escape.
- Roster compilation deterministically covers 1v0, 1v1, 2v1, 2v2, 3v2,
  and 3v3 with exact active-participant counts.
- Inactive participants are excluded from roles, distance, congestion,
  defensive geometry, actions, rewards, and PPO losses.
- Yellow-team congestion now uses yellow positions; the previous vector path
  implicitly measured blue positions for both controlled colors.

## Gates

- `just doctor`: passed with zero failures.
- `just lint`: Rust fmt/clippy, Ruff, formatting, mypy over 99 files, and
  TypeScript typecheck passed.
- `just build`: protocol, Rust workspace, Python compilation, and Vite passed.
- `just test`: Rust workspace including 13 physics regressions, 201 Python
  tests, and 7 viewer tests passed.
- `openspec validate m17-coordination-contact-curriculum --strict`: passed.

## CUDA integration and performance

A four-iteration, 64-world CUDA smoke completed 65,536 environment steps and
222 matches. Final cumulative throughput was 2,117.6 frames/s and
7.17 matches/s. This is above the approximately 1,885 frames/s observed at the
end of run 0006, so the new telemetry did not introduce a measured throughput
regression.

The smoke allocated every roster:

- 1v0: 1
- 1v1: 6
- 2v1: 3
- 2v2: 17
- 3v2: 5
- 3v3: 4

Contact telemetry reported ally/opponent seconds, 23/30 new deadlocks, and
35 escapes for the untrained policy. Rotation telemetry reported 14 attempts
and zero completions, matching the expected initial baseline.

A separate CUDA semantic-evaluation smoke correctly marked the initial
checkpoint promotion-ineligible: interception and save passed their floors,
while clearance, pass, and rotation failed theirs. The report and fallback
checkpoint remained readable.

## Known limitations

- Contact is inferred conservatively from canonical center distances because
  the Python batch API does not yet expose Rapier contact manifolds.
- Static obstacles are intentionally omitted from routine training. A future
  diagnostic holdout may add one without affecting the training distribution.
- Opponent diversity begins after iteration 100; the four-iteration smoke
  therefore correctly reports heuristic-only allocation.
