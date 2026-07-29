# Evidence

## M18 diagnosis

- PPO remained stable near KL 0.005 with low action saturation.
- Semantic success fell from 51.8% at iteration 25 to 47.6% at iteration 300.
- Both checkpoints passed three promotion gates, but the old selector preferred
  the later checkpoint's 4.2% rotation result despite lower global success and
  more unresolved trials.
- The final window had negative progress and the highest unresolved count,
  supporting multi-objective interference rather than optimizer collapse.

## M19 smoke

- Three CUDA iterations completed with 64 worlds and 49,152 environment steps.
- Throughput remained 1,956–1,999 frames/s.
- KL fell from 0.0100 to 0.0062 and clip fraction from 0.133 to 0.079.
- Allocation contained only approach, shot, and interception.
- Observed full-match allocation remained above the 5% foundation floor.
- A separate semantic integration smoke persisted phase, streak, gates, and
  checkpoint ranking successfully.

## Gates

- OpenSpec strict validation passed.
- Doctor reported zero failures.
- Rust workspace build and 13 physics correctness tests passed.
- Python suite passed: 207 tests.
- Replay viewer passed: 7 tests.
- Ruff, formatting, mypy, Clippy, and TypeScript checks passed.

## Rollback

`semantic_phased_curriculum` defaults to false, so existing configurations retain
their allocator. M19 has a separate configuration and run recipe; rollback is
selecting M18/M17 without checkpoint migration.
