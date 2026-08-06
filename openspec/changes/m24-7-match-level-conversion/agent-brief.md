# Prime Agent Brief — VSSS Lab, M24.7: cross the full-match gate

This is the persistent goal. Do not stop iterating until the Terminal Goal is reached. Work
only in `/home/rob/src/vsss-lab` on `main`. Re-read `AGENTS.md`, `mise.toml`, and
`openspec/changes/m24-7-match-level-conversion/proposal.md` first.

## Terminal goal (the only definition of done)

Full-match draw rate ≤ **0.70** (`semantic_max_match_draw_rate`) **sustained over ≥ 8
consecutive semantic evaluations**, with win rate ≥ 0.15 in the same window, in a full
50M-step run. Every candidate change is measured against this gate and the M24.6 baseline
below. Reaching any other intermediate number is progress, not completion.

## Baseline (M24.6, run `vsss-m24-6-run-0002`, early-stopped at iter 2,900)

- Full-match: win 0.089 / draw 0.834 / loss 0.077 (29,536 matches)
- Semantic success_rate peak 0.768 (iter 2,500); phase progression defense → integration
  (M24.5 never left defense)
- Curriculum: `shot` ball_angle level 0.375, approach spawn_distance maxed 1.0 with
  success 0.54–0.67; rotations attempted (6, 3 completed); skill goals 4,415 for / 6,129
  against; full_match goals +1.16 for/against
- Evidence: `openspec/changes/finishing-from-angle/evidence.md`

## Hypothesis queue (iterate in order; one change per cycle; each must be falsifiable)

1. **H1 — early-stop truncated a still-improving run.** Draw was still declining
   monotonically (0.883 → 0.834) when 12 consecutive holdout regressions fired the stop.
   Test: config-only — set `semantic_regression_patience` high (e.g. 10_000) or
   `semantic_regression_warmup_evaluations` to cover the phase transitions; rerun and
   compare the gate at full budget (3,052 iterations).
2. **H2 — approach ladder over-demand blocks conversion.** Cap `approach.spawn_distance`
   curriculum level (~0.8) or pace it against `shot.ball_angle`; observe whether shot
   success climbs past 0.6 and draw follows within 1,000 iterations.
3. **H3 — skill-against deficit.** Separate per-phase for/against accounting; skew the
   integration mixture toward ally-first scenarios.
4. **H4 — role dynamics deadlock possession** (opponent possession 139.7s vs ally 90.2s,
   deadlocks 23/14). Instrument possession/territory in the observatory first; change role
   shaping only if correlation with draw rate is shown.

If a hypothesis is falsified, record the data, move to the next. If all four are falsified,
formulate H5+ from the evidence; never stop at "no hypothesis left".

## The iteration loop (repeat until Terminal Goal)

1. `just doctor`; pull latest state (`git log --oneline -5`, read evidence.md + proposal).
2. Pick the next hypothesis; write a 3-line plan in `openspec/changes/m24-7-match-level-conversion/tasks.md` first.
3. Implement. Config/curriculum/observatory changes only, unless an ADR is accepted.
4. Rebuild native bindings after any Rust change: `uv run maturin develop --release`.
5. Any change that alters executed behaviour gets a **new fingerprint**: copy the config
   with a bumped `policy_id` and `seed`, e.g. `m24.7-h2-mappo-shared-v1`. Never reuse a
   fingerprint across behaviourally different configs.
6. Smoke run first (25 iterations / 409,600 steps) — must exit cleanly with
   `allocation_valid: true`.
7. Full run (50,000,000 steps ≈ 3,052 iterations ≈ 8–12h wall). Launch with nohup, keep a
   log, and set a watcher that records the gate readout. One training run at a time; no
   competing heavy processes.
8. Evaluate against the gate at every semantic eval and at the end:
   `total_match_outcomes` win/draw/loss from `metrics.jsonl`, `success_rate` from
   `semantic-evaluations.jsonl`.
9. Gate passed → write the ADR, update evidence.md, commit everything signed, promote
   `best-semantic.json` per repo procedure, and STOP.
10. Gate not passed → update evidence.md with the falsification data, commit, next cycle.
11. After every code change run the local gates: `just lint`, `just build`, `just test`.

## Standing constraints (violations must be reverted)

- `vsss-spec` cannot depend on physics, PyTorch, ROS, or Python. ROS/ZeroMQ/Docker/Ray and
  Python dicts never enter the simulation hot loop. Active artifacts stay in
  Linux-native storage, never `/mnt/*`.
- No global installs of CUDA, PyTorch, OpenCV-CUDA, ROS, or Gazebo (devbox rule).
- Contract changes (observation, reward, action space, checkpoint layout, config schema
  that checkpoints embed) require an accepted ADR, an OpenSpec delta, golden fixtures, and
  contract tests **before** the change. Do not pull work forward from later milestones.
- Commits: small, signed (ED25519, machine identity already configured), Conventional
  Commits, no push to `main` without a PR; protected branches.
- Do not delete checkpoints, run dirs, or registry entries. Rollback is config-only or by
  reverting a signed commit.
- Keep the Python/native parity: any behaviour change must land in both, with the
  equivalence tests green.

## Ops cheat sheet (the environment is broken in known ways)

- **mise is broken** (a dead T3-Code shim). Never rely on bare `python3`/`uv`/`just` via
  mise shims. Use:
  - `UV=~/.local/share/mise/installs/uv/0.11.32/uv-x86_64-unknown-linux-musl/uv`
  - Rust: `export PATH="$HOME/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:$PATH"
    RUSTUP_TOOLCHAIN=1.97.1-x86_64-unknown-linux-gnu`
  - Reading JSON: `/usr/bin/python3` (system interpreter, not the mise shim)
- Provision before any run: `$UV sync --group train --locked` and
  `$UV run maturin develop --release` (a stale native `.so` silently breaks parity — this
  bit us once).
- Allocate run dir: `$UV run python tools/next_run_dir.py vsss-m24-7-run`.
- Smoke: `nohup $UV run --group train python -m vsss_league.cli run --config <config>
  --match-config tests/golden/m1_match_config.json --match-state tests/golden/m1_match_state.json
  --run-dir <dir> --steps 409600 --capture-every 25 --capture-seconds 30 --checkpoint-every 25
  --device auto --num-envs 64 --semantic-eval-every 25 --semantic-eval-seeds 3 > <dir>/smoke.log 2>&1 &`
- Full run: same command with `--steps 50000000 --capture-seconds 60`, log to `<dir>/train.log`.
- Relevant config knobs (M24.6 file is the template):
  `semantic_regression_patience = 12`, `semantic_regression_warmup_evaluations = 16`,
  `semantic_max_match_draw_rate = 0.70`, `semantic_min_match_win_rate = 0.20`,
  `semantic_min_goals_per_minute = 0.2`, `semantic_phase_patience = 2`,
  `semantic_full_match_fraction = 0.35`, `strike_clearing_enabled = true`,
  `strike_clearing_distance = 0.16`, curriculum levels under `semantic_curriculum`.
- Gate readout after each eval:
  `/usr/bin/python3 -c "import json; rows=[json.loads(l) for l in open('<dir>/metrics.jsonl')]; r=rows[-1]; t=r['total_match_outcomes']; print({k: t[k] for k in t})"`
- Watcher pattern: poll `metrics.jsonl` every 60s, log iteration, dump the readout when a
  milestone iteration is reached (`/tmp/opencode/watch-m24-6-v3.sh` is a reference).
- Eval cadence: `--semantic-eval-every 25` → one eval per ~25 iterations; 8 consecutive
  gate-passing evals ≈ 200 iterations ≈ ~1.5h of wall time once the policy is there.

## Handoff contract

Each cycle ends with: signed commits, evidence.md updated with numbers (not prose),
tasks.md checked, gates green (`just lint`, `just build`, `just test`), and the next
hypothesis stated. The PR/commit notes include artifacts (run dir, checkpoint path),
evidence, known limitations, and rollback.
