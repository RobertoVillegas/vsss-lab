# Match-level conversion (next run after M24.6)

## Why

The M24.6 run (evidence.md in `finishing-from-angle`) proves the ADR 0027 mechanics and the
`shot ball_angle` curriculum: semantic success peaks at 0.768 (M24.5: 0.671), the curriculum
advances defense → rotation → integration (M24.5 never left defense), rotations are attempted
for the first time, and the full-match scoring ratio rises to +1.16. One number does not
follow: the full-match draw rate ends at 0.834, still far from the 0.70 gate, and the win
rate at 0.089. M24.5's stall was a *phase* stall — fixed. What remains is a *match-outcome*
stall: the policy is competent at skills and still draws four matches in five.

The evidence has three observations that any next-run hypothesis must explain:

1. The draw rate was **still declining monotonically** when the early-stop fired (0.883 at
   iter 1,000 → 0.834 at iter 2,900). The run was cut mid-`integration` phase on a holdout
   criterion that cannot distinguish phase-transition noise from real regression.
2. The **approach family is the weakest link**: success sits at 0.54–0.67 with
   `spawn_distance` pinned at the max level 1.0 while every other family ramps normally; the
   iter-1,000 semantic dip (0.560) and the worst match stretch (draw 0.883) coincide with
   the approach collapse.
3. **Skill scenarios score negatively**: 4,415 for vs 6,129 against (0.72); matches score
   positively (+1.16). The policy defends skill situations worse than it wins matches —
   consistent with a team that can finish but gets out-converted in chaotic settings.

## Milestone and non-goals

Proposed M24.7, evidence-driven: convert the proven skill layer into full-match wins
(draw ≤ 0.70 at the gate). Non-goals:

- no change to the strike/carry primitives (ADR 0022/0027 mechanics are accepted and
  measured; they are not the binding constraint);
- no change to the reward contract (ADRs 0015/0018/0020/0021) without a new ADR;
- no architecture change (shared CTDE policy, ADR 0008, is not in question).

## Hypotheses (ranked, each falsifiable)

### H1 — The early-stop truncated a run that was still converting

The gate was moving (draw −0.049 over the last third, win +0.037) when 12 consecutive
holdout regressions fired the stop. The regression window coincides with the `integration`
phase, which mixes multi-role scenarios whose per-trial success is naturally lower and
noisier. Falsifiable: rerun from the same config with the early-stop disabled (or
phase-aware: count holdout regressions per phase) and compare the gate at iter 3,052.

### H2 — The approach ladder over-demand blocks the conversion chain

`spawn_distance` at level 1.0 with success 0.54–0.67 means the policy is being fed its
hardest approach while the shot ladder (which needs a good approach to convert) is mid-ramp
(ball_angle pulled back to 0.375). Falsifiable: cap `approach.spawn_distance` at ~0.8 in
the curriculum (or pace it to the shot ladder) and observe whether shot success climbs past
0.6 and the draw rate follows within 1,000 iterations.

### H3 — The skill-against deficit is a defensive conversion problem, not a finishing one

Skill goals against (6,129) exceed for (4,415); the opponent's scripted scoring in skill
settings drags the integrated policy into defensive failure trials. Falsifiable: separate
the integrated phase's for/against accounting and measure whether the gate responds to
skewing the integration mixture toward fewer opponent-first scenarios.

### H4 — Match-level role dynamics still deadlock possession

Opponent possession seconds (139.7) exceed ally (90.2) and deadlocks persist (23 opponent /
14 ally) despite rotation attempts. The 0.70 draw gate may require the role formation
(ADRs 0023/0024/0026) to press, not merely to cover. Falsifiable: a match-relative
possession/territory term in the observatory (M24.1) correlated against draw rate per
checkpoint.

## Candidate solution space for M24.7

1. **Phase-aware early-stop** (tests H1): suppress the holdout-regression stop while the
   phase index is advancing; record the gate at full budget. Configuration-only, no
   contract change.
2. **Pace the approach ladder to the shot ladder** (tests H2): cap `approach.spawn_distance`
   during `foundation`/`defense`, release it only when `shot.ball_angle` ≥ 0.5. Curriculum
   policy change, no contract change.
3. **Integration-mixture accounting** (tests H3): log per-phase for/against and opponent
   mixture; skew integration scenarios toward ally-first starts. Observatory + curriculum
   change.
4. **Match-level behavior instrumentation before any shaping** (tests H4): extend the M24.1
   observatory with possession and territory metrics per checkpoint, and gate any role
   shaping change on their correlation with the draw rate. Instrumentation first, reward/
   role change only if the correlation is shown.

Steps 1–3 are cheap and can ride one run; step 4 may become its own ADR if the correlation
evidence points at role shaping.

## Success criteria

- Full-match draw rate ≤ 0.70 (gate) sustained over ≥ 8 consecutive evaluations, or
- draw rate < M24.6's 0.834 with a monotonic 500-iteration trend at stop, **and** the
  hypothesis tested (H1–H4) identified as the driver by its falsification data.

## Rollback

- Early-stop and curriculum pacing are configuration-only; the M24.6 run directory,
  checkpoints, and `best-semantic.json` are untouched and remain the comparison baseline.
