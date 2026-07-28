# M15 bounded probes — 2026-07-28

Revision under test: working tree after `36d91ef`.

## Paired controls

Both controls used five immutable seeds, both colors, and all six families
(60 trials each).

| Control | resolved drills/s | approach | interception | save | clearance | shot | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 7.81 | 0% | 0% | 0% | 0% | 0% | 0% |
| heuristic | 9.83 | 100% | 0% | 0% | 0% | 0% | 0% |

An initial probe exposed high incidental random interception/save rates. The
defenders were moved off the ballistic path and the controls were repeated;
random success then fell to zero in every family. The heuristic remains a
useful positive control only for approach, which makes the other skills genuine
learning targets rather than scripted demonstrations.

## Matched-compute smoke ablation

Each arm used two paired seeds, ten optimization iterations, eight worlds,
eight rollout steps, and 1,280 environment steps.

| Arm | semantic success | unresolved | full-match terminal score |
| --- | ---: | ---: | ---: |
| frozen M14 static | 0% | 33.3% | 0.50 |
| predicates only | 0% | 41.7% | 0.50 |
| terminal skill reward | 0% | 41.7% | 0.50 |
| additional dense shaping | 0% | 33.3% | 0.50 |
| 75% full-match control | 0% | 33.3% | 0.50 |

This is a smoke probe, not enough compute for promotion. It proves the arms run
at matched steps, but none learned a holdout success at this fidelity. The
terminal and dense arms therefore remain unproven; no shaping arm is selected.

## Decision

The entry decision is `reject_large_run`. M15 implementation may continue with
bounded probes, but no 50M-step command is authorized. Required remediation:

1. reduce incidental random success in interception/save difficulty cells;
2. demonstrate learned shot, clearance, and pass success above both controls;
3. rerun the matched-compute ablation at screening fidelity;
4. only then evaluate a checkpoint against frozen M14, heuristic, historical
   league policies, and five-seed immutable holdouts.

Canonical raw outputs remain in ignored local paths
`experiments/reports/m15/{random,heuristic,ablation}.json`.

## End-to-end throughput

A warmed three-iteration benchmark used 64 worlds and 6,144 environment steps
per device:

| Device | frames/s | matches/s | resolved drills/s | elapsed |
| --- | ---: | ---: | ---: | ---: |
| CPU | 2,284 | 3.72 | 3.72 | 2.690 s |
| CUDA | 2,700 | 4.83 | 4.83 | 2.275 s |

CUDA was available and used for policy inference/optimization; Rapier remained
CPU-authoritative. Total matched-ablation compute was approximately 28.8
seconds across five arms, excluding environment setup and the separate control
evaluations.

## Bounded candidate screen

`semantic-shared@50` trained for 204,800 environment steps on CUDA in 83.03 s
(2,467 frames/s), completing 132 drills/matches. It was then evaluated on five
immutable seeds, both colors, and all six families:

- semantic holdouts: 0/60 successes, 40 failures, 20 unresolved;
- heuristic full matches: 0 wins, 10 draws, 0 losses;
- frozen `directional-shared@425`: 0 wins, 8 draws, 2 losses;
- historical `directional-shared@1450`: 0 wins, 10 draws, 0 losses.

The candidate is physically valid but learned no holdout skill and regressed
against the frozen incumbent. This is direct rejection evidence, not merely an
absence of positive evidence. The local ignored checkpoint is
`experiments/reports/m15/candidate/candidate.pt`; it is not promoted or
published.
