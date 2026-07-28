# M15 frozen M14 control — 2026-07-28

M15 begins from an explicit control rather than treating its first generated
scenario as an improvement.

## Source identity

- code revision: `c6a247ad78bbbdaeecc79c8abcdded472f30af37`
- configuration: `experiments/configs/m14-mappo-adaptive.toml`
- scenario suite: `experiments/scenarios/m14-v1.json`
- promoted registry incumbent: M13 `directional-shared@425`
- incumbent terminal evidence: 0 wins, 9 draws, 1 loss over five independent
  paired-color seeds

M14 did not promote a replacement checkpoint. Its integration-scale curriculum
smoke produced six draws for both uniform and adaptive arms. Adaptive allocation
improved shaped return at that tiny budget but did not resolve a terminal
advantage. This is a control configuration, not a successful-policy claim.

## Current semantic coverage

The M14 suite contains nine static snapshots. Two trainable frontier snapshots
start with a moving ball (`interception vx=-0.3`, `defense vx=-0.2`); one
immutable mixed holdout uses `vy=0.2`. Approach, kickoff, clearance,
pass/receive, and shot snapshots use a static ball. Robot placement is inherited
from one base state.

Most importantly, curriculum outcome recording currently treats a blue goal as
success for every scenario kind. It does not distinguish a useful deflection,
save, interception, clearance, controlled reception, miss, opponent touch, or
unresolved timeout. M15 must compare against this exact behavior.

## Entry decision

No further 50-million-step run is authorized from this baseline. The next
high-budget command is published only after M15 generation, semantic predicates,
learnability, transfer, throughput, and paired promotion gates pass.
