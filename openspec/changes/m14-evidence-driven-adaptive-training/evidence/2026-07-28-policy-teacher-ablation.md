# Policy and teacher ablation evidence — 2026-07-28

## Comparable systems

The maintained RLGym stack uses an explicit action parser and normalized
observations; Nexto's released implementation uses a symmetric lookup action
table and permutation-aware entity processing. Recent MARL evidence reviewed
in the M14 design does not establish recurrence as universally superior: its
credible entry condition is partial observability.

## Decisions

- **Adapt:** a nine-action symmetric differential-drive lattice, evaluated at
  the same physical control frequency as continuous wheels.
- **Adapt:** shared GRU state has explicit world/agent axes and per-world reset.
- **Adapt:** entity attention consumes teammates and opponents as a set and
  exposes attention weights as diagnostic telemetry.
- **Adopt:** exact-simulator bounded CEM only for atomic skills; replay the
  winner and require an explicit success predicate.
- **Reject:** whole-match TAS and demonstrations that score through invalid
  physics.
- **Defer:** KAN and learned world-model actors until MLP/GRU/attention matched
  budget comparisons justify added complexity.

## Falsifiable gates

Resetting one world cannot change another world's memory. The action lattice
must be sign symmetric and bounded. Attention weights must cover all five
visible entities. Invalid or unsuccessful planner trajectories cannot enter a
demonstration artifact.

## Executed teacher smoke

`just m14-teacher-ablation experiments/reports/m14-teacher-smoke.json cuda 3`
executed the exact Rapier teacher, verified its winning atomic trajectory, and
compared scratch MAPPO, verified imitation, and imitation followed by MAPPO on
three paired seeds. All three arms finished `0-6-0` for a terminal score of
`0.5`. The imitation arms reached mean clone loss `0.036642`, but neither
improved terminal outcomes. The machine-readable decision is therefore
`no_terminal_advantage`: imitation remains available behind an experiment
boundary and is not promoted.
