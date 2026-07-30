# Design

## Time units

- One physics tick is 5 ms.
- One policy decision repeats four physics ticks and represents 20 ms.
- One full episode is bounded at 1,500 decisions or 30 virtual seconds.
- One PPO iteration collects 256 decisions per world, or 5.12 virtual seconds.
- Worlds persist across PPO iterations and reset only on episode completion.

## Credit assignment

Continuing rollout boundaries bootstrap from the critic value of the following
observation. Goal, draw, stagnation, semantic success, semantic failure, and
semantic timeout remain episode boundaries and block GAE from crossing into the
reset state.

Advantages continue to be normalized over active samples for PPO stability.
Raw matches are not averaged before GAE. The dashboard distinguishes fragment
return from the mean return of episodes completed during the iteration.

## Selection

Semantic checkpoints retain skill-family and idle-spin gates. A deterministic
paired-side match scorecard adds explicit win-rate and draw-rate gates. M22
requires at least 20% wins and at most 70% draws against the reference heuristic.

## Defensive transfer

Coverage is considered absent when a threatening ball is behind every active
robot and no robot is close enough to challenge. Under threat, the coverage
target moves from the goal center toward an interception point. Clearance and
shot generators include near-goal emergency and loose-finishing variants.
