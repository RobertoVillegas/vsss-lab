# M24 — Hierarchical trajectory control

## Problem

M23 demonstrates isolated approach, interception, clearance, save, and shot
skills, but full matches still enter low-motion valleys after ball contact.
Inspection of replay 425 shows that the policy emits non-zero wheel commands
while producing little translational motion and repeatedly fails to reacquire a
stationary ball. The cooperation phase then allocates most worlds to
`pass_receive`, asking the policy to coordinate two trajectories before a
single robot has a reliable approach-and-strike controller.

The current MAPPO actor controls both differential-drive wheels at 50 Hz. The
CoRL 2025 robot-soccer reference instead separates a slower MAPPO strategy from
verified locomotion and ball-manipulation skills. Its ablation reports smoother
ball trajectories and substantially better task performance than direct
end-to-end motor control.

## Change

- Add a deterministic, exact-simulator-compatible primitive action parser.
- Let the learned policy choose `stop`, `navigate`, or `strike` plus one of eight
  canonical directions.
- Make `strike` predict a reachable ball location, acquire a point behind the
  ball, align, and then drive through contact in the requested exit direction.
- Keep primitive execution stateless and causal so vector worlds, historical
  opponents, replay, and hardware-oriented evaluation use identical semantics.
- Add trajectory benchmarks for reacquisition, contact, exit-direction error,
  useful impulse, and time to contact.
- Mark replay episode identity explicitly and prevent prediction error samples
  from crossing resets.
- Rebalance the phased curriculum so cooperation cannot monopolize rollouts
  before individual reacquisition remains consolidated.
- Add paired MAPPO/IPPO short-run commands using the same primitive parser.
- Preserve M23 and direct continuous wheels as a rollback baseline.

## Evidence required before a long run

- Primitive geometry and symmetry unit tests.
- Exact Rapier tests for stationary and moving-ball reacquisition.
- Replay reset and prediction-metric contract tests.
- Paired short smoke runs for MAPPO and IPPO.
- Full local doctor, build, test, and lint gates.

## Non-goals

- Learned joint-level or wheel-level motor primitives.
- Camera-derived prediction as a training input.
- Physical-robot deployment or M12 hardware validation.
- Automatically changing rewards from evaluation results.
- Claiming MAPPO superiority over IPPO without multi-seed long-run evidence.
