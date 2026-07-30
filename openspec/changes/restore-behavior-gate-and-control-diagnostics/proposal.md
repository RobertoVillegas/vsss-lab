# Restore the Behavior Gate and the Control Diagnostics

## Why

M24.2 moved the policy from wheel commands to semantic tokens, and the guards and
diagnostics that watch a run were left measuring the old action space.

The idle-spin detector compares a wheel differential against a threshold
calibrated for a policy that emits wheels directly. Every parametric wheel command
comes from `go_to_target`, which spends at most a fixed fraction of the wheel
limit on turning, so the measured differential can never reach the threshold. The
detector cannot fire for any state or any token. Its ratio is therefore always
zero, the promotion ceiling built on that ratio is vacuously true, and because the
same eligibility flag gates curriculum phase advancement, phase promotion runs
unguarded. The behavior it was written for — a slow robot turning in place, remote
from the ball — remains fully reachable.

Two further gaps make a run report something other than what it did. The action
diagnostics average over every roster slot, including robots that are absent from
the field and whose tokens the parser discards, so a large minority of each
reported average is untrained noise. The heading-change statistic reports an exact
right angle whenever a direction vector is near zero and crosses episode
boundaries. The viewer renders one chart for a hybrid policy and silently drops
both the exploration deviation and the skill mix, which are the two series needed
to see exploration collapse. The paired evaluation throughput is computed and then
discarded before the run record is written, though it is part of the declared
entry gate.

Finally, a learned opponent's token is re-parsed on every physics substep in the
single environment while the vector environment parses it once. The same
checkpoint therefore acts on a tighter control loop during evaluation and replay
than during training, which biases every paired policy comparison and makes the
recorded opponent intent valid only for the first substep.

## Milestone and non-goals

Maintenance for the active M24.2 milestone. Non-goals:

- no change to any reward coefficient, gate threshold, or promotion rule in a
  configuration file;
- no change to the action space, the policy architecture, or the probability
  model;
- no new evaluation or gate.

## What changes

- express idle-spin turn intensity as a fraction of the turn authority the action
  parser can actually request, so one configured threshold keeps its meaning
  across action spaces;
- make the behavior gate read the run's configured thresholds instead of a
  hardcoded copy;
- parse a learned opponent once per decision, matching the learner and the vector
  environment, and leave per-substep re-planning to the scripted controller;
- restrict the action diagnostics to agents that are on the field, exclude
  episode boundaries and undirected vectors from the heading statistic, and
  persist the paired evaluation throughput;
- render exploration deviation and skill mix for a hybrid policy.

## Success criteria

- a turn-in-place command that a skill parser can produce raises the idle-spin
  flag and can fail the behavior ceiling;
- retuning the thresholds in a configuration moves the gate;
- one decision produces one opponent parse in both environments;
- reported action statistics contain no contribution from absent robots;
- the run record carries resolved drills per second.
