# ADR 0016: Difficulty must modulate the demand a drill is scored on

- Status: accepted
- Date: 2026-07-31

## Context

The semantic curriculum allocates drills by learning progress and lowers difficulty
for families a policy is failing. That mechanism cannot work if a family's difficulty
axes do not move the quantity the family is scored on.

Clearance is scored on the ball leaving the defensive third: the primary robot must
touch it and the ball must pass `x = -0.10` in the attacking frame. Its generator set
the ball's starting depth by a coin flip between `-0.62` and `-0.48`, independent of
every difficulty axis. The axes moved the incoming ball speed and the robot's spawn
offset, so the easy end of the drill asked for the same 0.38 m of ball displacement as
the hard end, inside the same 4.8 s horizon.

Measured on run `vsss-m24-3-run-0003` at iteration 1100, the consequence was total:
clearance scored **0 of 10 at every difficulty band**, with 38 of 40 trials touching the
ball 2.48 times on average and ending in `timeout`. The policy reached the ball and
could not move it far enough, at any difficulty. Because it never succeeded once, no
clearance reward ever arrived, so nothing taught the skill, and the difficulty
curriculum had no lower rung to offer.

That single family then blocked everything downstream. Advancing the `defense` phase
requires `clearance >= 0.35`; interception and save_deflection sat comfortably at 0.50
for 700 iterations. Clearance crossed 0.35 four times in 43 evaluations and never twice
consecutively, so with `phase_patience = 2` the maximum streak was one. The phases after
`defense` are `cooperation`, which teaches passing, and `rotation`, which teaches
rotation recovery. Neither received a single training allocation for the entire run, and
both scored 0.00 throughout. The policy learned to score directly and never learned to
play, because it was never shown play.

## Decision

A difficulty axis SHALL move the demand its family is scored on. For clearance the
ball's starting depth is interpolated across `spawn_distance` from just inside the
defensive third to deep in it, replacing the coin flip. The scored threshold and the
predicate are unchanged, so a clearance still means the same thing.

The scenario generator revision is bumped, which changes every parameter and state
digest. Holdouts remain immutable within a revision, and evaluations across revisions
are explicitly not comparable.

Verified without retraining: the same iteration-1100 checkpoint moves from clearance
0.00 to **0.47** — 10 of 10 at the easiest band, 9 of 10 at the next, 0 of 10 at the
two hardest, 19 `ball_cleared` where there had been none. The ramp is now a gradient the
learning-progress curriculum can climb, and 0.47 clears the phase gate.

## Alternatives considered and rejected on measurement

- **Raise the useful-contact coefficient.** The term measures the right quantity, signed
  `Δv_x` toward the goal on the contact edge, and contributes 4e-6 per decision at the
  policy's operating point. It is linear near zero, so no coefficient can make a 0.05 m/s
  touch a meaningful signal without making a 1.5 m/s strike pay several times a goal. It
  cannot bootstrap a skill from below, and scaling it was rejected on that structural
  argument rather than on taste.
- **Let the strike drive through harder.** The hypothesis was that the controller's
  proportional approach caps `forward` at `2 · distance`, so the 0.28 m drive-through
  offset limits authority to 0.56. Measured by sweeping that offset to 0.45, 0.60 and
  1.00: clearance moved from 0.00 to 0.03 and no further. The executor was not the
  binding constraint.
- **Lower the phase patience to one.** The phase would have advanced four times, but the
  policy would have entered `cooperation` still unable to clear. It treats the symptom.
- **Relax the clearance predicate.** That changes what a clearance means rather than how
  hard the instance is, and weakens the milestone's own claim.

## Consequences

- Difficulty for clearance now spans genuinely easy to genuinely hard, so the curriculum
  can prioritize the family and have somewhere to prioritize it to.
- The phase gate can open, which is the only path by which passing and rotation recovery
  are ever taught.
- All prior semantic evaluations are on the previous generator revision and are not
  comparable; the milestone needs a fresh baseline.
- The same audit should be applied to the other families: this ADR fixes the one case
  measurement exposed, and does not claim the rest were checked.
