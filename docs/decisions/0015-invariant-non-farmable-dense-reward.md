# ADR 0015: Invariant and non-farmable dense reward

- Status: proposed
- Date: 2026-07-30

## Context

M20 deliberately removed directional ball shaping because positive horizontal ball
velocity could be earned on paths that miss the goal, and it chose a goal-aperture
potential instead, on the stated principle that a static pose must not farm
reward. That decision stands: the absence of a raw dense term pointing at the goal
is intended, not drift, and this ADR does not propose reinstating one.

What the M24.2 audit found is that the accepted design does not hold in three
places.

First, the potential term is applied as `γ·Φ(s') − Φ(s)` with the shaping discount
equal to the training discount, which is the correct policy-invariant form, but
`Φ` is not treated as zero at a terminal state. The last transition of an episode
therefore pays for ending with good geometry. With the configured coefficient the
bias is on the order of a tenth of the draw penalty — small, but it rewards
drawing in a favorable pose, which is the loophole the potential was chosen to
close.

Second, the only dense term that can accumulate positively without bound,
`useful_touch_impulse`, is non-negative and triggered on the team's contact edge.
Sustained control pays nothing, a lateral pass pays nothing because only the
horizontal velocity component counts, and no contact can ever be penalized. A
robot oscillating at the boundary of the contact envelope re-triggers the edge on
every decision and collects on each re-entry. That is a farm of the same kind M20
set out to prevent, reachable without ever advancing the ball.

Third, the terminal pressures are incoherent with each other. The time penalty is
divided by the horizon, so an entire episode costs a twentieth of what a draw
costs, and stagnation is penalized twice as hard as a draw. Keeping the ball
jiggling enough to avoid the stagnation terminal while never attempting a goal is
therefore cheaper than either stagnating or risking a concession. The first M24.2
evaluation returned nine draws in ten paired matches. An untrained policy that
cannot finish explains that equally well, so this is not offered as a diagnosis —
only as the observation that the reward does not oppose the equilibrium and offers
a cheaper alternative to competing.

## Decision

Restore the invariance and remove the farm, without adding new shaping.

Treat the potential as zero at a terminal state, so the shaping term is provably
policy-invariant and cannot pay for how an episode ends. Make the useful-contact
term signed with respect to the attacking direction and evaluate it on the
controlled robot's contribution rather than on a team-level contact edge, so
moving the ball the wrong way costs what moving it the right way earns and
re-entering the envelope is not itself an event. Express the episode time cost and
the draw and stagnation terminals on one scale, so a stalemate is not the cheapest
available outcome.

Goals, semantic outcomes, and the geometry potential remain authoritative. No
field zone is hard-coded, and no term rewards a pose.

## Alternatives considered

- **Reinstate a ball-progress or ball-direction coefficient.** Rejected: M20
  removed it for a documented reason, and a non-potential dense term biases the
  optimum toward pushing the ball rather than scoring.
- **Raise the geometry coefficient instead.** It is preference-neutral by
  construction, so scaling it changes credit assignment speed, never what the
  policy prefers. It cannot substitute for a coherent terminal structure.
- **Penalize draws harder alone.** Leaves the contact farm intact, so the farm
  simply has to out-earn a larger penalty.

## Consequences

- The shaping term becomes exactly policy-invariant, so it can be tuned for credit
  assignment without anyone having to argue about whether it distorts the optimum.
- Contact stops being a free source of return, which removes the cheapest known
  alternative to attempting a goal.
- Absolute returns shift and are not comparable with earlier runs of the same
  configuration; the milestone needs a fresh baseline.
- Making contact signed introduces a penalty a policy can incur while learning to
  dribble, so the contact terms and the semantic drills need to be read together
  before the coefficient is retuned.
