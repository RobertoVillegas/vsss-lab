# Design

## What is actually wrong, and what is not

The shaping term is already the correct policy-invariant form,
`coefficient · (γ_shaping · Φ(s') − Φ(s))` with `γ_shaping` equal to the training
discount. The defect is only the terminal: `Φ(s')` is evaluated on the terminal
state instead of being zero, so the last transition of every episode pays
`coefficient · γ · Φ(s_T)`. With the configured coefficient that is on the order of
a tenth of the draw penalty. Small, but it is a payment for how an episode ends,
which is the loophole the potential was chosen to close.

Nothing else about the geometry potential needs to change. In particular, raising its
coefficient cannot substitute for a goal-directed term, because a potential is
preference-neutral by construction: it changes how fast credit propagates, never what
the policy prefers. Reading it as "the attacking reward" is a mistake this design
explicitly does not make.

## Contact attribution

The current term is `max(0, attack_sign · Δv_x)` gated on a team-level contact edge.
Three properties compound: the edge trigger pays only on entry, so sustained control
earns nothing; only the horizontal component counts, so a lateral pass earns nothing;
and the non-negative clamp means noise integrates upward and nothing can be
penalized. Together they make envelope oscillation profitable.

Signing the term and attributing it to the controlled robot's contribution removes
the farm without adding a new incentive: moving the ball toward the goal earns, moving
it away costs the same magnitude, and re-entering the envelope is no longer an event
in itself. The lateral component is left out of scope here — a pass is a semantic drill
with its own outcome predicate, and encoding pass value in the dense term would
reintroduce shaping by the back door.

## Terminal scale

Time is currently divided by the horizon, so its whole-episode budget is a twentieth
of the draw penalty, while stagnation is twice the draw. The three are put on one
scale so that the intended ordering — a goal beats a contested draw beats a stalemate
beats stagnation — is the ordering by return. The scale is chosen so the sum of dense
terms over an episode cannot exceed the gap between adjacent terminal outcomes.

## Validation

- A test that the terminal transition pays nothing for final geometry.
- A test that the ordering of episodes by return is invariant to the shaping
  coefficient.
- A contact test driving envelope oscillation and asserting no positive accumulation,
  and a signed-magnitude test for the wrong-way case.
- A terminal-ordering test over goal, contested draw, stalemate, and stagnation.
- The existing anti-farming tests stay green, since they encode the M20 principle this
  change restores rather than replaces.

## Compatibility and rollback

No configuration key is removed and no new dense term is introduced. Absolute returns
shift, so the milestone needs a fresh baseline; earlier runs of the same configuration
are not comparable. Rollback is `git revert`, which restores a shaping term that pays
at the terminal and a contact term that can be farmed.
