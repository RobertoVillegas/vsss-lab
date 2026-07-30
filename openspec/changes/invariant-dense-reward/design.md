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

## Contact impulse: sign it, keep the edge

The current term is `max(0, attack_sign · Δv_x)` gated on a team-level contact edge.
The clamp is the defect: velocity noise at the moment of re-entry is positive half the
time, so a robot hovering at the envelope boundary re-triggers the edge and collects
only the favorable half. Signed, those impulses cancel.

The edge itself is not a defect and stays. Paying on every contact step would make the
term a rate reward for ball advancement, which is precisely what M20 removed; the fix
must not reintroduce it through the back door. This is a revision of this change's own
first draft, which proposed removing the edge.

Per-robot attribution was also dropped. The state exposes team contact, not contact
ownership, so attributing the impulse to one robot would require new plumbing for no
additional protection — the sign already removes the farm.

The lateral component stays out of scope: a pass is a semantic drill with its own
outcome predicate, and encoding pass value densely would be new shaping.

## Terminal scale: left alone, deliberately

Time is divided by the horizon, so its whole-episode budget is a twentieth of the draw
penalty, while stagnation costs twice a draw. The audit read that as incoherent, but
the incoherence it described depended on the contact farm outweighing the draw penalty.
With the impulse signed, dense accumulation is approximately zero and the terminal
ordering — a goal, then a draw, then stagnation — already expresses the intent.

Retuning is therefore not part of this change. The open question is whether the time
cost is strong enough to prefer a fast goal, and that needs a run under the corrected
terms before anyone picks a number.

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
