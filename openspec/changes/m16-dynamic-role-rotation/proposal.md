# M16: Dynamic role rotation

M15 exposed a generalization failure: training skills reported high success while
immutable holdouts collapsed to zero. The shared policy also lacked a tactical signal
that could distinguish the player challenging the ball from the players supporting
and covering without binding those responsibilities to robot identity.

M16 adds an identity-free attacker/support/coverage assignment, role-conditioned
MAPPO, a rotation-recovery drill, multidimensional curriculum progression, graduated
holdouts, rotation telemetry and semantic regression stopping.

The goalkeeper is deliberately not a permanent player. Any robot may hold coverage,
leave it when another robot becomes the safer replacement, and later become attacker.
