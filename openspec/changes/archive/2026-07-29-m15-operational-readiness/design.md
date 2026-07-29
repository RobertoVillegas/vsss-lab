# Design

Clean teacher initialization is the default because the matched 163,840-step
probe produced positive mean progress and more resolved matches. An M14 policy
may optionally seed an M15 learner only when algorithm and architecture
match. Actor and critic parameters transfer because observations and physical
reward semantics remain compatible; optimizer, policy version, RNG, and
semantic curriculum state start fresh. The run records the source path, digest,
version, and reset boundary.

M15 uses a higher entropy coefficient and a bounded log-standard-deviation floor
to preserve exploration while short semantic drills teach missing behaviors.
Every configured checkpoint cadence evaluates immutable paired-color holdouts.
Selection ranks the minimum family success first, then macro success, then fewer
unresolved trials. This prevents approach or shooting from hiding zero defense.

The existing generic M14 recipes remain available. A distinct clean semantic
recipe selects the M15 configuration, evaluation cadence, and automatic run
naming. Warm start has a separate explicit recipe so the two protocols cannot
be confused.
