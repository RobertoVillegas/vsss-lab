# pSim 0.2.4 source review

Reviewed on 2026-07-28 against the official usage/API documentation, the PyPI
0.2.4 source distribution, and Julio De La Torre's public
`simulation_vsss` ROS/Gazebo repository.

## What pSim contains

pSim is a Python/Box2D alpha package exposing three façades over one simulator:

- `SimpleVSSSEnv` for manual and classical control;
- `VSSSGymEnv` for Gymnasium;
- `VSSSPettingZooEnv` implementing PettingZoo's parallel API.

Actions are normalized `[v, w]` body-velocity pairs. The simulator converts them
to a maximum `0.75 m/s` and `20 rad/s` from a 25 mm wheel radius, 75 mm track,
and 30 rad/s wheel limit. Physics advances at 60 Hz with six velocity and two
position iterations. Rendering is optional (`human`, `rgb_array`, or headless).

The source distribution vendors Box2D source plus CPython 3.10, 3.11, and 3.12
Linux shared objects. Its declared runtime dependencies are pygame and scipy,
with Gymnasium and PettingZoo extras. The package metadata says Python
`>=3.11,<3.12`, while its README recommends 3.12 and the archive ships a 3.12
binary; this inconsistency is another reason not to add it as a runtime
dependency.

## Geometry findings

pSim uses a `1.70 x 1.30 m` outer render extent. Its playable rectangle begins
100 mm from each x edge, so this is the same geometry as VSSS Lab's
`1.50 x 1.30 m` playing field plus two 100 mm-deep goals.

Useful calibrated details:

- 400 mm goal mouth and 100 mm goal depth;
- 70 mm clipped playing-field corners;
- 150 x 700 mm penalty areas;
- 200 mm center circle;
- 130 mm goal-area arcs;
- six 50 mm restart crosses;
- 80 mm pSim robot body, versus the 75 mm body in `simulation_vsss`;
- 21.25 mm pSim ball radius, versus 21.35 mm in `simulation_vsss`.

VSSS Lab keeps its canonical 75 mm robot and 21.5 mm ball, while adopting the
missing field form and presentation.

## Ideas adopted

- Declarative scenarios with fixed/ranged poses and per-actor behavior are a
  useful model for the existing curriculum system.
- Separate simple, Gymnasium, and parallel multi-agent façades validate the
  decision to keep a composable environment API over one canonical backend.
- Seeded resets, explicit `terminated`/`truncated`, and optional RGB/headless
  rendering are good public API expectations.
- Per-agent egocentric observations encode headings as sine/cosine and include
  relative distances/angles; these are useful candidates for observation
  ablations.
- Manual keyboard/gamepad control would be valuable later as a debugging input
  source, independent from the web replay renderer.

## Ideas deliberately not adopted

- A second canonical Box2D backend would split calibration and reduce native
  rollout throughput. Box2D can instead become an optional differential
  validation oracle.
- Directly assigning body linear/angular velocity bypasses actuator dynamics
  and wheel contact. VSSS Lab retains bounded wheel commands and acceleration.
- pSim's goal check only compares ball x with `0.765`, ignores y and ball radius,
  and encodes the negative-side goal as zero reward; it therefore cannot be the
  reference for scoring. VSSS Lab requires the full ball inside the mouth and
  emits an edge-triggered goal followed by a one-second grace.
- Its collision listener stores global booleans rather than actor/contact
  identities, and its reward system clears them while calculating one shared
  reward. This is insufficient for per-agent diagnostics.
- Random pose generation uses unbounded recursion and global NumPy seeding.
  VSSS Lab should retain bounded, per-world deterministic sampling.
- Reward, termination, and time bookkeeping are coupled. VSSS Lab keeps match
  rules independent from reward shaping.

## License boundary

The PyPI pSim 0.2.4 archive is GPLv3. The older `simulation_vsss` repository is
MIT. This review records behavior and public dimensions; VSSS Lab's
implementation is original and does not copy GPL source or bundled assets.

