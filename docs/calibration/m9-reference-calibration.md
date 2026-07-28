# M9 reference calibration

## Outcome

The Rapier reference backend passes the three first isolated golden phenomena
against committed analytic/reference values:

| Scenario | Reference | Measured | Absolute error | Tolerance |
|---|---:|---:|---:|---:|
| straight displacement | 0.500000 m | 0.4999996 m | 0.0000004 m | 0.015 m |
| center-turn heading | -2.116519 rad | -2.118927 rad | 0.002408 rad | 0.030 rad |
| passive ball speed | 0.086070 m/s | 0.086076 m/s | 0.000006 m/s | 0.010 m/s |

Reproduce with:

```bash
just calibrate-reference reports/m9/calibration.json
```

The JSON report is generated, not committed, so downstream runs can retain it
with their exact build metadata. The source manifest and acceptance thresholds
are versioned under `calibration/`.

## Provenance and assets

Field, robot, wheel, and ball values were extracted or adopted from Julio De La
Torre's public simulator and thesis review, then expressed in canonical SI
units. The six 800×800 visual-marker textures are inventoried by URL and
semantics but not copied: the repository does not state an asset reuse license
clearly enough to justify vendoring them. VSSS Lab can generate regulation-aware
markers independently in M12.

## Interpretation

These scenarios establish equation-level consistency for commanded straight
motion, differential center turning, and passive damping. They do not claim
full equivalence with Julio's Box2D training runtime or ROS 1/Gazebo Classic.
No reproducible legacy trajectory bundle or standalone container was available,
so impact, oblique rebound, compression, multi-body contact, and time-to-stop
remain future calibration rows rather than invented measurements.

The differences measured here are well inside their reviewable tolerances. A
future trace can replace a reference number without changing the runner or
metrics.

## Rollback

M9 adds fixtures, a report tool, tests, and documentation only. Removing those
files leaves simulation, training, protocol, and replay behavior unchanged.
