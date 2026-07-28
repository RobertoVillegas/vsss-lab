# Technical review: Julio De La Torre VSSS thesis

## Source

Julio De La Torre Vanegas, *Aprendizaje profundo por refuerzo para robots
colaborativos y/o competitivos*, Master's thesis, Centro de Investigación en
Matemáticas, A.C., 18 November 2024, 169 PDF pages.

The author-provided local copy was reviewed on 2026-07-28. Its SHA-256 is:

```text
d0e5752290a800f604e7b9f2f01bf937bf996558ffb2e938c6e3428b0a1bef09
```

The PDF is not copied into this repository. This note paraphrases its technical
claims and records hypotheses for VSSS Lab; it does not treat thesis measurements
as VSSS Lab benchmark evidence.

Related implementation:
<https://github.com/juliodltv/simulation_vsss>

## Training simulator and interface

The thesis separates a fast 2D training simulator from a higher-fidelity
ROS 1/Gazebo simulator. The fast simulator uses Box2D, Gymnasium, Pygame, and
differential-drive kinematics. On an AMD Ryzen 7 5700X it reports approximately
5,338 steps/s for a 3v3 match, compared with approximately 920 steps/s for the
cited rSim configuration.

Actions are normalized linear and angular velocity rather than direct wheel
speeds. If their absolute sum exceeds one, the pair is normalized before mapping
to physical limits. This is a useful action-adapter baseline; it is not a reason
to replace VSSS Lab's canonical wheel-action contract.

The selected structured observation includes:

- robot position, sine/cosine heading, linear and angular velocity;
- ball position and velocity;
- robot-to-ball distance and bearing;
- distances and bearings to allies and opponents;
- normalized linear and angular action.

VSSS Lab already covers the same physical information with team-relative,
agent-centric, permutation-safe observations. A future comparison should isolate
representation effects instead of changing physics and observation together.

## Rewards and curriculum

The single-agent reward combines:

- terminal goal reward of `+10` or `-10`;
- a per-step time penalty;
- dense ball-direction reward based on cosine similarity between ball velocity
  and vectors toward the allied/enemy goals.

The thesis reports reward-hacking-like behavior when positive dense reward lets
an agent delay episode termination. Its preferred formulation is primarily
penalty-shaped. This supports VSSS Lab's existing reward-hacking gates and
componentized reward pipeline.

The curriculum has two stages:

1. Spawn robot and ball close enough to learn orientation, approach, and scoring.
2. Expand initial conditions across the field with other players present.

The reported experiments show curriculum learning was essential when other
robots were present. This closely matches the staged skill curriculum already
implemented in M5 and supplies an external comparison target.

## Algorithms and results

For a single agent, the thesis tunes PPO and TD3 with Optuna and selects PPO
because it converges faster, runs at higher step throughput, and performs better
over 100 evaluation matches. It reports:

- PPO: accumulated reward `9.353 ± 0.350`, match duration `0.617 ± 0.041 s`;
- TD3: accumulated reward `8.248 ± 0.774`, match duration `0.764 ± 0.119 s`.

The single-agent scoring comparison reports `6.3967 ± 4.3899 s` for DRL versus
`39.1833 ± 18.9197 s` for the traditional attacker. With all other players
present, PPO without curriculum did not score within the 60 s limit, while
PPO plus curriculum reports `16.5523 ± 14.4869 s`.

For multi-agent training, the thesis chooses MATD3 with decentralized actors and
centralized critics. In cooperative cases it reports better average scoring time
than its traditional and expanded-single-policy baselines. The 3v3 mixed case did
not converge within the allocated training and underperformed the traditional
controller.

These numbers are evidence about that implementation, hardware, physics, reset
distribution, and metric—not directly comparable project benchmarks. We should
reproduce the tasks under our canonical fixtures before drawing algorithmic
conclusions.

## Role-assignment finding

In the 3v3 mixed environment, identically rewarded agents collided. The thesis
then assigns attacker, defender, and goalkeeper reward functions tied to each
agent. This improved positioning, but embeds tactical role in physical agent
identity.

VSSS Lab deliberately tests the alternative:

- shared/permutation-safe policy parameters;
- centralized training with decentralized execution;
- dynamic role emergence and assignment;
- goalkeeper defined from match context rather than a fixed robot;
- identity permutation and side-switch tests as blocking gates.

The thesis explicitly lists MAPPO as future work to reduce abrupt MATD3 policy
changes and improve stability. That independently supports our MAPPO baseline,
while MATD3 remains valuable as a later controlled comparison.

## Camera and visual-marker pipeline

The high-fidelity pipeline is directly relevant to M10–M12:

1. Correct radial and tangential camera distortion.
2. Apply perspective correction and standardize the 1.70 m by 1.30 m scene to
   680 by 520 pixels (`400 px/m`).
3. Collect regulated colors under varied illumination.
4. Compare HSV, HSL, LCh, YCrCb, XYZ, Lab, Luv, YUV, RGB, and OKLab.
5. Train SVM classifiers and use SHAP feature contributions to select the useful
   color-space channels.
6. Segment in HSL, then use K-means with seven clusters for six robots plus ball.
7. Classify objects from color content around each centroid.
8. Use Hungarian assignment to associate individual tags consistently.
9. Estimate ball position/velocity with a Kalman filter and robot
   position/orientation with an EKF; reject large measurement outliers.

The six simulator textures are 800×800 RGB images. Each contains one large
blue/yellow team rectangle and two smaller personal-color rectangles:

| Robot | Blue team personal colors | Yellow team personal colors |
|---|---|---|
| 0 | cyan + green | green + cyan |
| 1 | magenta + green | magenta + cyan |
| 2 | red + green | red + cyan |

Their layout also conveys orientation. This validates ADR-0010: team assignment,
personal marker, orientation cue, and logical robot identity must remain separate.

## Experiments to reproduce

Later milestones should add explicit, versioned experiments rather than silently
adopting thesis choices:

1. PPO/MAPPO versus TD3/MATD3 on identical canonical resets and budgets.
2. Direct wheel actions versus normalized linear/angular action adapter.
3. Thesis-style absolute observation versus VSSS Lab agent-centric
   permutation-safe observation.
4. Fixed role rewards versus dynamic role assignment under robot permutations.
5. Directional dense reward with automated episode-delay/reward-hacking tests.
6. Two-stage curriculum ablation with and without other players.
7. Camera pipeline comparison across HSL, Lab, and learned color classification
   under controlled illumination/occlusion suites.
8. Hungarian-plus-EKF association versus alternative trackers using ground-truth
   simulator identity and calibrated confidence.

## Adoption decisions

Adopt now:

- thesis as a documented external baseline;
- visual-marker semantics and future association tests;
- curriculum and action-adapter comparison fixtures;
- explicit MATD3 and fixed-role reward baselines in the experiment backlog.

Do not adopt blindly:

- fixed roles bound to robot identity;
- thesis metrics as project performance claims;
- HSL as a universal color space without our calibration data;
- camera observations in M8;
- ROS 1/Gazebo Classic as the target architecture.
