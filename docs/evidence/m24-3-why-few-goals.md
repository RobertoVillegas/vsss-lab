# M24.3: why the training goal counts collapsed, and three hypotheses that were wrong

Run `vsss-m24-3-run-0006` was stopped at iteration 450 because its goals fell away. Goals
for dropped from 41 to 26 per thirty-iteration block while goals against rose from 39 to 62.

## What it was not

**Not the idle-spin penalty.** The reward decomposition reports that term at 0.000000
throughout. Spinning fell from 0.094 to 0.018 over the same window, but nothing was paying
for it to fall.

**Not the entropy bonus on drive intensity.** Intensity fell from 0.901 to 0.579, and 0.5 is
the maximum-entropy point of a tanh-bounded parameter, which made the entropy bonus the
obvious suspect. An A/B of a hundred iterations per arm says otherwise: with the bonus the
intensity moved 0.442 to 0.648, and without it 0.442 to 0.609. Both arms rose, and they rose
together. Intensity converges to roughly 0.6 from either direction and with no entropy
pressure at all, so it is an attractor of the policy gradient rather than a regularization
artefact. The experiment also revealed its own flaw: it built the learner directly instead of
distilling the teacher first, so it started at 0.44 rather than 0.87 and could not reproduce
the fall it was meant to explain.

**Not the exploration asymmetry.** The learner samples its actions while its opponent plays
deterministically, which looked like a systematic handicap in self-play. Measured on the
final checkpoint over fourteen paired matches, a sampling copy against a deterministic copy
of the same policy scores 3-8-3 with goals 3-3, and deterministic against deterministic
scores 3-8-3 with goals 3-3. Identical. The sampling is real — heading draws differ by 2.4
radians on average — and it costs nothing in a match.

## What it was

The league. Split by opponent across the run:

| opponent | iterations | goals for | goals against | net |
| --- | --- | --- | --- | --- |
| scripted heuristic | 191 | 162 | 146 | +16 |
| league self and history | 260 | 345 | 512 | −167 |

Against the scripted controller the policy is positive. Against its own copies it is not, and
the reason is that some of those copies are better than it is:

| pairing | result | goals |
| --- | --- | --- |
| iteration 450 vs 100 | 1-5-4 | 1-4 |
| iteration 450 vs 200 | 0-7-3 | 0-3 |
| iteration 450 vs 300 | 2-7-1 | 2-1 |
| iteration 450 vs 400 | 2-7-1 | 2-1 |

The policy beats its recent past and loses to its distant past. With a history window of
sixteen entries and a checkpoint every twenty-five iterations, the sampled pool reaches back
past the versions that beat it, so the aggregate goal difference against the league is
negative by construction.

The league is not broken. It is reporting that training is not monotone: the policy improved,
gave ground between roughly iteration 100 and 300, and was recovering when the run stopped.

## What is now recorded

Nothing surfaced this. The heuristic scorecard cannot, since it is positive throughout, and
the incumbent bound only watches the promoted entry. Each evaluation now also scores the
candidate against the newest checkpoint at least a hundred iterations old, so non-monotone
training is a curve in the run record rather than something to be discovered by hand.
