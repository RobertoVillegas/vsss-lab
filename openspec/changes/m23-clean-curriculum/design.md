# Design

## Curriculum

Foundation, defense, cooperation, and rotation advance after their skill-family
floors pass twice consecutively and deterministic idle spin remains below the
behavior ceiling. A full-match win/draw gate cannot block those teaching
transitions because later phases provide capabilities needed to satisfy it.

Integration and final checkpoint promotion still require:

- at least 20% wins in paired-side deterministic evaluation;
- at most 70% draws;
- all configured semantic family floors;
- idle-spin ratio at or below 8%.

Semantic regression early stopping applies only during integration for phased
runs. Earlier phases continue learning until their skill gates pass or the
requested environment-step budget ends.

## Goal attribution

The vector environment latches the scoring event while the one-second closing
grace elapses. The terminal collector sees that latched result, but reward
calculation continues to consume only the original one-step event, preventing a
duplicate goal reward.

Telemetry distinguishes:

- all completed episodes;
- completed full matches and skill drills;
- full-match wins, draws, and losses;
- goals for/against in full matches;
- goals for/against inside drills.

Canonical JSONL additionally carries cumulative match outcomes and goal totals.

## Controlled baseline

M23 retains M22's network, optimizer, physics, and reward coefficients so the
curriculum and attribution corrections remain the controlled independent
variables. It changes the random seed and increases deterministic evaluation
from three to five seeds, producing ten paired-side matches per checkpoint.
