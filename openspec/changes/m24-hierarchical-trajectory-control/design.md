# Design

## Action contract

The `primitive` parser is categorical with 17 actions:

- `0`: stop;
- `1..8`: navigate along one of eight canonical unit directions;
- `9..16`: strike the ball toward one of eight canonical unit directions.

The actor stores the categorical index and log probability for PPO. A reversible
two-value token transports the selected primitive through existing rollout and
replay boundaries; only the environment parser converts that token into wheel
commands. Direct continuous and lattice action parsers remain unchanged.

Canonical directions are rotated by π for the yellow team, matching the
observation symmetry already used by the shared policy.

## Navigation

`navigate` selects a short target along the requested world direction and uses
the bounded differential-drive controller. This removes the need for the policy
to rediscover left/right wheel mixing while retaining control over direction.

## Strike

The strike controller:

1. predicts a bounded short-horizon ball point from current position and
   velocity;
2. estimates robot arrival time from translation and heading error;
3. chooses the earliest reachable prediction;
4. constructs an acquisition point behind the ball relative to the requested
   exit direction;
5. navigates to acquisition until position and angular gates pass;
6. drives through the ball toward a point beyond it.

The parser never reads future simulator frames and does not use the viewer's
camera estimator. Candidate commands are tested through the exact Rapier
environment rather than a separate approximate dynamics engine.

## Curriculum

M24 keeps semantic drills but caps frontier concentration and reserves
rehearsal/full-match capacity. Cooperation is taught through redirection rather
than requiring the receiver to stop the ball. Phase promotion remains based on
immutable holdouts; long-run selection remains outcome-gated.

## Evaluation

Trajectory evaluation is independent from reward. Each trial records:

- first-contact latency;
- minimum robot-ball distance;
- translational-motion ratio;
- useful ball impulse;
- requested versus observed exit-direction error;
- reacquisition success after a forced loss of contact.

Prediction metrics carry episode IDs. Pending predictions are discarded at
reset before matching them with truth from another episode.

## Rollback

Set `action_parser = "continuous"` and use the M23 configuration. Checkpoints
remain configuration-fingerprinted, so primitive and direct-wheel policies
cannot be loaded into the wrong parser accidentally.
