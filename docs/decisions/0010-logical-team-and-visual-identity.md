# ADR-0010: Separate logical, team, and visual identity

- Status: accepted
- Date: 2026-07-28
- Owners: Roberto Villegas

## Context

VSSS uses an overhead camera, removable blue/yellow team tags, and personal
robot tags. Team color may change between matches. Simulator assets commonly
encode both team and robot number in a top texture, but RL policies must remain
safe under robot permutation and side switching.

## Decision

Model stable logical robot identity, per-match team assignment, and
camera-observable visual marker as separate concepts. Canonical physics state
continues to use `robot_id` and team. Marker profiles belong to match/presentation
metadata until M12 introduces observations and association confidence.

External protocol envelopes identify the controller slot and match side, never a
permanent physical color or tactical role. Replays may carry marker-profile
metadata additively. Policies cannot receive raw marker identity as a privileged
feature.

## Consequences

Viewers can resemble the real overhead feed, while later vision pipelines can
measure association errors against ground truth. Team tag changes, substitutions,
and dynamic goalkeeper roles do not alter policy or physical identity. M8 does
not implement vision; M10/M12 can add rendered/sensed markers without changing
the learner API.

## Alternatives considered

Encoding team and jersey number in `robot_id` is simpler but breaks side
switching and encourages role specialization. Treating markers as viewer-only
sprites loses the future perception contract. Adding pixels to M8 pulls vision
ahead of its validated structured-state prerequisite.

## Validation and rollback

Permutation and side-switch contract tests remain blocking. M8 protocol tests
must prove controller routing is by assigned slot. M12 will add marker
association fixtures. Rollback removes optional marker metadata without changing
canonical state identifiers or replay snapshots.
