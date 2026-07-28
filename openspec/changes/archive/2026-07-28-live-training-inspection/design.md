## Context

Training writes immutable replay/checkpoint files and append-only metrics. The
browser may poll while any of those artifacts are being created.

## Goals / Non-Goals

**Goals:**

- Follow newly completed captures without refreshing the page.
- Loop the latest capture at a useful inspection speed.
- Keep exact historical replay controls.
- Report the latest durable checkpoint and complete metric record.

**Non-Goals:**

- Stream partially written replay frames.
- Control or terminate the trainer from the browser.
- Claim that 1× playback implies a policy is ready for physical robots.

## Decisions

1. TanStack Query owns server-state caching and two-second polling. React
   effects remain limited to browser animation, keyboard listeners, and
   resetting local transport state when an immutable replay changes.
2. The server only discovers canonical numbered filenames. It tolerates a
   partially appended final metrics line by returning the previous valid line.
3. Live-follow selects each newest completed replay and starts it in a loop.
   Manual iteration selection enters history mode until the operator explicitly
   resumes following.
4. Playback 1× uses the canonical 20 ms control period (50 Hz). The default 4×
   changes only visualization speed, never physics or policy inference.
5. RL outputs are normalized controls and must be scaled once, at the
   environment boundary, to the calibrated physical wheel limit. The 30 rad/s
   reference matches Julio's operational Gazebo command limit.
6. Sustained training resumes from the latest durable registry checkpoint.
   Checkpoint and capture intervals are independent to bound disk use.

## Validation and Rollback

Contract tests cover ordered discovery and concurrent metric reads. Frontend
tests/typechecking/build and an isolated browser session validate the UI.
Rollback removes polling and supplemental metadata without changing run files.
The action-scale correction is not rollback-compatible with pre-fix learned
weights; those weights remain historical artifacts and must not be promoted.
