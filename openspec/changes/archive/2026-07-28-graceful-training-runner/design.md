## Context

Interrupting PPO during an optimizer step can leave partially updated weights
with an unchanged policy version. A signal must therefore request a stop rather
than raise immediately.

## Decisions

1. SIGINT and SIGTERM set a stop flag. The current iteration completes, then an
   unscheduled checkpoint is saved and registered when necessary.
2. Progress is line-oriented terminal output so it works in a normal shell,
   logs, and remote sessions without a TUI dependency.
3. Trainer and viewer remain independently runnable. `league-live` and
   `league-live-resume` are convenience supervisors whose exit trap stops only
   the viewer they launched.

## Validation and Rollback

Exercise normal completion, resume, and SIGINT against a disposable run.
Rollback removes signal/progress handling and combined recipes without changing
existing run artifacts.
