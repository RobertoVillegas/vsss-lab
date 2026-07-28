## Context

The old reference backend saturated wheel targets instantly and used only front
field walls, allowing robots through goal mouths and producing severe jitter.
The learner also executed one world at a time and exposed only ad-hoc lines.

## Decisions

1. Derive maximum wheel angular acceleration from configured actuator force,
   robot mass, and wheel radius; approach targets at each fixed physics step.
2. Model goal boxes with side and back colliders. Ball goal events remain based
   on crossing the goal line inside the mouth.
3. Regularize changes in normalized policy action, separately from the physical
   actuator response.
4. `auto` chooses CUDA when PyTorch reports it available and otherwise emits an
   explicit warning. Explicit unavailable CUDA is an error.
5. Batch actor/critic/PPO tensor operations across vector worlds. Rapier worlds
   remain independent and CPU-backed; native parallel stepping is follow-up.
6. Use Rich Live with a table grouped above Progress; non-TTY output is stable
   aligned text suitable for logs.

## Evidence and rollback

The measured CPU/CUDA comparison and public physical references live in
`docs/evidence/m12-physical-controls-cuda-dashboard.md`. Configuration
fingerprints prevent unsafe resume. Device and world count can be overridden
without removing CUDA support; action regularization can be set to zero.
