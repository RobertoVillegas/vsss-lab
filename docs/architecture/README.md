# Architecture

The canonical contracts, simulation, RL environment, learning, and competition planes
remain separate. M0 defines repository boundaries only; binding decisions begin with
reviewed ADRs in M1.

## Training and evaluation

Rapier owns deterministic physics and batched world stepping. Python exposes the
environment and orchestrates MAPPO in PyTorch; CUDA accelerates policy inference
and optimization while native CPU workers advance independent physics worlds.
Rendering consumes recorded events and never participates in the training loop.

M13 keeps reward construction in the environment, optimizer constraints in the
learner, and terminal checkpoint evaluation in the league plane. That separation
lets historical policies retain their original reward fingerprint while new
policies use directional shaping and an exploration floor. See
`docs/evidence/m13-directional-reward.md` for the measured rationale and the
checkpoint-selection protocol.
