# Public sources informing the M6 MARL design

This note records inspiration, not vendored code. VSSS Lab keeps an explicit
small learner because its native simulator and identity contracts are
project-specific.

## Adopted

### TorchRL multi-agent PPO

The [TorchRL multi-agent PPO tutorial](https://docs.pytorch.org/rl/0.12/tutorials/multiagent_ppo.html)
separates a decentralized parameter-shared actor from either a local IPPO critic
or centralized MAPPO critic, represents the agent axis explicitly, and warns
against normalizing advantage across the agent dimension accidentally.

VSSS Lab adopts the actor/critic separation, `[batch, agent, ...]` tensor shape,
TensorDict trajectories, shared parameters, GAE, and clipped PPO. The native
environment remains direct rather than wrapping VMAS.

### MAPPO

Yu et al., [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent
Games](https://arxiv.org/abs/2103.01955), establishes PPO variants as strong,
simple cooperative MARL baselines and motivates implementation-level discipline
before adding algorithmic complexity.

VSSS Lab adopts IPPO as the local-critic control and MAPPO as CTDE. It does not
claim paper-level sample efficiency from the M6 smoke workloads.

### BenchMARL

[BenchMARL](https://github.com/facebookresearch/BenchMARL) distinguishes IPPO
from MAPPO by critic observability, supports parameter sharing within agent
groups, exposes Deep Sets models, uses typed configuration, and emphasizes
reproducible comparisons.

VSSS Lab adopts typed algorithm configs, same-seed comparisons, team-level
parameter sharing, and explicit model/algorithm separation. It does not add
Hydra or the complete benchmark framework because that would obscure this
milestone's end-to-end path.

### PettingZoo Parallel API

The [PettingZoo Parallel API](https://pettingzoo.farama.org/main/api/parallel/)
models simultaneous actions and observations in partially observable stochastic
games.

VSSS Lab retains its M3 Parallel adapter for ecosystem compatibility but keeps
M6 rollouts tensor-native so Python dictionaries stay outside the hot loop.

## Deliberately deferred

- distributed collectors and Ray;
- recurrent actors;
- attention/GNN comparisons;
- large benchmark matrices and statistical league evaluation;
- framework-owned environment wrappers around the native simulator.

These become justified only after a measured bottleneck or later PRD milestone.
