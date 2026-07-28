## Why

M13 can produce physically valid, GPU-trained policies, but a single reward
configuration and a latest-checkpoint lineage do not establish that learning is
robust. The observed clustering, stalemates, exploration saturation, and reward
regressions require an outer experimental loop that selects training scenarios,
reward weights, and checkpoints from objective match evidence.

Recent work changes the recommended path. Automatic curricula now target
scenarios at the learner's current frontier instead of applying uniform domain
randomization. Multi-agent world models improve sample efficiency when real
interactions are scarce, but VSSS Lab already owns a fast, exact simulator, so
replacing it with learned dynamics would add model bias without removing the
current bottleneck. Recurrent policies remain the smallest credible response to
camera partial observability. KAN policies are promising compact ablations, not
an established replacement for MLP or recurrent actors.

## What Changes

- Define a deterministic evaluation suite independent from shaped training
  reward and latest-checkpoint order.
- Add a Ballchasing-style derived analytics layer for possession, territorial
  pressure, positioning, movement, coordination, and event attribution.
- Add an automatic scenario curriculum driven by measured learning progress,
  with replay of failures and coverage of routine, frontier, and holdout cases.
- Search bounded reward-component weights and selected learner
  hyperparameters with multi-fidelity, multi-seed Optuna studies.
- Compare feed-forward MAPPO with a compact recurrent MAPPO policy using
  camera-realistic observations before adopting memory.
- Generate verified demonstrations for atomic soccer skills with the exact
  simulator and a bounded planner, then compare cold-start, imitation warm
  start, and RL fine-tuning.
- Maintain a small league of complementary checkpoints and distill only when
  the population demonstrates a reproducible advantage over the best single
  policy.
- Benchmark RLGym/Nexto-inspired environment decomposition, batched entity
  observations, action abstractions, staged skills, and historical ratings.
- Benchmark a device-resident batched simulation prototype before considering
  NVIDIA Warp/Newton or another GPU physics path.
- Record experiment lineage, objectives, uncertainty, compute, and promotion
  decisions as machine-readable artifacts.
- Require a dated prior-art review before each major implementation block and
  record which ideas were adopted, adapted, deferred, or rejected.

## PRD Milestone

M14 — evidence-driven adaptive training after the fresh M13 baseline and paired
evaluation are complete.

## Explicit Non-goals

- No LLM, generative reward judge, natural-language mutation, or API dependency
  is part of the training loop.
- No GAN-generated state distribution.
- No learned world model replacing Rapier.
- No Isaac Sim, Omniverse, Cosmos, or OmniDreams runtime dependency.
- No KAN policy as the default actor.
- No online reward-weight mutation inside an active PPO lineage.
- No promotion based on training return, a single seed, or the latest
  checkpoint alone.
- No direct reward for Ballchasing-style diagnostic metrics without a separate
  anti-farming ablation.
