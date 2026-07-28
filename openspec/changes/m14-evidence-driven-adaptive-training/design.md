## Research conclusion

There is no single universal SOTA MARL algorithm that can be transplanted into
VSSS. Results remain benchmark- and observability-dependent, and recent
multi-agent world-model papers primarily optimize sample efficiency where
environment interaction is expensive. VSSS Lab's differentiator is the
opposite: its authoritative dynamics are available, deterministic, and cheaper
than learning an approximate replacement.

The recommended system therefore keeps Rust/Rapier as the source of truth and
improves what is trained, how it is evaluated, and which experiments receive
compute. The order matters: evaluation first, curriculum and reward search
second, memory and demonstration ablations third, population consolidation
last.

## Decision 1: promotion is a constrained evaluation problem

Training reward SHALL remain an optimization signal, not the promotion
objective. Every candidate is evaluated from both team colors against:

- the deterministic heuristic;
- the current promoted policy;
- a bounded sample of historical policies;
- holdout initial states and physics/perception randomizations.

The primary score is terminal goal difference and win/draw/loss confidence.
Secondary objectives are non-terminal rate, ball-contact rate, teammate
congestion, defensive failures, wheel effort, action jerk, inference latency,
and simulator throughput. Physical invalidity, non-finite telemetry, checkpoint
incompatibility, or a statistically unresolved regression blocks promotion.

At least three independent learner seeds are required during screening and at
least five for a final promotion comparison. Reports include bootstrap
confidence intervals and paired seeds. This explicitly addresses the high
variance and reproducibility weakness of automated reward search.

## Decision 2: use a learning-progress scenario teacher, not a GAN

The scenario space is a typed, bounded vector over ball/robot poses and
velocities, score/time context, opponent family, observation noise, latency,
friction, damping, and actuator response. Every generated state must pass the
same canonical and overlap validation as a normal reset.

Scenarios are bucketed by interpretable soccer skills: kickoff, approach,
interception, clearance, defense, pass/receive, shot, congestion recovery, and
mixed play. The teacher assigns compute using absolute learning progress and
maintains a mixture of:

- 20% routine scenarios, preventing catastrophic forgetting;
- 50% frontier scenarios with neither near-zero nor near-perfect success;
- 20% prioritized failures, deduplicated by scenario features;
- 10% immutable holdouts, sampled for evaluation but never optimized against.

Mutation operates on the typed parameters and rejects impossible states. A GAN
is unnecessary: the domain is low-dimensional, constraints are exact, and
generator realism is less useful than coverage, validity, and controllable
difficulty. Quality-diversity archives may later retain difficult scenarios,
but scenario descriptors must correspond to soccer behavior rather than raw
coordinates.

## Decision 3: optimize a reward DSL through staged, multi-seed search

The existing named reward components form a bounded reward DSL. Optuna may
select component weights, curriculum thresholds, entropy schedule, and a small
set of PPO hyperparameters. It SHALL NOT synthesize arbitrary executable code.

Search follows three fidelities:

1. smoke: short deterministic runs reject invalid, unstable, idle, or
   physically exploitative candidates;
2. screen: three seeds and a reduced evaluation league eliminate dominated
   trials with a multiobjective Pareto study;
3. confirm: five or more paired seeds at the target budget compare finalists
   against the promoted baseline.

Correctness gates precede performance scoring, mirroring the transferable
principle in Kimi K3's verifiable kernel tasks: an invalid solution earns no
credit regardless of speed. Study storage records config hashes, code commit,
seed sets, parent trial, pruning reason, objective vector, and artifacts.
Changing a promoted reward configuration starts a new checkpoint lineage.

Meta-gradient reward learning and differentiable evolutionary reward design
remain research-only. Their bilevel gradients, bias, and failure attribution
are not justified until the bounded black-box baseline is reproducible.

## Decision 4: test memory only where the observation is non-Markov

The authoritative simulator state already contains positions, headings,
velocities, and actions; adding memory there can increase cost without adding
information. The meaningful experiment is the M12 camera-derived observation
contract with latency, occlusion, noise, and dropped detections.

Run a controlled ablation:

- current frame-stacked MLP actor and centralized critic;
- shared GRU actor with per-world, per-agent hidden state and the same critic;
- optional lightweight sequence model only if GRU wins on holdout perception
  conditions at matched parameter count and wall-clock budget.

Hidden state resets at episode boundaries and cannot cross worlds. Replay,
checkpoint, and viewer artifacts expose the architecture and memory horizon.
The 2026 streaming-RL work on recurrent trace units is not a direct fit because
VSSS uses batched PPO rather than batch-size-one updates.

KAN is limited to a later matched-budget actor/value ablation. Current online-RL
evidence shows comparable performance with fewer parameters, not consistent
superiority. It is better suited to offline interpretability or compact
function approximation than to becoming the default policy now.

## Decision 5: use exact planning as a teacher for atomic skills

A bounded CEM or trajectory-optimization teacher may search wheel commands in
the authoritative simulator for short, verifiable skills such as interception,
clearance, shooting, and pass reception. Demonstrations are accepted only when
they satisfy terminal or geometric success predicates without invalid contact.

The experiment compares:

- MAPPO from scratch;
- behavior cloning from verified demonstrations;
- behavior cloning followed by MAPPO fine-tuning.

Whole-match TAS is a non-goal: the branching factor, opponent non-stationarity,
and sim-to-real sensitivity make it a poor first teacher. Distillation is
adopted only if it reduces time-to-success without lowering final league
performance.

## Decision 6: defer learned world models

DIMA, GAWM, MATWM, and SeqWM show that structured multi-agent world models can
improve sample efficiency and model teammate intentions. They do not establish
that an approximate model beats an already parallel, exact simulator on total
wall-clock learning for this project.

A world model may be reconsidered for:

- camera-state belief estimation under occlusion;
- residual dynamics learned from physical-robot telemetry;
- opponent-intention prediction;
- uncertainty-aware short-horizon planning.

It must never become the canonical physics source, and imagined rollouts must
be labeled separately from authoritative transitions.

## Decision 7: borrow NVIDIA's throughput architecture, not its 3D stack

OmniDreams is a real-time, action-conditioned video world model for closed-loop
autonomous-driving sensor simulation. It solves photorealistic long-tail camera
generation from 21,000 hours of driving data. Cosmos 3 expands this into an
omnimodal world model over language, images, video, audio, and actions. Neither
is proportionate to a calibrated, top-down 2D VSSS field: generated RGB frames
would be slower and less geometrically trustworthy than rendering exact state,
and the 3070 cannot train or practically host these foundation models.

Isaac Lab, Warp, and Newton expose a more relevant lesson. Their throughput
comes from thousands of vectorized worlds, device-resident tensors, fused
kernels, CUDA-graphable execution, procedural randomization, and decoupled
sensor/control frequencies. VSSS should borrow those properties without
adopting USD, Omniverse, PhysX, or a second production physics engine.

M14 therefore includes a measured accelerator feasibility spike:

1. profile the Rust/Rapier step, Python conversion, tensor copies, reward,
   inference, and PPO update separately;
2. remove avoidable host allocations and use structure-of-arrays batches;
3. keep observations, actions, rewards, and resets on device across a rollout
   where the current architecture permits it;
4. evaluate CUDA graphs or fused custom kernels for reward/reset/observation;
5. prototype only VSSS primitives—circles, oriented boxes, walls, goals, and
   contacts—in a device kernel and compare them against Rapier golden traces.

Warp/Newton is an optional prototype vehicle, not a dependency decision. Its
current stack targets 3D robotics, has moving APIs, and would create Python/JIT
and alternate-physics ownership inside the hottest path. A Rust/CUDA kernel may
be smaller if profiling proves physics is the dominant cost. Adoption requires
material end-to-end speedup, deterministic reset, contact/goal parity, and no
loss in physical validity. Otherwise Rapier remains authoritative.

Cosmos-style generation may later augment the M12 perception dataset with
lighting, blur, occlusion, lens, and surface variations. It may not generate
labels: canonical state produces geometry and labels, while any image generator
is merely an appearance transform. Ordinary procedural rendering and
camera-domain randomization must be benchmarked first.

## Transferable lessons from Kimi K3

Kimi K3 is not evidence that VSSS needs an LLM. The applicable system patterns
are:

- progressively increase task difficulty instead of paying full cost from the
  first update;
- diversify configurable environments to avoid harness overfitting;
- use deterministic verifiers and anti-reward-hacking checks;
- train specialists when objectives conflict, then distill only after each is
  independently strong;
- avoid waiting for stragglers in heterogeneous long rollouts;
- co-design kernels, batching, storage, and learning rather than judging an
  algorithm by model architecture alone.

For VSSS, skill buckets replace language domains, exact match predicates replace
generative judges, and checkpoint specialists replace MoE experts. Kimi's
architecture, scale, optimizer, and LLM reward models are not transferred.

## Evidence reviewed

Primary recent references:

- Moonshot AI, *Kimi K3 Technical Report* (2026):
  https://github.com/MoonshotAI/Kimi-K3
- Farr et al., *Streaming Reinforcement Learning under Partial Observability
  with Real-Time Recurrent Learning* (2026):
  https://arxiv.org/abs/2605.24709
- Zhao et al., *Empowering Multi-Robot Cooperation via Sequential World
  Models* (2025): https://arxiv.org/abs/2509.13095
- Zhang et al., *Revisiting Multi-Agent World Modeling from a
  Diffusion-Inspired Perspective* (2025):
  https://arxiv.org/abs/2505.20922
- Deihim et al., *Transformer World Model for Sample Efficient Multi-Agent
  Reinforcement Learning* (2025): https://arxiv.org/abs/2506.18537
- Shi et al., *Global-Aware World Model for Multi-Agent Reinforcement Learning*
  (2025): https://arxiv.org/abs/2501.10116
- Abouelazm et al., *Automatic Curriculum Learning for Driving Scenarios*
  (2025): https://arxiv.org/abs/2505.08264
- Batra et al., *Quality Diversity for Robot Learning: Limitations and Future
  Directions* (2024): https://arxiv.org/abs/2407.17515
- Kich et al., *Kolmogorov-Arnold Network for Online Reinforcement Learning*
  (2024): https://arxiv.org/abs/2408.04841
- NVIDIA, *OmniDreams: Real-Time Generative World Model for Closed-Loop
  Autonomous Vehicle Simulation* (2026):
  https://arxiv.org/abs/2606.03159
- NVIDIA, *Cosmos 3: Omnimodal World Models for Physical AI* (2026):
  https://arxiv.org/abs/2606.02800
- NVIDIA, *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal
  Robot Learning* (2025): https://arxiv.org/abs/2511.04831
- NVIDIA Warp, GPU kernels and differentiable simulation:
  https://github.com/NVIDIA/warp
- Newton Physics, GPU-accelerated simulation built on Warp:
  https://github.com/newton-physics/newton

Older papers are retained only as provenance for MAPPO, automatic environment
design, population-based training, and Optuna-style black-box optimization; no
recommendation rests on novelty claims from 2017.

## Compatibility and rollback

M14 adds experiment orchestration and optional policy variants. Canonical
physics and match contracts remain unchanged. Existing M12/M13 checkpoints and
replays remain readable.

Every study is additive and lineage-isolated. Rollback disables the scenario
teacher, selects the promoted fixed experiment, and removes no historical
artifacts. No reward or architecture candidate may overwrite the promoted
configuration.

## Validation

- Determinism tests cover scenario generation, study resumes, seed allocation,
  pruning, and promotion.
- Property tests reject overlap, out-of-field, non-finite, and physically
  impossible generated starts.
- Reward-search tests prove that evaluation objectives do not read shaped
  training return.
- Recurrent-policy tests cover hidden-state shape, world isolation, episode
  reset, checkpoint round trip, and CPU/CUDA parity within tolerance.
- Planner demonstrations replay successfully in the authoritative simulator.
- Benchmarks report matches/s, frames/s, learner time, evaluator time, GPU
  utilization, and total wall-clock cost.
- Accelerator prototypes replay the same contact and goal corpus as Rapier and
  report parity failures separately from throughput.
