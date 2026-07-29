# Evidence

## External review

| Source | Relevant finding | Decision in VSSS Lab |
| --- | --- | --- |
| [RLBot flatbuffers-python](https://github.com/RLBot/flatbuffers-python) | Rust/PyO3 protocol code generation with Python typing | Keep our existing FlatBuffers boundary; use this as a packaging benchmark, not a dependency |
| [RocketLeague-PPO-Bot](https://github.com/jszerlag/RocketLeague-PPO-Bot) | 256-wide ReLU/LayerNorm PPO, large vector batches, randomized starts, contact-causal rewards | Isolate width, normalization, activation, PPO epochs, and useful impulse in separate arms |
| [RLBotPack](https://github.com/RLBot/RLBotPack) | Reproducible catalog of heterogeneous opponents | Retain as a future league-opponent packaging reference; it does not answer PPO architecture |
| [Bot-Wheels algorithms](https://bot-wheels.github.io/docs/research/research-and-documentation-of-algorithm-categories-for-rocket-league-bot/) | PPO is suitable but sensitive to reward and hyperparameter design | Keep MAPPO and measure PPO diagnostics before changing algorithms |
| [Bot-Wheels rewards](https://bot-wheels.github.io/docs/research/research-about-different-reward-functions/) | Mix sparse, dense, event, and contextual rewards while auditing exploitation | Add an event-causal reward without replacing terminal and semantic metrics |
| [Rocket League RL agent](https://sohum-padhye.medium.com/building-a-reinforcement-learning-agent-that-can-play-rocket-league-5df59c69b1f5) | Continuous velocity shaping can teach braking before contact; 256-wide Tanh was used | Reduce directional shaping only in the causal-reward arm and reject on semantic regressions |
| [RLGym PPO Guide](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/intro.md) | Separate collection throughput from learner consumption and use randomized states/no-touch termination | Report matched environment steps and throughput separately; retain our semantic state curriculum and stagnation terminal |

## Controlled protocol

- Two paired seeds per arm.
- Equal worlds, rollout length, environment steps, physics, curriculum, opponent,
  and immutable semantic holdouts.
- Width-only comparison keeps Tanh, rewards, LayerNorm, and PPO epochs fixed.
- Architecture selection order: semantic promotion floors, semantic success,
  terminal score, unresolved rate, then throughput.
- A short screen may nominate a longer-run candidate but cannot replace the
  production default.

## CUDA screen

The persisted report is
`experiments/reports/m18/ppo-ablation.json`. Each arm received 81,920
environment steps over the same two seeds.

| Arm | Actor params | Train frames/s | PPO KL | Clip fraction | Semantic success | Floors passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128_tanh` | 54,148 | 2,162 | 0.0021 | 0.020 | 0.0% | 0/5 |
| `256_tanh` | 206,596 | 2,109 | 0.0035 | 0.041 | 8.9% | 0/5 |
| `256_tanh_ln` | 208,132 | 2,104 | 0.0101 | 0.114 | 0.0% | 0/5 |
| `256_relu_ln` | 208,132 | 2,085 | 0.0076 | 0.089 | 11.6% | 2/5 |
| `256_relu_ln_2epoch` | 208,132 | 2,049 | 0.0101 | 0.112 | 0.0% | 0/5 |
| `256_relu_ln_impulse` | 208,132 | 2,055 | 0.0077 | 0.091 | 8.9% | 1/5 |

The width-only arm costs 3.8 times the actor parameters and 2.4% throughput
while producing more early semantic successes than the 128 control. Width is
therefore useful but insufficient. ReLU plus LayerNorm is the confirmation
candidate because it alone clears the interception and save-deflection floors.
LayerNorm with Tanh and the two-epoch variant regress, so neither component is
credited independently. The useful-impulse arm clears pass-receive and has the
lowest unresolved rate, but remains a separate reward hypothesis rather than
part of the architecture winner.

All terminal scores remain 0.5 at this fidelity. The report consequently marks
the winner as requiring longer confirmation and does not change the M17
production default.

## Rollback

The new configuration fields default to the M17 behavior (`tanh`, no
LayerNorm, no useful-impulse reward). Rollback is therefore a configuration
change; existing checkpoints remain loadable through neutral legacy defaults.
