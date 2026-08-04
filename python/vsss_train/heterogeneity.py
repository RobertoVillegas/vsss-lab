"""Measure how much a role-conditioned policy actually depends on its role signal.

ADR 0026's addendum asks for direct evidence that the role machinery earns its keep instead of
assuming it. Following the heterogeneity-gain definition of Amir, Bettini and Prorok
(arXiv:2506.09434), this module compares a checkpoint's progress when it receives the real
transient role assignment against the same policy evaluated with the role columns of the
observation context removed or scrambled. A gain near zero means the conditioning is dead
weight and the role stack should be simplified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
)
from vsss_train.marl import (
    RoleSharedActor,
    SharedActor,
    TeamBatch,
)
from vsss_train.marl_env import MarlMatchEnv

# The observation context is [time, score, event bits (4)] ++ role_features (5); only the role
# columns are touched by an ablation.
ROLE_FEATURE_WIDTH = 5
ABLATION_MODES = ("uniform", "shuffle", "none")
ROLE_FEATURE_WIDTH = 5


@dataclass(frozen=True)
class HeterogeneityGain:
    """Paired comparison between the conditioned policy and its role-ablated counterpart."""

    ablation: str
    seeds: int
    conditioned_progress: float
    ablated_progress: float
    gain: float
    conditioned_std: float
    ablated_std: float


def ablate_role_features(
    observation: TeamBatch,
    mode: str,
    rng: np.random.Generator,
) -> TeamBatch:
    """Return the observation with the role columns of context removed or scrambled.

    `uniform` makes every robot believe it is the attacker, the closest analogue of the
    homogeneous-policy arm in the literature; `shuffle` randomly reassigns the role one-hots
    across teammates on every decision; `none` zeroes the block entirely.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"ablation mode must be one of {ABLATION_MODES}")
    context = observation.context.clone()
    roles = context[..., -ROLE_FEATURE_WIDTH:]
    if mode == "none":
        roles.zero_()
    elif mode == "uniform":
        roles.zero_()
        roles[..., 0] = 1.0
    else:
        order = torch.from_numpy(rng.permutation(roles.shape[-2]))
        context[..., -ROLE_FEATURE_WIDTH:] = roles.index_select(-2, order)
    return observation._replace(context=context)


def measure_heterogeneity_gain(
    actor: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    config_json: str,
    state_json: str,
    *,
    stage: int,
    seeds: range,
    horizon: int,
    action_repeat: int = 4,
    action_parser: str = "continuous",
    ablation: str = "uniform",
) -> HeterogeneityGain:
    """Run paired episodes and report the progress lost when roles are ablated.

    Both arms start from the same seeded reset, so any difference in progress comes from the
    role signal rather than from initial conditions.
    """
    if ablation not in ABLATION_MODES:
        raise ValueError(f"ablation mode must be one of {ABLATION_MODES}")
    actor.eval()
    device = next(actor.parameters()).device
    conditioned_scores: list[float] = []
    ablated_scores: list[float] = []
    for seed in seeds:
        for arm, scores in (
            ("conditioned", conditioned_scores),
            ("ablated", ablated_scores),
        ):
            environment = MarlMatchEnv(
                config_json,
                state_json,
                stage=stage,
                horizon=horizon,
                action_repeat=action_repeat,
                action_parser=action_parser,
            )
            observation = environment.reset(seed)
            environment.mark_progress_origin()
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                if arm == "ablated":
                    observation = ablate_role_features(observation, ablation, rng)
                with torch.no_grad():
                    action = actor.deterministic_action(observation.to(device)).cpu().numpy()
                observation, _, done, _ = environment.step(action)
            scores.append(environment.progress_score())

    def _spread(scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        return float(np.std(np.asarray(scores), ddof=1))

    conditioned = float(np.mean(conditioned_scores))
    ablated = float(np.mean(ablated_scores))
    return HeterogeneityGain(
        ablation=ablation,
        seeds=len(seeds),
        conditioned_progress=conditioned,
        ablated_progress=ablated,
        gain=conditioned - ablated,
        conditioned_std=_spread(conditioned_scores),
        ablated_std=_spread(ablated_scores),
    )
