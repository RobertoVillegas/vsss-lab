"""Bounded exact-simulator planning and verified demonstration contracts."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from vsss_train.marl import SharedActor, TeamBatch, stack_team_batches
from vsss_train.marl_env import MarlMatchEnv

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SkillResult:
    score: float
    success: bool
    physically_valid: bool
    terminal_reason: str


@dataclass(frozen=True)
class Demonstration:
    skill: str
    seed: int
    actions: tuple[tuple[float, float], ...]
    score: float
    terminal_reason: str


Rollout = Callable[[FloatArray], SkillResult]


def plan_atomic_skill(
    rollout: Rollout,
    *,
    skill: str,
    seed: int,
    horizon: int = 20,
    population: int = 128,
    elites: int = 16,
    generations: int = 6,
) -> Demonstration:
    """Use bounded CEM, then replay the winner through the exact verifier."""
    if horizon <= 0 or population <= 1 or not 1 <= elites < population or generations <= 0:
        raise ValueError("invalid bounded planner dimensions")
    generator = np.random.default_rng(seed)
    mean = np.zeros((horizon, 2), dtype=np.float32)
    deviation = np.ones((horizon, 2), dtype=np.float32)
    best: FloatArray | None = None
    for _ in range(generations):
        candidates = np.clip(
            generator.normal(mean, deviation, size=(population, horizon, 2)),
            -1.0,
            1.0,
        ).astype(np.float32)
        candidates[0] = mean
        candidates[1] = 1.0
        if population > 2:
            candidates[2] = -1.0
        scored = [(rollout(candidate), candidate) for candidate in candidates]
        valid = [
            (result.score, candidate) for result, candidate in scored if result.physically_valid
        ]
        if not valid:
            raise ValueError("planner produced no physically valid trajectory")
        valid.sort(key=lambda item: item[0], reverse=True)
        selected = np.stack([candidate for _, candidate in valid[:elites]])
        mean = selected.mean(axis=0).astype(np.float32)
        deviation = np.maximum(selected.std(axis=0), 0.05).astype(np.float32)
        best = valid[0][1]
    if best is None:
        raise AssertionError("planner failed to select a trajectory")
    verified = rollout(best.copy())
    if not verified.physically_valid:
        raise ValueError("winning trajectory failed exact physics replay")
    if not verified.success:
        raise ValueError("winning trajectory failed skill predicate")
    return Demonstration(
        skill=skill,
        seed=seed,
        actions=tuple((float(action[0]), float(action[1])) for action in best),
        score=verified.score,
        terminal_reason=verified.terminal_reason,
    )


def write_demonstrations(path: Path, demonstrations: tuple[Demonstration, ...]) -> None:
    payload = {
        "schema_version": 1,
        "verified_exact_simulator": True,
        "demonstrations": [asdict(item) for item in demonstrations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class ExactApproachRollout:
    """Replay candidate commands through Rapier for one verified approach skill."""

    def __init__(self, config_json: str, state_json: str, *, seed: int) -> None:
        self.config_json = config_json
        self.state_json = state_json
        self.seed = seed
        self.initial_state = json.loads(state_json)
        self.initial_state.update(
            tick=0,
            simulation_time=0.0,
            score_blue=0,
            score_yellow=0,
            events=0,
        )
        self.initial_state["ball"].update(x=-0.10, y=0.0, vx=0.0, vy=0.0, omega=0.0)

    def __call__(self, actions: FloatArray) -> SkillResult:
        environment = MarlMatchEnv(
            self.config_json,
            self.state_json,
            stage=7,
            horizon=len(actions),
            action_repeat=1,
        )
        environment.reset_state(copy.deepcopy(self.initial_state))
        environment.mark_progress_origin()
        minimum_distance = float("inf")
        for command in actions:
            team_actions = np.zeros((3, 2), dtype=np.float32)
            team_actions[0] = command
            _, _, done, info = environment.step(team_actions)
            minimum_distance = min(minimum_distance, float(info["closest_ball_distance"]))
            if done:
                break
        finite = bool(np.isfinite(environment.state).all())
        score = environment.progress_score()
        return SkillResult(
            score=score,
            success=score >= 0.03 or minimum_distance <= 0.09,
            physically_valid=finite,
            terminal_reason="approach"
            if score >= 0.03
            else "contact"
            if minimum_distance <= 0.09
            else "timeout",
        )


def behavior_clone_demonstration(
    actor: SharedActor,
    demonstration: Demonstration,
    config_json: str,
    state_json: str,
    *,
    epochs: int = 20,
) -> float:
    """Clone one exact verified trajectory before optional MAPPO fine-tuning."""
    if epochs <= 0:
        raise ValueError("behavior cloning epochs must be positive")
    rollout = ExactApproachRollout(config_json, state_json, seed=demonstration.seed)
    verified = rollout(np.asarray(demonstration.actions, dtype=np.float32))
    if not verified.physically_valid or not verified.success:
        raise ValueError("demonstration no longer passes exact verification")
    environment = MarlMatchEnv(
        config_json,
        state_json,
        stage=7,
        horizon=len(demonstration.actions),
        action_repeat=1,
    )
    observation = environment.reset_state(copy.deepcopy(rollout.initial_state))
    observations: list[TeamBatch] = []
    targets: list[torch.Tensor] = []
    for command in demonstration.actions:
        observations.append(observation)
        team_actions = np.zeros((3, 2), dtype=np.float32)
        team_actions[0] = command
        targets.append(torch.from_numpy(team_actions.copy()))
        observation, _, done, _ = environment.step(team_actions)
        if done:
            break
    batch = stack_team_batches(observations)
    target = torch.stack(targets)
    device = next(actor.parameters()).device
    batch = batch.to(device)
    target = target.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss = torch.zeros((), device=device)
    for _ in range(epochs):
        mean, _ = actor(batch)
        loss = (torch.tanh(mean) - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return float(loss.detach())
