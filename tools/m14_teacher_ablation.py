"""Compare scratch MAPPO, verified imitation, and imitation plus MAPPO.

DEPRECATED: M14 is closed and this study is pinned to its configuration; it does
not evaluate the M24.2 parametric action space. See docs/tooling-status.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import cast

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import load_marl_config
from vsss_train.marl import SharedActor
from vsss_train.marl_ppo import MarlLearner
from vsss_train.teacher import (
    ExactApproachRollout,
    behavior_clone_demonstration,
    plan_atomic_skill,
)


@dataclass(frozen=True)
class TeacherArm:
    arm: str
    wins: int
    draws: int
    losses: int
    terminal_score: float
    mean_clone_loss: float | None
    compute_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/m14-mappo-adaptive.toml")
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/m14-teacher.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()
    base = replace(
        load_marl_config(arguments.config),
        device=arguments.device,
        adaptive_curriculum=False,
        scenario_suite="",
        policy_architecture="mlp",
        action_parser="continuous",
        num_envs=8,
        rollout_steps=8,
        horizon=60,
        epochs=1,
        minibatch_size=96,
    )
    seeds = tuple(base.seed + 1_200_000 + index for index in range(arguments.seeds))
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    results: dict[str, list[tuple[int, int, int, float | None, float]]] = {
        "scratch_mappo": [],
        "verified_imitation": [],
        "imitation_plus_mappo": [],
    }
    for seed in seeds:
        demonstration = plan_atomic_skill(
            ExactApproachRollout(config_json, state_json, seed=seed),
            skill="approach",
            seed=seed,
            horizon=100,
            population=16,
            elites=4,
            generations=3,
        )
        for arm in results:
            started = time.perf_counter()
            config = replace(base, seed=seed)
            learner = MarlLearner(config)
            clone_loss: float | None = None
            if arm != "scratch_mappo":
                clone_loss = behavior_clone_demonstration(
                    cast(SharedActor, learner.actor),
                    demonstration,
                    config_json,
                    state_json,
                    epochs=5,
                )
            if arm != "verified_imitation":
                session = create_rollout_session(config, config_json, state_json)
                for iteration in range(1, 4):
                    train_iteration(
                        learner,
                        None,
                        config_json,
                        state_json,
                        iteration=iteration,
                        seed=seed + iteration,
                        opponent_id="heuristic",
                        checkpoint=None,
                        session=session,
                    )
            scorecard = evaluate_checkpoint_scorecard(
                learner.actor,
                config_json,
                state_json,
                checkpoint=Path("in-memory.pt"),
                policy_version=learner.policy_version,
                seeds=(seed + 1_300_000,),
                ticks=60,
                action_parser=learner.config.action_parser,
            )
            results[arm].append(
                (
                    scorecard.wins,
                    scorecard.draws,
                    scorecard.losses,
                    clone_loss,
                    time.perf_counter() - started,
                )
            )
    arms = tuple(_summarize(name, values) for name, values in results.items())
    payload = {
        "schema_version": 1,
        "verified_exact_teacher": True,
        "paired_seeds": list(seeds),
        "arms": [asdict(arm) for arm in arms],
        "decision": (
            "no_terminal_advantage"
            if len({arm.terminal_score for arm in arms}) == 1
            else max(arms, key=lambda arm: (arm.terminal_score, -arm.compute_seconds)).arm
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _summarize(
    name: str,
    values: list[tuple[int, int, int, float | None, float]],
) -> TeacherArm:
    wins = sum(value[0] for value in values)
    draws = sum(value[1] for value in values)
    losses = sum(value[2] for value in values)
    losses_clone = [value[3] for value in values if value[3] is not None]
    matches = wins + draws + losses
    return TeacherArm(
        arm=name,
        wins=wins,
        draws=draws,
        losses=losses,
        terminal_score=(wins + 0.5 * draws) / matches,
        mean_clone_loss=fmean(losses_clone) if losses_clone else None,
        compute_seconds=sum(value[4] for value in values),
    )


if __name__ == "__main__":
    main()
