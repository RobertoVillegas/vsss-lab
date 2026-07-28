"""Compare fixed-reward uniform and adaptive curricula on paired seeds."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_ppo import MarlLearner


@dataclass(frozen=True)
class ArmResult:
    arm: str
    seeds: tuple[int, ...]
    wins: int
    draws: int
    losses: int
    terminal_score: float
    mean_return: float
    mean_coordination_failure: float
    compute_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/m14-mappo-adaptive.toml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/reports/m14-curriculum.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.seeds < 2:
        parser.error("--seeds must be at least 2")
    base = replace(
        load_marl_config(arguments.config),
        device=arguments.device,
        num_envs=8,
        rollout_steps=8,
        horizon=60,
        epochs=1,
        minibatch_size=96,
    )
    seeds = tuple(base.seed + 600_000 + index for index in range(arguments.seeds))
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    arms = tuple(
        _run_arm(
            replace(
                base,
                adaptive_curriculum=adaptive,
                scenario_suite=base.scenario_suite if adaptive else "",
            ),
            name,
            seeds,
            config_json,
            state_json,
        )
        for name, adaptive in (("uniform", False), ("adaptive", True))
    )
    payload = {
        "schema_version": 1,
        "fixed_reward": True,
        "paired_seeds": list(seeds),
        "arms": [asdict(arm) for arm in arms],
        "terminal_score_delta": arms[1].terminal_score - arms[0].terminal_score,
        "decision": (
            "advance_curriculum"
            if arms[1].terminal_score > arms[0].terminal_score
            else "no_terminal_advantage"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _run_arm(
    config: MarlConfig,
    name: str,
    seeds: tuple[int, ...],
    config_json: str,
    state_json: str,
) -> ArmResult:
    started = time.perf_counter()
    returns: list[float] = []
    coordination: list[float] = []
    wins = draws = losses = 0
    for seed in seeds:
        seeded = replace(config, seed=seed)
        learner = MarlLearner(seeded)
        session = create_rollout_session(seeded, config_json, state_json)
        for iteration in range(1, 4):
            result = train_iteration(
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
            returns.append(result.return_total)
            coordination.append(result.losses["action_saturation"])
        scorecard = evaluate_checkpoint_scorecard(
            learner.actor,
            config_json,
            state_json,
            checkpoint=Path("in-memory.pt"),
            policy_version=learner.policy_version,
            seeds=(seed + 700_000,),
            ticks=60,
        )
        wins += scorecard.wins
        draws += scorecard.draws
        losses += scorecard.losses
    matches = wins + draws + losses
    return ArmResult(
        arm=name,
        seeds=seeds,
        wins=wins,
        draws=draws,
        losses=losses,
        terminal_score=(wins + 0.5 * draws) / matches,
        mean_return=fmean(returns),
        mean_coordination_failure=fmean(coordination),
        compute_seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
