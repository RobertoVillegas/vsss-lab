"""Run a short training probe and report whether the policy keeps going for the ball.

Run 0008 collapsed over roughly seven hundred iterations: strikes fell, `stop` rose, unresolved
drills rose, and full-match scoring reached zero. A reward change that claims to prevent that
has to be measured against those same quantities rather than argued for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import MarlLearner

ROOT = Path(__file__).parents[1]


def probe(config_overrides: dict[str, object], iterations: int, seed: int) -> list[dict]:
    config = load_marl_config(ROOT / "experiments/configs/m24-3-mappo-circular.toml")
    config = replace(config, seed=seed, **config_overrides)
    match_config = (ROOT / "tests/golden/m1_match_config.json").read_text()
    match_state = (ROOT / "tests/golden/m1_match_state.json").read_text()
    learner = MarlLearner(config)
    session = create_rollout_session(config, match_config, match_state)

    history = []
    for iteration in range(iterations):
        if iteration and iteration % 25 == 0:
            recent = history[-25:]
            print(
                "    %4d  remates %.3f  stop %.3f  sin resolver %.3f"
                % (
                    iteration,
                    sum(r["strike_fraction"] for r in recent) / len(recent),
                    sum(r["stop_fraction"] for r in recent) / len(recent),
                    sum(r["unresolved_share"] for r in recent) / len(recent),
                ),
                flush=True,
            )
        result = train_iteration(
            learner,
            None,
            match_config,
            match_state,
            iteration=iteration,
            seed=config.seed + iteration,
            opponent_id="heuristic",
            checkpoint=None,
            session=session,
        )
        stats = result.policy_stats or {}
        outcomes = ((result.curriculum or {}).get("outcomes")) or {}
        resolved = sum(outcomes.values()) or 1
        history.append(
            {
                "iteration": iteration,
                "strike_fraction": float(stats.get("strike_fraction", 0.0)),
                "stop_fraction": float(stats.get("stop_fraction", 0.0)),
                "mean_intensity": float(stats.get("mean_intensity", 0.0)),
                "unresolved_share": outcomes.get("unresolved", 0) / resolved,
                "success_share": outcomes.get("success", 0) / resolved,
                "goal_scored": float((result.reward_terms or {}).get("goal_scored", 0.0)),
            }
        )
    return history


def summarize(name: str, history: list[dict]) -> str:
    def window(rows: list[dict], key: str) -> float:
        return sum(row[key] for row in rows) / max(1, len(rows))

    early, late = history[: len(history) // 4], history[-len(history) // 4 :]
    fields = (
        "strike_fraction",
        "stop_fraction",
        "mean_intensity",
        "unresolved_share",
        "goal_scored",
    )
    parts = [f"{name:<22}"]
    for field in fields:
        parts.append(f"{window(early, field):.3f}->{window(late, field):.3f}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    variants = {
        "timeout libre (0008)": {"semantic_timeout_penalty": 0.0},
        "timeout = fracaso": {"semantic_timeout_penalty": None},
    }
    print(f"{arguments.iterations} iteraciones por variante, primer cuarto -> último cuarto")
    header = f"{'variante':<22}" + "  ".join(
        f"{n:>13}" for n in ("remates", "stop", "intensidad", "sin resolver", "goles")
    )
    print(header, flush=True)
    results = {}
    for name, overrides in variants.items():
        print(name, flush=True)
        history = probe(overrides, arguments.iterations, arguments.seed)
        results[name] = history
        print(summarize(name, history), flush=True)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
