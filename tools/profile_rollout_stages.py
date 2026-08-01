"""Attribute rollout time to the stages ADR 0019 migrates, one slice at a time.

The migration's own rule is that the thirteen-times estimate is an extrapolation and must be
replaced by measurement as it is approached. That is only possible with a measurement anyone
can repeat, on the live configuration rather than a benchmark shaped to flatter the port. This
runs the real vector environment and reports where an iteration's seconds actually go.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from vsss_league.training import create_rollout_session
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_env import team_action_width

ROOT = Path(__file__).parents[1]
# One entry point per stage ADR 0019 migrates. Cumulative time is the measure that matters here:
# a stage is worth porting for everything it calls, and self time hides that cost in numpy and
# math leaves. The consequence is that stages overlap — the reward's goal geometry performs its
# own role assignment — so the shares are read individually and never summed.
STAGES: tuple[tuple[str, str], ...] = (
    ("observations", "build_team_observation"),
    ("role assignment", "assign_roles"),
    ("reward geometry", "_goal_geometry_metrics"),
    ("action execution", "circular_primitive_wheel_actions"),
    ("idle spin", "_idle_spin_flags"),
    ("free ball", "_restart_free_ball"),
    ("native physics", "step_repeated"),
)


@dataclass(frozen=True)
class Stage:
    """One named stage of the rollout and the time attributed to it."""

    name: str
    seconds: float
    calls: int


def profile_stages(profiler: cProfile.Profile) -> tuple[list[Stage], float]:
    """Group profiled functions into the stages the migration is organized around."""
    stats = pstats.Stats(profiler)
    measured: dict[str, tuple[float, int]] = {}
    total = 0.0
    for (_, _, function), entry in stats.stats.items():  # type: ignore[attr-defined]
        calls, _, self_time, cumulative, _ = entry
        total += self_time
        for name, entry_point in STAGES:
            if entry_point in function:
                seconds, seen = measured.get(name, (0.0, 0))
                measured[name] = (seconds + cumulative, seen + calls)
    stages = [Stage(name, *measured.get(name, (0.0, 0))) for name, _ in STAGES]
    return sorted(stages, key=lambda stage: -stage.seconds), total


def run(config: MarlConfig, config_json: str, state_json: str, decisions: int) -> dict[str, Any]:
    """Step the live environment for a fixed number of decisions under the profiler."""
    session = create_rollout_session(config, config_json, state_json)
    environment = session.environment
    for world in range(environment.num_envs):
        environment.reset(world, config.seed + world)
    generator = np.random.default_rng(config.seed)
    width = team_action_width(config.action_parser)
    shape = (environment.num_envs, 3, width)

    for _ in range(8):  # warm the allocator and the branch predictors before measuring
        environment.step(generator.uniform(-1.0, 1.0, shape).astype(np.float32), None)

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    for _ in range(decisions):
        environment.step(generator.uniform(-1.0, 1.0, shape).astype(np.float32), None)
    profiler.disable()
    wall = time.perf_counter() - started

    stages, profiled = profile_stages(profiler)
    return {
        "worlds": environment.num_envs,
        "decisions": decisions,
        "wall_seconds": wall,
        "profiled_seconds": profiled,
        "stages": [
            {"stage": stage.name, "seconds": stage.seconds, "calls": stage.calls}
            for stage in stages
        ],
    }


def report(result: dict[str, Any]) -> str:
    """Render the measurement so a share is readable without arithmetic."""
    profiled = float(result["profiled_seconds"])
    lines = [
        f"{result['decisions']} decisions across {result['worlds']} worlds",
        f"wall {result['wall_seconds']:.2f} s, profiled {profiled:.2f} s"
        " (the profiler's own overhead inflates both)",
        "",
        f"{'stage':<20}{'cumulative':>12}{'share':>9}{'calls':>12}",
    ]
    for stage in result["stages"]:
        share = 100.0 * float(stage["seconds"]) / profiled if profiled else 0.0
        lines.append(
            f"{stage['stage']:<20}{stage['seconds']:>12.3f}{share:>8.1f}%{stage['calls']:>12,}"
        )
    lines.append("")
    lines.append("Stages nest, so the shares overlap and must not be summed.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/configs/m24-3-mappo-circular.toml"
    )
    parser.add_argument(
        "--match-config", type=Path, default=ROOT / "tests/golden/m1_match_config.json"
    )
    parser.add_argument(
        "--match-state", type=Path, default=ROOT / "tests/golden/m1_match_state.json"
    )
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--worlds", type=int, default=None, help="override the config's num_envs")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    config = load_marl_config(arguments.config)
    if arguments.worlds is not None:
        config = replace(config, num_envs=arguments.worlds)
    result = run(
        config,
        arguments.match_config.read_text(),
        arguments.match_state.read_text(),
        arguments.decisions,
    )
    print(report(result))
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
