"""Audit whether each skill family's difficulty axes move the demand it is scored on.

A difficulty curriculum can only rescue a failing family if lowering difficulty makes the
family easier at the thing it scores. Two families have already been found where it did
not: clearance held the ball's depth constant across every axis, and interception scores
exactly one half because its bands are a cliff rather than a ramp.

The scripted controller is the reference. It is not a policy under evaluation, so its
success rate isolates the scenario from the learner: if the reference cannot solve a
family's easiest band, the band is not easy, and no amount of training or reward will make
the family learnable from below.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import MarlLearner
from vsss_train.semantic_evaluation import IdleSpinThresholds, evaluate_semantic_skills
from vsss_train.semantic_scenarios import (
    SKILL_FAMILIES,
    SemanticScenario,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

LADDER = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)
# A reference controller that cannot clear half of the easiest band leaves the difficulty
# curriculum no rung to stand on.
EASIEST_BAND_FLOOR = 0.5


@dataclass(frozen=True)
class Cell:
    family: str
    level: float
    attempts: int
    successes: int
    invalid: str | None = None

    @property
    def rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


def scenarios_for(
    family: str,
    level: float,
    state: dict[str, object],
    config: dict[str, object],
    *,
    seeds: int,
) -> tuple[SemanticScenario, ...]:
    return tuple(
        compile_skill_scenario(
            SkillScenarioParameters(
                schema_version=1,
                family=family,  # type: ignore[arg-type]
                seed=601 + index * 37,
                controlled_team=team,  # type: ignore[arg-type]
                difficulty=SkillDifficulty(level, level, level, level, level),
                horizon=240,
                holdout=True,
            ),
            state,
            config,
        )
        for team in ("blue", "yellow")
        for index in range(seeds)
    )


def audit(
    config_text: str,
    state_text: str,
    *,
    seeds: int,
    probe: str = "heuristic",
    learner: MarlLearner | None = None,
) -> list[Cell]:
    config = json.loads(config_text)
    state = json.loads(state_text)
    cells: list[Cell] = []
    for family in SKILL_FAMILIES:
        for level in LADDER:
            try:
                batch = scenarios_for(family, level, state, config, seeds=seeds)
            except ValueError as error:
                # A declared difficulty that cannot be compiled is a generator defect, not
                # a policy result, and the spec requires zero invalid states.
                cells.append(Cell(family, level, 0, 0, invalid=str(error)))
                continue
            report = evaluate_semantic_skills(
                learner.actor if learner is not None else None,
                batch,
                config_text,
                state_text,
                control="policy" if learner is not None else probe,  # type: ignore[arg-type]
                device=learner.device if learner is not None else "cpu",
                action_parser=(
                    learner.config.action_parser if learner is not None else "continuous"
                ),
                idle_spin=IdleSpinThresholds(
                    angular_speed=(
                        learner.config.idle_spin_angular_speed if learner is not None else 1.0
                    )
                ),
            )
            successes = sum(trial.status == "success" for trial in report.trials)
            cells.append(Cell(family, level, len(report.trials), successes))
    return cells


def classify(rates: list[float], invalid: list[float]) -> str:
    """Name the shape of a difficulty ladder, so a defect is read rather than inferred."""
    if invalid:
        return "invalid-generation"
    easiest, hardest = rates[0], rates[-1]
    live = sum(1 for rate in rates if rate > 0.0)
    if hardest > easiest + 0.2:
        return "inverted"
    if max(rates) - min(rates) < 0.2:
        return "no-gradient"
    if easiest < EASIEST_BAND_FLOOR:
        # The scripted controller cannot shoot or pass, so this is not proof of a defect.
        return "beyond-reference"
    if live <= 2:
        return "cliff"
    return "ramp"


def verdicts(cells: list[Cell]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for family in SKILL_FAMILIES:
        row = [cell for cell in cells if cell.family == family]
        rates = [cell.rate for cell in row]
        invalid = [cell.level for cell in row if cell.invalid]
        result[family] = {
            "rates": rates,
            "easiest": rates[0],
            "hardest": rates[-1],
            "live_bands": sum(1 for rate in rates if rate > 0.0),
            "invalid_levels": invalid,
            "verdict": classify(rates, invalid),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-config", type=Path, default=Path("tests/golden/m1_match_config.json")
    )
    parser.add_argument(
        "--match-state", type=Path, default=Path("tests/golden/m1_match_state.json")
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--probe", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--config", type=Path, help="training config for a policy probe")
    parser.add_argument("--checkpoint", type=Path, help="probe with this trained policy instead")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config_text = arguments.match_config.read_text()
    state_text = arguments.match_state.read_text()
    learner = None
    if arguments.checkpoint is not None:
        if arguments.config is None:
            raise SystemExit("--checkpoint requires --config")
        learner = MarlLearner(load_marl_config(arguments.config))
        learner.load(arguments.checkpoint)
    cells = audit(
        config_text,
        state_text,
        seeds=arguments.seeds,
        probe=arguments.probe,
        learner=learner,
    )
    summary = verdicts(cells)

    header = "family".ljust(19) + "".join(f"{level:>7.2f}" for level in LADDER) + "   verdict"
    print(header)
    print("-" * len(header))
    for family, entry in summary.items():
        rates = entry["rates"]
        assert isinstance(rates, list)
        invalid_levels = entry["invalid_levels"]
        assert isinstance(invalid_levels, list)
        row = family.ljust(19) + "".join(
            "    n/a" if level in invalid_levels else f"{rate:>7.2f}"
            for level, rate in zip(LADDER, rates, strict=True)
        )
        print(f"{row}   {entry['verdict']}")
    print()
    probe_name = (
        f"trained policy {arguments.checkpoint.name}"
        if arguments.checkpoint is not None
        else f"scripted {arguments.probe}"
    )
    print(f"probe: {probe_name}, {arguments.seeds * 2} attempts per cell")
    print(f"easiest band must reach {EASIEST_BAND_FLOOR:.2f}; difficulty must cost at least 0.20")
    print("ramp = usable ladder | cliff = solvable then impossible | no-gradient = axes do")
    print("nothing | inverted = harder is easier | beyond-reference = the scripted control")
    print("cannot perform this skill, so the ladder is untested rather than proven broken")
    usable = [f for f, e in summary.items() if e["verdict"] == "ramp"]
    print(f"families with a usable ladder: {', '.join(usable) if usable else 'none'}")
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(
                {
                    "ladder": list(LADDER),
                    "easiest_band_floor": EASIEST_BAND_FLOOR,
                    "attempts_per_cell": arguments.seeds * 2,
                    "probe": probe_name,
                    "families": summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
