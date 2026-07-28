"""Typed deterministic M15 semantic skill scenario compilation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Literal

from vsss_train.scenarios import Scenario, validate_scenario

SkillFamily = Literal[
    "approach",
    "interception",
    "save_deflection",
    "clearance",
    "shot",
    "pass_receive",
]
ControlledTeam = Literal["blue", "yellow"]

GENERATOR_REVISION = "m15.1"
DIFFICULTY_AXES = (
    "ball_speed",
    "ball_angle",
    "spawn_distance",
    "target_width",
    "opponent_pressure",
)
SKILL_FAMILIES: tuple[SkillFamily, ...] = (
    "approach",
    "interception",
    "save_deflection",
    "clearance",
    "shot",
    "pass_receive",
)


@dataclass(frozen=True)
class SkillDifficulty:
    """Independent normalized difficulty axes in [0, 1]."""

    ball_speed: float = 0.0
    ball_angle: float = 0.0
    spawn_distance: float = 0.0
    target_width: float = 0.0
    opponent_pressure: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"difficulty {name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class SkillScenarioParameters:
    schema_version: int
    family: SkillFamily
    seed: int
    controlled_team: ControlledTeam
    difficulty: SkillDifficulty
    horizon: int = 250
    holdout: bool = False
    generator_revision: str = GENERATOR_REVISION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported semantic scenario parameter schema")
        if self.family not in (
            "approach",
            "interception",
            "save_deflection",
            "clearance",
            "shot",
            "pass_receive",
        ):
            raise ValueError(f"unsupported skill family: {self.family}")
        if self.controlled_team not in ("blue", "yellow"):
            raise ValueError("controlled_team must be blue or yellow")
        if self.horizon <= 0:
            raise ValueError("semantic scenario horizon must be positive")
        if self.generator_revision != GENERATOR_REVISION:
            raise ValueError("unsupported semantic scenario generator revision")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SkillContext:
    family: SkillFamily
    controlled_team: ControlledTeam
    controlled_robot_id: str
    support_robot_id: str | None
    target_goal_x: float
    own_goal_x: float
    target_y: float
    target_half_width: float
    initial_ball_speed: float
    initial_threat: bool
    horizon: int
    parameter_hash: str
    state_hash: str


@dataclass(frozen=True)
class SemanticScenario:
    scenario: Scenario
    parameters: SkillScenarioParameters
    context: SkillContext


@dataclass(frozen=True)
class SemanticSelection:
    scenario: SemanticScenario | None
    source: Literal["full_match", "routine", "frontier", "failure"]


class SemanticSkillCurriculum:
    """Learning-progress scheduler with mirrored colors and failure rehearsal."""

    def __init__(
        self,
        base_state: dict[str, Any],
        config: dict[str, Any],
        *,
        seed: int,
        full_match_fraction: float = 0.25,
        window: int = 24,
    ) -> None:
        if not 0.0 <= full_match_fraction <= 1.0:
            raise ValueError("full_match_fraction must be in [0, 1]")
        if window < 4:
            raise ValueError("curriculum window must be at least four")
        self.base_state = copy.deepcopy(base_state)
        self.config = config
        self.seed = seed
        self.full_match_fraction = full_match_fraction
        self.window = window
        self.levels = {
            family: {axis: 0.05 for axis in DIFFICULTY_AXES} for family in SKILL_FAMILIES
        }
        self.outcomes: dict[SkillFamily, deque[float]] = defaultdict(
            lambda: deque(maxlen=window * 2)
        )
        self.failures: dict[str, SkillScenarioParameters] = {}
        self.counts: dict[str, int] = defaultdict(int)

    def select_training(self, index: int) -> SemanticSelection:
        generator = random.Random(self.seed + index * 104_729)
        total_allocated = sum(self.counts.values())
        required_full_matches = math.ceil((total_allocated + 1) * self.full_match_fraction)
        if self.counts["full_match"] < required_full_matches:
            self.counts["full_match"] += 1
            return SemanticSelection(None, "full_match")
        if self.failures and generator.random() < 0.15:
            key = sorted(self.failures)[generator.randrange(len(self.failures))]
            parameters = self.failures[key]
            source: Literal["failure", "routine", "frontier"] = "failure"
        else:
            family = SKILL_FAMILIES[index % len(SKILL_FAMILIES)]
            levels = self.levels[family]
            source = "routine" if generator.random() < 0.30 else "frontier"
            amounts = {
                axis: max(0.0, value - 0.15) if source == "routine" else value
                for axis, value in levels.items()
            }
            parameters = SkillScenarioParameters(
                schema_version=1,
                family=family,
                seed=self.seed + index,
                controlled_team="blue" if (index // len(SKILL_FAMILIES)) % 2 == 0 else "yellow",
                difficulty=SkillDifficulty(
                    **{axis: _jitter(amount, generator) for axis, amount in amounts.items()}
                ),
            )
        self.counts[source] += 1
        return SemanticSelection(
            compile_skill_scenario(parameters, self.base_state, self.config),
            source,
        )

    def record(self, scenario: SemanticScenario, *, success: bool) -> None:
        if scenario.parameters.holdout:
            raise ValueError("immutable holdouts cannot be recorded for training")
        family = scenario.parameters.family
        history = self.outcomes[family]
        history.append(float(success))
        if success:
            self.failures.pop(scenario.parameters.digest, None)
        else:
            self.failures.setdefault(scenario.parameters.digest, scenario.parameters)
        if len(history) < self.window * 2:
            return
        values = tuple(history)
        previous = sum(values[: self.window]) / self.window
        current = sum(values[self.window :]) / self.window
        learning_progress = current - previous
        axis = DIFFICULTY_AXES[(len(history) // self.window - 2) % len(DIFFICULTY_AXES)]
        if current >= 0.70 and learning_progress >= -0.05:
            self.levels[family][axis] = min(1.0, self.levels[family][axis] + 0.05)
        elif current < 0.35 and learning_progress <= 0.0:
            self.levels[family][axis] = max(0.0, self.levels[family][axis] - 0.025)

    def holdouts(
        self,
        *,
        seeds: tuple[int, ...] = (10_007, 10_009, 10_037, 10_039, 10_061),
    ) -> tuple[SemanticScenario, ...]:
        """Build an immutable, paired-color suite outside the training allocator."""
        scenarios = []
        for family in SKILL_FAMILIES:
            for team in ("blue", "yellow"):
                for seed in seeds:
                    scenarios.append(
                        compile_skill_scenario(
                            SkillScenarioParameters(
                                schema_version=1,
                                family=family,
                                seed=seed,
                                controlled_team=team,
                                difficulty=SkillDifficulty(0.65, 0.65, 0.65, 0.65, 0.65),
                                holdout=True,
                            ),
                            self.base_state,
                            self.config,
                        )
                    )
        return tuple(scenarios)

    def telemetry(self, *, reset: bool = False) -> dict[str, object]:
        total = sum(self.counts.values())
        observed_full_match_fraction = self.counts["full_match"] / total if total else 0.0
        result: dict[str, object] = {
            "schema_version": 1,
            "levels": dict(self.levels),
            "allocation": dict(self.counts),
            "failure_count": len(self.failures),
            "observed_full_match_fraction": observed_full_match_fraction,
            "allocation_valid": (
                not total or observed_full_match_fraction + 1e-12 >= self.full_match_fraction
            ),
            "success_rate": {
                family: sum(values) / len(values) if values else None
                for family in SKILL_FAMILIES
                if (values := self.outcomes[family])
            },
        }
        if reset:
            self.counts.clear()
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generator_revision": GENERATOR_REVISION,
            "levels": copy.deepcopy(self.levels),
            "outcomes": {family: list(self.outcomes[family]) for family in SKILL_FAMILIES},
            "failures": {
                digest: asdict(parameters) for digest, parameters in self.failures.items()
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("generator_revision") != GENERATOR_REVISION
        ):
            raise ValueError("incompatible semantic curriculum state")
        levels = state.get("levels")
        outcomes = state.get("outcomes")
        failures = state.get("failures")
        if not isinstance(levels, dict) or not isinstance(outcomes, dict):
            raise ValueError("semantic curriculum state is incomplete")
        for family in SKILL_FAMILIES:
            raw_axes = levels.get(family)
            if not isinstance(raw_axes, dict) or set(raw_axes) != set(DIFFICULTY_AXES):
                raise ValueError(f"invalid difficulty state for {family}")
            self.levels[family] = {
                axis: _bounded_number(raw_axes[axis]) for axis in DIFFICULTY_AXES
            }
            raw_outcomes = outcomes.get(family, [])
            if not isinstance(raw_outcomes, list):
                raise ValueError(f"invalid outcome history for {family}")
            self.outcomes[family].clear()
            self.outcomes[family].extend(float(value) for value in raw_outcomes)
        self.failures.clear()
        if isinstance(failures, dict):
            for digest, raw in failures.items():
                if not isinstance(raw, dict):
                    raise ValueError("invalid semantic failure state")
                difficulty = raw.get("difficulty")
                if not isinstance(difficulty, dict):
                    raise ValueError("semantic failure lacks difficulty")
                parameters = SkillScenarioParameters(
                    **{
                        **raw,
                        "difficulty": SkillDifficulty(**difficulty),
                    }
                )
                if parameters.digest != digest:
                    raise ValueError("semantic failure digest mismatch")
                self.failures[str(digest)] = parameters


def compile_skill_scenario(
    parameters: SkillScenarioParameters,
    base_state: dict[str, Any],
    config: dict[str, Any],
) -> SemanticScenario:
    """Compile one deterministic skill instance and validate canonical physics."""
    generator = random.Random(parameters.seed)
    difficulty = parameters.difficulty
    attack_sign = 1.0 if parameters.controlled_team == "blue" else -1.0
    state = copy.deepcopy(base_state)
    state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    _reset_motion(state)
    controlled = [robot for robot in state["robots"] if robot["team"] == parameters.controlled_team]
    opponents = [robot for robot in state["robots"] if robot["team"] != parameters.controlled_team]
    controlled.sort(key=lambda robot: str(robot["id"]))
    opponents.sort(key=lambda robot: str(robot["id"]))
    primary_index = parameters.seed % len(controlled)
    primary = controlled[primary_index]
    support = controlled[(primary_index + 1) % len(controlled)]
    reserve = controlled[(primary_index + 2) % len(controlled)]
    lane_sign = -1.0 if generator.random() < 0.5 else 1.0
    lane = lane_sign * _lerp(0.04, 0.28, difficulty.ball_angle)
    lane += generator.uniform(-0.015, 0.015)
    spawn_distance = _lerp(0.13, 0.34, difficulty.spawn_distance)
    speed = _lerp(0.12, 1.05, difficulty.ball_speed)
    target_half_width = _lerp(0.17, 0.06, difficulty.target_width)
    own_goal_x = -attack_sign * float(config["field"]["length"]) / 2
    target_goal_x = -own_goal_x
    target_y = attack_sign * lane

    _park_robot(reserve, attack_sign, -0.58, -lane_sign * 0.48, 0.0)
    _park_opponents(opponents, attack_sign, difficulty.opponent_pressure, lane_sign)

    family = parameters.family
    initial_threat = False
    support_id: str | None = None
    if family == "approach":
        ball = (-0.05, lane)
        _set_ball(state, attack_sign, *ball, speed=0.0, heading=0.0)
        _place_behind(
            primary,
            attack_sign,
            ball,
            spawn_distance,
            heading_error=0.22,
            generator=generator,
        )
        _park_robot(support, attack_sign, -0.52, -lane_sign * 0.32, 0.0)
    elif family in ("interception", "save_deflection"):
        ball = (-0.28 if family == "interception" else -0.46, lane)
        heading = math.pi + lane_sign * _lerp(0.02, 0.32, difficulty.ball_angle)
        _set_ball(state, attack_sign, *ball, speed=speed, heading=heading)
        intercept_x = ball[0] - _lerp(0.10, 0.24, difficulty.spawn_distance)
        intercept_y = lane + lane_sign * _lerp(
            0.11,
            0.22,
            difficulty.spawn_distance,
        )
        intercept_heading = math.atan2(ball[1] - intercept_y, ball[0] - intercept_x)
        _park_robot(
            primary,
            attack_sign,
            intercept_x,
            intercept_y,
            intercept_heading,
        )
        _park_robot(support, attack_sign, -0.58, -lane_sign * 0.34, 0.0)
        initial_threat = True
    elif family == "clearance":
        ball = (-0.48, lane * 0.55)
        heading = math.pi + lane_sign * _lerp(0.0, 0.22, difficulty.ball_angle)
        _set_ball(state, attack_sign, *ball, speed=speed * 0.45, heading=heading)
        clearance_x = ball[0] - _lerp(0.08, 0.14, difficulty.spawn_distance)
        clearance_y = ball[1] + lane_sign * _lerp(
            0.10,
            0.18,
            difficulty.spawn_distance,
        )
        _park_robot(
            primary,
            attack_sign,
            clearance_x,
            clearance_y,
            math.atan2(ball[1] - clearance_y, ball[0] - clearance_x),
        )
        _park_robot(support, attack_sign, -0.24, -lane_sign * 0.38, 0.0)
        initial_threat = True
    elif family == "shot":
        ball = (0.28, lane * 0.45)
        _set_ball(state, attack_sign, *ball, speed=speed * 0.18, heading=0.0)
        _place_behind(
            primary,
            attack_sign,
            ball,
            spawn_distance * 0.75,
            heading_error=0.18,
            generator=generator,
        )
        _park_robot(support, attack_sign, -0.20, -lane_sign * 0.38, 0.0)
    else:
        passer = support
        receiver = primary
        support_id = str(passer["id"])
        passer_position = (-0.30, -lane_sign * 0.26)
        receiver_position = (0.08, lane_sign * 0.18)
        target_y = attack_sign * receiver_position[1]
        heading = math.atan2(
            receiver_position[1] - passer_position[1],
            receiver_position[0] - passer_position[0],
        )
        robot_config = config["robot"]
        ball_config = config["ball"]
        contact_offset = (
            math.hypot(
                float(robot_config["length"]),
                float(robot_config["width"]),
            )
            / 2
            + float(ball_config["radius"])
            + 1e-7
        )
        ball_position = (
            passer_position[0] + contact_offset * math.cos(heading),
            passer_position[1] + contact_offset * math.sin(heading),
        )
        launch_heading = heading + lane_sign * _lerp(
            0.45,
            1.00,
            difficulty.ball_angle,
        )
        _set_ball(
            state,
            attack_sign,
            *ball_position,
            speed=speed * 0.65,
            heading=launch_heading,
        )
        _park_robot(passer, attack_sign, *passer_position, heading)
        _park_robot(receiver, attack_sign, *receiver_position, heading + math.pi)

    scenario = Scenario(
        scenario_id=(
            f"m15-{family}-{parameters.controlled_team}-{parameters.seed}-{parameters.digest[:8]}"
        ),
        kind=_legacy_kind(family),
        role="holdout" if parameters.holdout else "frontier",
        state=state,
        immutable=parameters.holdout,
    )
    validate_scenario(scenario, config)
    _validate_not_terminal(state, config)
    state_hash = scenario.digest
    context = SkillContext(
        family=family,
        controlled_team=parameters.controlled_team,
        controlled_robot_id=str(primary["id"]),
        support_robot_id=support_id,
        target_goal_x=target_goal_x,
        own_goal_x=own_goal_x,
        target_y=target_y,
        target_half_width=target_half_width,
        initial_ball_speed=math.hypot(float(state["ball"]["vx"]), float(state["ball"]["vy"])),
        initial_threat=initial_threat,
        horizon=parameters.horizon,
        parameter_hash=parameters.digest,
        state_hash=state_hash,
    )
    return SemanticScenario(scenario, parameters, context)


def _legacy_kind(family: SkillFamily) -> Any:
    return "defense" if family == "save_deflection" else family


def _reset_motion(state: dict[str, Any]) -> None:
    state["ball"].update(vx=0.0, vy=0.0, omega=0.0)
    for robot in state["robots"]:
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0, enabled=True)


def _set_ball(
    state: dict[str, Any],
    attack_sign: float,
    logical_x: float,
    logical_y: float,
    *,
    speed: float,
    heading: float,
) -> None:
    state["ball"].update(
        x=attack_sign * logical_x,
        y=attack_sign * logical_y,
        vx=attack_sign * speed * math.cos(heading),
        vy=attack_sign * speed * math.sin(heading),
        omega=0.0,
    )


def _park_robot(
    robot: dict[str, Any],
    attack_sign: float,
    logical_x: float,
    logical_y: float,
    logical_heading: float,
) -> None:
    robot["pose"].update(
        x=attack_sign * logical_x,
        y=attack_sign * logical_y,
        theta=_angle(logical_heading + (0.0 if attack_sign > 0 else math.pi)),
    )


def _place_behind(
    robot: dict[str, Any],
    attack_sign: float,
    ball: tuple[float, float],
    distance: float,
    *,
    heading_error: float,
    generator: random.Random,
) -> None:
    y_offset = generator.uniform(-0.035, 0.035)
    heading = math.atan2(-y_offset, distance) + generator.uniform(
        -heading_error,
        heading_error,
    )
    _park_robot(
        robot,
        attack_sign,
        ball[0] - distance,
        ball[1] + y_offset,
        heading,
    )


def _park_opponents(
    opponents: list[dict[str, Any]],
    attack_sign: float,
    pressure: float,
    lane_sign: float,
) -> None:
    forward = _lerp(0.56, 0.18, pressure)
    positions = (
        (forward, lane_sign * 0.42),
        (0.52, -lane_sign * 0.42),
        (0.62, 0.0),
    )
    for robot, (x, y) in zip(opponents, positions, strict=True):
        _park_robot(robot, attack_sign, x, y, math.pi)


def _validate_not_terminal(state: dict[str, Any], config: dict[str, Any]) -> None:
    half_length = float(config["field"]["length"]) / 2
    ball_radius = float(config["ball"]["radius"])
    if abs(float(state["ball"]["x"])) + ball_radius >= half_length:
        raise ValueError("semantic scenario starts at a goal boundary")


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _jitter(amount: float, generator: random.Random) -> float:
    return min(1.0, max(0.0, amount + generator.uniform(-0.08, 0.08)))


def _bounded_number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("difficulty state must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("difficulty state must be finite and in [0, 1]")
    return result


def _angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi
