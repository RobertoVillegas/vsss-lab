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
    "rotation_recovery",
]
ControlledTeam = Literal["blue", "yellow"]
Roster = Literal["1v0", "1v1", "2v1", "2v2", "3v2", "3v3"]

GENERATOR_REVISION = "m24.3-ladders"
DIFFICULTY_AXES = (
    "ball_speed",
    "ball_angle",
    "spawn_distance",
    "target_width",
    "opponent_pressure",
)
# Which axes a family actually responds to. Measured with tools/audit_skill_difficulty:
# an axis absent here was inert for that family under both a scripted and a trained probe,
# so advancing it only made the curriculum believe it was exploring a dimension that does
# not exist. Declaring the map keeps the difficulty space honest.
FAMILY_AXES: dict[str, tuple[str, ...]] = {
    "approach": ("spawn_distance",),
    "interception": ("ball_speed",),
    "save_deflection": ("ball_speed",),
    "clearance": ("spawn_distance", "ball_speed"),
    "shot": ("spawn_distance", "ball_speed"),
    "pass_receive": ("ball_angle", "target_width"),
    "rotation_recovery": ("spawn_distance",),
}
SKILL_FAMILIES: tuple[SkillFamily, ...] = (
    "approach",
    "interception",
    "save_deflection",
    "clearance",
    "shot",
    "pass_receive",
    "rotation_recovery",
)

PHASES: tuple[tuple[str, tuple[SkillFamily, ...], dict[SkillFamily, float], float], ...] = (
    (
        "foundation",
        ("approach", "shot", "interception"),
        {"approach": 0.75, "shot": 0.70, "interception": 0.35},
        0.15,
    ),
    (
        "defense",
        ("interception", "save_deflection", "clearance"),
        {"interception": 0.35, "save_deflection": 0.35, "clearance": 0.35},
        0.25,
    ),
    (
        "cooperation",
        ("pass_receive",),
        {"pass_receive": 0.20, "clearance": 0.30},
        0.20,
    ),
    (
        "rotation",
        ("rotation_recovery",),
        {"rotation_recovery": 0.15, "pass_receive": 0.15, "save_deflection": 0.30},
        0.25,
    ),
    ("integration", SKILL_FAMILIES, {}, 1.0),
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
    roster: Roster = "3v3"
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
            "rotation_recovery",
        ):
            raise ValueError(f"unsupported skill family: {self.family}")
        if self.controlled_team not in ("blue", "yellow"):
            raise ValueError("controlled_team must be blue or yellow")
        if self.roster not in ("1v0", "1v1", "2v1", "2v2", "3v2", "3v3"):
            raise ValueError(f"unsupported semantic roster: {self.roster}")
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
    relay_robot_id: str | None
    target_goal_x: float
    own_goal_x: float
    target_y: float
    target_half_width: float
    initial_ball_speed: float
    initial_threat: bool
    horizon: int
    parameter_hash: str
    state_hash: str
    roster: Roster = "3v3"


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
        phased: bool = False,
        phase_patience: int = 2,
        phase_rehearsal_fraction: float = 0.20,
        phase_full_match_floor: float = 0.0,
    ) -> None:
        if not 0.0 <= full_match_fraction <= 1.0:
            raise ValueError("full_match_fraction must be in [0, 1]")
        if window < 4:
            raise ValueError("curriculum window must be at least four")
        if phase_patience <= 0:
            raise ValueError("phase patience must be positive")
        if not 0.0 <= phase_rehearsal_fraction <= 1.0:
            raise ValueError("phase rehearsal fraction must be in [0, 1]")
        if not 0.0 <= phase_full_match_floor <= 1.0:
            raise ValueError("phase full-match floor must be in [0, 1]")
        self.base_state = copy.deepcopy(base_state)
        self.config = config
        self.seed = seed
        self.full_match_fraction = full_match_fraction
        self.window = window
        self.phased = phased
        self.phase_patience = phase_patience
        self.phase_rehearsal_fraction = phase_rehearsal_fraction
        self.phase_full_match_floor = phase_full_match_floor
        self.phase_index = 0
        self.phase_gate_streak = 0
        self.levels = {
            family: {axis: 0.05 for axis in DIFFICULTY_AXES} for family in SKILL_FAMILIES
        }
        self.outcomes: dict[SkillFamily, deque[float]] = defaultdict(
            lambda: deque(maxlen=window * 2)
        )
        self.failures: dict[str, SkillScenarioParameters] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.family_counts: dict[SkillFamily, int] = defaultdict(int)
        self.roster_counts: dict[Roster, int] = defaultdict(int)
        self.updates: dict[SkillFamily, int] = defaultdict(int)

    def select_training(self, index: int) -> SemanticSelection:
        generator = random.Random(self.seed + index * 104_729)
        total_allocated = sum(self.counts.values())
        required_full_matches = math.ceil((total_allocated + 1) * self.current_full_match_fraction)
        if self.counts["full_match"] < required_full_matches:
            self.counts["full_match"] += 1
            return SemanticSelection(None, "full_match")
        eligible_failures = tuple(
            key
            for key, parameters in sorted(self.failures.items())
            if parameters.family in self.training_families
        )
        if eligible_failures and generator.random() < 0.15:
            key = eligible_failures[generator.randrange(len(eligible_failures))]
            parameters = self.failures[key]
            source: Literal["failure", "routine", "frontier"] = "failure"
        else:
            family = self._select_family(generator)
            levels = self.levels[family]
            source = "routine" if generator.random() < 0.30 else "frontier"
            amounts = {
                axis: max(0.05, value - 0.15) if source == "routine" else value
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
                roster=_roster_for(family, source),
            )
        self.counts[source] += 1
        self.family_counts[parameters.family] += 1
        self.roster_counts[parameters.roster] += 1
        return SemanticSelection(
            compile_skill_scenario(parameters, self.base_state, self.config),
            source,
        )

    def _select_family(self, generator: random.Random) -> SkillFamily:
        """Bias practice toward weak skills while retaining a rehearsal floor."""
        families = self.training_families
        if self.phased and self.phase_index > 0 and self.phase_index < len(PHASES) - 1:
            previous = tuple(
                dict.fromkeys(
                    family
                    for _, phase_families, _, _ in PHASES[: self.phase_index]
                    for family in phase_families
                )
            )
            if previous and generator.random() < self.phase_rehearsal_fraction:
                return previous[generator.randrange(len(previous))]
        if generator.random() < 0.20:
            return families[generator.randrange(len(families))]
        weights = []
        for family in families:
            history = self.outcomes[family]
            success_rate = sum(history) / len(history) if history else 0.5
            # Every family remains sampleable; weak skills receive up to 26x
            # the weight of a mastered one.
            weights.append(0.04 + (1.0 - success_rate) ** 2)
        return generator.choices(families, weights=weights, k=1)[0]

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase_index][0] if self.phased else "adaptive_all"

    @property
    def training_families(self) -> tuple[SkillFamily, ...]:
        return PHASES[self.phase_index][1] if self.phased else SKILL_FAMILIES

    @property
    def current_full_match_fraction(self) -> float:
        if not self.phased:
            return self.full_match_fraction
        phase_fraction = PHASES[self.phase_index][3]
        selected = self.full_match_fraction if phase_fraction == 1.0 else phase_fraction
        return max(selected, self.phase_full_match_floor)

    def observe_holdout_rates(
        self,
        rates: dict[str, float],
        *,
        behavior_eligible: bool = True,
    ) -> bool:
        """Advance after consecutive paired holdouts clear the current phase."""
        if not self.phased or self.phase_index >= len(PHASES) - 1:
            return False
        gates = PHASES[self.phase_index][2]
        passed = behavior_eligible and all(
            rates.get(family, 0.0) >= floor for family, floor in gates.items()
        )
        self.phase_gate_streak = self.phase_gate_streak + 1 if passed else 0
        if self.phase_gate_streak < self.phase_patience:
            return False
        self.phase_index += 1
        self.phase_gate_streak = 0
        return True

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
        # Only advance an axis this family responds to, so difficulty tracks the demand
        # instead of drifting into a dimension that changes nothing.
        active = FAMILY_AXES.get(family, DIFFICULTY_AXES)
        axis = active[self.updates[family] % len(active)]
        if current >= 0.70 and learning_progress >= -0.05:
            self.levels[family][axis] = min(1.0, self.levels[family][axis] + 0.05)
        elif current < 0.35 and learning_progress <= 0.0:
            self.levels[family][axis] = max(0.05, self.levels[family][axis] - 0.025)
        self.updates[family] += 1

    def holdouts(
        self,
        *,
        seeds: tuple[int, ...] = (10_007, 10_009, 10_037, 10_039, 10_061),
        bands: tuple[float, ...] = (0.10, 0.25, 0.40, 0.65),
    ) -> tuple[SemanticScenario, ...]:
        """Build immutable paired-color ladders outside the training allocator."""
        scenarios = []
        for family in SKILL_FAMILIES:
            for team in ("blue", "yellow"):
                for band in bands:
                    for seed in seeds:
                        scenarios.append(
                            compile_skill_scenario(
                                SkillScenarioParameters(
                                    schema_version=1,
                                    family=family,
                                    seed=seed + round(band * 10_000),
                                    controlled_team=team,
                                    difficulty=SkillDifficulty(band, band, band, band, band),
                                    roster=_roster_for(family, "frontier"),
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
            "allocation_by_family": dict(self.family_counts),
            "allocation_by_roster": dict(self.roster_counts),
            "failure_count": len(self.failures),
            "observed_full_match_fraction": observed_full_match_fraction,
            "allocation_valid": (
                not total
                or observed_full_match_fraction + 1e-12 >= self.current_full_match_fraction
            ),
            "phase": self.phase_name,
            "phase_index": self.phase_index,
            "phase_gate_streak": self.phase_gate_streak,
            "training_families": self.training_families,
            "success_rate": {
                family: sum(values) / len(values) if values else None
                for family in SKILL_FAMILIES
                if (values := self.outcomes[family])
            },
        }
        if reset:
            self.counts.clear()
            self.family_counts.clear()
            self.roster_counts.clear()
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
            "updates": {family: self.updates[family] for family in SKILL_FAMILIES},
            "phase_index": self.phase_index,
            "phase_gate_streak": self.phase_gate_streak,
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
        updates = state.get("updates", {})
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
        self.updates.clear()
        raw_phase_index = state.get("phase_index", 0)
        raw_phase_streak = state.get("phase_gate_streak", 0)
        if not isinstance(raw_phase_index, int) or not isinstance(raw_phase_streak, int):
            raise ValueError("invalid semantic curriculum phase state")
        self.phase_index = raw_phase_index
        self.phase_gate_streak = raw_phase_streak
        if not 0 <= self.phase_index < len(PHASES):
            raise ValueError("invalid semantic curriculum phase")
        if isinstance(updates, dict):
            self.updates.update({family: int(updates.get(family, 0)) for family in SKILL_FAMILIES})
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
    relay_id: str | None = None
    if family == "approach":
        ball = (-0.05, lane)
        _set_ball(state, attack_sign, *ball, speed=0.0, heading=0.0)
        # A fixed 0.13-0.34 m reach was solved at every level by every controller, so the
        # axis carried no information. Reaching the ball is the demand: it now spans an
        # easy step to most of a half.
        _place_behind(
            primary,
            attack_sign,
            ball,
            _lerp(0.12, 0.62, difficulty.spawn_distance),
            heading_error=_lerp(0.05, 1.10, difficulty.ball_angle),
            generator=generator,
        )
        _park_robot(support, attack_sign, -0.52, -lane_sign * 0.32, 0.0)
    elif family in ("interception", "save_deflection"):
        ball = (-0.28 if family == "interception" else -0.46, lane)
        if family == "save_deflection":
            # A fixed angular deflection eventually points the ball away from the goal, and
            # the validator then rejects the scenario for not being a save at all. Aiming at
            # a point inside the goal mouth keeps every band goal-bound by construction and
            # still lets difficulty walk the shot toward the post.
            goal_half_width = float(config["field"]["goal_width"]) / 2
            aim_y = lane_sign * _lerp(0.0, 0.70, difficulty.ball_angle) * goal_half_width
            heading = math.atan2(aim_y - lane, own_goal_x * attack_sign - ball[0])
        else:
            heading = math.pi + lane_sign * _lerp(0.02, 0.32, difficulty.ball_angle)
        # The shared speed ramp reached 1.05 m/s, which no controller can intercept from
        # behind, so every band above the middle was dead. This one spans reachable to
        # marginal instead of reachable to impossible.
        _set_ball(
            state,
            attack_sign,
            *ball,
            speed=_lerp(0.10, 0.58, difficulty.ball_speed),
            heading=heading,
        )
        intercept_x = ball[0] - _lerp(0.10, 0.24, difficulty.spawn_distance)
        # save_deflection starts deeper, so without this the hardest band parks a robot
        # outside the field and the scenario cannot be compiled at all.
        intercept_x = max(-0.69, intercept_x)
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
        emergency = generator.random() < 0.55
        # A clearance is scored on the ball leaving the defensive third, so the ball's
        # depth *is* the demand. It used to be a coin flip between two deep positions,
        # which left the easy end of every difficulty axis asking for the same 0.38 m of
        # displacement as the hard end. Lowering difficulty could then never make the
        # drill learnable, which is the one thing the difficulty curriculum exists to do.
        ball = (
            -_lerp(0.20, 0.62, difficulty.spawn_distance),
            lane * (0.35 if emergency else 0.55),
        )
        heading = math.pi + lane_sign * _lerp(0.0, 0.22, difficulty.ball_angle)
        _set_ball(state, attack_sign, *ball, speed=speed * 0.28, heading=heading)
        clearance_x = ball[0] - _lerp(0.08, 0.14, difficulty.spawn_distance)
        clearance_x = max(-0.69, clearance_x)
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
    elif family == "rotation_recovery":
        ball = (-0.08, lane * 0.45)
        _set_ball(state, attack_sign, *ball, speed=speed * 0.35, heading=math.pi)
        _park_robot(
            primary,
            attack_sign,
            ball[0] - spawn_distance * 0.65,
            ball[1] + lane_sign * 0.10,
            -lane_sign * 0.25,
        )
        # The previous attacker begins beyond the play and must rotate through
        # coverage. The former coverage player advances into support behind the
        # incoming challenger: all three responsibilities must turn over.
        # The rotation this scores is the support travelling back into coverage, and that
        # journey was a fixed 0.7 m at every difficulty. It is now the axis.
        # Success needs a three-way role turnover, so the easy end starts the support
        # already deeper than the relay: the permutation nearly holds and only the touch is
        # missing. The hard end strands it beyond the play, which is the full rotation.
        _park_robot(
            support,
            attack_sign,
            _lerp(-0.52, 0.40, difficulty.spawn_distance),
            -lane_sign * 0.24,
            math.pi,
        )
        _park_robot(reserve, attack_sign, -0.38, lane_sign * 0.30, 0.0)
        support_id = str(support["id"])
        relay_id = str(reserve["id"])
        initial_threat = True
    elif family == "shot":
        loose_finish = generator.random() < 0.50
        # Both finishes sat within a third of a metre of the goal, so difficulty cost
        # nothing. Range is the demand: a tap-in at the easy end, a long strike at the hard.
        # Easy is a short range, not point blank: the defenders are parked in front of the
        # goal, and a ball placed on top of them cannot be compiled at all.
        ball = (
            _lerp(0.42, 0.02, difficulty.spawn_distance),
            lane * (0.30 if loose_finish else 0.45),
        )
        # The loose ball used to roll toward the goal, so a faster ball arrived closer to
        # scoring on its own and the speed axis made the family easier. It now drifts across
        # the shooting line, which is what makes a finish harder.
        _set_ball(
            state,
            attack_sign,
            *ball,
            speed=_lerp(0.0, 0.42, difficulty.ball_speed),
            heading=lane_sign * math.pi / 2,
        )
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
        # The launch was deflected 0.45 to 1.00 rad off the receiver even at difficulty
        # zero, so the ball never arrived and no controller ever completed a pass. The
        # deviation now starts at a true pass and grows into a mis-hit.
        launch_heading = heading + lane_sign * _lerp(
            0.0,
            0.55,
            difficulty.ball_angle,
        )
        # A pass that never arrives cannot be received at any difficulty, and the shared
        # ramp started at 0.08 m/s. The floor now always reaches the receiver, and the
        # demand lives in the launch deviation and the reception speed limit.
        _set_ball(
            state,
            attack_sign,
            *ball_position,
            speed=_lerp(0.62, 1.05, difficulty.ball_speed),
            heading=launch_heading,
        )
        _park_robot(passer, attack_sign, *passer_position, heading)
        _park_robot(receiver, attack_sign, *receiver_position, heading + math.pi)

    _apply_roster(
        controlled,
        opponents,
        primary=primary,
        support=support,
        reserve=reserve,
        roster=parameters.roster,
    )
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
    _validate_reachable(state, primary, parameters, config)
    if family in ("interception", "save_deflection"):
        initial_threat = _trajectory_intersects_goal(state, own_goal_x, config)
        if not initial_threat:
            raise ValueError(f"{family} must begin with a goal-bound trajectory")
    elif family == "clearance":
        initial_threat = _trajectory_intersects_goal(state, own_goal_x, config)
    state_hash = scenario.digest
    context = SkillContext(
        family=family,
        controlled_team=parameters.controlled_team,
        controlled_robot_id=str(primary["id"]),
        support_robot_id=support_id,
        relay_robot_id=relay_id,
        target_goal_x=target_goal_x,
        own_goal_x=own_goal_x,
        target_y=target_y,
        target_half_width=target_half_width,
        initial_ball_speed=math.hypot(float(state["ball"]["vx"]), float(state["ball"]["vy"])),
        initial_threat=initial_threat,
        horizon=parameters.horizon,
        parameter_hash=parameters.digest,
        state_hash=state_hash,
        roster=parameters.roster,
    )
    return SemanticScenario(scenario, parameters, context)


def _legacy_kind(family: SkillFamily) -> Any:
    return "defense" if family in ("save_deflection", "rotation_recovery") else family


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


def _roster_for(
    family: SkillFamily,
    source: Literal["routine", "frontier", "failure"],
) -> Roster:
    """Use the smallest roster expressing a skill, then add realistic pressure."""
    frontier = source != "routine"
    if family in ("approach", "shot"):
        return "1v1" if frontier else "1v0"
    if family in ("interception", "save_deflection"):
        return "2v1" if frontier else "1v1"
    if family == "clearance":
        return "2v2" if frontier else "1v1"
    if family == "pass_receive":
        return "2v2" if frontier else "2v1"
    return "3v3" if frontier else "3v2"


def _apply_roster(
    controlled: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
    *,
    primary: dict[str, Any],
    support: dict[str, Any],
    reserve: dict[str, Any],
    roster: Roster,
) -> None:
    controlled_count, opponent_count = (int(value) for value in roster.split("v", 1))
    controlled_priority = (primary, support, reserve)
    active_controlled = {str(robot["id"]) for robot in controlled_priority[:controlled_count]}
    active_opponents = {str(robot["id"]) for robot in opponents[:opponent_count]}
    for robot in controlled:
        robot["enabled"] = str(robot["id"]) in active_controlled
    for robot in opponents:
        robot["enabled"] = str(robot["id"]) in active_opponents


def _validate_not_terminal(state: dict[str, Any], config: dict[str, Any]) -> None:
    half_length = float(config["field"]["length"]) / 2
    ball_radius = float(config["ball"]["radius"])
    if abs(float(state["ball"]["x"])) + ball_radius >= half_length:
        raise ValueError("semantic scenario starts at a goal boundary")


def _validate_reachable(
    state: dict[str, Any],
    primary: dict[str, Any],
    parameters: SkillScenarioParameters,
    config: dict[str, Any],
) -> None:
    robot = config["robot"]
    ball = config["ball"]
    wheel = config["wheel"]
    contact_distance = math.hypot(float(robot["length"]), float(robot["width"])) / 2 + float(
        ball["radius"]
    )
    maximum_speed = float(config["max_wheel_speed"]) * float(wheel["radius"])
    duration = parameters.horizon * float(config["control_period"])
    pose = primary["pose"]
    separation = math.dist(
        (float(pose["x"]), float(pose["y"])),
        (float(state["ball"]["x"]), float(state["ball"]["y"])),
    )
    ball_travel = (
        math.hypot(
            float(state["ball"]["vx"]),
            float(state["ball"]["vy"]),
        )
        * duration
    )
    if separation > contact_distance + (maximum_speed * duration) + ball_travel:
        raise ValueError("controlled robot cannot reach the ball within the horizon")


def _trajectory_intersects_goal(
    state: dict[str, Any],
    goal_x: float,
    config: dict[str, Any],
) -> bool:
    ball = state["ball"]
    velocity_x = float(ball["vx"])
    if abs(velocity_x) < 1e-9 or (goal_x - float(ball["x"])) * velocity_x <= 0:
        return False
    crossing_time = (goal_x - float(ball["x"])) / velocity_x
    crossing_y = float(ball["y"]) + float(ball["vy"]) * crossing_time
    aperture = float(config["field"]["goal_width"]) / 2 - float(config["ball"]["radius"])
    return crossing_time >= 0.0 and abs(crossing_y) <= aperture


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
