"""Typed scenario suites and deterministic learning-progress allocation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ScenarioKind = Literal[
    "kickoff",
    "approach",
    "interception",
    "clearance",
    "defense",
    "pass_receive",
    "shot",
    "congestion_recovery",
    "mixed",
]
SuiteRole = Literal["routine", "frontier", "failure", "holdout"]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    kind: ScenarioKind
    role: SuiteRole
    state: dict[str, object]
    immutable: bool = False
    parent: str | None = None

    @property
    def digest(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class BucketProgress:
    kind: ScenarioKind
    recent_success: float
    previous_success: float
    samples: int

    @property
    def learning_progress(self) -> float:
        return abs(self.recent_success - self.previous_success)


@dataclass(frozen=True)
class Allocation:
    routine: int
    frontier: int
    failure: int
    holdout: int
    frontier_kinds: tuple[ScenarioKind, ...]


@dataclass(frozen=True)
class CurriculumSelection:
    scenario: Scenario
    source: Literal["routine", "frontier", "failure"]


class ScenarioCurriculum:
    """Stateful deterministic teacher with deduplicated failure rehearsal."""

    def __init__(
        self,
        scenarios: tuple[Scenario, ...],
        config: dict[str, object],
        *,
        seed: int,
        history: int = 32,
    ) -> None:
        if not scenarios:
            raise ValueError("curriculum requires scenarios")
        for scenario in scenarios:
            validate_scenario(scenario, config)
        if not any(scenario.role == "routine" for scenario in scenarios):
            raise ValueError("curriculum requires routine scenarios")
        if not any(scenario.role == "frontier" for scenario in scenarios):
            raise ValueError("curriculum requires frontier scenarios")
        if not any(scenario.role == "holdout" and scenario.immutable for scenario in scenarios):
            raise ValueError("curriculum requires immutable holdouts")
        self.scenarios = scenarios
        self.config = config
        self.seed = seed
        self.history = history
        self._outcomes: dict[ScenarioKind, deque[float]] = defaultdict(
            lambda: deque(maxlen=2 * history)
        )
        self._failures: dict[str, Scenario] = {}
        self._descriptor_failures: set[str] = set()
        self._counts: Counter[str] = Counter()

    @property
    def holdouts(self) -> tuple[Scenario, ...]:
        return tuple(scenario for scenario in self.scenarios if scenario.role == "holdout")

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def select_training(self, index: int) -> CurriculumSelection:
        """Select a 20/50/20 training mixture; holdouts never enter optimization."""
        slot = index % 90
        source: Literal["routine", "frontier", "failure"]
        if slot < 20:
            source = "routine"
        elif slot < 70:
            source = "frontier"
        else:
            source = "failure"
        candidates = self._candidates(source)
        if not candidates:
            source = "frontier"
            candidates = self._candidates(source)
        ranked = sorted(candidates, key=lambda item: item.scenario_id)
        generator = random.Random(self.seed + index)
        scenario = ranked[generator.randrange(len(ranked))]
        self._counts[source] += 1
        return CurriculumSelection(scenario, source)

    def record(self, scenario: Scenario, *, success: bool) -> None:
        self._outcomes[scenario.kind].append(float(success))
        if success:
            self._failures.pop(scenario.digest, None)
        elif not scenario.immutable:
            self._failures.setdefault(scenario.digest, scenario)

    def ingest_failure_descriptor(self, *, kind: str, digest: str) -> bool:
        """Route a deduplicated replay failure to a matching rehearsal scenario."""
        if digest in self._descriptor_failures:
            return False
        candidates = sorted(
            (
                scenario
                for scenario in self.scenarios
                if scenario.kind == kind and scenario.role != "holdout"
            ),
            key=lambda scenario: scenario.scenario_id,
        )
        if not candidates:
            return False
        self._descriptor_failures.add(digest)
        self._failures.setdefault(digest, candidates[0])
        return True

    def telemetry(self, *, reset: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "mixture": {
                "routine": self._counts["routine"],
                "frontier": self._counts["frontier"],
                "failure": self._counts["failure"],
                "holdout": len(self.holdouts),
            },
            "deduplicated_failures": len(self._failures),
            "replay_failure_descriptors": len(self._descriptor_failures),
            "learning_progress": {
                kind: _window_progress(values, self.history)
                for kind, values in sorted(self._outcomes.items())
            },
        }
        if reset:
            self._counts.clear()
        return result

    def _candidates(self, source: str) -> tuple[Scenario, ...]:
        if source == "failure":
            return tuple(self._failures.values())
        role = "routine" if source == "routine" else "frontier"
        candidates = tuple(scenario for scenario in self.scenarios if scenario.role == role)
        if source != "frontier":
            return candidates
        progress = {
            kind: _window_progress(values, self.history) for kind, values in self._outcomes.items()
        }
        if not progress:
            return candidates
        maximum = max((progress.get(item.kind, 0.0) for item in candidates), default=0.0)
        selected = tuple(
            item for item in candidates if progress.get(item.kind, 0.0) >= maximum - 1e-12
        )
        return selected or candidates


def validate_scenario(
    scenario: Scenario,
    config: dict[str, object],
) -> None:
    """Reject non-finite, out-of-field, and overlapping canonical states."""
    field = _mapping(config["field"])
    robot_config = _mapping(config["robot"])
    ball_config = _mapping(config["ball"])
    half_length = _number(field["length"]) / 2
    half_width = _number(field["width"]) / 2
    robot_radius = math.hypot(_number(robot_config["length"]), _number(robot_config["width"])) / 2
    ball_radius = _number(ball_config["radius"])
    state = scenario.state
    ball = _mapping(state["ball"])
    ball_position = (_number(ball["x"]), _number(ball["y"]))
    _finite(ball_position, "ball")
    if abs(ball_position[0]) > half_length or abs(ball_position[1]) > half_width:
        raise ValueError("ball outside playable field")
    robots = state["robots"]
    if not isinstance(robots, list) or len(robots) != 6:
        raise ValueError("scenario requires exactly six robots")
    positions: list[tuple[str, tuple[float, float]]] = []
    for raw in robots:
        robot = _mapping(raw)
        pose = _mapping(robot["pose"])
        position = (_number(pose["x"]), _number(pose["y"]))
        _finite((*position, _number(pose["theta"])), f"robot {robot['id']}")
        if abs(position[0]) + robot_radius > half_length:
            raise ValueError(f"robot {robot['id']} outside field length")
        if abs(position[1]) + robot_radius > half_width:
            raise ValueError(f"robot {robot['id']} outside field width")
        if math.dist(position, ball_position) < robot_radius + ball_radius:
            raise ValueError(f"robot {robot['id']} overlaps ball")
        positions.append((str(robot["id"]), position))
    for index, (first_id, first) in enumerate(positions):
        for second_id, second in positions[index + 1 :]:
            if math.dist(first, second) < 2 * robot_radius:
                raise ValueError(f"robots {first_id} and {second_id} overlap")


def allocate_scenarios(
    progress: tuple[BucketProgress, ...],
    *,
    batch_size: int,
) -> Allocation:
    """Allocate 20/50/20/10 while selecting learnable, progressing buckets."""
    if batch_size < 10:
        raise ValueError("curriculum batch_size must be at least 10")
    if not progress:
        raise ValueError("curriculum requires bucket progress")
    holdout = max(1, round(batch_size * 0.10))
    failure = max(1, round(batch_size * 0.20))
    routine = max(1, round(batch_size * 0.20))
    frontier = batch_size - holdout - failure - routine
    eligible = [
        bucket
        for bucket in progress
        if bucket.samples >= 2 and 0.15 <= bucket.recent_success <= 0.85
    ]
    if not eligible:
        eligible = list(progress)
    ranked = sorted(
        eligible,
        key=lambda bucket: (-bucket.learning_progress, bucket.kind),
    )
    return Allocation(
        routine=routine,
        frontier=frontier,
        failure=failure,
        holdout=holdout,
        frontier_kinds=tuple(bucket.kind for bucket in ranked),
    )


def mutate_scenario(
    parent: Scenario,
    config: dict[str, object],
    *,
    seed: int,
    position_sigma: float = 0.03,
    attempts: int = 32,
) -> Scenario:
    """Bounded mutation with rejection; immutable holdouts cannot be mutated."""
    if parent.immutable or parent.role == "holdout":
        raise ValueError("immutable holdout scenarios cannot be mutated")
    generator = random.Random(seed)
    for attempt in range(attempts):
        state = json.loads(json.dumps(parent.state))
        for robot in state["robots"]:
            robot["pose"]["x"] += generator.gauss(0.0, position_sigma)
            robot["pose"]["y"] += generator.gauss(0.0, position_sigma)
        state["ball"]["x"] += generator.gauss(0.0, position_sigma)
        state["ball"]["y"] += generator.gauss(0.0, position_sigma)
        candidate = Scenario(
            scenario_id=f"{parent.scenario_id}-s{seed}-a{attempt}",
            kind=parent.kind,
            role=parent.role,
            state=state,
            parent=parent.scenario_id,
        )
        try:
            validate_scenario(candidate, config)
        except ValueError:
            continue
        return candidate
    raise ValueError("unable to produce a valid bounded scenario mutation")


def write_suite(path: Path, scenarios: tuple[Scenario, ...]) -> None:
    """Persist a deduplicated, role-explicit suite."""
    digests = [scenario.digest for scenario in scenarios]
    if len(digests) != len(set(digests)):
        raise ValueError("scenario suite contains duplicate canonical states")
    payload = {
        "schema_version": 1,
        "scenarios": [{**asdict(scenario), "digest": scenario.digest} for scenario in scenarios],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_suite(path: Path, config: dict[str, object]) -> tuple[Scenario, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported scenario suite schema")
    base_state: dict[str, object] | None = None
    if "base_state" in document:
        base_path = (path.parent / str(document["base_state"])).resolve()
        base_state = json.loads(base_path.read_text(encoding="utf-8"))
    scenarios = tuple(
        Scenario(
            scenario_id=str(item["scenario_id"]),
            kind=item["kind"],
            role=item["role"],
            state=_scenario_state(item, base_state),
            immutable=bool(item.get("immutable", False)),
            parent=item.get("parent"),
        )
        for item in document["scenarios"]
    )
    for scenario in scenarios:
        validate_scenario(scenario, config)
    return scenarios


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("scenario field must be a mapping")
    return value


def _finite(values: tuple[float, ...], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite value")


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("scenario numeric field must be a number")
    return float(value)


def _scenario_state(
    item: dict[str, object],
    base_state: dict[str, object] | None,
) -> dict[str, object]:
    if "state" in item:
        return _mapping(item["state"])
    if base_state is None:
        raise ValueError("scenario requires state or suite base_state")
    state = copy.deepcopy(base_state)
    patch = _mapping(item.get("patch", {}))
    ball_patch = _mapping(patch.get("ball", {}))
    _mapping(state["ball"]).update(ball_patch)
    return state


def _window_progress(values: deque[float], history: int) -> float:
    if len(values) < 2:
        return 0.0
    recent = tuple(values)[-history:]
    previous = tuple(values)[-2 * history : -history]
    if not previous:
        midpoint = max(1, len(recent) // 2)
        previous, recent = recent[:midpoint], recent[midpoint:]
    if not recent or not previous:
        return 0.0
    return abs(sum(recent) / len(recent) - sum(previous) / len(previous))
