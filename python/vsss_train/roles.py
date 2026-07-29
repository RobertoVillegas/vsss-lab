"""Dynamic, identity-free tactical responsibility assignment for VSSS teams."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

Role = Literal["attacker", "support", "coverage"]
ROLES: tuple[Role, ...] = ("attacker", "support", "coverage")
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class RoleAssignment:
    roles: tuple[Role, Role, Role]
    changed: tuple[bool, bool, bool]
    cost: float
    uncovered: bool


class DynamicRoleAssigner:
    """Assign responsibilities with hysteresis; no robot identity owns a role."""

    def __init__(self, *, switch_penalty: float = 0.18, emergency_margin: float = 0.20) -> None:
        self.switch_penalty = switch_penalty
        self.emergency_margin = emergency_margin
        self.previous: tuple[Role, Role, Role] | None = None

    def reset(self) -> None:
        self.previous = None

    def assign(self, state: Sequence[float] | Any, team: int) -> RoleAssignment:
        result = assign_roles(
            state,
            team,
            previous=self.previous,
            switch_penalty=self.switch_penalty,
            emergency_margin=self.emergency_margin,
        )
        self.previous = result.roles
        return result


def assign_roles(
    state: Sequence[float] | Any,
    team: int,
    *,
    previous: tuple[Role, Role, Role] | None = None,
    switch_penalty: float = 0.18,
    emergency_margin: float = 0.20,
) -> RoleAssignment:
    """Minimize joint tactical cost over all six role permutations."""
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    attack_sign = 1.0 if team == 0 else -1.0
    robots = [_robot(state, slot) for slot in range(6) if int(_robot(state, slot)[1]) == team]
    if len(robots) != 3:
        raise ValueError("role assignment requires exactly three controlled robots")
    ball_x, ball_y = float(state[5]), float(state[6])
    ball_vx, ball_vy = float(state[7]), float(state[8])
    own_goal_x = -attack_sign * 0.75
    projected_x = max(-0.75, min(0.75, ball_x + ball_vx * 0.35))
    projected_y = max(-0.55, min(0.55, ball_y + ball_vy * 0.35))

    costs: list[dict[Role, float]] = []
    for robot in robots:
        x, y = float(robot[2]), float(robot[3])
        speed = max(0.20, math.hypot(float(robot[5]), float(robot[6])) + 0.20)
        time_to_ball = math.hypot(projected_x - x, projected_y - y) / speed
        goal_side = attack_sign * (ball_x - x)
        attack_angle = abs(math.atan2(projected_y - y, projected_x - x))
        support_x = ball_x - attack_sign * 0.22
        support_y = max(-0.42, min(0.42, ball_y * 0.55))
        coverage_y = max(-0.24, min(0.24, ball_y * 0.65))
        costs.append(
            {
                "attacker": time_to_ball + 0.20 * attack_angle + 0.45 * max(0.0, -goal_side),
                "support": math.hypot(support_x - x, support_y - y)
                + 0.35 * max(0.0, attack_sign * (x - ball_x)),
                "coverage": math.hypot(own_goal_x - x, coverage_y - y)
                + 1.1 * max(0.0, attack_sign * (x - ball_x)),
            }
        )

    active = [bool(float(robot[10])) for robot in robots]
    active_count = sum(active)
    active_roles = set(ROLES[:active_count])

    def raw_cost(roles: tuple[Role, Role, Role]) -> float:
        return sum(
            costs[index][role]
            + (
                1_000.0
                if (active[index] and role not in active_roles)
                or (not active[index] and role in active_roles)
                else 0.0
            )
            for index, role in enumerate(roles)
        )

    candidates = []
    for roles in itertools.permutations(ROLES):
        typed_roles = cast(tuple[Role, Role, Role], roles)
        switches = (
            sum(role != previous[index] for index, role in enumerate(typed_roles))
            if previous is not None
            else 0
        )
        candidates.append((raw_cost(typed_roles) + switch_penalty * switches, typed_roles))
    selected_cost, selected = min(candidates, key=lambda item: (item[0], item[1]))
    if previous is not None:
        raw_selected = min(
            (
                (raw_cost(typed), typed)
                for roles in itertools.permutations(ROLES)
                if (typed := cast(tuple[Role, Role, Role], roles))
            ),
            key=lambda item: (item[0], item[1]),
        )
        if raw_cost(previous) - raw_selected[0] >= emergency_margin:
            selected_cost, selected = raw_selected

    coverage = robots[selected.index("coverage")]
    uncovered = active_count == 3 and (
        math.hypot(float(coverage[2]) - own_goal_x, float(coverage[3])) > 0.62
        and attack_sign * ball_x < 0.0
    )
    changed = cast(
        tuple[bool, bool, bool],
        tuple(
            previous is not None and role != previous[index] for index, role in enumerate(selected)
        ),
    )
    return RoleAssignment(selected, changed, float(selected_cost), uncovered)


def role_features(assignment: RoleAssignment) -> FloatArray:
    rows = []
    for role, changed in zip(assignment.roles, assignment.changed, strict=True):
        rows.append(
            [
                float(role == "attacker"),
                float(role == "support"),
                float(role == "coverage"),
                float(changed),
                float(assignment.uncovered),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _robot(state: Sequence[float] | Any, slot: int) -> Sequence[float]:
    start = 10 + slot * 11
    return state[start : start + 11]
