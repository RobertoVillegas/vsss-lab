from __future__ import annotations

from pathlib import Path

import numpy as np
from vsss_env._native import BatchSimulator
from vsss_train.marl import RoleSharedActor, build_team_observation
from vsss_train.roles import DynamicRoleAssigner, assign_roles

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def state() -> np.ndarray:
    return np.asarray(BatchSimulator(CONFIG, STATE, 1).reset()[0], dtype=np.float32)


def place(value: np.ndarray, slot: int, x: float, y: float) -> None:
    base = 10 + slot * 11
    value[base + 2 : base + 4] = (x, y)
    value[base + 5 : base + 7] = (0.0, 0.0)


def test_roles_are_unique_and_visible_without_robot_identity_features() -> None:
    value = state()
    assignment = assign_roles(value, 0)
    assert set(assignment.roles) == {"attacker", "support", "coverage"}
    observation = build_team_observation(value, team=0, role_assignment=assignment)
    assert observation.context.shape == (3, 9)
    assert (observation.context[:, 4:7].sum(dim=-1) == 1).all()
    assert RoleSharedActor(16).deterministic_action(observation).shape == (3, 2)


def test_goalkeeper_responsibility_rotates_when_geometry_changes() -> None:
    value = state()
    value[5:9] = (0.20, 0.0, 0.0, 0.0)
    place(value, 0, -0.66, 0.0)
    place(value, 1, -0.15, 0.22)
    place(value, 2, 0.12, 0.0)
    first = assign_roles(value, 0)
    first_coverage = first.roles.index("coverage")

    # The previous keeper joins the play while another robot recovers behind it.
    place(value, first_coverage, 0.10, 0.0)
    replacement = (first_coverage + 1) % 3
    place(value, replacement, -0.68, 0.02)
    second = assign_roles(value, 0)
    assert second.roles.index("coverage") == replacement
    assert second.roles.index("coverage") != first_coverage


def test_hysteresis_blocks_marginal_role_churn_but_allows_emergency_rotation() -> None:
    value = state()
    assigner = DynamicRoleAssigner(switch_penalty=0.50, emergency_margin=0.20)
    first = assigner.assign(value, 0)
    value[5] += 0.005
    stable = assigner.assign(value, 0)
    assert stable.roles == first.roles

    coverage = stable.roles.index("coverage")
    place(value, coverage, 0.65, 0.0)
    candidate = (coverage + 1) % 3
    place(value, candidate, -0.70, 0.0)
    emergency = assigner.assign(value, 0)
    assert emergency.roles.index("coverage") == candidate


def test_ball_behind_defensive_line_is_uncovered_until_challenged() -> None:
    value = state()
    value[5:9] = (-0.68, 0.05, 0.0, 0.0)
    place(value, 0, -0.46, -0.10)
    place(value, 1, -0.20, 0.18)
    place(value, 2, 0.10, 0.0)

    assert assign_roles(value, 0).uncovered

    place(value, 0, -0.60, 0.05)
    assert not assign_roles(value, 0).uncovered
