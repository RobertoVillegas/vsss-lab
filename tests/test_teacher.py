from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from vsss_train.teacher import SkillResult, plan_atomic_skill, write_demonstrations


def target_rollout(actions: NDArray[np.float32]) -> SkillResult:
    score = -float(np.square(actions - 0.5).mean())
    return SkillResult(
        score=score,
        success=score > -0.25,
        physically_valid=True,
        terminal_reason="interception",
    )


def test_planner_is_reproducible_and_verifies_winner(tmp_path: Path) -> None:
    first = plan_atomic_skill(
        target_rollout, skill="interception", seed=14, population=32, elites=4
    )
    second = plan_atomic_skill(
        target_rollout, skill="interception", seed=14, population=32, elites=4
    )
    assert first == second
    output = tmp_path / "demonstrations.json"
    write_demonstrations(output, (first,))
    assert '"verified_exact_simulator": true' in output.read_text()


def test_planner_rejects_invalid_physics_even_with_high_score() -> None:
    def exploit(actions: NDArray[np.float32]) -> SkillResult:
        return SkillResult(999.0, True, False, "invalid")

    with pytest.raises(ValueError, match="no physically valid"):
        plan_atomic_skill(exploit, skill="shot", seed=1, population=8, elites=2)


def test_planner_rejects_unmet_skill_predicate() -> None:
    def failure(actions: NDArray[np.float32]) -> SkillResult:
        return SkillResult(1.0, False, True, "timeout")

    with pytest.raises(ValueError, match="skill predicate"):
        plan_atomic_skill(failure, skill="clearance", seed=1, population=8, elites=2)
