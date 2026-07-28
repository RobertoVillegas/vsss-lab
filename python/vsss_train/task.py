"""Native single-robot go-to-target curriculum task."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from vsss_env._native import BatchSimulator

FloatArray = NDArray[np.float32]
ROBOT_OFFSET = 10
ROBOT_WIDTH = 11


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    min_distance: float
    max_distance: float
    max_bearing: float
    heading_noise: float
    position_radius: float
    promotion_threshold: float


STAGES = (
    CurriculumStage("C0", 0.15, 0.30, 0.0, 0.0, 0.0, 0.90),
    CurriculumStage("C1", 0.15, 0.45, 0.35, 0.15, 0.0, 0.90),
    CurriculumStage("C2", 0.20, 0.60, 0.75, 0.35, 0.10, 0.90),
    CurriculumStage("C3", 0.20, 0.75, 1.40, 0.70, 0.20, 0.92),
    CurriculumStage("C4", 0.20, 0.90, math.pi, 1.20, 0.30, 0.94),
    CurriculumStage("C5", 0.20, 1.00, math.pi, math.pi, 0.45, 0.95),
)


class GoToTargetEnv:
    """One controlled differential-drive robot with a stationary ball target."""

    observation_size = 7
    action_size = 2

    def __init__(
        self,
        config_json: str,
        state_json: str,
        *,
        stage: int = 0,
        max_steps: int = 3_000,
        success_radius: float = 0.09,
        action_repeat: int = 4,
    ) -> None:
        if stage not in range(len(STAGES)):
            raise ValueError("stage must be in [0, 5]")
        self._config_json = config_json
        self._template = json.loads(state_json)
        self._native = BatchSimulator(config_json, state_json, 1)
        self.stage = stage
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.action_repeat = action_repeat
        self._steps = 0
        self._distance = 0.0

    def reset(self, seed: int) -> FloatArray:
        rng = np.random.default_rng(seed)
        spec = STAGES[self.stage]
        start_angle = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(0.0, spec.position_radius)
        x = float(radius * math.cos(start_angle))
        y = float(radius * math.sin(start_angle))
        bearing = float(rng.uniform(-spec.max_bearing, spec.max_bearing))
        distance = float(rng.uniform(spec.min_distance, spec.max_distance))
        target_angle = start_angle + bearing
        target_x = float(np.clip(x + distance * math.cos(target_angle), -0.62, 0.62))
        target_y = float(np.clip(y + distance * math.sin(target_angle), -0.52, 0.52))
        desired = math.atan2(target_y - y, target_x - x)
        theta = desired + float(rng.uniform(-spec.heading_noise, spec.heading_noise))

        snapshot = copy.deepcopy(self._template)
        snapshot.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
        snapshot["ball"].update(x=target_x, y=target_y, vx=0.0, vy=0.0, omega=0.0)
        for index, robot in enumerate(snapshot["robots"]):
            robot["pose"].update(
                x=x if index == 0 else (-0.68 if index < 3 else 0.68),
                y=y if index == 0 else (-0.45 + 0.18 * index),
                theta=theta if index == 0 else 0.0,
            )
            robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
            robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
        self._native.restore(0, json.dumps(snapshot, separators=(",", ":")))
        state = self._native.snapshots()
        row = self._native.step(np.zeros((1, 6, 2), dtype=np.float32))[0]
        del state
        self._steps = 0
        self._distance = self._target_distance(row)
        return self._observation(row)

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, object]]:
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, 0] = np.clip(action, -1.0, 1.0)
        state = self._native.step(actions)[0]
        for _ in range(self.action_repeat - 1):
            state = self._native.step(actions)[0]
        self._steps += 1
        distance = self._target_distance(state)
        success = distance <= self.success_radius
        truncated = self._steps >= self.max_steps and not success
        progress = self._distance - distance
        reward = 8.0 * progress - 0.002 + (1.0 if success else 0.0)
        self._distance = distance
        return (
            self._observation(state),
            reward,
            success,
            truncated,
            {
                "success": success,
                "distance": distance,
                "stage": STAGES[self.stage].name,
            },
        )

    @staticmethod
    def _target_distance(state: FloatArray) -> float:
        dx = float(state[5] - state[ROBOT_OFFSET + 2])
        dy = float(state[6] - state[ROBOT_OFFSET + 3])
        return math.hypot(dx, dy)

    @staticmethod
    def _observation(state: FloatArray) -> FloatArray:
        base = ROBOT_OFFSET
        theta = float(state[base + 4])
        return np.asarray(
            [
                (state[5] - state[base + 2]) / 1.5,
                (state[6] - state[base + 3]) / 1.3,
                math.cos(theta),
                math.sin(theta),
                state[base + 5],
                state[base + 6],
                state[base + 7],
            ],
            dtype=np.float32,
        )

    def promote(self, success_rate: float) -> bool:
        """Advance one stage after meeting its deterministic threshold."""
        if success_rate < STAGES[self.stage].promotion_threshold or self.stage == 5:
            return False
        self.stage += 1
        return True
