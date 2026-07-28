"""PettingZoo and Gymnasium adapters around the native batch hot loop."""

from collections.abc import Mapping
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from vsss_env._native import BatchSimulator
from vsss_env.components import (
    ActionAdapter,
    GlobalObservation,
    GoalTermination,
    ObservationBuilder,
    RewardTerm,
    TerminationCondition,
    WheelVelocityAction,
    ZeroReward,
)

FloatArray = NDArray[np.float32]
AGENTS = tuple(f"{team}_{index}" for team in ("blue", "yellow") for index in range(3))
STATE_WIDTH = BatchSimulator.state_width()
FLOAT_LIMIT = np.finfo(np.float32).max


class ParallelVSSSEnv(ParallelEnv[str, FloatArray, FloatArray]):
    """Six-agent simultaneous VSSS environment."""

    metadata: ClassVar[dict[str, object]] = {
        "name": "vsss_lab_parallel_v0",
        "render_modes": [],
    }

    def __init__(
        self,
        config_json: str,
        state_json: str,
        observation_builder: ObservationBuilder | None = None,
        action_adapter: ActionAdapter | None = None,
        reward_term: RewardTerm | None = None,
        termination: TerminationCondition | None = None,
    ) -> None:
        self.possible_agents = list(AGENTS)
        self.agents = list(AGENTS)
        self._native = BatchSimulator(config_json, state_json, 1)
        self._observation = observation_builder or GlobalObservation()
        self._action = action_adapter or WheelVelocityAction()
        self._reward = reward_term or ZeroReward()
        self._termination = termination or GoalTermination()
        self._state = np.zeros(STATE_WIDTH, dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(-FLOAT_LIMIT, FLOAT_LIMIT, (STATE_WIDTH,), np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(-1.0, 1.0, (2,), np.float32) for agent in self.possible_agents
        }

    def reset(
        self, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, FloatArray], dict[str, dict[str, object]]]:
        del seed, options
        self.agents = list(self.possible_agents)
        self._state = self._native.reset()[0]
        return self._observations(), {agent: {} for agent in self.agents}

    def step(
        self, actions: Mapping[str, FloatArray]
    ) -> tuple[
        dict[str, FloatArray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, object]],
    ]:
        previous = self._state
        native_actions = np.zeros((1, 6, 2), dtype=np.float32)
        for index, agent in enumerate(self.agents):
            native_actions[0, index] = self._action.parse(actions[agent], previous, index)
        self._state = self._native.step(native_actions)[0]
        terminated = self._termination.evaluate(self._state)
        current_agents = list(self.agents)
        observations = self._observations()
        rewards = {
            agent: self._reward.compute(previous, self._state, index)
            for index, agent in enumerate(current_agents)
        }
        terminations = {agent: terminated for agent in current_agents}
        truncations = {agent: False for agent in current_agents}
        infos: dict[str, dict[str, object]] = {agent: {} for agent in current_agents}
        if terminated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> FloatArray:
        """Return the centralized global state."""
        return self._state.copy()

    def observation_space(self, agent: str) -> spaces.Space[FloatArray]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[FloatArray]:
        return self.action_spaces[agent]

    def _observations(self) -> dict[str, FloatArray]:
        return {
            agent: self._observation.build(self._state, index)
            for index, agent in enumerate(self.agents)
        }


class TeamEnv(gym.Env[FloatArray, FloatArray]):
    """Gymnasium adapter controlling all six robots."""

    def __init__(self, config_json: str, state_json: str) -> None:
        self.metadata = {"render_modes": []}
        self._native = BatchSimulator(config_json, state_json, 1)
        self.observation_space = spaces.Box(-FLOAT_LIMIT, FLOAT_LIMIT, (STATE_WIDTH,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (6, 2), np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[FloatArray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        return self._native.reset()[0], {}

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, object]]:
        state = self._native.step(np.ascontiguousarray(action[None], dtype=np.float32))[0]
        return state, 0.0, bool(int(state[-1]) & 0b11), False, {}


class SingleRobotEnv(TeamEnv):
    """Gymnasium adapter controlling one robot while others remain stopped."""

    def __init__(self, config_json: str, state_json: str, robot_index: int = 0) -> None:
        super().__init__(config_json, state_json)
        if robot_index not in range(6):
            raise ValueError("robot_index must be in [0, 6)")
        self.robot_index = robot_index
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, object]]:
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, self.robot_index] = action
        state = self._native.step(actions)[0]
        return state, 0.0, bool(int(state[-1]) & 0b11), False, {}
