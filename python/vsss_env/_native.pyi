"""Typing surface for the native extension."""

import numpy as np
from numpy.typing import NDArray

class BatchSimulator:
    """Native independent-world simulator."""

    def __init__(self, config_json: str, state_json: str, num_worlds: int) -> None: ...
    @property
    def num_worlds(self) -> int: ...
    @staticmethod
    def state_width() -> int: ...
    def reset(self) -> NDArray[np.float32]: ...
    def step(self, actions: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def step_repeated(self, actions: NDArray[np.float32], repeats: int) -> NDArray[np.float32]: ...
    def observations(
        self,
        teams: NDArray[np.int64],
        roles: NDArray[np.float32],
        field_length: float,
        field_width: float,
        match_duration: float,
    ) -> list[NDArray[np.float32]]: ...
    def team_roles(
        self, teams: NDArray[np.int64], hysteretic: bool
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.bool_],
        NDArray[np.float64],
        list[list[str]],
    ]: ...
    def circular_wheel_actions(
        self,
        teams: NDArray[np.int64],
        tokens: NDArray[np.float32],
        ball_deceleration: float,
    ) -> NDArray[np.float32]: ...
    def scripted_actions(self, teams: NDArray[np.int64]) -> NDArray[np.float32]: ...
    def contacts(
        self,
        teams: NDArray[np.int64],
        previous_ball: NDArray[np.float32],
        ally_streaks: NDArray[np.int64],
        opponent_streaks: NDArray[np.int64],
        contact_distance: float,
        grace_steps: int,
        meaningful_ball_displacement: float,
        robot: tuple[float, float, float],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]: ...
    def idle_spin(
        self,
        teams: NDArray[np.int64],
        actions: NDArray[np.float32],
        angular_speed: float,
        drive: float,
        speed: float,
        ball_distance: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32]]: ...
    def goal_geometry(
        self,
        teams: NDArray[np.int64],
        field_length: float,
        goal_width: float,
        ball_radius: float,
    ) -> NDArray[np.float64]: ...
    def restart_roles(
        self, index: int, team: int
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.bool_],
        NDArray[np.float64],
        list[list[str]],
    ]: ...
    def snapshots(self) -> list[str]: ...
    def restore(self, index: int, snapshot_json: str) -> None: ...
    def restore_state(self, index: int, snapshot_json: str) -> NDArray[np.float32]: ...
