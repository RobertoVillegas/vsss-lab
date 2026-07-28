"""M9 isolated reference-physics calibration scenarios."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from vsss_env._native import BatchSimulator

ROOT = Path(__file__).resolve().parents[2]


def run_calibration(
    config_path: Path = ROOT / "tests/golden/m1_match_config.json",
    state_path: Path = ROOT / "tests/golden/m1_match_state.json",
    scenarios_path: Path = ROOT / "calibration/golden-scenarios-v1.json",
) -> dict[str, Any]:
    config_text = config_path.read_text()
    state_data = json.loads(state_path.read_text())
    state_data["tick"] = 0
    state_data["simulation_time"] = 0.0
    state_data["score_blue"] = 0
    state_data["score_yellow"] = 0
    state_data["events"] = 0
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_state = json.loads(json.dumps(state_data))
        if "initial_ball" in scenario:
            scenario_state["ball"].update(scenario["initial_ball"])
        simulator = BatchSimulator(config_text, json.dumps(scenario_state), 1)
        initial = simulator.reset()[0]
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, 0] = scenario["wheel_action"]
        current = initial
        for _ in range(int(scenario["ticks"])):
            current = simulator.step(actions)[0]
        metric = str(scenario["metric"])
        if metric == "robot_0_displacement_m":
            measured = math.hypot(
                float(current[12] - initial[12]),
                float(current[13] - initial[13]),
            )
        elif metric == "robot_0_heading_rad":
            measured = float(current[14])
        elif metric == "ball_speed_mps":
            measured = math.hypot(float(current[7]), float(current[8]))
        else:
            raise ValueError(f"unknown metric: {metric}")
        reference = float(scenario["reference"])
        error = abs(measured - reference)
        tolerance = float(scenario["tolerance"])
        results.append(
            {
                "id": scenario["id"],
                "metric": metric,
                "reference": reference,
                "measured": measured,
                "absolute_error": error,
                "tolerance": tolerance,
                "passed": error <= tolerance,
            }
        )
    return {
        "schema_version": 1,
        "passed": all(row["passed"] for row in results),
        "results": results,
    }
