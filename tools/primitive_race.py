"""Which primitive resolves each foundation drill first, and how often.

The timeout penalty pays for resolving. If navigate resolves the touch-based families faster
than strike does, then charging the timeout necessarily biases the policy toward navigate,
whatever the family is nominally teaching.
"""

import json
import math
from pathlib import Path

import numpy as np
from vsss_baselines import DynamicTeamController
from vsss_env._native import BatchSimulator
from vsss_train.primitives import circular_primitive_wheel_actions
from vsss_train.roles import DynamicRoleAssigner
from vsss_train.semantic_scenarios import (
    GENERATOR_REVISION,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)
from vsss_train.skill_predicates import SkillEvaluator, SkillStatus, skill_frame_from_native

cfgj = Path("tests/golden/m1_match_config.json").read_text()
stj = Path("tests/golden/m1_match_state.json").read_text()
base, cfg = json.loads(stj), json.loads(cfgj)
rb, ba, gw = cfg["robot"], cfg["ball"], cfg["field"]["goal_width"]
GOAL_X = cfg["field"]["length"] / 2.0


def token(skill: str, heading: float, intensity: float = 1.0) -> list[float]:
    index = {"stop": -1.0, "navigate": 0.0, "strike": 1.0}[skill]
    wrapped = (heading + math.pi) % (2.0 * math.pi) - math.pi
    return [index, wrapped / math.pi, intensity * 2.0 - 1.0]


def scripted(skill: str, state, slot: int, attack_sign: float):
    """Always ask for one primitive: navigate at the ball, strike toward the goal."""
    base_index = 10 + slot * 11
    if skill == "navigate":
        heading = math.atan2(state[6] - state[base_index + 3], state[5] - state[base_index + 2])
    else:
        heading = math.atan2(-state[6], attack_sign * GOAL_X - state[5])
    return token(skill, heading * attack_sign if attack_sign < 0 else heading)


def race(family: str, skill: str, difficulty: float, trials: int = 20):
    resolved = steps_to = successes = 0
    for seed in range(trials):
        parameters = SkillScenarioParameters(
            schema_version=1,
            family=family,
            seed=seed,
            controlled_team="blue",
            difficulty=SkillDifficulty(spawn_distance=difficulty, ball_speed=0.1),
            roster="3v3",
            horizon=240,
            holdout=False,
            generator_revision=GENERATOR_REVISION,
        )
        scenario = compile_skill_scenario(parameters, base, cfg)
        slot = int(scenario.context.controlled_robot_id[1:])
        evaluator = SkillEvaluator(
            scenario.context,
            robot_radius=(rb["length"] ** 2 + rb["width"] ** 2) ** 0.5 / 2,
            ball_radius=ba["radius"],
            goal_half_width=gw / 2,
        )
        simulator = BatchSimulator(cfgj, json.dumps(scenario.scenario.state), 1)
        state = np.asarray(simulator.reset())[0]
        assigner, yellow = DynamicRoleAssigner(), DynamicTeamController(3, -1)
        for step in range(240):
            roles = assigner.assign(state, 0)
            outcome = evaluator.observe(
                skill_frame_from_native(
                    state,
                    step=step,
                    events=int(state[-1]),
                    role_assignment=roles,
                    controlled_team="blue",
                )
            )
            if outcome.status is not SkillStatus.RUNNING:
                resolved += 1
                steps_to += step
                successes += outcome.status is SkillStatus.SUCCESS
                break
            tokens = np.zeros((3, 3), dtype=np.float32)
            tokens[:, 0] = -1.0
            tokens[slot] = scripted(skill, state, slot, 1.0)
            blue = circular_primitive_wheel_actions(state, team=0, tokens=tokens)
            command = np.concatenate([blue, yellow.actions(state)])[None].astype(np.float32) * 12
            state = np.asarray(simulator.step_repeated(command, 4))[0]
    mean_steps = steps_to / resolved if resolved else 240
    return resolved / trials, successes / trials, mean_steps


print("what the drill asks for, and which primitive delivers it first")
print(f"{'family':<16}{'primitive':<12}{'resolved':>10}{'success':>10}{'steps':>8}")
for family in ("approach", "interception", "shot"):
    for skill in ("navigate", "strike"):
        rate, success, steps = race(family, skill, 0.1)
        print(f"{family:<16}{skill:<12}{rate:>10.2f}{success:>10.2f}{steps:>8.0f}")
