"""Does the policy pick its primitive per family, or the same one everywhere?

Interception rewards blocking and punishes striking; shot needs the strike. A policy that
reads the state should ask for different things in the two. If it asks for the same, the
per-family signal is being averaged into a global preference and no reward weight fixes that.
"""

import json
from pathlib import Path

import numpy as np
import torch
from vsss_env._native import BatchSimulator
from vsss_train.config import load_marl_config
from vsss_train.marl import build_team_observation, stack_team_batches
from vsss_train.marl_ppo import load_policy_actor
from vsss_train.primitives import CircularPrimitiveSet
from vsss_train.roles import assign_roles
from vsss_train.semantic_scenarios import (
    GENERATOR_REVISION,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

cfgj = Path("tests/golden/m1_match_config.json").read_text()
base = json.loads(Path("tests/golden/m1_match_state.json").read_text())
cfg = json.loads(cfgj)
config = load_marl_config("experiments/configs/m24-3-mappo-circular.toml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def opening_tokens(actor, family: str, seeds: int = 24) -> dict[str, float]:
    """What the striker asks for on the drill's opening states, before anything has happened."""
    counts = {"stop": 0, "navigate": 0, "strike": 0}
    for seed in range(seeds):
        parameters = SkillScenarioParameters(
            schema_version=1,
            family=family,
            seed=seed,
            controlled_team="blue",
            difficulty=SkillDifficulty(spawn_distance=0.1, ball_speed=0.1),
            roster="3v3",
            horizon=240,
            holdout=False,
            generator_revision=GENERATOR_REVISION,
        )
        scenario = compile_skill_scenario(parameters, base, cfg)
        slot = int(scenario.context.controlled_robot_id[1:])
        simulator = BatchSimulator(cfgj, json.dumps(scenario.scenario.state), 1)
        state = np.asarray(simulator.reset())[0]
        observation = stack_team_batches(
            [build_team_observation(state, team=0, role_assignment=assign_roles(state, 0))]
        ).to(device)
        with torch.no_grad():
            token = actor.deterministic_action(observation).cpu().numpy()[0][slot]
        counts[CircularPrimitiveSet.decode(np.clip(token, -1, 1).astype(np.float32)).skill] += 1
    total = sum(counts.values())
    return {name: value / total for name, value in counts.items()}


checkpoints = {
    "0009 sana  it1500": "/home/rob/runs/vsss-m24-3-run-0009/checkpoints/iteration-001500.pt",
    "0011 caida it0175": "/home/rob/runs/vsss-m24-3-run-0011/checkpoints/iteration-000175.pt",
}
families = ("interception", "shot", "approach")
print(f"{'checkpoint':<20}" + "".join(f"{name:>32}" for name in families))
print(f"{'':<20}" + "".join(f"{'stop/nav/strike':>32}" for _ in families))
for label, path in checkpoints.items():
    if not Path(path).exists():
        print(f"{label:<20}  falta {path}")
        continue
    actor, _ = load_policy_actor(Path(path), config, device)
    cells = []
    for family in families:
        share = opening_tokens(actor, family)
        cells.append(f"{share['stop']:.2f}/{share['navigate']:.2f}/{share['strike']:.2f}".rjust(32))
    print(f"{label:<20}" + "".join(cells))
