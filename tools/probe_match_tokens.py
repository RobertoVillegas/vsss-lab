"""What each policy asks for during a full match, split by where the ball is.

Drills say the 0011 policy picks its primitive per family. A match is not a drill: the two
score very differently there, so the question is what each asks for when it has the ball in
the attacking half, which is where a goal comes from.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from vsss_league.training import create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import team_action_width
from vsss_train.marl_ppo import load_policy_actor
from vsss_train.primitives import CircularPrimitiveSet

cfgj = Path("tests/golden/m1_match_config.json").read_text()
stj = Path("tests/golden/m1_match_state.json").read_text()
# The checkpoint validates a fingerprint of the config it was trained under, so the actor is
# loaded with that one and only the environment is rebuilt as plain matches.
trained = load_marl_config("experiments/configs/m24-3-mappo-circular.toml")
config = replace(
    trained,
    num_envs=32,
    semantic_curriculum=False,
    semantic_phased_curriculum=False,
    adaptive_curriculum=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def match_tokens(actor, steps: int = 200) -> dict[str, dict[str, float]]:
    session = create_rollout_session(config, cfgj, stj)
    environment = session.environment
    for world in range(environment.num_envs):
        environment.reset(world, 4000 + world)
    zones = {
        "ataque (bola x>0.2)": {"stop": 0, "navigate": 0, "strike": 0},
        "medio": {"stop": 0, "navigate": 0, "strike": 0},
        "defensa (bola x<-0.2)": {"stop": 0, "navigate": 0, "strike": 0},
    }
    width = team_action_width(config.action_parser)
    for _ in range(steps):
        observation = environment.current_observations().to(device)
        with torch.no_grad():
            tokens = actor.deterministic_action(observation).cpu().numpy()
        for world in range(environment.num_envs):
            ball_x = float(environment.states[world][5])
            key = (
                "ataque (bola x>0.2)"
                if ball_x > 0.2
                else "defensa (bola x<-0.2)"
                if ball_x < -0.2
                else "medio"
            )
            for slot in range(3):
                skill = CircularPrimitiveSet.decode(
                    np.clip(tokens[world][slot], -1, 1).astype(np.float32)
                ).skill
                zones[key][skill] += 1
        environment.step(tokens.astype(np.float32).reshape(-1, 3, width), None)
    return {
        zone: {k: v / max(1, sum(counts.values())) for k, v in counts.items()}
        for zone, counts in zones.items()
    }


for label, path in (
    ("0009 it1500", "/home/rob/runs/vsss-m24-3-run-0009/checkpoints/iteration-001500.pt"),
    ("0011 it0175", "/home/rob/runs/vsss-m24-3-run-0011/checkpoints/iteration-000175.pt"),
):
    actor, _ = load_policy_actor(Path(path), trained, device)
    print(label)
    for zone, share in match_tokens(actor).items():
        print(
            f"   {zone:<24} stop {share['stop']:.2f}  navigate {share['navigate']:.2f}"
            f"  strike {share['strike']:.2f}"
        )
