from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_profile_emits_phase_evidence(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "tools.profile_m14",
            "--steps",
            "2",
            "--output",
            str(output),
        ),
        check=True,
    )
    payload = output.read_text()
    for phase in (
        "observation_cpu",
        "host_to_device",
        "inference",
        "device_to_host",
        "physics_reward_reset",
        "rollout_end_to_end",
        "ppo_update",
    ):
        assert phase in payload
