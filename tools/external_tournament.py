"""Run one isolated Rust-versus-Python external-controller match."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Build participants, run the match, and preserve its replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:42043")
    arguments = parser.parse_args()
    if arguments.ticks <= 0:
        raise ValueError("ticks must be positive")

    root = Path(__file__).parents[1]
    subprocess.run(
        ["cargo", "build", "-p", "vsss-match-server", "-p", "vsss-controller"],
        cwd=root,
        check=True,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            root / "target/debug/vsss-match-server",
            arguments.endpoint,
            root / "tests/golden/m1_match_config.json",
            root / "tests/golden/m1_match_state.json",
            arguments.output,
            str(arguments.ticks),
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    participants: list[subprocess.Popen[str]] = []
    try:
        assert server.stdout is not None
        ready = server.stdout.readline()
        if not ready.startswith("READY "):
            raise RuntimeError(f"server failed before readiness: {ready}")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "python")
        participants = [
            subprocess.Popen(
                [root / "target/debug/vsss-controller", arguments.endpoint],
                cwd=root,
                text=True,
            ),
            subprocess.Popen(
                [
                    root / ".venv/bin/python",
                    "-m",
                    "vsss_controller.sample",
                    "--endpoint",
                    arguments.endpoint,
                ],
                cwd=root,
                env=environment,
                text=True,
            ),
        ]
        for participant in participants:
            if participant.wait(timeout=30) != 0:
                raise RuntimeError("controller exited unsuccessfully")
        if server.wait(timeout=30) != 0:
            raise RuntimeError("match server exited unsuccessfully")
        remainder = server.stdout.read()
        sys.stdout.write(ready + remainder)
    finally:
        for process in [*participants, server]:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
