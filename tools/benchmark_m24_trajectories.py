"""Run M24 trajectory primitives through exact simulator fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsss_train.trajectory_benchmark import benchmark_primitives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-config",
        type=Path,
        default=Path("tests/golden/m1_match_config.json"),
    )
    parser.add_argument(
        "--match-state",
        type=Path,
        default=Path("tests/golden/m1_match_state.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = benchmark_primitives(
        arguments.match_config.read_text(),
        arguments.match_state.read_text(),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary.write_text(payload)
    temporary.replace(arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
