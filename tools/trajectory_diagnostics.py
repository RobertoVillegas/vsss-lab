"""Print episode-aware trajectory diagnostics for a captured replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsss_train.trajectory_diagnostics import analyze_trajectory_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze_trajectory_replay(arguments.replay).to_dict(), indent=2))


if __name__ == "__main__":
    main()
