"""Report causal vision accuracy from a completed replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsss_vision.metrics import analyze_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--analysis", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze_replay(arguments.replay, arguments.analysis).to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
