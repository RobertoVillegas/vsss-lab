"""Export versioned M14 analytics from one canonical replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from vsss_eval import analyze_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--team-csv", type=Path)
    arguments = parser.parse_args()
    analytics = analyze_replay(arguments.replay)
    analytics.write_json(arguments.json)
    if arguments.team_csv is not None:
        analytics.write_team_csv(arguments.team_csv)
    print(arguments.json.resolve())


if __name__ == "__main__":
    main()
