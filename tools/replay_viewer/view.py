"""Validate and summarize an M4 JSONL replay."""

import argparse
import json
from pathlib import Path

from vsss_eval import inspect_replay


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_replay(args.replay), sort_keys=True))


if __name__ == "__main__":
    main()
