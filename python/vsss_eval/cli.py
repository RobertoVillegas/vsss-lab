"""Command-line entry point for scripted evaluation."""

import argparse
from pathlib import Path

from vsss_eval.match import run_scripted_match, summary_json


def main() -> None:
    """Run a scripted match from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    summary = run_scripted_match(
        args.config.read_text(),
        args.state.read_text(),
        args.ticks,
        args.replay,
        args.seed,
    )
    print(summary_json(summary))


if __name__ == "__main__":
    main()
