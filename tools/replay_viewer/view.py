"""Validate and summarize an M4 JSONL replay."""

import argparse
import json
from pathlib import Path

from vsss_eval import inspect_replay, render_svg, replay_frames


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--frame", type=int, default=-1)
    args = parser.parse_args()
    if args.svg is not None:
        header = json.loads(args.replay.read_text().splitlines()[0])
        frames = replay_frames(args.replay)
        args.svg.write_text(render_svg(frames[args.frame], header["config"]))
    print(json.dumps(inspect_replay(args.replay), sort_keys=True))


if __name__ == "__main__":
    main()
