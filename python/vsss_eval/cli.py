"""Command-line entry point for scripted evaluation."""

import argparse
import json
from pathlib import Path

from vsss_eval.match import run_scripted_match, summary_json
from vsss_eval.visual import UdpFrameSink


def main() -> None:
    """Run a scripted match from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--live-target")
    parser.add_argument("--live-sample-every", type=int, default=4)
    args = parser.parse_args()
    config_json = args.config.read_text()
    observers: tuple[UdpFrameSink, ...] = ()
    live: UdpFrameSink | None = None
    if args.live_target:
        host, port = args.live_target.rsplit(":", 1)
        live = UdpFrameSink(
            json.loads(config_json),
            (host, int(port)),
            sample_every=args.live_sample_every,
        )
        observers = (live,)
    summary = run_scripted_match(
        config_json,
        args.state.read_text(),
        args.ticks,
        args.replay,
        args.seed,
        observers,
    )
    if live is not None:
        live.close()
    print(summary_json(summary))


if __name__ == "__main__":
    main()
