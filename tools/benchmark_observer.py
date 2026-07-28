"""Measure scripted-match observer overhead with identical replay recording."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from vsss_eval import LatestFrameSink, run_scripted_match

ROOT = Path(__file__).parents[1]


def measure(
    config: str, state: str, ticks: int, repeats: int, sample_every: int
) -> tuple[list[float], list[float]]:
    """Return interleaved unwatched and watched elapsed seconds."""
    elapsed: dict[bool, list[float]] = {False: [], True: []}
    with tempfile.TemporaryDirectory(prefix="vsss-observer-") as directory:
        for repeat in range(repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for watched in order:
                sink = (LatestFrameSink(sample_every=sample_every),) if watched else ()
                start = time.perf_counter()
                run_scripted_match(
                    config,
                    state,
                    ticks,
                    Path(directory) / f"{watched}-{repeat}.jsonl",
                    observers=sink,
                )
                elapsed[watched].append(time.perf_counter() - start)
    return elapsed[False], elapsed[True]


def main() -> None:
    """Run the observer benchmark and print stable JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()
    config = (ROOT / "tests/golden/m1_match_config.json").read_text()
    state = (ROOT / "tests/golden/m1_match_state.json").read_text()
    unwatched, watched = measure(config, state, args.ticks, args.repeats, args.sample_every)
    base = statistics.median(unwatched)
    observed = statistics.median(watched)
    print(
        json.dumps(
            {
                "ticks": args.ticks,
                "repeats": args.repeats,
                "sample_every": args.sample_every,
                "unwatched_median_seconds": round(base, 6),
                "watched_median_seconds": round(observed, 6),
                "observer_overhead_percent": round((observed / base - 1.0) * 100.0, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
