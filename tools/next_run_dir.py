"""Atomically allocate a sequential training run directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def allocate_run_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for candidate in root.glob(f"{prefix}-*"):
        suffix = candidate.name.removeprefix(f"{prefix}-")
        if suffix.isdigit():
            existing.append(int(suffix))

    sequence = max(existing, default=0) + 1
    while True:
        candidate = root / f"{prefix}-{sequence:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            sequence += 1
            continue
        return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--root", type=Path, default=Path("/home/rob/runs"))
    arguments = parser.parse_args()
    print(allocate_run_dir(arguments.root, arguments.prefix))


if __name__ == "__main__":
    main()
