"""Serve a captured run and its web replay viewer on loopback."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REPLAY_NAME = re.compile(r"iteration-(\d+)\.jsonl")


@dataclass(frozen=True)
class ReplayInfo:
    """Metadata needed by the run picker."""

    iteration: int
    filename: str
    bytes: int


def discover_replays(run_dir: Path) -> list[ReplayInfo]:
    """Return canonical iteration files in numeric order."""
    replay_dir = run_dir.resolve() / "replays"
    found: list[ReplayInfo] = []
    if not replay_dir.is_dir():
        return found
    for path in replay_dir.iterdir():
        match = REPLAY_NAME.fullmatch(path.name)
        if path.is_file() and match is not None:
            found.append(ReplayInfo(int(match.group(1)), path.name, path.stat().st_size))
    return sorted(found, key=lambda replay: replay.iteration)


def resolve_replay(run_dir: Path, filename: str) -> Path | None:
    """Resolve only a replay currently discoverable in the configured run."""
    names = {replay.filename for replay in discover_replays(run_dir)}
    if filename not in names:
        return None
    return run_dir.resolve() / "replays" / filename


def make_handler(run_dir: Path, static_dir: Path) -> type[SimpleHTTPRequestHandler]:
    """Create a request handler closed over validated roots."""
    resolved_run = run_dir.resolve()
    resolved_static = static_dir.resolve()

    class ReplayRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(resolved_static), **kwargs)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/iterations":
                self._send_json(
                    {
                        "run_dir": str(resolved_run),
                        "replays": [asdict(replay) for replay in discover_replays(resolved_run)],
                    }
                )
                return
            if path.startswith("/api/replays/"):
                filename = unquote(path.removeprefix("/api/replays/"))
                replay = resolve_replay(resolved_run, filename)
                if replay is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Replay not found")
                    return
                self._send_file(replay, "application/x-ndjson")
                return
            if path != "/" and not (resolved_static / path.lstrip("/")).is_file():
                self.path = "/"
            super().do_GET()

        def _send_json(self, value: object) -> None:
            payload = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path: Path, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as replay:
                self.copyfile(replay, self.wfile)

    return ReplayRequestHandler


def main() -> None:
    """Run the local viewer server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path(__file__).parents[2] / "web" / "replay-viewer" / "dist",
    )
    args = parser.parse_args()
    if not args.run_dir.resolve().is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")
    if not args.static_dir.resolve().is_dir():
        parser.error(f"web build does not exist: {args.static_dir}; run just web-build")
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.run_dir, args.static_dir)
    )
    print(f"VSSS replay viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Run: {args.run_dir.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
