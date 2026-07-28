import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_platform_manifest_is_m0_and_digest_pinned() -> None:
    manifest = json.loads((ROOT / "platform" / "manifest.json").read_text())
    assert manifest["milestone"] == "M0"
    assert all("@sha256:" in image for image in manifest["images"].values())


def test_active_artifact_paths_are_linux_native() -> None:
    forbidden = ("/mnt/c", "/mnt/d", "/mnt/g")
    documented = (ROOT / "README.md").read_text()
    assert all(path in documented for path in forbidden)
