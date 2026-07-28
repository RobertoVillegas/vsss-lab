import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "next_run_dir.py"
SPEC = importlib.util.spec_from_file_location("next_run_dir", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
allocate_run_dir = MODULE.allocate_run_dir


def test_allocate_run_dir_uses_next_sequence(tmp_path: Path) -> None:
    (tmp_path / "vsss-training-run-0003").mkdir()
    (tmp_path / "vsss-training-run-not-a-number").mkdir()

    first = allocate_run_dir(tmp_path, "vsss-training-run")
    second = allocate_run_dir(tmp_path, "vsss-training-run")

    assert first == tmp_path / "vsss-training-run-0004"
    assert second == tmp_path / "vsss-training-run-0005"
    assert first.is_dir()
    assert second.is_dir()
