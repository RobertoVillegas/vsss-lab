"""M4 scripted controller and match regression tests."""

import json
import socket
import threading
import time
import zlib
from pathlib import Path

import numpy as np
from vsss_baselines import DynamicTeamController, go_to_target
from vsss_env._native import BatchSimulator
from vsss_eval import (
    LatestFrameSink,
    MetricsSink,
    UdpFrameSink,
    inspect_replay,
    render_svg,
    replay_frames,
    run_scripted_match,
)

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def initial_row() -> np.ndarray:
    """Return the canonical M3 kickoff tensor."""
    return np.asarray(BatchSimulator(CONFIG, STATE, 1).reset()[0], dtype=np.float32)


def test_target_ahead_drives_straight_and_actions_are_bounded() -> None:
    action = go_to_target((0.0, 0.0, 0.0), (0.5, 0.0))
    assert action[0] == action[1] > 0.0
    assert np.all(np.abs(action) <= 1.0)


def test_dynamic_assignment_ignores_physical_ids() -> None:
    state = initial_row()
    controller = DynamicTeamController(0, 1)
    expected_roles = controller.assign(state)
    expected_actions = controller.actions(state)
    renamed = state.copy()
    renamed[[10, 21, 32]] = renamed[[32, 10, 21]]
    assert controller.assign(renamed) == expected_roles
    np.testing.assert_array_equal(controller.actions(renamed), expected_actions)


def test_scripted_match_and_replay_are_byte_reproducible(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = run_scripted_match(CONFIG, STATE, 120, first_path, seed=7)
    second = run_scripted_match(CONFIG, STATE, 120, second_path, seed=7)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    inspected = inspect_replay(first_path)
    assert inspected["ticks"] == 120
    assert inspected["final_checksum"] == first.final_checksum


def test_live_observer_is_bounded_and_does_not_change_match(tmp_path: Path) -> None:
    headless_path = tmp_path / "headless.jsonl"
    observed_path = tmp_path / "observed.jsonl"
    headless = run_scripted_match(CONFIG, STATE, 40, headless_path, seed=7)
    live = LatestFrameSink()
    metrics = MetricsSink()
    observed = run_scripted_match(
        CONFIG,
        STATE,
        40,
        observed_path,
        seed=7,
        observers=(live, metrics),
    )
    assert observed == headless
    assert observed_path.read_bytes() == headless_path.read_bytes()
    assert live.published == 40
    assert live.dropped == 39
    assert live.consume_latest() is not None
    assert live.consume_latest() is None
    assert metrics.frames == 40
    assert metrics.last_tick == replay_frames(observed_path)[-1].tick


def test_replay_and_live_use_equivalent_visual_frames(tmp_path: Path) -> None:
    replay_path = tmp_path / "match.jsonl"
    live = LatestFrameSink()
    run_scripted_match(CONFIG, STATE, 5, replay_path, observers=(live,))
    live_frame = live.consume_latest()
    frames = replay_frames(replay_path)
    assert live_frame is not None
    assert live_frame == frames[-1]
    assert [frame.tick for frame in frames] == [43, 44, 45, 46, 47]


def test_live_observer_samples_by_simulation_tick_count(tmp_path: Path) -> None:
    replay_path = tmp_path / "sampled.jsonl"
    live = LatestFrameSink(sample_every=4)
    run_scripted_match(CONFIG, STATE, 10, replay_path, observers=(live,))
    assert live.seen == 3
    assert live.published == 3
    assert live.dropped == 2
    assert live.consume_latest() == replay_frames(replay_path)[8]


def test_headless_svg_projection_is_deterministic(tmp_path: Path) -> None:
    replay_path = tmp_path / "match.jsonl"
    run_scripted_match(CONFIG, STATE, 2, replay_path)
    frame = replay_frames(replay_path)[-1]
    config = json.loads(CONFIG)
    first = render_svg(frame, config)
    second = render_svg(frame, config)
    assert first == second
    assert ">R0</text>" in first
    assert "<svg " in first


def test_slow_live_consumer_never_blocks_match(tmp_path: Path) -> None:
    replay_path = tmp_path / "live.jsonl"
    live = LatestFrameSink()
    summary: list[object] = []

    producer = threading.Thread(
        target=lambda: summary.append(
            run_scripted_match(CONFIG, STATE, 500, replay_path, observers=(live,))
        )
    )
    producer.start()
    consumed = 0
    while producer.is_alive():
        consumed += int(live.consume_latest() is not None)
        time.sleep(0.002)
    producer.join(timeout=0.1)
    assert not producer.is_alive()
    assert len(summary) == 1
    assert live.published == 500
    assert live.dropped > 0
    assert consumed < live.published


def test_udp_live_sink_sends_self_describing_frame(tmp_path: Path) -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    target = receiver.getsockname()
    live = UdpFrameSink(json.loads(CONFIG), (str(target[0]), int(target[1])), sample_every=4)
    run_scripted_match(
        CONFIG,
        STATE,
        1,
        tmp_path / "udp.jsonl",
        observers=(live,),
    )
    datagram = receiver.recv(1_400)
    assert datagram.startswith(b"VSS1")
    packet = json.loads(zlib.decompress(datagram[4:]))
    live.close()
    receiver.close()
    assert packet["type"] == "visual_frame"
    assert packet["sequence"] == 0
    assert packet["sample_every"] == 4
    assert packet["config"]["field"]["length"] == 1.5
    assert packet["frame"]["snapshot"]["tick"] == 43
