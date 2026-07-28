"""Scripted evaluation and replay utilities."""

from vsss_eval.analytics import AnalyticsEvent, ReplayAnalytics, analyze_replay
from vsss_eval.match import MatchSummary, run_scripted_match
from vsss_eval.render import render_svg
from vsss_eval.replay import inspect_replay, replay_frames
from vsss_eval.visual import (
    FrameSink,
    LatestFrameSink,
    MetricsSink,
    NullSink,
    UdpFrameSink,
    VisualFrame,
)

__all__ = [
    "AnalyticsEvent",
    "FrameSink",
    "LatestFrameSink",
    "MatchSummary",
    "MetricsSink",
    "NullSink",
    "ReplayAnalytics",
    "UdpFrameSink",
    "VisualFrame",
    "analyze_replay",
    "inspect_replay",
    "render_svg",
    "replay_frames",
    "run_scripted_match",
]
