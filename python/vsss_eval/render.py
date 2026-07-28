"""Deterministic SVG projection for canonical VSSS visual frames."""

import math
from typing import Any

from vsss_eval.visual import VisualFrame


def render_svg(
    frame: VisualFrame,
    config: dict[str, Any],
    *,
    width: int = 900,
    height: int = 780,
) -> str:
    """Render an exact visual frame as a stable standalone SVG."""
    field = config["field"]
    robot = config["robot"]
    ball = config["ball"]
    field_length = float(field["length"])
    field_width = float(field["width"])
    margin = 50.0
    scale = min((width - 2 * margin) / field_length, (height - 2 * margin) / field_width)
    pitch_width = field_length * scale
    pitch_height = field_width * scale
    origin_x = (width - pitch_width) / 2
    origin_y = (height - pitch_height) / 2

    def point(x: float, y: float) -> tuple[float, float]:
        return (width / 2 + x * scale, height / 2 - y * scale)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#15202b"/>',
        f'<rect x="{origin_x:.3f}" y="{origin_y:.3f}" width="{pitch_width:.3f}" '
        f'height="{pitch_height:.3f}" fill="#176b3a" stroke="#f4f4f4" stroke-width="3"/>',
        f'<line x1="{width / 2:.3f}" y1="{origin_y:.3f}" x2="{width / 2:.3f}" '
        f'y2="{origin_y + pitch_height:.3f}" stroke="#f4f4f4" stroke-width="2"/>',
        f'<circle cx="{width / 2:.3f}" cy="{height / 2:.3f}" r="{0.2 * scale:.3f}" '
        'fill="none" stroke="#f4f4f4" stroke-width="2"/>',
    ]
    robot_width = float(robot["length"]) * scale
    robot_height = float(robot["width"]) * scale
    for item in frame.snapshot["robots"]:
        pose = item["pose"]
        x, y = point(float(pose["x"]), float(pose["y"]))
        angle = -math.degrees(float(pose["theta"]))
        color = "#2795ff" if item["team"] == "blue" else "#ffd629"
        lines.append(
            f'<g transform="translate({x:.3f} {y:.3f}) rotate({angle:.3f})">'
            f'<rect x="{-robot_width / 2:.3f}" y="{-robot_height / 2:.3f}" '
            f'width="{robot_width:.3f}" height="{robot_height:.3f}" rx="3" '
            f'fill="{color}" stroke="#111" stroke-width="2"/>'
            f'<line x1="0" y1="0" x2="{robot_width / 2:.3f}" y2="0" '
            'stroke="#111" stroke-width="3"/></g>'
        )
        lines.append(
            f'<text x="{x:.3f}" y="{y - robot_height / 2 - 6:.3f}" '
            f'text-anchor="middle" fill="#fff" font-family="monospace" font-size="14">'
            f"{item['id']}</text>"
        )
    ball_x, ball_y = point(float(frame.snapshot["ball"]["x"]), float(frame.snapshot["ball"]["y"]))
    lines.extend(
        [
            f'<circle cx="{ball_x:.3f}" cy="{ball_y:.3f}" '
            f'r="{float(ball["radius"]) * scale:.3f}" fill="#ff6b22" stroke="#111" '
            'stroke-width="2"/>',
            f'<text x="20" y="30" fill="#fff" font-family="monospace" font-size="18">'
            f"tick {frame.tick} · t={frame.simulation_time:.3f}s · "
            f"blue {frame.snapshot['score_blue']}-{frame.snapshot['score_yellow']} yellow"
            "</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"
