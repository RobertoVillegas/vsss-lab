"""Map a candidate ball-position potential before anything is wired to it.

The requirement is a gradient that pays for carrying the ball toward a scoring position and
pays almost nothing for carrying it along a touchline or into a corner, where a goal is
possible but not direct and the carrier gets blocked or stuck. The angular width of the goal
mouth as seen from the ball has that shape for free: it is maximal in front of the goal, and it
collapses at grazing incidence no matter how close the corner is.
"""

import json
import math
from pathlib import Path

cfg = json.loads(Path("tests/golden/m1_match_config.json").read_text())
LENGTH = cfg["field"]["length"]
WIDTH = cfg["field"]["width"]
GOAL_X = LENGTH / 2.0
HALF_GOAL = cfg["field"]["goal_width"] / 2.0


def subtended(ball_x: float, ball_y: float) -> float:
    """Angular width of the goal mouth seen from the ball, normalized to its maximum."""
    near = math.atan2(HALF_GOAL - ball_y, GOAL_X - ball_x)
    far = math.atan2(-HALF_GOAL - ball_y, GOAL_X - ball_x)
    return abs(near - far)


# The maximum is on the goal line at the centre; normalize against a point just in front of it.
PEAK = subtended(GOAL_X - 0.02, 0.0)


def potential(ball_x: float, ball_y: float) -> float:
    return min(1.0, subtended(ball_x, ball_y) / PEAK)


print(f"campo {LENGTH:.2f} x {WIDTH:.2f}, arco en x={GOAL_X:.2f}, |y|<={HALF_GOAL:.2f}")
print("potencial normalizado, 1.00 = justo frente al arco\n")
ys = [0.0, 0.15, 0.30, 0.45, 0.60]
xs = [0.72, 0.60, 0.45, 0.30, 0.10, -0.20, -0.50]
print(f"{'x \\\\ |y|':>9}" + "".join(f"{y:>8.2f}" for y in ys))
for x in xs:
    print(f"{x:>9.2f}" + "".join(f"{potential(x, y):>8.3f}" for y in ys))

print()
print("las rutas que importan:")
corner = potential(0.70, 0.62)
front_far = potential(0.10, 0.0)
front_near = potential(0.60, 0.0)
touchline = potential(0.45, 0.60)
print(f"  esquina, pegado al arco  (x=0.70, y=0.62): {corner:.3f}")
print(f"  orilla a media altura    (x=0.45, y=0.60): {touchline:.3f}")
print(f"  centro, lejos            (x=0.10, y=0.00): {front_far:.3f}")
print(f"  centro, cerca            (x=0.60, y=0.00): {front_near:.3f}")
print()
print(f"  arrastrar por la orilla hasta la esquina paga {corner - touchline:+.3f}")
print(f"  llevarla al centro desde lejos paga          {front_near - front_far:+.3f}")
