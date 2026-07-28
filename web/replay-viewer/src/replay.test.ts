import { describe, expect, it } from "vitest";

import { clampedFrame, frameLabel, parseReplay } from "./replay";

const header = {
  type: "header",
  ticks: 1,
  policies: { blue: "blue@1", yellow: "yellow@0" },
  config: {
    control_period: 0.02,
    field: { length: 1.5, width: 1.3, goal_depth: 0.1, goal_width: 0.4 },
    robot: { length: 0.075, width: 0.075 },
    ball: { radius: 0.0215 },
  },
};
const frame = {
  type: "tick",
  index: 1,
  events: 0,
  rewards: [0],
  snapshot: {
    tick: 4,
    simulation_time: 0.02,
    score_blue: 0,
    score_yellow: 0,
    robots: [],
    ball: { x: 0, y: 0 },
  },
};

describe("replay helpers", () => {
  it("parses canonical JSONL", () => {
    expect(parseReplay(`${JSON.stringify(header)}\n${JSON.stringify(frame)}\n`).frames).toHaveLength(1);
  });

  it("clamps stepping and formats its position", () => {
    expect(clampedFrame(2, -10, 5)).toBe(0);
    expect(clampedFrame(2, 10, 5)).toBe(4);
    expect(frameLabel(2, 5)).toBe("3 / 5");
  });
});
