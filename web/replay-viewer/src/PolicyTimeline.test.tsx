import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PolicyTimeline } from "./PolicyTimeline";
import type { Replay } from "./types";

const replay = {
  header: {
    type: "header",
    ticks: 2,
    policies: { blue: "m24@1", yellow: "heuristic" },
    action_parser: "primitive",
    config: {
      control_period: 0.02,
      max_wheel_speed: 30,
      field: { length: 1.5, width: 1.3, goal_depth: 0.1, goal_width: 0.4 },
      robot: { length: 0.075, width: 0.075 },
      ball: { radius: 0.0215 },
    },
  },
  frames: [0, 1].map((index) => ({
    type: "tick",
    index,
    episode: 0,
    events: 0,
    rewards: [],
    actions: [],
    policy_intents: Array.from({ length: 6 }, () => ({
      action_index: 1,
      skill: "navigate",
      direction_index: 0,
      direction: "E",
      confidence: 0.5,
      phase: "navigate",
      target: { x: 0.1, y: 0 },
      exit_direction: { x: 1, y: 0 },
      ball_distance: 0.2,
      top_actions: [],
    })),
    snapshot: {
      tick: index,
      simulation_time: (index + 1) * 0.02,
      score_blue: 0,
      score_yellow: 0,
      robots: [],
      ball: { x: 0, y: 0 },
    },
  })),
} satisfies Replay;

describe("PolicyTimeline", () => {
  it("groups primitive decisions and seeks from event marks", () => {
    const seek = vi.fn();
    render(
      <PolicyTimeline
        replay={replay}
        events={[{
          time: 0.02,
          kind: "touch",
          team: "blue",
          robot_id: "R0",
          x: 0,
          y: 0,
          attribution: null,
          related_team: null,
        }]}
        frameIndex={0}
        selectedActor={0}
        onSelectActor={vi.fn()}
        onSeek={seek}
      />,
    );
    expect(screen.getAllByTitle(/NAV-E/)).toHaveLength(6);
    fireEvent.click(screen.getByLabelText(/Seek to touch/));
    expect(seek).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole("button", { name: "HIDE CHANNELS" }));
    expect(screen.queryByTitle(/NAV-E/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /SHOW CHANNELS/ }));
    expect(screen.getAllByTitle(/NAV-E/)).toHaveLength(6);
  });

  it("explains legacy replays instead of rendering empty actor lanes", () => {
    const legacyReplay = {
      ...replay,
      frames: replay.frames.map(({ policy_intents: _policyIntents, ...frame }) => frame),
    };

    const { container } = render(
      <PolicyTimeline
        replay={legacyReplay}
        events={[]}
        frameIndex={0}
        selectedActor={0}
        onSelectActor={vi.fn()}
        onSeek={vi.fn()}
      />,
    );

    expect(container.querySelector(".intent-lane")).toBeNull();
    expect(container.querySelector(".timeline-empty")?.textContent).toMatch(
      /predates policy-intent telemetry/i,
    );
  });
});
