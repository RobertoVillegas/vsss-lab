export interface ReplayInfo {
  iteration: number;
  filename: string;
  bytes: number;
}

export interface ReplayIndex {
  run_dir: string;
  replays: ReplayInfo[];
}

export interface Pose {
  x: number;
  y: number;
  theta: number;
}

export interface Robot {
  id: string;
  team: "blue" | "yellow";
  enabled: boolean;
  pose: Pose;
}

export interface Snapshot {
  tick: number;
  simulation_time: number;
  score_blue: number;
  score_yellow: number;
  robots: Robot[];
  ball: { x: number; y: number };
}

export interface ReplayHeader {
  type: "header";
  ticks: number;
  policies: { blue: string; yellow: string };
  config: {
    control_period: number;
    field: { length: number; width: number; goal_depth: number; goal_width: number };
    robot: { length: number; width: number };
    ball: { radius: number };
  };
}

export interface ReplayFrame {
  type: "tick";
  index: number;
  events: number;
  rewards: number[];
  snapshot: Snapshot;
}

export interface Replay {
  header: ReplayHeader;
  frames: ReplayFrame[];
}
