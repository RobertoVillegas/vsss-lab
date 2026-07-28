export interface ReplayInfo {
  iteration: number;
  filename: string;
  bytes: number;
  outcome: "win" | "loss" | "draw";
  score_blue: number;
  score_yellow: number;
  goals: number;
  simulation_seconds: number;
}

export interface CheckpointInfo {
  iteration: number;
  filename: string;
  bytes: number;
}

export interface TrainingMetric {
  iteration: number;
  policy_version: number;
  frames: number;
  return_total: number;
  progress: number;
}

export interface ReplayIndex {
  run_dir: string;
  replays: ReplayInfo[];
  checkpoints: CheckpointInfo[];
  latest_metric: TrainingMetric | null;
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
  wheel_speed_left: number;
  wheel_speed_right: number;
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
    max_wheel_speed: number;
    field: { length: number; width: number; goal_depth: number; goal_width: number };
    robot: { length: number; width: number };
    wheel?: { radius: number; axle_track: number };
    ball: { radius: number };
  };
}

export interface ReplayFrame {
  type: "tick";
  index: number;
  events: number;
  rewards: number[];
  actions: number[][];
  snapshot: Snapshot;
}

export interface Replay {
  header: ReplayHeader;
  frames: ReplayFrame[];
}
