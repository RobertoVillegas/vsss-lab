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
  matches: number;
  losses: {
    policy_loss?: number;
    value_loss?: number;
    entropy?: number;
    approx_kl?: number;
    clip_fraction?: number;
    mean_abs_action?: number;
    action_saturation?: number;
  };
  terminations?: {
    goal?: number;
    draw?: number;
    stagnation?: number;
  };
  environment_steps?: number;
  total_matches?: number;
  performance?: {
    frames_per_second?: number;
    matches_per_second?: number;
    iterations_per_second?: number;
  };
  exploration?: {
    actor_log_std?: number[];
  };
  curriculum?: {
    schema_version?: number;
    levels?: Record<string, Record<string, number>>;
    success_rate?: Record<string, number>;
    outcomes?: {
      success?: number;
      failure?: number;
      unresolved?: number;
    };
    trials?: SemanticTrial[];
  };
}

export interface SemanticTrial {
  schema_version: number;
  scenario_id: string;
  family: string;
  controlled_team: "blue" | "yellow";
  difficulty: Record<string, number>;
  parameter_hash: string;
  state_hash: string;
  status: "success" | "failure" | "unresolved";
  reason: string;
  steps: number;
  controlled_touches: number;
  opponent_touches: number;
}

export interface MetricHistory {
  metrics: TrainingMetric[];
}

export interface ReplayIndex {
  run_dir: string;
  replays: ReplayInfo[];
  checkpoints: CheckpointInfo[];
  latest_metric: TrainingMetric | null;
}

export interface ReplayAnalytics {
  schema_version: number;
  definition_version: string;
  sampled_seconds: number;
  teams: Record<"blue" | "yellow", {
    possession_seconds: number;
    passes: number;
    assists: number;
    shots: number;
    saves: number;
    clearances: number;
    interceptions: number;
    double_commit_seconds: number;
    congestion_seconds: number;
  }>;
  events: {
    time: number;
    kind: string;
    team: "blue" | "yellow";
    robot_id: string | null;
    x: number;
    y: number;
  }[];
  ball_heatmap: number[][];
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
  semantic_context?: TrainingMetric["curriculum"];
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
  perception?: {
    policy_visible: boolean;
    camera: {
      capture_time: number;
      arrival_time: number;
      ball: { x: number; y: number } | null;
      robots: {
        association: {
          marker_id: number | null;
          confidence: number;
          ambiguous: boolean;
        };
      }[];
    };
    ball_estimate: {
      effective_time: number;
      update_time: number;
      state: [number, number, number, number, number, number];
      measurement_accepted: boolean;
      rejection_reason: string | null;
    } | null;
    ball_prediction: {
      samples: [number, number, number][];
      uncertainty: [number, number, number][];
      stale: boolean;
      model_id: string;
    } | null;
    goalkeeper_interception: {
      team: "blue" | "yellow";
      elapsed: number;
      x: number;
      y: number;
      model_id: string;
    } | null;
  };
}

export interface Replay {
  header: ReplayHeader;
  frames: ReplayFrame[];
}
