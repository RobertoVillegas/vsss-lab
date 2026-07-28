import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TrainingMetric } from "./types";

type ChartDatum = TrainingMetric & {
  policy_loss?: number;
  value_loss?: number;
  entropy?: number;
  approx_kl?: number;
  clip_fraction?: number;
  mean_abs_action?: number;
  action_saturation?: number;
  goals?: number;
  draws?: number;
  stagnations?: number;
  frames_per_second?: number;
  matches_per_second?: number;
  log_std_left?: number;
  log_std_right?: number;
};

export default function TrainingCharts({ metrics }: { metrics: TrainingMetric[] }) {
  const data = useMemo<ChartDatum[]>(
    () => metrics.map((metric) => ({
      ...metric,
      policy_loss: metric.losses.policy_loss,
      value_loss: metric.losses.value_loss,
      entropy: metric.losses.entropy,
      approx_kl: metric.losses.approx_kl,
      clip_fraction: metric.losses.clip_fraction,
      mean_abs_action: metric.losses.mean_abs_action,
      action_saturation: metric.losses.action_saturation,
      goals: metric.terminations?.goal,
      draws: metric.terminations?.draw,
      stagnations: metric.terminations?.stagnation,
      frames_per_second: metric.performance?.frames_per_second,
      matches_per_second: metric.performance?.matches_per_second,
      log_std_left: metric.exploration?.actor_log_std?.[0],
      log_std_right: metric.exploration?.actor_log_std?.[1],
    })),
    [metrics],
  );
  if (!data.length) {
    return (
      <div className="empty-state">
        <strong>No training metrics yet</strong>
        <span>Start or resume a run; this view updates every two seconds.</span>
      </div>
    );
  }
  return (
    <div className="charts-shell">
      <header>
        <div>
          <p className="eyebrow">LIVE SCALARS · {data.length.toLocaleString()} SAMPLED POINTS</p>
          <h2>Training diagnostics</h2>
        </div>
        <span>iteration {data.at(-1)?.iteration.toLocaleString()}</span>
      </header>
      <div className="chart-grid">
        <TrainingChart title="RETURN · PROGRESS" data={data}>
          <Line dataKey="return_total" name="return" stroke="#71e1ae" dot={false} />
          <Line dataKey="progress" name="progress" stroke="#49a7ff" dot={false} />
        </TrainingChart>
        <TrainingChart title="OPTIMIZATION" data={data}>
          <Line dataKey="policy_loss" name="policy loss" stroke="#71e1ae" dot={false} />
          <Line dataKey="value_loss" name="value loss" stroke="#ffd84a" dot={false} />
          <Line dataKey="entropy" name="entropy" stroke="#ff8a62" dot={false} />
        </TrainingChart>
        <TrainingChart title="POLICY HEALTH" data={data}>
          <Line dataKey="approx_kl" name="approx KL" stroke="#71e1ae" dot={false} />
          <Line dataKey="clip_fraction" name="clip fraction" stroke="#ffd84a" dot={false} />
          <Line dataKey="mean_abs_action" name="mean |action|" stroke="#49a7ff" dot={false} />
          <Line
            dataKey="action_saturation"
            name="saturation"
            stroke="#ff8a62"
            dot={false}
          />
        </TrainingChart>
        <TrainingChart title="THROUGHPUT" data={data}>
          <Line dataKey="frames_per_second" name="frames/s" stroke="#71e1ae" dot={false} />
          <Line dataKey="matches_per_second" name="matches/s" stroke="#49a7ff" dot={false} />
        </TrainingChart>
        <TrainingChart title="TERMINATIONS" data={data}>
          <Line dataKey="goals" name="goals" stroke="#71e1ae" dot={false} />
          <Line dataKey="draws" name="draws" stroke="#ffd84a" dot={false} />
          <Line dataKey="stagnations" name="stagnation" stroke="#ff8a62" dot={false} />
        </TrainingChart>
        <TrainingChart title="EXPLORATION · LOG STD" data={data}>
          <Line dataKey="log_std_left" name="left wheel" stroke="#71e1ae" dot={false} />
          <Line dataKey="log_std_right" name="right wheel" stroke="#49a7ff" dot={false} />
        </TrainingChart>
      </div>
    </div>
  );
}

function TrainingChart({
  title,
  data,
  children,
}: {
  title: string;
  data: ChartDatum[];
  children: React.ReactNode;
}) {
  return (
    <article className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} syncId="training" margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#183027" strokeDasharray="3 3" />
          <XAxis dataKey="iteration" stroke="#678075" tick={{ fontSize: 9 }} minTickGap={32} />
          <YAxis stroke="#678075" tick={{ fontSize: 9 }} width={58} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#0b1914", border: "1px solid #315044", fontSize: 10 }}
            labelFormatter={(value) => `iteration ${Number(value).toLocaleString()}`}
          />
          <Legend wrapperStyle={{ fontSize: 9 }} />
          {children}
        </LineChart>
      </ResponsiveContainer>
    </article>
  );
}
