import { useMemo, useState } from "react";
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

import type { SemanticTrial, TrainingMetric } from "./types";

const SKILL_FAMILIES = [
  "approach",
  "interception",
  "save_deflection",
  "clearance",
  "shot",
  "pass_receive",
  "rotation_recovery",
] as const;

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
  log_std_intensity?: number;
  normalized_entropy?: number;
  stop_fraction?: number;
  navigate_fraction?: number;
  strike_fraction?: number;
  mean_intensity?: number | null;
  direction_change_mean_degrees?: number | null;
  direction_change_p95_degrees?: number | null;
  full_match_allocation?: number;
  frontier_allocation?: number;
  routine_allocation?: number;
  failure_allocation?: number;
  curriculum_phase?: number;
  skill_success?: number;
  skill_failure?: number;
  skill_unresolved?: number;
  approach_success?: number;
  interception_success?: number;
  save_deflection_success?: number;
  clearance_success?: number;
  shot_success?: number;
  pass_receive_success?: number;
};

export default function TrainingCharts({ metrics }: { metrics: TrainingMetric[] }) {
  const [familyFilter, setFamilyFilter] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [outcomeFilter, setOutcomeFilter] = useState("all");
  const [difficultyFilter, setDifficultyFilter] = useState("all");
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
      log_std_intensity: metric.exploration?.actor_log_std?.[2],
      heading_concentration: metric.exploration?.heading_concentration,
      normalized_entropy: metric.exploration?.normalized_entropy,
      stop_fraction: metric.policy_stats?.stop_fraction,
      navigate_fraction: metric.policy_stats?.navigate_fraction,
      strike_fraction: metric.policy_stats?.strike_fraction,
      mean_intensity: metric.policy_stats?.mean_intensity,
      direction_change_mean_degrees: metric.policy_stats?.direction_change_mean_degrees,
      direction_change_p95_degrees: metric.policy_stats?.direction_change_p95_degrees,
      full_match_allocation: metric.curriculum?.allocation?.full_match,
      frontier_allocation: metric.curriculum?.allocation?.frontier,
      routine_allocation: metric.curriculum?.allocation?.routine,
      failure_allocation: metric.curriculum?.allocation?.failure,
      curriculum_phase: metric.curriculum?.phase_index,
      skill_success: metric.curriculum?.outcomes?.success,
      skill_failure: metric.curriculum?.outcomes?.failure,
      skill_unresolved: metric.curriculum?.outcomes?.unresolved,
      approach_success: metric.curriculum?.success_rate?.approach,
      interception_success: metric.curriculum?.success_rate?.interception,
      save_deflection_success: metric.curriculum?.success_rate?.save_deflection,
      clearance_success: metric.curriculum?.success_rate?.clearance,
      shot_success: metric.curriculum?.success_rate?.shot,
      pass_receive_success: metric.curriculum?.success_rate?.pass_receive,
    })),
    [metrics],
  );
  const trials = useMemo(
    () => metrics.flatMap((metric) => (
      metric.curriculum?.trials?.map((trial) => ({ ...trial, iteration: metric.iteration })) ?? []
    )).filter((trial) => (
      (familyFilter === "all" || trial.family === familyFilter)
      && (teamFilter === "all" || trial.controlled_team === teamFilter)
      && (outcomeFilter === "all" || trial.status === outcomeFilter)
      && matchesDifficulty(trial, difficultyFilter)
    )),
    [difficultyFilter, familyFilter, metrics, outcomeFilter, teamFilter],
  );
  const categorical = metrics.some((metric) => metric.exploration?.kind === "categorical");
  const hybrid = metrics.some(
    (metric) => metric.exploration?.kind === "hybrid" || metric.exploration?.kind === "circular",
  );
  const circular = metrics.some((metric) => metric.exploration?.kind === "circular");
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
        {categorical ? (
          <TrainingChart title="CATEGORICAL EXPLORATION" data={data}>
            <Line dataKey="normalized_entropy" name="normalized entropy" stroke="#bd8cff" dot={false} />
            <Line dataKey="stop_fraction" name="stop" stroke="#82998f" dot={false} />
            <Line dataKey="navigate_fraction" name="navigate" stroke="#49a7ff" dot={false} />
            <Line dataKey="strike_fraction" name="strike" stroke="#ff8a62" dot={false} />
          </TrainingChart>
        ) : hybrid ? (
          <>
            <TrainingChart title="PARAMETRIC CONTROL · Δ HEADING" data={data}>
              <Line dataKey="direction_change_mean_degrees" name="mean Δ heading °" stroke="#49a7ff" dot={false} />
              <Line dataKey="direction_change_p95_degrees" name="p95 Δ heading °" stroke="#ff8a62" dot={false} />
            </TrainingChart>
            <TrainingChart title="PARAMETRIC CONTROL · INTENSITY" data={data}>
              <Line dataKey="mean_intensity" name="mean intensity" stroke="#71e1ae" dot={false} />
            </TrainingChart>
            {circular ? (
              <TrainingChart title="HEADING CONCENTRATION" data={data}>
                <Line dataKey="heading_concentration" name="concentration" stroke="#bd8cff" dot={false} />
              </TrainingChart>
            ) : null}
            <TrainingChart title="SKILL MIX" data={data}>
              <Line dataKey="stop_fraction" name="stop" stroke="#82998f" dot={false} />
              <Line dataKey="navigate_fraction" name="navigate" stroke="#49a7ff" dot={false} />
              <Line dataKey="strike_fraction" name="strike" stroke="#ff8a62" dot={false} />
            </TrainingChart>
            <TrainingChart title="EXPLORATION · LOG STD" data={data}>
              {circular ? (
                <Line dataKey="log_std_left" name="intensity" stroke="#ff8a62" dot={false} />
              ) : (
                <>
                  <Line dataKey="log_std_left" name="direction x" stroke="#71e1ae" dot={false} />
                  <Line dataKey="log_std_right" name="direction y" stroke="#49a7ff" dot={false} />
                  <Line dataKey="log_std_intensity" name="intensity" stroke="#ff8a62" dot={false} />
                </>
              )}
            </TrainingChart>
          </>
        ) : (
          <TrainingChart title="EXPLORATION · LOG STD" data={data}>
            <Line dataKey="log_std_left" name="left wheel" stroke="#71e1ae" dot={false} />
            <Line dataKey="log_std_right" name="right wheel" stroke="#49a7ff" dot={false} />
          </TrainingChart>
        )}
        <TrainingChart title="SEMANTIC OUTCOMES" data={data}>
          <Line dataKey="skill_success" name="success" stroke="#71e1ae" dot={false} />
          <Line dataKey="skill_failure" name="failure" stroke="#ff8a62" dot={false} />
          <Line dataKey="skill_unresolved" name="unresolved" stroke="#ffd84a" dot={false} />
        </TrainingChart>
        <TrainingChart title="SKILL SUCCESS RATE" data={data}>
          {SKILL_FAMILIES.map((family, index) => (
            <Line
              key={family}
              dataKey={`${family}_success`}
              name={family}
              stroke={[
                "#71e1ae",
                "#49a7ff",
                "#ffd84a",
                "#ff8a62",
                "#bd8cff",
                "#eaf4ef",
                "#49d6c8",
              ][index]}
              dot={false}
            />
          ))}
        </TrainingChart>
        <TrainingChart title="CURRICULUM ALLOCATION" data={data}>
          <Line dataKey="full_match_allocation" name="full match" stroke="#eaf4ef" dot={false} />
          <Line dataKey="frontier_allocation" name="frontier" stroke="#ff8a62" dot={false} />
          <Line dataKey="routine_allocation" name="routine" stroke="#49a7ff" dot={false} />
          <Line dataKey="failure_allocation" name="failure rehearsal" stroke="#ffd84a" dot={false} />
        </TrainingChart>
        <TrainingChart title="CURRICULUM PHASE" data={data}>
          <Line dataKey="curriculum_phase" name="phase index" stroke="#71e1ae" dot={false} />
        </TrainingChart>
      </div>
      <SemanticTrials
        trials={trials}
        familyFilter={familyFilter}
        teamFilter={teamFilter}
        outcomeFilter={outcomeFilter}
        difficultyFilter={difficultyFilter}
        setFamilyFilter={setFamilyFilter}
        setTeamFilter={setTeamFilter}
        setOutcomeFilter={setOutcomeFilter}
        setDifficultyFilter={setDifficultyFilter}
      />
    </div>
  );
}

type TimelineTrial = SemanticTrial & { iteration: number };

function SemanticTrials({
  trials,
  familyFilter,
  teamFilter,
  outcomeFilter,
  difficultyFilter,
  setFamilyFilter,
  setTeamFilter,
  setOutcomeFilter,
  setDifficultyFilter,
}: {
  trials: TimelineTrial[];
  familyFilter: string;
  teamFilter: string;
  outcomeFilter: string;
  difficultyFilter: string;
  setFamilyFilter: (value: string) => void;
  setTeamFilter: (value: string) => void;
  setOutcomeFilter: (value: string) => void;
  setDifficultyFilter: (value: string) => void;
}) {
  return (
    <section className="semantic-trials">
      <header>
        <div>
          <p className="eyebrow">SEMANTIC DRILL TIMELINE</p>
          <h2>Scenario outcomes</h2>
        </div>
        <span>{trials.length.toLocaleString()} matching trials</span>
      </header>
      <div className="semantic-filters">
        <Filter label="Family" value={familyFilter} setValue={setFamilyFilter}
          options={["all", ...SKILL_FAMILIES]} />
        <Filter label="Color" value={teamFilter} setValue={setTeamFilter}
          options={["all", "blue", "yellow"]} />
        <Filter label="Outcome" value={outcomeFilter} setValue={setOutcomeFilter}
          options={["all", "success", "failure", "unresolved"]} />
        <Filter label="Difficulty" value={difficultyFilter} setValue={setDifficultyFilter}
          options={["all", "beginner", "developing", "advanced"]} />
      </div>
      <div className="semantic-table" role="table" aria-label="Semantic skill outcomes">
        {trials.slice(-100).reverse().map((trial) => (
          <div className="semantic-row" role="row" key={`${trial.iteration}-${trial.scenario_id}`}>
            <span>ITER {trial.iteration.toString().padStart(6, "0")}</span>
            <strong>{trial.family.replace("_", " ").toUpperCase()}</strong>
            <span className={trial.controlled_team}>{trial.controlled_team.toUpperCase()}</span>
            <span>{difficultyMean(trial).toFixed(2)}</span>
            <span className={trial.status}>{trial.status.toUpperCase()}</span>
            <span>{trial.reason.replaceAll("_", " ")}</span>
            <span>{trial.steps} steps</span>
          </div>
        ))}
        {!trials.length ? <p>No semantic trials match these filters.</p> : null}
      </div>
    </section>
  );
}

function Filter({
  label,
  value,
  setValue,
  options,
}: {
  label: string;
  value: string;
  setValue: (value: string) => void;
  options: readonly string[];
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => setValue(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option.toUpperCase()}</option>
        ))}
      </select>
    </label>
  );
}

function difficultyMean(trial: SemanticTrial): number {
  const values = Object.values(trial.difficulty);
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function matchesDifficulty(trial: SemanticTrial, filter: string): boolean {
  const value = difficultyMean(trial);
  if (filter === "beginner") return value < 0.33;
  if (filter === "developing") return value >= 0.33 && value < 0.66;
  if (filter === "advanced") return value >= 0.66;
  return true;
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
