import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { exportReplay } from "./exportReplay";
import type { ExportFormat, ExportProgress } from "./exportReplay";
import { FieldCanvas } from "./FieldCanvas";
import { PolicyTimeline } from "./PolicyTimeline";
import { clampedFrame, frameLabel, parseReplay } from "./replay";
import type { MetricHistory, Replay, ReplayAnalytics, ReplayIndex } from "./types";

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
const POLL_INTERVAL_MS = 2_000;
const TrainingCharts = lazy(() => import("./TrainingCharts"));
const ICONS = {
  back: "↶",
  previous: "‹",
  play: "▶",
  pause: "Ⅱ",
  next: "›",
  forward: "↷",
};

export default function App() {
  const [activeView, setActiveView] = useState<"replay" | "metrics">("metrics");
  const [historicalSelection, setHistoricalSelection] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [followLatest, setFollowLatest] = useState(true);
  const [loop, setLoop] = useState(true);
  const [filter, setFilter] = useState("all");
  const [selectedActor, setSelectedActor] = useState(0);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [exportError, setExportError] = useState("");
  const frameRef = useRef(0);
  const fieldCanvasRef = useRef<HTMLCanvasElement>(null);

  const indexQuery = useQuery({
    queryKey: ["training-run-index"],
    queryFn: async (): Promise<ReplayIndex> => {
      const response = await fetch("/api/iterations");
      if (!response.ok) throw new Error("Could not discover training iterations.");
      return response.json() as Promise<ReplayIndex>;
    },
    refetchInterval: POLL_INTERVAL_MS,
  });
  const index = indexQuery.data;
  const metricsQuery = useQuery({
    queryKey: ["training-metrics"],
    queryFn: async (): Promise<MetricHistory> => {
      const response = await fetch("/api/metrics");
      if (!response.ok) throw new Error("Could not load training metrics.");
      return response.json() as Promise<MetricHistory>;
    },
    enabled: activeView === "metrics",
    refetchInterval: 5_000,
  });
  const visibleReplays = useMemo(
    () => index?.replays.filter((item) => (
      filter === "all"
      || item.outcome === filter
      || filter === "goals" && item.goals > 0
    )) ?? [],
    [filter, index?.replays],
  );
  const latestFilename = visibleReplays.at(-1)?.filename ?? "";
  const selected = followLatest ? latestFilename : historicalSelection;
  const replayQuery = useQuery({
    queryKey: ["replay", selected],
    enabled: Boolean(selected),
    queryFn: async (): Promise<Replay> => {
      const response = await fetch(`/api/replays/${encodeURIComponent(selected)}`);
      if (!response.ok) throw new Error(`Could not load ${selected}.`);
      return parseReplay(await response.text());
    },
    retry: 10,
    retryDelay: (attempt) => Math.min(500 * 2 ** attempt, POLL_INTERVAL_MS),
    staleTime: Number.POSITIVE_INFINITY,
  });
  const replay = replayQuery.data ?? null;
  const analyticsQuery = useQuery({
    queryKey: ["replay-analytics", selected],
    enabled: Boolean(selected) && activeView === "replay",
    queryFn: async (): Promise<ReplayAnalytics> => {
      const response = await fetch(
        `/api/replays/${encodeURIComponent(selected)}/analytics`,
      );
      if (!response.ok) throw new Error(`Could not analyze ${selected}.`);
      return response.json() as Promise<ReplayAnalytics>;
    },
    staleTime: Number.POSITIVE_INFINITY,
  });
  const analytics = analyticsQuery.data;
  const error = indexQuery.error ?? replayQuery.error;
  const loading = indexQuery.isPending || replayQuery.isFetching;

  useEffect(() => {
    if (!replay) return;
    setFrameIndex(0);
    setPlaying(followLatest);
  }, [followLatest, replay]);

  useEffect(() => {
    frameRef.current = frameIndex;
  }, [frameIndex]);

  useEffect(() => {
    if (!playing || !replay) return;
    let animation = 0;
    let previous = performance.now();
    let accumulated = 0;
    const period = replay.header.config.control_period * 1000;
    const animate = (now: number) => {
      accumulated += (now - previous) * speed;
      previous = now;
      const advance = Math.floor(accumulated / period);
      if (advance > 0) {
        accumulated -= advance * period;
        const requested = frameRef.current + advance;
        const next = loop
          ? requested % replay.frames.length
          : Math.min(replay.frames.length - 1, requested);
        setFrameIndex(next);
        if (!loop && next === replay.frames.length - 1) setPlaying(false);
      }
      animation = requestAnimationFrame(animate);
    };
    animation = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animation);
  }, [loop, playing, replay, speed]);

  const move = useCallback(
    (delta: number) => {
      if (!replay) return;
      setPlaying(false);
      setFrameIndex((current) => clampedFrame(current, delta, replay.frames.length));
    },
    [replay],
  );

  const beginExport = useCallback(
    async (format: ExportFormat) => {
      const canvas = fieldCanvasRef.current;
      if (!canvas || !replay || exportProgress) return;
      const originalFrame = frameRef.current;
      setPlaying(false);
      setExportError("");
      setExportProgress({ completed: 0, total: 1, format });
      try {
        await exportReplay(format, {
          canvas,
          replay,
          speed,
          seek: async (requested) => {
            setFrameIndex(requested);
            await nextPaint();
          },
          onProgress: setExportProgress,
        });
      } catch (cause) {
        setExportError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setFrameIndex(originalFrame);
        setExportProgress(null);
      }
    },
    [exportProgress, replay, speed],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!replay || event.target instanceof HTMLSelectElement) return;
      if (event.code === "Space") {
        event.preventDefault();
        setPlaying((value) => !value);
      } else if (event.code === "ArrowLeft") move(event.shiftKey ? -100 : -1);
      else if (event.code === "ArrowRight") move(event.shiftKey ? 100 : 1);
      else if (event.code === "Home") move(-replay.frames.length);
      else if (event.code === "End") move(replay.frames.length);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move, replay]);

  const frame = replay?.frames[frameIndex];
  const latestReplayIteration = index?.replays.at(-1)?.iteration;
  const latestCheckpoint = index?.checkpoints.at(-1);
  const selectedIntent = frame?.policy_intents?.[selectedActor];
  const closestRobot = useMemo(() => {
    if (!frame) return null;
    return frame.snapshot.robots
      .map((robot, actor) => ({ robot, actor }))
      .filter(({ robot }) => robot.enabled)
      .map(({ robot, actor }) => ({
        actor,
        distance: Math.hypot(
          robot.pose.x - frame.snapshot.ball.x,
          robot.pose.y - frame.snapshot.ball.y,
        ),
      }))
      .sort((first, second) => first.distance - second.distance)[0] ?? null;
  }, [frame]);
  const recentChanges = useMemo(() => {
    if (!replay) return 0;
    const window = replay.frames.slice(Math.max(0, frameIndex - 49), frameIndex + 1);
    let changes = 0;
    let previous: number | undefined;
    for (const candidate of window) {
      const action = candidate.policy_intents?.[selectedActor]?.action_index;
      if (action !== undefined && previous !== undefined && action !== previous) changes += 1;
      previous = action;
    }
    return changes;
  }, [frameIndex, replay, selectedActor]);
  return (
    <main className={`app-shell ${activeView === "metrics" ? "metrics-mode" : ""}`}>
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true"><i /><i /><i /></div>
        <div>
          <p className="eyebrow">VSSS LAB · REPLAY STUDIO</p>
          <h1>Training run explorer</h1>
        </div>
        <nav className="view-tabs" aria-label="Viewer mode">
          <button
            className={activeView === "replay" ? "active" : ""}
            onClick={() => setActiveView("replay")}
          >
            REPLAY
          </button>
          <button
            className={activeView === "metrics" ? "active" : ""}
            onClick={() => setActiveView("metrics")}
          >
            TRAINING METRICS
          </button>
        </nav>
        <div className="run-state">
          <span className="pulse" />
          <div>
            <small>{followLatest ? "LIVE · POLLING 2S" : "HISTORY MODE"}</small>
            <strong>checkpoint {latestCheckpoint?.iteration ?? "—"} · replay {latestReplayIteration ?? "—"}</strong>
          </div>
        </div>
      </header>

      <section className={`workspace ${activeView === "metrics" ? "metrics-workspace" : ""}`}>
        <aside className="sidebar">
          <label htmlFor="iteration">Captured iteration</label>
          <select
            id="iteration"
            value={selected}
            onChange={(event) => {
              setFollowLatest(false);
              setHistoricalSelection(event.target.value);
            }}
            disabled={!index?.replays.length}
          >
            {visibleReplays.map((item) => (
              <option key={item.filename} value={item.filename}>
                {item.outcome.toUpperCase()} · {item.score_blue}–{item.score_yellow} · Iter {item.iteration.toString().padStart(4, "0")}
              </option>
            ))}
          </select>
          <label htmlFor="result-filter">Replay filter</label>
          <select
            id="result-filter"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="all">All results</option>
            <option value="win">Wins</option>
            <option value="loss">Losses</option>
            <option value="draw">Draws</option>
            <option value="goals">Any goal</option>
          </select>
          <button
            className={`follow-button ${followLatest ? "active" : ""}`}
            onClick={() => {
              setFollowLatest(true);
            }}
          >
            {followLatest ? "● Following latest" : "Follow latest"}
          </button>

          <div className="metric-grid">
            <Metric label="Tick" value={frame?.snapshot.tick ?? "—"} />
            <Metric label="Sim time" value={frame ? `${frame.snapshot.simulation_time.toFixed(2)}s` : "—"} />
            <Metric label="Blue" value={frame?.snapshot.score_blue ?? "—"} tone="blue" />
            <Metric label="Yellow" value={frame?.snapshot.score_yellow ?? "—"} tone="yellow" />
          </div>
          <section className="quick-insights">
            <p className="side-heading">AT A GLANCE</p>
            <dl className="details">
              <div>
                <dt>Selected</dt>
                <dd>{selectedIntent
                  ? `${selectedIntent.skill.toUpperCase()} ${selectedIntent.direction ?? ""}`
                  : "LEGACY"}</dd>
              </div>
              <div>
                <dt>Decision changes</dt>
                <dd>{selectedIntent ? `${recentChanges} / recent 1s` : "—"}</dd>
              </div>
              <div>
                <dt>Closest to ball</dt>
                <dd>{closestRobot
                  ? `${closestRobot.actor >= 3 ? "Y" : "B"}${closestRobot.actor % 3} · ${closestRobot.distance.toFixed(2)}m`
                  : "—"}</dd>
              </div>
              <div>
                <dt>Ball region</dt>
                <dd>{frame
                  ? Math.abs(frame.snapshot.ball.x) < 0.15
                    ? "MIDFIELD"
                    : frame.snapshot.ball.x < 0 ? "BLUE HALF" : "YELLOW HALF"
                  : "—"}</dd>
              </div>
            </dl>
          </section>
        </aside>

        <section className={`stage ${activeView === "metrics" ? "metrics-stage" : ""}`}>
          {error ? <div className="empty-state"><strong>Replay unavailable</strong><span>{String(error)}</span></div> : null}
          {activeView === "replay" && replayQuery.isPending && replayQuery.failureCount > 0 ? (
            <div className="loading">Replay is still being recorded · retrying…</div>
          ) : null}
          {activeView === "replay" && loading ? <div className="loading">Loading recorded frames…</div> : null}
          {activeView === "replay" && replay && frame && !error ? (
            <FieldCanvas
              ref={fieldCanvasRef}
              header={replay.header}
              frame={frame}
              layers={{ truth: true, measured: false, estimated: false, predicted: true }}
              selectedActor={selectedActor}
            />
          ) : null}
          {activeView === "metrics" ? (
            <Suspense fallback={<div className="loading">Loading chart engine…</div>}>
              <TrainingCharts metrics={metricsQuery.data?.metrics ?? []} />
            </Suspense>
          ) : (
            <span className="recorded-badge">{followLatest ? "● LIVE INSPECT" : "● RECORDED"}</span>
          )}
        </section>

        <ActorTelemetry
          actions={frame?.actions}
          robots={frame?.snapshot.robots}
          roles={frame?.roles}
          roleChanges={frame?.role_changes}
          intents={frame?.policy_intents}
          selectedActor={selectedActor}
          onSelectActor={setSelectedActor}
          maxWheelSpeed={replay?.header.config.max_wheel_speed ?? 1}
          wheelRadius={replay?.header.config.wheel?.radius ?? 0.025}
          axleTrack={replay?.header.config.wheel?.axle_track ?? 0.06}
        />
      </section>

      {activeView === "replay" ? <footer className="transport">
        {replay ? (
          <PolicyTimeline
            replay={replay}
            events={analytics?.events ?? []}
            frameIndex={frameIndex}
            selectedActor={selectedActor}
            onSelectActor={setSelectedActor}
            onSeek={(target) => {
              setPlaying(false);
              setFrameIndex(target);
            }}
          />
        ) : null}
        <div className="timeline-row">
          <span>{frameLabel(frameIndex, replay?.frames.length ?? 0)}</span>
          <input
            aria-label="Replay timeline"
            type="range"
            min="0"
            max={Math.max(0, (replay?.frames.length ?? 1) - 1)}
            value={frameIndex}
            onChange={(event) => {
              setPlaying(false);
              setFrameIndex(Number(event.target.value));
            }}
          />
          <span>{replay ? `${(replay.frames.at(-1)?.snapshot.simulation_time ?? 0).toFixed(2)}s` : "0.00s"}</span>
        </div>
        <div className="transport-row">
          <div className="key-hint"><kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> step · <kbd>Shift</kbd> skip</div>
          <div className="buttons">
            <Control label="Back 100 frames" icon={ICONS.back} onClick={() => move(-100)} />
            <Control label="Previous frame" icon={ICONS.previous} onClick={() => move(-1)} />
            <button
              className="play-button"
              aria-label={playing ? "Pause" : "Play"}
              disabled={!replay}
              onClick={() => {
                if (replay && frameIndex === replay.frames.length - 1) setFrameIndex(0);
                setPlaying((value) => !value);
              }}
            >
              {playing ? ICONS.pause : ICONS.play}
            </button>
            <Control label="Next frame" icon={ICONS.next} onClick={() => move(1)} />
            <Control label="Forward 100 frames" icon={ICONS.forward} onClick={() => move(100)} />
          </div>
          <div className="transport-options">
            <div className="export-controls">
              <button
                disabled={!replay || Boolean(exportProgress)}
                onClick={() => void beginExport("webm")}
              >
                VIDEO
              </button>
              <button
                disabled={!replay || Boolean(exportProgress)}
                onClick={() => void beginExport("gif")}
              >
                GIF
              </button>
              {exportProgress ? (
                <span>
                  {exportProgress.format.toUpperCase()}{" "}
                  {Math.round(100 * exportProgress.completed / exportProgress.total)}%
                </span>
              ) : exportError ? <span className="export-error">{exportError}</span> : null}
            </div>
            <label className="loop-toggle">
              <input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} />
              LOOP
            </label>
            <label className="speed">
              SPEED
              <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
                {SPEEDS.map((value) => <option key={value} value={value}>{value}×</option>)}
              </select>
            </label>
          </div>
        </div>
      </footer> : null}
    </main>
  );
}

function Metric({ label, value, tone = "" }: { label: string; value: string | number; tone?: string }) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function Control({ label, icon, onClick }: { label: string; icon: string; onClick: () => void }) {
  return <button className="control-button" aria-label={label} onClick={onClick}>{icon}</button>;
}

async function nextPaint(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function ActorTelemetry({
  actions,
  robots,
  roles,
  roleChanges,
  intents,
  selectedActor,
  onSelectActor,
  maxWheelSpeed,
  wheelRadius,
  axleTrack,
}: {
  actions?: number[][];
  robots?: Replay["frames"][number]["snapshot"]["robots"];
  roles?: Replay["frames"][number]["roles"];
  roleChanges?: Replay["frames"][number]["role_changes"];
  intents?: Replay["frames"][number]["policy_intents"];
  selectedActor: number;
  onSelectActor: (index: number) => void;
  maxWheelSpeed: number;
  wheelRadius: number;
  axleTrack: number;
}) {
  return (
    <aside className="telemetry">
      <p className="side-heading">ACTOR CONTROL · LIVE FRAME</p>
      <div className="actor-list">
        {Array.from({ length: 6 }, (_, index) => {
          const commandLeft = actions?.[index]?.[0] ?? 0;
          const commandRight = actions?.[index]?.[1] ?? 0;
          const left = robots?.[index]?.wheel_speed_left ?? 0;
          const right = robots?.[index]?.wheel_speed_right ?? 0;
          const linear = wheelRadius * (left + right) / 2;
          const angular = wheelRadius * (right - left) / axleTrack;
          const intensity = Math.min(1, Math.max(Math.abs(left), Math.abs(right)) / maxWheelSpeed);
          const intent = intents?.[index];
          const selected = selectedActor === index;
          const direction = Math.abs(linear) < 0.03 && Math.abs(angular) > 0.2
            ? angular > 0 ? "TURN LEFT" : "TURN RIGHT"
            : linear > 0.03 ? "FORWARD"
            : linear < -0.03 ? "REVERSE"
            : "IDLE";
          return (
            <button
              className={`actor-card ${selected ? "selected" : ""}`}
              key={index}
              onClick={() => onSelectActor(index)}
              aria-expanded={selected}
            >
              <div className="actor-title">
                <span className={index >= 3 ? "dot yellow" : "dot"} />
                <strong>{index >= 3 ? "Y" : "B"}{index % 3}</strong>
                <em>{roles?.[index]?.toUpperCase() ?? direction}{roleChanges?.[index] ? " ↻" : ""}</em>
              </div>
              <div className="primitive-summary">
                <strong>
                  {intent
                    ? intent.skill === "stop"
                      ? "STOP"
                      : `${intent.skill.toUpperCase()} · ${intent.direction}`
                    : direction}
                </strong>
                <span>{intent ? `${(intent.confidence * 100).toFixed(1)}%` : "LEGACY"}</span>
              </div>
              <div className="throttle"><i style={{ width: `${intensity * 100}%` }} /></div>
              <dl className={selected ? "expanded" : ""}>
                {selected && intent ? (
                  <>
                    <div><dt>PHASE</dt><dd>{intent.phase.toUpperCase()}</dd></div>
                    <div><dt>BALL DIST</dt><dd>{intent.ball_distance.toFixed(3)} m</dd></div>
                    <div><dt>TARGET</dt><dd>{intent.target.x.toFixed(2)}, {intent.target.y.toFixed(2)}</dd></div>
                  </>
                ) : null}
                <div><dt>CMD L/R</dt><dd>{commandLeft.toFixed(1)} / {commandRight.toFixed(1)}</dd></div>
                <div><dt>APPLIED</dt><dd>{left.toFixed(1)} / {right.toFixed(1)} rad/s</dd></div>
                <div><dt>LINEAR</dt><dd>{linear.toFixed(2)} m/s</dd></div>
                <div><dt>TURN</dt><dd>{angular.toFixed(2)} rad/s</dd></div>
              </dl>
              {selected && intent ? (
                <div className="top-actions">
                  {intent.top_actions.map((candidate) => (
                    <span key={candidate.action_index}>
                      {candidate.label} {(candidate.probability * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              ) : null}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
