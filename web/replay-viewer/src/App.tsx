import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { FieldCanvas } from "./FieldCanvas";
import { clampedFrame, frameLabel, parseReplay } from "./replay";
import type { Replay, ReplayIndex } from "./types";

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
const POLL_INTERVAL_MS = 2_000;
const ICONS = {
  back: "↶",
  previous: "‹",
  play: "▶",
  pause: "Ⅱ",
  next: "›",
  forward: "↷",
};

export default function App() {
  const [historicalSelection, setHistoricalSelection] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [followLatest, setFollowLatest] = useState(true);
  const [loop, setLoop] = useState(true);
  const [filter, setFilter] = useState("all");
  const frameRef = useRef(0);

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
    staleTime: Number.POSITIVE_INFINITY,
  });
  const replay = replayQuery.data ?? null;
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
  const iteration = index?.replays.find((item) => item.filename === selected)?.iteration;
  const selectedInfo = index?.replays.find((item) => item.filename === selected);
  const latestReplayIteration = index?.replays.at(-1)?.iteration;
  const latestCheckpoint = index?.checkpoints.at(-1);
  const reward = useMemo(
    () => frame?.rewards.reduce((sum, value) => sum + value, 0) ?? 0,
    [frame],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true"><i /><i /><i /></div>
        <div>
          <p className="eyebrow">VSSS LAB · REPLAY STUDIO</p>
          <h1>Training run explorer</h1>
        </div>
        <div className="run-state">
          <span className="pulse" />
          <div>
            <small>{followLatest ? "LIVE · POLLING 2S" : "HISTORY MODE"}</small>
            <strong>checkpoint {latestCheckpoint?.iteration ?? "—"} · replay {latestReplayIteration ?? "—"}</strong>
          </div>
        </div>
      </header>

      <section className="workspace">
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

          <div className="section-rule" />
          <p className="side-heading">POLICY MATCHUP</p>
          <Policy team="BLUE" value={replay?.header.policies.blue} />
          <Policy team="YELLOW" value={replay?.header.policies.yellow} yellow />
          <div className="section-rule" />
          <dl className="details">
            <div><dt>Iteration</dt><dd>{iteration ?? "—"}</dd></div>
            <div><dt>Result</dt><dd>{selectedInfo?.outcome.toUpperCase() ?? "—"} {selectedInfo ? `${selectedInfo.score_blue}–${selectedInfo.score_yellow}` : ""}</dd></div>
            <div><dt>Frames</dt><dd>{replay?.frames.length.toLocaleString() ?? "—"}</dd></div>
            <div><dt>Event mask</dt><dd>{frame?.events ?? "—"}</dd></div>
            <div><dt>Σ reward</dt><dd>{reward.toFixed(4)}</dd></div>
            <div><dt>Train return</dt><dd>{index?.latest_metric?.return_total.toFixed(3) ?? "—"}</dd></div>
            <div><dt>Progress</dt><dd>{index?.latest_metric?.progress.toFixed(3) ?? "—"}</dd></div>
          </dl>
        </aside>

        <section className="stage">
          {error ? <div className="empty-state"><strong>Replay unavailable</strong><span>{String(error)}</span></div> : null}
          {loading ? <div className="loading">Loading recorded frames…</div> : null}
          {replay && frame && !error ? <FieldCanvas header={replay.header} frame={frame} /> : null}
          <span className="recorded-badge">{followLatest ? "● LIVE INSPECT" : "● RECORDED"}</span>
        </section>

        <ActorTelemetry
          actions={frame?.actions}
          robots={frame?.snapshot.robots}
          maxWheelSpeed={replay?.header.config.max_wheel_speed ?? 1}
          wheelRadius={replay?.header.config.wheel?.radius ?? 0.025}
          axleTrack={replay?.header.config.wheel?.axle_track ?? 0.06}
        />
      </section>

      <footer className="transport">
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
      </footer>
    </main>
  );
}

function Metric({ label, value, tone = "" }: { label: string; value: string | number; tone?: string }) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function Policy({ team, value, yellow = false }: { team: string; value?: string; yellow?: boolean }) {
  return <div className="policy"><span className={yellow ? "dot yellow" : "dot"} /><div><small>{team}</small><strong>{value ?? "—"}</strong></div></div>;
}

function Control({ label, icon, onClick }: { label: string; icon: string; onClick: () => void }) {
  return <button className="control-button" aria-label={label} onClick={onClick}>{icon}</button>;
}

function ActorTelemetry({
  actions,
  robots,
  maxWheelSpeed,
  wheelRadius,
  axleTrack,
}: {
  actions?: number[][];
  robots?: Replay["frames"][number]["snapshot"]["robots"];
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
          const direction = Math.abs(linear) < 0.03 && Math.abs(angular) > 0.2
            ? angular > 0 ? "TURN LEFT" : "TURN RIGHT"
            : linear > 0.03 ? "FORWARD"
            : linear < -0.03 ? "REVERSE"
            : "IDLE";
          return (
            <article className="actor-card" key={index}>
              <div className="actor-title">
                <span className={index >= 3 ? "dot yellow" : "dot"} />
                <strong>{index >= 3 ? "Y" : "B"}{index % 3}</strong>
                <em>{direction}</em>
              </div>
              <div className="throttle"><i style={{ width: `${intensity * 100}%` }} /></div>
              <dl>
                <div><dt>CMD L/R</dt><dd>{commandLeft.toFixed(1)} / {commandRight.toFixed(1)}</dd></div>
                <div><dt>APPLIED</dt><dd>{left.toFixed(1)} / {right.toFixed(1)} rad/s</dd></div>
                <div><dt>LINEAR</dt><dd>{linear.toFixed(2)} m/s</dd></div>
                <div><dt>TURN</dt><dd>{angular.toFixed(2)} rad/s</dd></div>
              </dl>
            </article>
          );
        })}
      </div>
    </aside>
  );
}
