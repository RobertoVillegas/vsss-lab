import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { FieldCanvas } from "./FieldCanvas";
import { clampedFrame, frameLabel, parseReplay } from "./replay";
import type { Replay, ReplayIndex } from "./types";

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const ICONS = {
  back: "↶",
  previous: "‹",
  play: "▶",
  pause: "Ⅱ",
  next: "›",
  forward: "↷",
};

export default function App() {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  const [selected, setSelected] = useState("");
  const [replay, setReplay] = useState<Replay | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const frameRef = useRef(0);

  useEffect(() => {
    fetch("/api/iterations")
      .then((response) => {
        if (!response.ok) throw new Error("Could not discover replay iterations.");
        return response.json() as Promise<ReplayIndex>;
      })
      .then((value) => {
        setIndex(value);
        if (value.replays.length) setSelected(value.replays.at(-1)!.filename);
        else setError("This run has no captured iteration replays.");
      })
      .catch((reason: unknown) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setPlaying(false);
    fetch(`/api/replays/${encodeURIComponent(selected)}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load ${selected}.`);
        return response.text();
      })
      .then((text) => {
        setReplay(parseReplay(text));
        setFrameIndex(0);
        setError("");
      })
      .catch((reason: unknown) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, [selected]);

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
        const next = Math.min(replay.frames.length - 1, frameRef.current + advance);
        setFrameIndex(next);
        if (next === replay.frames.length - 1) setPlaying(false);
      }
      animation = requestAnimationFrame(animate);
    };
    animation = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animation);
  }, [playing, replay, speed]);

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
          <div><small>LOCAL CAPTURE</small><strong>{index?.replays.length ?? 0} iterations</strong></div>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <label htmlFor="iteration">Captured iteration</label>
          <select
            id="iteration"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
            disabled={!index?.replays.length}
          >
            {index?.replays.map((item) => (
              <option key={item.filename} value={item.filename}>
                Iteration {item.iteration.toString().padStart(4, "0")}
              </option>
            ))}
          </select>

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
            <div><dt>Frames</dt><dd>{replay?.frames.length.toLocaleString() ?? "—"}</dd></div>
            <div><dt>Event mask</dt><dd>{frame?.events ?? "—"}</dd></div>
            <div><dt>Σ reward</dt><dd>{reward.toFixed(4)}</dd></div>
          </dl>
        </aside>

        <section className="stage">
          {error ? <div className="empty-state"><strong>Replay unavailable</strong><span>{error}</span></div> : null}
          {loading ? <div className="loading">Loading recorded frames…</div> : null}
          {replay && frame && !error ? <FieldCanvas header={replay.header} frame={frame} /> : null}
          <span className="recorded-badge">● RECORDED</span>
        </section>
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
          <label className="speed">
            SPEED
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              {SPEEDS.map((value) => <option key={value} value={value}>{value}×</option>)}
            </select>
          </label>
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
