import { useMemo, useState } from "react";

import type { Replay, ReplayAnalytics } from "./types";

interface Props {
  replay: Replay;
  events: ReplayAnalytics["events"];
  frameIndex: number;
  selectedActor: number;
  onSelectActor: (index: number) => void;
  onSeek: (index: number) => void;
}

interface Segment {
  start: number;
  end: number;
  label: string;
  skill: string;
}

interface TimelineEvent {
  index: number;
  time: number;
  kind: string;
  level: number;
}

const KEY_EVENT_KINDS = new Set([
  "goal",
  "own_goal",
  "forced_own_goal",
  "save",
  "shot",
  "assist",
]);

export function PolicyTimeline({
  replay,
  events,
  frameIndex,
  selectedActor,
  onSelectActor,
  onSeek,
}: Props) {
  const [showLanes, setShowLanes] = useState(true);
  const lanes = useMemo(
    () => Array.from({ length: 6 }, (_, actor) => ({
      actor,
      segments: buildSegments(replay, actor),
      hasIntent: replay.frames.some((frame) => Boolean(frame.policy_intents?.[actor])),
    })).filter((lane) => lane.hasIntent),
    [replay],
  );
  const finalTime = replay.frames.at(-1)?.snapshot.simulation_time ?? 0;
  const keyEvents = useMemo(
    () => assignEventLevels(events
      .filter((event) => KEY_EVENT_KINDS.has(event.kind))
      .map((event) => ({
        index: nearestFrame(replay, event.time),
        time: event.time,
        kind: event.kind,
        level: 0,
      })), finalTime),
    [events, finalTime, replay],
  );
  return (
    <section className="policy-timeline" aria-label="Policy intent timeline">
      <div className="timeline-toolbar">
        <span>POLICY INTENT</span>
        <button
          type="button"
          aria-expanded={showLanes}
          onClick={() => setShowLanes((current) => !current)}
        >
          {showLanes ? "HIDE CHANNELS" : `SHOW CHANNELS · ${lanes.length}`}
        </button>
      </div>
      <div className="event-rail">
        {keyEvents.map((event, eventIndex) => (
          <button
            className={`event-mark event-${event.kind.replaceAll("_", "-")}`}
            key={`${event.kind}-${event.time}-${eventIndex}`}
            style={{
              left: `${percentage(event.time, finalTime)}%`,
              bottom: `${event.level * 13 - 6}px`,
            }}
            title={`${event.time.toFixed(2)}s · ${event.kind.replaceAll("_", " ")}`}
            aria-label={`Seek to ${event.kind} at ${event.time.toFixed(2)} seconds`}
            onClick={() => onSeek(event.index)}
          >
            <i aria-hidden="true">{eventIcon(event.kind)}</i>
          </button>
        ))}
        {!keyEvents.length ? <span className="no-key-events">NO KEY EVENTS</span> : null}
      </div>
      {showLanes ? <div className="intent-lanes">
        {lanes.map(({ actor, segments }) => (
          <div
            className={`intent-lane ${selectedActor === actor ? "selected" : ""}`}
            key={actor}
          >
            <button className="lane-label" onClick={() => onSelectActor(actor)}>
              {actor >= 3 ? "Y" : "B"}{actor % 3}
            </button>
            <div className="lane-track">
              {segments.map((segment) => (
                <button
                  className={`intent-segment skill-${segment.skill}`}
                  key={`${actor}-${segment.start}`}
                  style={{
                    left: `${100 * segment.start / replay.frames.length}%`,
                    width: `${100 * (segment.end - segment.start + 1) / replay.frames.length}%`,
                  }}
                  title={`${segment.label} · frames ${segment.start + 1}–${segment.end + 1}`}
                  onClick={() => {
                    onSelectActor(actor);
                    onSeek(segment.start);
                  }}
                >
                  <span>{segment.label}</span>
                </button>
              ))}
              <i
                className="timeline-playhead"
                style={{ left: `${100 * frameIndex / Math.max(1, replay.frames.length - 1)}%` }}
              />
            </div>
          </div>
        ))}
        {!lanes.length ? (
          <div className="timeline-empty">
            This replay predates policy-intent telemetry · start a new M24 run
          </div>
        ) : null}
      </div> : null}
    </section>
  );
}

function assignEventLevels(events: TimelineEvent[], finalTime: number): TimelineEvent[] {
  const lastPosition = [-Infinity, -Infinity, -Infinity];
  return [...events].sort((first, second) => first.time - second.time).map((event) => {
    const position = percentage(event.time, finalTime);
    const available = lastPosition.findIndex((previous) => position - previous >= 1.5);
    const level = available >= 0 ? available : 2;
    lastPosition[level] = position;
    return { ...event, level };
  });
}

function eventIcon(kind: string): string {
  if (kind.includes("goal")) return "G";
  if (kind === "shot") return "↗";
  if (kind === "save") return "S";
  if (kind === "assist") return "A";
  return "·";
}

function buildSegments(replay: Replay, actor: number): Segment[] {
  const result: Segment[] = [];
  replay.frames.forEach((frame, index) => {
    const intent = frame.policy_intents?.[actor];
    const label = intent
      ? intent.skill === "stop"
        ? "STOP"
        : `${intent.skill === "navigate" ? "NAV" : "STR"}-${intent.direction}`
      : "WHEEL";
    const skill = intent?.skill ?? "legacy";
    const current = result.at(-1);
    if (current?.label === label) current.end = index;
    else result.push({ start: index, end: index, label, skill });
  });
  if (result.length <= 400) return result;
  const bucketSize = Math.ceil(replay.frames.length / 400);
  const compressed: Segment[] = [];
  for (let start = 0; start < replay.frames.length; start += bucketSize) {
    const end = Math.min(replay.frames.length - 1, start + bucketSize - 1);
    const middle = Math.floor((start + end) / 2);
    const intent = replay.frames[middle]?.policy_intents?.[actor];
    const label = intent
      ? intent.skill === "stop"
        ? "STOP"
        : `${intent.skill === "navigate" ? "NAV" : "STR"}-${intent.direction}`
      : "WHEEL";
    const skill = intent?.skill ?? "legacy";
    const current = compressed.at(-1);
    if (current?.label === label) current.end = end;
    else compressed.push({ start, end, label, skill });
  }
  return compressed;
}

function nearestFrame(replay: Replay, time: number): number {
  const index = replay.frames.findIndex((frame) => frame.snapshot.simulation_time >= time);
  return index >= 0 ? index : Math.max(0, replay.frames.length - 1);
}

function percentage(time: number, finalTime: number): number {
  return finalTime > 0 ? Math.min(100, Math.max(0, 100 * time / finalTime)) : 0;
}
