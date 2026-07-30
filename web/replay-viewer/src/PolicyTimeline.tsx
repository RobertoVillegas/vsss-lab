import { useMemo } from "react";

import type { Replay, ReplayAnalytics } from "./types";

interface Props {
  replay: Replay;
  events: ReplayAnalytics["events"];
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

export function PolicyTimeline({
  replay,
  events,
  selectedActor,
  onSelectActor,
  onSeek,
}: Props) {
  const lanes = useMemo(
    () => Array.from({ length: 6 }, (_, actor) => buildSegments(replay, actor)),
    [replay],
  );
  const finalTime = replay.frames.at(-1)?.snapshot.simulation_time ?? 0;
  const resetMarkers = useMemo(
    () => replay.frames.flatMap((frame, index) => (
      index > 0 && frame.episode !== replay.frames[index - 1]?.episode
        ? [{ index, time: frame.snapshot.simulation_time, kind: "episode reset" }]
        : []
    )),
    [replay],
  );
  return (
    <section className="policy-timeline" aria-label="Policy intent timeline">
      <div className="event-rail">
        {[...events.map((event) => ({
          index: nearestFrame(replay, event.time),
          time: event.time,
          kind: event.kind,
        })), ...resetMarkers].map((event, index) => (
          <button
            className={`event-mark event-${event.kind.replaceAll("_", "-")}`}
            key={`${event.kind}-${event.time}-${index}`}
            style={{ left: `${percentage(event.time, finalTime)}%` }}
            title={`${event.time.toFixed(2)}s · ${event.kind}`}
            aria-label={`Seek to ${event.kind} at ${event.time.toFixed(2)} seconds`}
            onClick={() => onSeek(event.index)}
          >
            <i />
          </button>
        ))}
      </div>
      <div className="intent-lanes">
        {lanes.map((segments, actor) => (
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
            </div>
          </div>
        ))}
      </div>
    </section>
  );
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
