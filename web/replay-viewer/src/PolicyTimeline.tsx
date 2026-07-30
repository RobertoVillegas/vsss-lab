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
}

export function PolicyTimeline({
  replay,
  events,
  frameIndex,
  selectedActor,
  onSelectActor,
  onSeek,
}: Props) {
  const [showLanes, setShowLanes] = useState(true);
  const [clusterPositions, setClusterPositions] = useState<Record<number, number>>({});
  const lanes = useMemo(
    () => Array.from({ length: 6 }, (_, actor) => ({
      actor,
      segments: buildSegments(replay, actor),
      hasIntent: replay.frames.some((frame) => Boolean(frame.policy_intents?.[actor])),
    })).filter((lane) => lane.hasIntent),
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
  const eventClusters = useMemo(
    () => clusterEvents([
      ...events.map((event) => ({
        index: nearestFrame(replay, event.time),
        time: event.time,
        kind: event.kind,
      })),
      ...resetMarkers,
    ], finalTime),
    [events, finalTime, replay, resetMarkers],
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
        {eventClusters.map((cluster, clusterIndex) => {
          const position = clusterPositions[clusterIndex] ?? 0;
          const event = cluster.events[position % cluster.events.length]!;
          const kinds = [...new Set(cluster.events.map((item) => item.kind))];
          return (
          <button
            className={`event-mark event-${event.kind.replaceAll("_", "-")}`}
            key={`${cluster.start}-${clusterIndex}`}
            style={{ left: `${cluster.position}%` }}
            title={`${event.time.toFixed(2)}s · ${event.kind}${
              cluster.events.length > 1 ? ` · ${position + 1}/${cluster.events.length}, click to cycle` : ""
            }`}
            aria-label={`Seek to ${event.kind} at ${event.time.toFixed(2)} seconds${
              cluster.events.length > 1 ? `; ${cluster.events.length} nearby events` : ""
            }`}
            onClick={() => {
              onSeek(event.index);
              if (cluster.events.length > 1) {
                setClusterPositions((current) => ({
                  ...current,
                  [clusterIndex]: (position + 1) % cluster.events.length,
                }));
              }
            }}
          >
            <i aria-hidden="true">{eventIcon(kinds)}</i>
            {cluster.events.length > 1 ? <b>{cluster.events.length}</b> : null}
          </button>
          );
        })}
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

function clusterEvents(events: TimelineEvent[], finalTime: number) {
  const clusters: Array<{
    start: number;
    position: number;
    events: TimelineEvent[];
  }> = [];
  for (const event of [...events].sort((first, second) => first.time - second.time)) {
    const position = percentage(event.time, finalTime);
    const current = clusters.at(-1);
    if (current && position - current.position < 1.5) {
      current.events.push(event);
      current.position = current.events.reduce(
        (sum, item) => sum + percentage(item.time, finalTime),
        0,
      ) / current.events.length;
    } else {
      clusters.push({ start: event.time, position, events: [event] });
    }
  }
  return clusters;
}

function eventIcon(kinds: string[]): string {
  if (kinds.length > 1) return "＋";
  const kind = kinds[0] ?? "";
  if (kind.includes("goal") || kind === "shot") return "●";
  if (kind === "touch") return "×";
  if (kind === "pass") return "→";
  if (kind === "interception" || kind === "save") return "◇";
  if (kind === "episode reset") return "↻";
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
