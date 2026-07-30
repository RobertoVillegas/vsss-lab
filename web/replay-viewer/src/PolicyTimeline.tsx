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
  actor: number | null;
}

const GLOBAL_EVENT_KINDS = new Set(["goal", "own_goal", "forced_own_goal"]);
const ACTOR_EVENT_KINDS = new Set([
  "save",
  "shot",
  "assist",
  "pass",
  "interception",
  "clearance",
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
  const timelineEvents = useMemo(
    () => events
      .filter((event) => GLOBAL_EVENT_KINDS.has(event.kind) || ACTOR_EVENT_KINDS.has(event.kind))
      .map((event) => ({
        index: nearestFrame(replay, event.time),
        time: event.time,
        kind: event.kind,
        actor: actorForEvent(event.team, event.robot_id),
      })),
    [events, replay],
  );
  const globalEvents = timelineEvents.filter((event) => GLOBAL_EVENT_KINDS.has(event.kind));
  const resetFrames = useMemo(
    () => replay.frames.flatMap((frame, index) => (
      index > 0 && frame.episode !== replay.frames[index - 1]?.episode ? [index] : []
    )),
    [replay],
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
      <div className="timeline-body">
        {resetFrames.map((index) => (
          <i
            className="episode-reset-line"
            key={`reset-${index}`}
            title={`Episode ${replay.frames[index]?.episode ?? "?"} reset`}
            style={{ left: `calc(32px + (100% - 32px) * ${index / replay.frames.length})` }}
          />
        ))}
        <div className="event-rail">
        {globalEvents.map((event, eventIndex) => (
          <button
            className={`event-mark event-${event.kind.replaceAll("_", "-")}`}
            key={`${event.kind}-${event.time}-${eventIndex}`}
            style={{ left: `${percentage(event.time, finalTime)}%` }}
            title={`${event.time.toFixed(2)}s · ${event.kind.replaceAll("_", " ")}`}
            aria-label={`Seek to ${event.kind} at ${event.time.toFixed(2)} seconds`}
            onClick={() => onSeek(event.index)}
          >
            <i aria-hidden="true">{eventIcon(event.kind)}</i>
          </button>
        ))}
        {!globalEvents.length ? <span className="no-key-events">NO GOALS</span> : null}
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
              {timelineEvents
                .filter((event) => event.actor === actor && ACTOR_EVENT_KINDS.has(event.kind))
                .map((event, eventIndex) => (
                  <button
                    className={`actor-event actor-event-${event.kind.replaceAll("_", "-")}`}
                    key={`${event.kind}-${event.time}-${eventIndex}`}
                    style={{ left: `${percentage(event.time, finalTime)}%` }}
                    title={`${event.time.toFixed(2)}s · ${event.kind.replaceAll("_", " ")}`}
                    aria-label={`Seek ${actor >= 3 ? "Y" : "B"}${actor % 3} to ${event.kind} at ${event.time.toFixed(2)} seconds`}
                    onClick={() => {
                      onSelectActor(actor);
                      onSeek(event.index);
                    }}
                  >
                    {eventIcon(event.kind)}
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
      </div>
    </section>
  );
}

function actorForEvent(team: string, robotId: string | null): number | null {
  const match = robotId?.match(/(\d+)$/);
  if (!match) return null;
  const member = Number(match[1]);
  if (member < 0 || member > 2) return null;
  return (team === "yellow" ? 3 : 0) + member;
}

function eventIcon(kind: string): string {
  if (kind.includes("goal")) return "G";
  if (kind === "shot") return "↗";
  if (kind === "save") return "S";
  if (kind === "assist") return "A";
  if (kind === "pass") return "→";
  if (kind === "interception") return "◇";
  if (kind === "clearance") return "↥";
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
