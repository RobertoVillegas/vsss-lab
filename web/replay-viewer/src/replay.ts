import type { Replay, ReplayFrame, ReplayHeader } from "./types";

export function parseReplay(text: string): Replay {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error("Replay has no recorded frames.");
  }
  const header = JSON.parse(lines[0]) as ReplayHeader;
  if (header.type !== "header" || !header.config?.field) {
    throw new Error("Replay header is invalid.");
  }
  const frames = lines.slice(1).map((line) => JSON.parse(line) as ReplayFrame);
  if (frames.some((frame) => frame.type !== "tick" || !frame.snapshot)) {
    throw new Error("Replay contains an invalid frame.");
  }
  return { header, frames };
}

export function clampedFrame(current: number, delta: number, count: number): number {
  return Math.max(0, Math.min(count - 1, current + delta));
}

export function frameLabel(current: number, count: number): string {
  return count === 0 ? "0 / 0" : `${current + 1} / ${count}`;
}
