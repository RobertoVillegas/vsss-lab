import { GIFEncoder, applyPalette, quantize } from "gifenc";

import type { Replay } from "./types";

const VIDEO_FPS = 30;
const GIF_MAX_FRAMES = 600;
const GIF_MAX_WIDTH = 640;

export type ExportFormat = "webm" | "gif";

export interface ExportProgress {
  completed: number;
  total: number;
  format: ExportFormat;
}

interface ExportOptions {
  canvas: HTMLCanvasElement;
  replay: Replay;
  speed: number;
  seek: (frame: number) => Promise<void>;
  onProgress: (progress: ExportProgress) => void;
}

export function sampledFrameIndices(frameCount: number, desiredFrames: number): number[] {
  if (frameCount <= 0 || desiredFrames <= 0) return [];
  const outputFrames = Math.min(frameCount, desiredFrames);
  if (outputFrames === 1) return [0];
  return Array.from(
    { length: outputFrames },
    (_, index) => Math.round(index * (frameCount - 1) / (outputFrames - 1)),
  );
}

export async function exportReplay(
  format: ExportFormat,
  options: ExportOptions,
): Promise<void> {
  if (format === "webm") await exportWebm(options);
  else await exportGif(options);
}

async function exportWebm(options: ExportOptions): Promise<void> {
  if (!("MediaRecorder" in window) || !options.canvas.captureStream) {
    throw new Error("This browser cannot encode canvas video.");
  }
  const mimeType = videoMimeType();
  const stream = options.canvas.captureStream(VIDEO_FPS);
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });
  const finished = new Promise<void>((resolve, reject) => {
    recorder.addEventListener("stop", () => resolve(), { once: true });
    recorder.addEventListener("error", () => reject(new Error("Video encoding failed.")), {
      once: true,
    });
  });
  const controlPeriod = options.replay.header.config.control_period;
  const duration = Math.max(
    0.5,
    options.replay.frames.length * controlPeriod / Math.max(options.speed, 0.01),
  );
  const outputFrames = Math.max(1, Math.ceil(duration * VIDEO_FPS));
  const indices = sampledFrameIndices(options.replay.frames.length, outputFrames);
  recorder.start(100);
  for (let index = 0; index < indices.length; index += 1) {
    await options.seek(indices[index]);
    options.onProgress({ completed: index + 1, total: indices.length, format: "webm" });
    await delay(1000 / VIDEO_FPS);
  }
  recorder.requestData();
  await delay(100);
  recorder.stop();
  await finished;
  stream.getTracks().forEach((track) => track.stop());
  downloadBlob(
    new Blob(chunks, { type: recorder.mimeType || "video/webm" }),
    replayFilename(options.replay, "webm"),
  );
}

async function exportGif(options: ExportOptions): Promise<void> {
  const scale = Math.min(1, GIF_MAX_WIDTH / options.canvas.width);
  const width = Math.max(1, Math.round(options.canvas.width * scale));
  const height = Math.max(1, Math.round(options.canvas.height * scale));
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const context = output.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create the GIF canvas.");
  const desiredFrames = Math.min(GIF_MAX_FRAMES, options.replay.frames.length);
  const indices = sampledFrameIndices(options.replay.frames.length, desiredFrames);
  const replaySeconds =
    options.replay.frames.length
    * options.replay.header.config.control_period
    / Math.max(options.speed, 0.01);
  const delayMs = Math.max(20, Math.round(replaySeconds * 1000 / indices.length));
  const encoder = GIFEncoder();
  for (let index = 0; index < indices.length; index += 1) {
    await options.seek(indices[index]);
    context.drawImage(options.canvas, 0, 0, width, height);
    const rgba = context.getImageData(0, 0, width, height).data;
    const palette = quantize(rgba, 256, { format: "rgb444" });
    const pixels = applyPalette(rgba, palette, "rgb444");
    encoder.writeFrame(pixels, width, height, { palette, delay: delayMs });
    options.onProgress({ completed: index + 1, total: indices.length, format: "gif" });
    if (index % 4 === 0) await delay(0);
  }
  encoder.finish();
  const encoded = encoder.bytesView();
  const payload = new Uint8Array(encoded.byteLength);
  payload.set(encoded);
  downloadBlob(
    new Blob([payload.buffer], { type: "image/gif" }),
    replayFilename(options.replay, "gif"),
  );
}

function videoMimeType(): string {
  return [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ].find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
}

function replayFilename(replay: Replay, extension: string): string {
  const policy = replay.header.policies.blue.replaceAll(/[^a-zA-Z0-9_-]+/g, "-");
  return `vsss-${policy}-${replay.frames.length}-frames.${extension}`;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
