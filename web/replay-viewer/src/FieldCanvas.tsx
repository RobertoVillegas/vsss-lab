import { useEffect, useRef, useState } from "react";

import type { ReplayFrame, ReplayHeader } from "./types";

interface Props {
  header: ReplayHeader;
  frame: ReplayFrame;
}

export function FieldCanvas({ header, frame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!parent) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width === 0 || size.height === 0) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, size.width);
    const height = Math.max(280, size.height);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);

    const field = header.config.field;
    const margin = Math.max(28, Math.min(width, height) * 0.07);
    const scale = Math.min(
      (width - margin * 2) / (field.length + field.goal_depth * 2),
      (height - margin * 2) / field.width,
    );
    const pitchWidth = field.length * scale;
    const pitchHeight = field.width * scale;
    const left = (width - pitchWidth) / 2;
    const top = (height - pitchHeight) / 2;
    const point = (x: number, y: number): [number, number] => [
      width / 2 + x * scale,
      height / 2 - y * scale,
    ];

    const gradient = context.createRadialGradient(
      width / 2,
      height / 2,
      10,
      width / 2,
      height / 2,
      Math.max(width, height) * 0.7,
    );
    gradient.addColorStop(0, "#17372d");
    gradient.addColorStop(1, "#07100d");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#123c2c";
    context.fillRect(left, top, pitchWidth, pitchHeight);
    context.strokeStyle = "rgba(224, 244, 235, 0.72)";
    context.lineWidth = 2;
    context.strokeRect(left, top, pitchWidth, pitchHeight);
    context.beginPath();
    context.moveTo(width / 2, top);
    context.lineTo(width / 2, top + pitchHeight);
    context.stroke();
    context.beginPath();
    context.arc(width / 2, height / 2, 0.2 * scale, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.arc(width / 2, height / 2, 2.5, 0, Math.PI * 2);
    context.fillStyle = "rgba(224, 244, 235, 0.8)";
    context.fill();

    const goalWidth = field.goal_width * scale;
    const goalDepth = field.goal_depth * scale;
    context.strokeStyle = "rgba(224, 244, 235, 0.5)";
    context.strokeRect(left - goalDepth, height / 2 - goalWidth / 2, goalDepth, goalWidth);
    context.strokeRect(left + pitchWidth, height / 2 - goalWidth / 2, goalDepth, goalWidth);

    const robotWidth = header.config.robot.length * scale;
    const robotHeight = header.config.robot.width * scale;
    context.font = "600 11px ui-monospace, monospace";
    context.textAlign = "center";
    for (const robot of frame.snapshot.robots) {
      if (!robot.enabled) continue;
      const [x, y] = point(robot.pose.x, robot.pose.y);
      context.save();
      context.translate(x, y);
      context.rotate(-robot.pose.theta);
      context.fillStyle = robot.team === "blue" ? "#49a7ff" : "#ffd84a";
      context.shadowColor = robot.team === "blue" ? "#248cff" : "#f4c600";
      context.shadowBlur = 10;
      context.fillRect(-robotWidth / 2, -robotHeight / 2, robotWidth, robotHeight);
      context.shadowBlur = 0;
      context.strokeStyle = "#07100d";
      context.lineWidth = 2;
      context.strokeRect(-robotWidth / 2, -robotHeight / 2, robotWidth, robotHeight);
      context.beginPath();
      context.moveTo(0, 0);
      context.lineTo(robotWidth / 2, 0);
      context.stroke();
      context.restore();
      context.fillStyle = "rgba(244, 255, 250, 0.82)";
      context.fillText(robot.id, x, y - robotHeight / 2 - 7);
    }

    const [ballX, ballY] = point(frame.snapshot.ball.x, frame.snapshot.ball.y);
    context.beginPath();
    context.arc(ballX, ballY, Math.max(5, header.config.ball.radius * scale), 0, Math.PI * 2);
    context.fillStyle = "#ff7547";
    context.shadowColor = "#ff5228";
    context.shadowBlur = 14;
    context.fill();
    context.shadowBlur = 0;
    context.strokeStyle = "#321108";
    context.stroke();
  }, [frame, header, size]);

  return <canvas aria-label="Recorded VSSS field" ref={canvasRef} />;
}
