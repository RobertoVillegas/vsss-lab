import { useEffect, useRef, useState } from "react";

import type { ReplayFrame, ReplayHeader } from "./types";

interface Props {
  header: ReplayHeader;
  frame: ReplayFrame;
  layers: {
    truth: boolean;
    measured: boolean;
    estimated: boolean;
    predicted: boolean;
  };
}

export function FieldCanvas({ header, frame, layers }: Props) {
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

    const chamfer = Math.min(0.07 * scale, pitchWidth / 8, pitchHeight / 8);
    const tracePitch = () => {
      context.beginPath();
      context.moveTo(left + chamfer, top);
      context.lineTo(left + pitchWidth - chamfer, top);
      context.lineTo(left + pitchWidth, top + chamfer);
      context.lineTo(left + pitchWidth, top + pitchHeight - chamfer);
      context.lineTo(left + pitchWidth - chamfer, top + pitchHeight);
      context.lineTo(left + chamfer, top + pitchHeight);
      context.lineTo(left, top + pitchHeight - chamfer);
      context.lineTo(left, top + chamfer);
      context.closePath();
    };

    context.fillStyle = "#123c2c";
    tracePitch();
    context.fill();
    context.strokeStyle = "rgba(224, 244, 235, 0.72)";
    context.lineWidth = 2;
    tracePitch();
    context.stroke();
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
    const penaltyDepth = 0.15 * scale;
    const penaltyWidth = 0.70 * scale;
    context.strokeStyle = "rgba(224, 244, 235, 0.5)";
    context.strokeRect(left - goalDepth, height / 2 - goalWidth / 2, goalDepth, goalWidth);
    context.strokeRect(left + pitchWidth, height / 2 - goalWidth / 2, goalDepth, goalWidth);
    context.strokeRect(left, height / 2 - penaltyWidth / 2, penaltyDepth, penaltyWidth);
    context.strokeRect(
      left + pitchWidth - penaltyDepth,
      height / 2 - penaltyWidth / 2,
      penaltyDepth,
      penaltyWidth,
    );

    // Goal-area arcs and restart crosses mirror the calibrated 1.70 x 1.30 m
    // reference layout while staying in canonical, field-centered coordinates.
    for (const xSign of [-1, 1]) {
      const [arcX, arcY] = point(xSign * (field.length / 2 - 0.07), 0);
      context.beginPath();
      if (xSign < 0) {
        context.arc(arcX, arcY, 0.13 * scale, -Math.PI / 2, Math.PI / 2);
      } else {
        context.arc(arcX, arcY, 0.13 * scale, Math.PI / 2, Math.PI * 1.5);
      }
      context.stroke();
    }

    const drawCross = (x: number, y: number) => {
      const [crossX, crossY] = point(x, y);
      const radius = 0.025 * scale;
      context.beginPath();
      context.moveTo(crossX - radius, crossY);
      context.lineTo(crossX + radius, crossY);
      context.moveTo(crossX, crossY - radius);
      context.lineTo(crossX, crossY + radius);
      context.stroke();
    };
    for (const x of [-0.375, 0.375]) {
      for (const y of [-0.40, 0, 0.40]) drawCross(x, y);
    }

    const perception = frame.perception;
    const prediction = perception?.ball_prediction;
    if (layers.predicted && prediction && prediction.samples.length > 1) {
      context.beginPath();
      prediction.samples.forEach((sample, index) => {
        const [sampleX, sampleY] = point(sample[1], sample[2]);
        if (index === 0) context.moveTo(sampleX, sampleY);
        else context.lineTo(sampleX, sampleY);
      });
      context.strokeStyle = prediction.stale ? "rgba(255, 184, 77, 0.6)" : "rgba(72, 224, 255, 0.8)";
      context.lineWidth = 2;
      context.setLineDash([6, 5]);
      context.stroke();
      context.setLineDash([]);
      prediction.samples.forEach((sample, index) => {
        if (index === 0 || index % 2 !== 0) return;
        const [sampleX, sampleY] = point(sample[1], sample[2]);
        context.fillStyle = "rgba(132, 234, 255, 0.9)";
        context.font = "10px ui-monospace, monospace";
        context.fillText(`+${sample[0].toFixed(1)}s`, sampleX, sampleY - 8);
      });
    }
    const estimate = perception?.ball_estimate;
    if (layers.estimated && estimate) {
      const [estimateX, estimateY] = point(estimate.state[0], estimate.state[3]);
      context.beginPath();
      context.arc(estimateX, estimateY, Math.max(8, header.config.ball.radius * scale * 1.5), 0, Math.PI * 2);
      context.strokeStyle = estimate.measurement_accepted ? "#48e0ff" : "#ffb84d";
      context.lineWidth = 2;
      context.stroke();
    }
    const measuredBall = perception?.camera.ball;
    if (layers.measured && measuredBall) {
      const [measuredX, measuredY] = point(measuredBall.x, measuredBall.y);
      context.strokeStyle = "rgba(255, 255, 255, 0.75)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(measuredX - 5, measuredY);
      context.lineTo(measuredX + 5, measuredY);
      context.moveTo(measuredX, measuredY - 5);
      context.lineTo(measuredX, measuredY + 5);
      context.stroke();
    }

    const robotWidth = header.config.robot.length * scale;
    const robotHeight = header.config.robot.width * scale;
    context.font = "600 11px ui-monospace, monospace";
    context.textAlign = "center";
    for (const robot of layers.truth ? frame.snapshot.robots : []) {
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
      const tagColors = ["#45e2b5", "#e05cff", "#ff625c"];
      context.fillStyle = robot.team === "blue" ? "#248cff" : "#f4c600";
      context.fillRect(
        -robotWidth * 0.32,
        -robotHeight * 0.40,
        robotWidth * 0.30,
        robotHeight * 0.80,
      );
      context.fillStyle = tagColors[Number.parseInt(robot.id.replace(/\D/g, ""), 10) % 3] ?? tagColors[0];
      context.fillRect(
        robotWidth * 0.08,
        robotHeight * 0.08,
        robotWidth * 0.25,
        robotHeight * 0.25,
      );
      context.beginPath();
      context.moveTo(0, 0);
      context.lineTo(robotWidth / 2, 0);
      context.stroke();
      context.restore();
      context.fillStyle = "rgba(244, 255, 250, 0.82)";
      context.fillText(robot.id, x, y - robotHeight / 2 - 7);
    }

    if (layers.truth) {
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
    }
  }, [frame, header, layers, size]);

  return <canvas aria-label="Recorded VSSS field" ref={canvasRef} />;
}
