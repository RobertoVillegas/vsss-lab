# M24.2 Parametric Soccer Primitives

## Why

M24 proved that semantic primitives are inspectable, but its eight canonical
headings force axis-aligned and diagonal motion. A differential-drive robot
needs continuous approach angles and controllable authority to acquire and
redirect a moving ball.

## What changes

- keep `stop`, `navigate`, and `strike` as categorical policy decisions;
- parameterize active skills with a continuous unit heading and intensity;
- train the joint categorical and bounded-Gaussian policy with MAPPO;
- preserve legacy M24 checkpoints and replays under the old parser;
- expose requested angle, intensity, and heading-change diagnostics.

## Success criteria

- headings between the old 45-degree bins execute without quantization;
- the policy crosses the ±π boundary without a discontinuity;
- replay intent reports the exact angle and requested intensity;
- a short training run completes optimization, checkpointing, and capture;
- legacy primitive tests remain green.
