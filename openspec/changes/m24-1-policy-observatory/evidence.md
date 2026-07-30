# Evidence

## Contract

- Replay schema v2 records `policy_intents` independently from actuator
  `actions`.
- Primitive intent includes action index, skill, direction, confidence, top
  alternatives, execution phase, target, exit direction, and ball distance.
- Replay v1 remains accepted by offline inspection and all new web fields are
  optional for legacy captures.

## Validation

- A real one-iteration M24 CPU run generated 2,048 environment steps, ten
  completed matches, a checkpoint, categorical metrics, and a replay v2.
- Backend contract tests confirm six actuator records and six separate policy
  intent slots, including null intent for a heuristic opponent.
- Timeline component tests confirm primitive grouping, event seeking, and an
  explicit compatibility message for captures without policy intent.
- Full Rust, Python, and web tests pass.

## Browser QA

Agent Browser exercised the production Vite build on loopback:

- metrics loaded categorical exploration and curriculum allocation;
- selecting B1 collapsed B0 and expanded B1 at the same playback frame;
- an interception marker sought the replay slider to its corresponding frame;
- the selected actor target and requested exit vector rendered on the field;
- intent lanes rendered only for actors with recorded policy decisions;
- browser console contained no errors.

Long timelines cap rendering at 400 representative segments per actor while
event markers retain exact seek targets.

## Viewer refinement

- The replay cursor is visible across every available intent lane.
- Selecting a decision segment selects its actor, pauses playback, and seeks to
  the segment's first frame.
- Intent channels can be collapsed without hiding the event rail.
- The primary event rail follows a Ballchasing-style signal hierarchy: goals,
  and own goals are global markers; shots, saves, assists, passes,
  interceptions, and clearances belong to the actor lane that produced them.
- Episode boundaries render as a shared vertical dashed line through every
  visible actor lane. Nearby events retain their exact independent seek target
  instead of being merged into an ambiguous cluster.
- Legacy captures and heuristic-only actors no longer create inert `WHEEL`
  lanes.
- The left rail keeps four compact, frame-synchronous insights: selected
  primitive, recent decision changes, closest actor to the ball, and ball
  region.
