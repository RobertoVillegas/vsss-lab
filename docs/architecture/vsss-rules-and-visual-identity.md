# VSSS rules and visual identity

This note records the rules baseline used by VSSS Lab and the boundary between
simulation identity and camera-observable identity. It is architecture input,
not a replacement for the rule book or event-specific captain meeting.

## Sources and version caveat

The LARC-LARS 2025 event page links a PDF titled `LARC-VSSS-2025.pdf`, while
the document body identifies itself as the 2023 Série A rules. Event organizers
can add or adapt rules, so competition work must pin the exact rule artifact and
its checksum instead of assuming the URL year is normative.

Primary sources consulted on 2026-07-28:

- LARC-LARS 2025 VSSS category page:
  <https://www.femexrobotica.org/larc-lars2025/en/portfolio-item/larc-vsss/>
- Linked Série A rules:
  <https://www.femexrobotica.org/larc-lars2025/wp-content/uploads/2025/06/LARC-VSSS-2025.pdf>
- RoboCup Brasil VSSS competition archive:
  <https://robocup.org.br/wiki/doku.php?id=very_small>
- VSSS League:
  <https://vsssleague.github.io/vss/index.html>

## Rules that affect the platform

- Two teams field at most three robots each.
- Field dimensions are 1.50 m by 1.30 m; goals are 0.40 m wide and extend
  0.10 m beyond the field.
- Robots are limited to 0.075 m per side, or 0.08 m including a removable
  uniform.
- The ball is orange, approximately 0.0427 m in diameter and 0.046 kg.
- Team identification uses blue or yellow removable tags and may change from
  match to match.
- Each robot also has a personal identification tag. The appendix standardizes
  tag geometry and permitted colors for up to ten distinguishable robots.
- A team camera or sensor is fixed above midfield, at least 2 m high, and must
  not need repositioning after side switching.
- Roles are contextual: the goalkeeper is determined by behavior in the goal
  area, not by permanent robot identity.

## Architecture consequence

Three identities must remain separate:

1. `robot_id`: stable logical participant identity in canonical state.
2. `team_assignment`: blue/yellow assignment for a particular match and side.
3. `visual_marker`: camera-observable removable tag pattern and colors.

Policies consume canonical team-relative observations and must not specialize by
physical marker. Simulation and replay viewers may render a marker profile, but
that profile is presentation/perception metadata rather than policy identity.
Future camera frames will be transformed into estimated canonical state through
an explicit association step with confidence, timestamp, and visibility.

This separation supports:

- switching blue/yellow tags without changing policy identity;
- simulating occlusion, glare, color drift, and mistaken association;
- comparing ground truth against vision estimates;
- using official marker geometry in M10/M12 without contaminating the physics
  or learner hot loops.

## Julio simulator reference

`juliodltv/simulation_vsss` uses six MIT-licensed top textures named
`blue_0..2` and `yellow_0..2`, selected from team and robot number in the URDF.
This validates the need for observable per-robot markers. VSSS Lab will generate
its own rule-derived marker assets rather than copy those PNGs, preserving a
traceable relationship with the pinned competition rule set.
