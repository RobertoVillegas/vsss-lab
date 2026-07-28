//! Interactive native viewer for deterministic VSSS JSONL replays.

use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    process::ExitCode,
};

use bevy::{
    color::palettes::css::{BLUE, DARK_GREEN, ORANGE, WHITE, YELLOW},
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use serde::Deserialize;

const SCALE: f32 = 500.0;

#[derive(Deserialize)]
struct Header {
    config: Config,
}

#[derive(Deserialize, Resource)]
struct Config {
    field: Field,
    robot: RobotGeometry,
    ball: BallGeometry,
}

#[derive(Deserialize)]
struct Field {
    length: f32,
    width: f32,
}

#[derive(Deserialize)]
struct RobotGeometry {
    length: f32,
    width: f32,
}

#[derive(Deserialize)]
struct BallGeometry {
    radius: f32,
}

#[derive(Deserialize)]
struct TickRecord {
    actions: Vec<[f32; 2]>,
    events: u32,
    snapshot: Snapshot,
}

#[derive(Deserialize)]
struct Snapshot {
    tick: u64,
    simulation_time: f32,
    score_blue: u32,
    score_yellow: u32,
    ball: Ball,
    robots: Vec<Robot>,
}

#[derive(Deserialize)]
struct Ball {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

#[derive(Deserialize)]
struct Robot {
    id: String,
    team: String,
    pose: Pose,
    twist: Twist,
}

#[derive(Deserialize)]
struct Pose {
    x: f32,
    y: f32,
    theta: f32,
}

#[derive(Deserialize)]
struct Twist {
    vx: f32,
    vy: f32,
}

#[derive(Resource)]
struct Playback {
    frames: Vec<TickRecord>,
    index: usize,
    paused: bool,
    speed: f32,
    accumulator: f32,
}

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: vsss-viewer-2d <replay.jsonl>");
        return ExitCode::from(2);
    };
    let (config, frames) = match load_replay(Path::new(&path)) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("cannot load replay: {error}");
            return ExitCode::from(1);
        }
    };
    if frames.is_empty() {
        eprintln!("cannot load replay: no tick records");
        return ExitCode::from(1);
    }

    App::new()
        .insert_resource(config)
        .insert_resource(Playback {
            frames,
            index: 0,
            paused: true,
            speed: 1.0,
            accumulator: 0.0,
        })
        .insert_resource(ClearColor(Color::srgb(0.08, 0.12, 0.16)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "VSSS replay viewer".into(),
                resolution: (900, 780).into(),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
        })
        .add_systems(Update, (playback_controls, advance_playback, draw_scene))
        .run();
    ExitCode::SUCCESS
}

fn load_replay(path: &Path) -> Result<(Config, Vec<TickRecord>), Box<dyn std::error::Error>> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header: Header = serde_json::from_str(&lines.next().ok_or("empty replay")??)?;
    let mut frames = Vec::new();
    for line in lines {
        frames.push(serde_json::from_str(&line?)?);
    }
    Ok((header.config, frames))
}

#[allow(clippy::needless_pass_by_value)]
fn playback_controls(input: Res<ButtonInput<KeyCode>>, mut playback: ResMut<Playback>) {
    if input.just_pressed(KeyCode::Space) {
        playback.paused = !playback.paused;
    }
    if input.just_pressed(KeyCode::ArrowRight) {
        playback.index = (playback.index + 1).min(playback.frames.len() - 1);
        playback.paused = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        playback.index = playback.index.saturating_sub(1);
        playback.paused = true;
    }
    if input.just_pressed(KeyCode::Home) {
        playback.index = 0;
        playback.paused = true;
    }
    if input.just_pressed(KeyCode::End) {
        playback.index = playback.frames.len() - 1;
        playback.paused = true;
    }
    if input.just_pressed(KeyCode::Equal) {
        playback.speed = (playback.speed * 2.0).min(16.0);
    }
    if input.just_pressed(KeyCode::Minus) {
        playback.speed = (playback.speed / 2.0).max(0.25);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn advance_playback(time: Res<Time>, mut playback: ResMut<Playback>) {
    if playback.paused || playback.index + 1 >= playback.frames.len() {
        return;
    }
    advance_by(&mut playback, time.delta_secs());
}

fn advance_by(playback: &mut Playback, delta_seconds: f32) {
    playback.accumulator += delta_seconds * playback.speed;
    while playback.index + 1 < playback.frames.len()
        && playback.accumulator
            >= playback.frames[playback.index + 1].snapshot.simulation_time
                - playback.frames[playback.index].snapshot.simulation_time
    {
        playback.accumulator -= playback.frames[playback.index + 1].snapshot.simulation_time
            - playback.frames[playback.index].snapshot.simulation_time;
        playback.index += 1;
    }
}

#[allow(clippy::needless_pass_by_value)]
fn draw_scene(
    config: Res<Config>,
    playback: Res<Playback>,
    mut gizmos: Gizmos,
    mut window: Single<&mut Window>,
) {
    let frame = &playback.frames[playback.index];
    let state = &frame.snapshot;
    let field_size = Vec2::new(config.field.length, config.field.width) * SCALE;
    gizmos.rect_2d(Isometry2d::IDENTITY, field_size, WHITE);
    gizmos.line_2d(
        Vec2::new(0.0, -field_size.y / 2.0),
        Vec2::new(0.0, field_size.y / 2.0),
        WHITE,
    );
    gizmos.circle_2d(Vec2::ZERO, 0.2 * SCALE, WHITE);

    for (robot, action) in state.robots.iter().zip(&frame.actions) {
        let position = world_point(robot.pose.x, robot.pose.y);
        let color = if robot.team == "blue" { BLUE } else { YELLOW };
        let transform = Isometry2d::new(position, Rot2::radians(robot.pose.theta));
        gizmos.rect_2d(
            transform,
            Vec2::new(config.robot.length, config.robot.width) * SCALE,
            color,
        );
        let heading = Vec2::from_angle(robot.pose.theta) * config.robot.length * SCALE;
        gizmos.line_2d(position, position + heading, color);
        let velocity = Vec2::new(robot.twist.vx, robot.twist.vy) * SCALE * 0.2;
        gizmos.line_2d(position, position + velocity, ORANGE);
        let action_span = Vec2::new(action[0] - action[1], action[0] + action[1]) * 6.0;
        gizmos.line_2d(position - action_span, position + action_span, WHITE);
    }

    let ball = world_point(state.ball.x, state.ball.y);
    gizmos.circle_2d(ball, config.ball.radius * SCALE, ORANGE);
    gizmos.line_2d(
        ball,
        ball + Vec2::new(state.ball.vx, state.ball.vy) * SCALE * 0.2,
        ORANGE,
    );
    for previous in playback.frames[..=playback.index].iter().rev().take(120) {
        gizmos.circle_2d(
            world_point(previous.snapshot.ball.x, previous.snapshot.ball.y),
            1.5,
            DARK_GREEN,
        );
    }

    window.title = format!(
        "VSSS | tick {} | {:.3}s | blue {}-{} yellow | {} | {:.2}x | events 0x{:x} | {}",
        state.tick,
        state.simulation_time,
        state.score_blue,
        state.score_yellow,
        if playback.paused { "paused" } else { "playing" },
        playback.speed,
        frame.events,
        state
            .robots
            .iter()
            .map(|robot| robot.id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
}

fn world_point(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y) * SCALE
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn loads_embedded_config_and_exact_ticks() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!("vsss-viewer-{suffix}.jsonl"));
        fs::write(
            &path,
            concat!(
                "{\"config\":{\"field\":{\"length\":1.5,\"width\":1.3},",
                "\"robot\":{\"length\":0.075,\"width\":0.075},",
                "\"ball\":{\"radius\":0.0215}}}\n",
                "{\"actions\":[],\"events\":0,\"snapshot\":{\"tick\":43,",
                "\"simulation_time\":0.215,\"score_blue\":0,\"score_yellow\":0,",
                "\"ball\":{\"x\":0.0,\"y\":0.0,\"vx\":0.0,\"vy\":0.0},",
                "\"robots\":[]}}\n"
            ),
        )
        .unwrap();
        let (config, frames) = load_replay(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert!((config.field.length - 1.5).abs() < f32::EPSILON);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].snapshot.tick, 43);
    }

    #[test]
    fn playback_uses_recorded_simulation_time() {
        let frame = |tick, simulation_time| TickRecord {
            actions: Vec::new(),
            events: 0,
            snapshot: Snapshot {
                tick,
                simulation_time,
                score_blue: 0,
                score_yellow: 0,
                ball: Ball {
                    x: 0.0,
                    y: 0.0,
                    vx: 0.0,
                    vy: 0.0,
                },
                robots: Vec::new(),
            },
        };
        let mut playback = Playback {
            frames: vec![frame(1, 0.0), frame(2, 0.02), frame(3, 0.04)],
            index: 0,
            paused: false,
            speed: 1.0,
            accumulator: 0.0,
        };
        advance_by(&mut playback, 0.039);
        assert_eq!(playback.index, 1);
        advance_by(&mut playback, 0.001);
        assert_eq!(playback.index, 2);
    }
}
