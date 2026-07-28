//! Interactive native viewer for deterministic VSSS JSONL replays.

use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, Read},
    net::UdpSocket,
    path::Path,
    process::ExitCode,
};

use bevy::{
    color::palettes::css::{BLUE, DARK_GREEN, ORANGE, WHITE, YELLOW},
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use flate2::read::ZlibDecoder;
use serde::Deserialize;

const SCALE: f32 = 500.0;

#[derive(Deserialize)]
struct Header {
    config: Config,
}

#[derive(Deserialize)]
struct Config {
    field: Field,
    robot: RobotGeometry,
    ball: BallGeometry,
}

#[derive(Resource)]
struct ViewerConfig(Option<Config>);

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
    #[serde(default)]
    rewards: Option<Vec<f32>>,
    snapshot: Snapshot,
}

#[derive(Deserialize)]
struct LivePacket {
    r#type: String,
    version: u32,
    sequence: u64,
    sample_every: u64,
    send_errors: u64,
    config: Config,
    frame: TickRecord,
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
    live: bool,
    transport_dropped: u64,
    send_errors: u64,
}

#[derive(Resource)]
struct LiveReceiver {
    socket: UdpSocket,
    last_sequence: Option<u64>,
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let Some(first) = arguments.first() else {
        eprintln!("usage: vsss-viewer-2d <replay.jsonl> | --listen 127.0.0.1:42042");
        return ExitCode::from(2);
    };
    let (config, frames, live_receiver) = if first == "--listen" {
        let Some(address) = arguments.get(1) else {
            eprintln!("--listen requires an address");
            return ExitCode::from(2);
        };
        let socket = match UdpSocket::bind(address) {
            Ok(socket) => socket,
            Err(error) => {
                eprintln!("cannot bind live viewer: {error}");
                return ExitCode::from(1);
            }
        };
        if let Err(error) = socket.set_nonblocking(true) {
            eprintln!("cannot configure live viewer: {error}");
            return ExitCode::from(1);
        }
        (
            None,
            Vec::new(),
            Some(LiveReceiver {
                socket,
                last_sequence: None,
            }),
        )
    } else {
        let (config, frames) = match load_replay(Path::new(first)) {
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
        (Some(config), frames, None)
    };
    let live = live_receiver.is_some();

    let mut app = App::new();
    app.insert_resource(ViewerConfig(config))
        .insert_resource(Playback {
            frames,
            index: 0,
            paused: !live,
            speed: 1.0,
            accumulator: 0.0,
            live,
            transport_dropped: 0,
            send_errors: 0,
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
        .add_systems(
            Update,
            (
                receive_live,
                playback_controls,
                advance_playback,
                draw_scene,
            ),
        );
    if let Some(receiver) = live_receiver {
        app.insert_resource(receiver);
    }
    app.run();
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
fn receive_live(
    receiver: Option<ResMut<LiveReceiver>>,
    mut config: ResMut<ViewerConfig>,
    mut playback: ResMut<Playback>,
) {
    let Some(mut receiver) = receiver else {
        return;
    };
    let mut datagram = [0_u8; 1_400];
    loop {
        match receiver.socket.recv(&mut datagram) {
            Ok(length) => {
                let Ok(packet) = decode_live_packet(&datagram[..length]) else {
                    continue;
                };
                if let Some(previous) = receiver.last_sequence {
                    playback.transport_dropped +=
                        packet.sequence.saturating_sub(previous.saturating_add(1));
                }
                receiver.last_sequence = Some(packet.sequence);
                playback.send_errors = packet.send_errors;
                config.0 = Some(packet.config);
                playback.frames.push(packet.frame);
                if playback.frames.len() > 120 {
                    playback.frames.remove(0);
                }
                playback.index = playback.frames.len() - 1;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(_) => break,
        }
    }
}

fn decode_live_packet(datagram: &[u8]) -> Result<LivePacket, Box<dyn std::error::Error>> {
    if !datagram.starts_with(b"VSS1") {
        return Err("unsupported live packet".into());
    }
    let mut decoder = ZlibDecoder::new(&datagram[4..]);
    let mut json = String::new();
    decoder.read_to_string(&mut json)?;
    let packet: LivePacket = serde_json::from_str(&json)?;
    if packet.r#type != "visual_frame" || packet.version != 1 || packet.sample_every == 0 {
        return Err("invalid live packet envelope".into());
    }
    Ok(packet)
}

#[allow(clippy::needless_pass_by_value)]
fn playback_controls(input: Res<ButtonInput<KeyCode>>, mut playback: ResMut<Playback>) {
    if playback.frames.is_empty() {
        return;
    }
    if input.just_pressed(KeyCode::Space) {
        playback.paused = !playback.paused;
    }
    if input.just_pressed(KeyCode::ArrowRight) {
        if playback.live {
            return;
        }
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
    config: Res<ViewerConfig>,
    playback: Res<Playback>,
    mut gizmos: Gizmos,
    mut window: Single<&mut Window>,
) {
    let Some(config) = config.0.as_ref() else {
        window.title = "VSSS live viewer | waiting for frames".into();
        return;
    };
    if playback.frames.is_empty() {
        return;
    }
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
        "VSSS | tick {} | {:.3}s | blue {}-{} yellow | {} | {:.2}x | reward {:.3} | events 0x{:x} | drops {}+{} | {}",
        state.tick,
        state.simulation_time,
        state.score_blue,
        state.score_yellow,
        if playback.paused { "paused" } else { "playing" },
        playback.speed,
        frame
            .rewards
            .as_ref()
            .map_or(0.0, |rewards| rewards.iter().sum()),
        frame.events,
        playback.transport_dropped,
        playback.send_errors,
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
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    use flate2::{Compression, write::ZlibEncoder};

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
            rewards: None,
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
            live: false,
            transport_dropped: 0,
            send_errors: 0,
        };
        advance_by(&mut playback, 0.039);
        assert_eq!(playback.index, 1);
        advance_by(&mut playback, 0.001);
        assert_eq!(playback.index, 2);
    }

    #[test]
    fn decodes_python_compatible_live_packet() {
        let json = concat!(
            "{\"type\":\"visual_frame\",\"version\":1,\"sequence\":7,",
            "\"sample_every\":4,\"send_errors\":2,",
            "\"config\":{\"field\":{\"length\":1.5,\"width\":1.3},",
            "\"robot\":{\"length\":0.075,\"width\":0.075},",
            "\"ball\":{\"radius\":0.0215}},",
            "\"frame\":{\"actions\":[],\"events\":0,\"rewards\":[1.0],",
            "\"snapshot\":{\"tick\":43,\"simulation_time\":0.215,",
            "\"score_blue\":0,\"score_yellow\":0,",
            "\"ball\":{\"x\":0.0,\"y\":0.0,\"vx\":0.0,\"vy\":0.0},",
            "\"robots\":[]}}}"
        );
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(json.as_bytes()).unwrap();
        let mut datagram = b"VSS1".to_vec();
        datagram.extend(encoder.finish().unwrap());
        let packet = decode_live_packet(&datagram).unwrap();
        assert_eq!(packet.sequence, 7);
        assert_eq!(packet.send_errors, 2);
        assert_eq!(packet.frame.snapshot.tick, 43);
        assert_eq!(packet.frame.rewards.unwrap(), vec![1.0]);
    }
}
