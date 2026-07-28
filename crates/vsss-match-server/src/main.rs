//! Executable heterogeneous match server.

use std::{collections::BTreeMap, env, path::PathBuf, time::Duration};

use sha2::{Digest, Sha256};
use vsss_match_server::{
    Clock, ControllerIdentity, FallbackPolicy, MatchArtifact, MatchMachine, MatchMetadata,
    MatchOutcome, RouterTransport, SessionRegistry, SystemClock,
};
use vsss_physics_rapier::RapierBackend;
use vsss_protocol::{
    EnvelopeMeta, MAX_MESSAGE_BYTES, PROTOCOL_VERSION, decode_envelope, encode_capabilities,
    encode_match_result, encode_observation, encode_reset, wire,
};
use vsss_spec::{ControlMode, MatchConfig, MatchState, RobotAction, serialization};

type AnyError = Box<dyn std::error::Error>;

#[tokio::main]
async fn main() -> Result<(), AnyError> {
    let arguments = Arguments::parse()?;
    run(arguments).await
}

#[allow(clippy::too_many_lines)]
async fn run(arguments: Arguments) -> Result<(), AnyError> {
    let config_json = std::fs::read_to_string(&arguments.config)?;
    let state_json = std::fs::read_to_string(&arguments.state)?;
    let config: MatchConfig = serialization::from_json(&config_json)?;
    let initial: MatchState = serialization::from_json(&state_json)?;
    let backend = RapierBackend::new(config.clone(), initial)?;
    let clock = SystemClock::default();
    let mut machine =
        MatchMachine::new(backend, clock.clone(), config.clone(), FallbackPolicy::Zero)?;
    let mut transport = if arguments.endpoint.starts_with("tcp://0.0.0.0:") {
        RouterTransport::bind_private(&arguments.endpoint, MAX_MESSAGE_BYTES).await?
    } else {
        RouterTransport::bind(&arguments.endpoint, MAX_MESSAGE_BYTES).await?
    };
    println!("READY {}", transport.endpoint());

    let match_id = *b"external-match01";
    let mut sessions = SessionRegistry::default();
    let mut routes = BTreeMap::new();
    while routes.len() < 2 {
        let incoming = transport.receive().await?;
        let envelope = decode_envelope(&incoming.payload)?;
        let hello = envelope
            .wire()
            .payload_as_hello()
            .ok_or("expected Hello during negotiation")?;
        let name = hello.controller_name().to_owned();
        let session = sessions.negotiate(
            ControllerIdentity {
                routing_id: incoming.routing_id.clone(),
                name,
            },
            envelope,
            0,
            duration_ns(config.control_period.get())?,
            MAX_MESSAGE_BYTES,
        )?;
        let slot = session.slot;
        routes.insert(slot.0, incoming.routing_id.clone());
        let response = encode_capabilities(
            server_meta(match_id, slot, 1, 0, 0, &clock),
            slot,
            duration_ns(config.control_period.get())?,
            u32::try_from(MAX_MESSAGE_BYTES)?,
        );
        transport.send(&incoming.routing_id, response).await?;
    }

    let state = machine.start()?.clone();
    let config_hash: [u8; 32] = Sha256::digest(config_json.as_bytes()).into();
    for (slot, route) in routes.iter().map(|(slot, route)| (*slot, route)) {
        transport
            .send(
                route,
                encode_reset(
                    server_meta(
                        match_id,
                        wire::ControllerSlot(slot),
                        2,
                        state.tick,
                        machine.deadline_ns(),
                        &clock,
                    ),
                    &config_json,
                    &config_hash,
                    &state_json,
                    config.seed,
                ),
            )
            .await?;
    }

    let metadata = MatchMetadata {
        match_id: "external-match01".to_owned(),
        config,
        blue_controller: sessions
            .get(&routes[&wire::ControllerSlot::Blue.0])
            .ok_or("missing blue session")?
            .identity
            .name
            .clone(),
        yellow_controller: sessions
            .get(&routes[&wire::ControllerSlot::Yellow.0])
            .ok_or("missing yellow session")?
            .identity
            .name
            .clone(),
        protocol_version: PROTOCOL_VERSION,
        build_revision: option_env!("VSSS_BUILD_REVISION")
            .unwrap_or("development")
            .to_owned(),
    };
    let mut artifact = MatchArtifact::create(&arguments.output, &metadata)?;
    let mut server_sequence = 3_u64;
    for _ in 0..arguments.control_ticks {
        let current = machine.state().ok_or("machine has no state")?.clone();
        let state_json = serialization::to_json(&current)?;
        let state_hash: [u8; 32] = Sha256::digest(state_json.as_bytes()).into();
        let deadline = machine.deadline_ns();
        for (slot, route) in routes.iter().map(|(slot, route)| (*slot, route)) {
            transport
                .send(
                    route,
                    encode_observation(
                        server_meta(
                            match_id,
                            wire::ControllerSlot(slot),
                            server_sequence,
                            current.tick,
                            deadline,
                            &clock,
                        ),
                        &state_json,
                        &state_hash,
                    ),
                )
                .await?;
        }
        server_sequence += 1;
        let mut received = [false; 2];
        while !received.into_iter().all(|value| value) {
            let now = clock.now_ns();
            let Some(remaining) = deadline.checked_sub(now) else {
                break;
            };
            let Ok(result) =
                tokio::time::timeout(Duration::from_nanos(remaining), transport.receive()).await
            else {
                break;
            };
            let incoming = result?;
            let envelope = decode_envelope(&incoming.payload)?;
            let meta = envelope.meta();
            sessions.accept(
                &incoming.routing_id,
                meta.controller_slot,
                meta.sequence,
                clock.now_ns(),
            )?;
            if let Some(action) = envelope.wire().payload_as_action() {
                let index = slot_index(meta.controller_slot)?;
                machine.submit(
                    meta.controller_slot,
                    meta.server_tick,
                    std::array::from_fn(|robot| {
                        let command = action.robots().get(robot);
                        RobotAction {
                            mode: if command.mode() == wire::ControlMode::WheelVelocity {
                                ControlMode::WheelVelocity
                            } else {
                                ControlMode::BodyVelocity
                            },
                            left: command.first(),
                            right: command.second(),
                        }
                    }),
                )?;
                received[index] = true;
            }
        }
        let now = clock.now_ns();
        if let Some(remaining) = deadline.checked_sub(now) {
            tokio::time::sleep(Duration::from_nanos(remaining)).await;
        }
        let advance = machine
            .advance_if_due()?
            .ok_or("deadline did not advance match")?;
        artifact.record(&advance)?;
        if advance.finished {
            break;
        }
    }
    let final_state = machine.state().ok_or("missing final state")?.clone();
    let digest = artifact.finish(MatchOutcome::Completed)?;
    let digest_bytes = decode_hex_32(&digest)?;
    for (slot, route) in routes.iter().map(|(slot, route)| (*slot, route)) {
        transport
            .send(
                route,
                encode_match_result(
                    server_meta(
                        match_id,
                        wire::ControllerSlot(slot),
                        server_sequence,
                        final_state.tick,
                        0,
                        &clock,
                    ),
                    final_state.score_blue,
                    final_state.score_yellow,
                    &digest_bytes,
                    "completed",
                ),
            )
            .await?;
    }
    println!(
        "RESULT blue={} yellow={} replay={} sha256={digest}",
        final_state.score_blue,
        final_state.score_yellow,
        arguments.output.display()
    );
    Ok(())
}

struct Arguments {
    endpoint: String,
    config: PathBuf,
    state: PathBuf,
    output: PathBuf,
    control_ticks: u64,
}

impl Arguments {
    fn parse() -> Result<Self, AnyError> {
        let mut values = env::args().skip(1);
        Ok(Self {
            endpoint: values.next().ok_or("missing endpoint")?,
            config: values.next().ok_or("missing config")?.into(),
            state: values.next().ok_or("missing state")?.into(),
            output: values.next().ok_or("missing output")?.into(),
            control_ticks: values.next().ok_or("missing control ticks")?.parse()?,
        })
    }
}

fn server_meta(
    match_id: [u8; 16],
    slot: wire::ControllerSlot,
    sequence: u64,
    tick: u64,
    deadline: u64,
    clock: &impl Clock,
) -> EnvelopeMeta {
    EnvelopeMeta {
        match_id,
        controller_slot: slot,
        sequence,
        server_tick: tick,
        sent_monotonic_ns: clock.now_ns(),
        deadline_monotonic_ns: deadline,
    }
}

fn duration_ns(seconds: f32) -> Result<u64, AnyError> {
    Ok(u64::try_from(
        Duration::try_from_secs_f32(seconds)?.as_nanos(),
    )?)
}

fn slot_index(slot: wire::ControllerSlot) -> Result<usize, AnyError> {
    match slot {
        wire::ControllerSlot::Blue => Ok(0),
        wire::ControllerSlot::Yellow => Ok(1),
        _ => Err("invalid action slot".into()),
    }
}

fn decode_hex_32(value: &str) -> Result<[u8; 32], AnyError> {
    if value.len() != 64 {
        return Err("invalid SHA-256 hex length".into());
    }
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    Ok(bytes)
}
