//! Rust codec tests for valid and invalid protocol messages.

use vsss_protocol::{
    DecodeError, EnvelopeMeta, ROBOTS_PER_TEAM, RobotCommand, decode_envelope, encode_action,
    encode_capabilities, encode_hello, encode_match_result, encode_observation, encode_reset, wire,
};

const META: EnvelopeMeta = EnvelopeMeta {
    match_id: *b"golden-match-001",
    controller_slot: wire::ControllerSlot::Blue,
    sequence: 7,
    server_tick: 42,
    sent_monotonic_ns: 1_000,
    deadline_monotonic_ns: 2_000,
};

#[test]
fn hello_round_trips() {
    let bytes = encode_hello(META, "golden-controller", "vsss-rust", "0.0.0");
    let envelope = decode_envelope(&bytes).expect("valid hello");

    assert_eq!(envelope.meta(), META);
    let hello = envelope.wire().payload_as_hello().expect("hello payload");
    assert_eq!(hello.controller_name(), "golden-controller");
    assert_eq!(hello.min_protocol_version(), 1);
    assert_eq!(hello.max_protocol_version(), 1);
}

#[test]
fn action_round_trips() {
    let commands = [
        RobotCommand {
            mode: wire::ControlMode::WheelVelocity,
            first: 1.0,
            second: 2.0,
        },
        RobotCommand {
            mode: wire::ControlMode::BodyVelocity,
            first: -0.5,
            second: 0.25,
        },
        RobotCommand {
            mode: wire::ControlMode::WheelVelocity,
            first: 0.0,
            second: 0.0,
        },
    ];
    let bytes = encode_action(META, commands);
    let envelope = decode_envelope(&bytes).expect("valid action");
    let action = envelope.wire().payload_as_action().expect("action payload");

    assert_eq!(action.robots().len(), ROBOTS_PER_TEAM);
    assert_eq!(
        action.robots().get(1).mode(),
        wire::ControlMode::BodyVelocity
    );
    assert!((action.robots().get(1).first() - (-0.5)).abs() < f32::EPSILON);
}

#[test]
fn rejects_wrong_identifier_before_parsing() {
    let mut bytes = encode_hello(META, "controller", "sdk", "1");
    bytes[4..8].copy_from_slice(b"NOPE");

    assert!(matches!(
        decode_envelope(&bytes),
        Err(DecodeError::WrongIdentifier)
    ));
}

#[test]
fn rejects_non_finite_actions() {
    let commands = [RobotCommand {
        mode: wire::ControlMode::WheelVelocity,
        first: f32::NAN,
        second: 0.0,
    }; ROBOTS_PER_TEAM];

    assert!(matches!(
        decode_envelope(&encode_action(META, commands)),
        Err(DecodeError::NonFiniteAction { robot: 0 })
    ));
}

#[test]
fn rust_decodes_committed_golden_buffers() {
    let hello = include_bytes!("../../../tests/golden/m8_hello_v1.vsss");
    let action = include_bytes!("../../../tests/golden/m8_action_v1.vsss");

    assert_eq!(
        decode_envelope(hello)
            .expect("valid golden hello")
            .wire()
            .payload_as_hello()
            .expect("hello payload")
            .controller_name(),
        "golden-python"
    );
    assert_eq!(
        decode_envelope(action)
            .expect("valid golden action")
            .wire()
            .payload_as_action()
            .expect("action payload")
            .robots()
            .len(),
        ROBOTS_PER_TEAM
    );
}

#[test]
fn server_payloads_round_trip() {
    let digest = [7_u8; 32];
    let capabilities = encode_capabilities(META, wire::ControllerSlot::Blue, 20_000_000, 4096);
    let reset = encode_reset(META, "{}", &digest, "{}", 42);
    let observation = encode_observation(META, "{}", &digest);
    let result = encode_match_result(META, 2, 1, &digest, "completed");

    assert_eq!(
        decode_envelope(&capabilities)
            .expect("capabilities")
            .wire()
            .payload_as_capabilities()
            .expect("payload")
            .max_message_bytes(),
        4096
    );
    assert_eq!(
        decode_envelope(&reset)
            .expect("reset")
            .wire()
            .payload_as_reset()
            .expect("payload")
            .seed(),
        42
    );
    assert_eq!(
        decode_envelope(&observation)
            .expect("observation")
            .wire()
            .payload_as_observation()
            .expect("payload")
            .state_sha256()
            .len(),
        32
    );
    assert_eq!(
        decode_envelope(&result)
            .expect("result")
            .wire()
            .payload_as_match_result()
            .expect("payload")
            .score_blue(),
        2
    );
}
