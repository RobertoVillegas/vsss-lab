//! Deterministic authoritative-machine contract tests.

use std::{cell::Cell, rc::Rc};

use vsss_match_server::{
    Clock, ControllerIdentity, FallbackPolicy, LeaseAdjudication, MatchMachine, SessionError,
    SessionRegistry, SessionState, TickDecision,
};
use vsss_physics_rapier::RapierBackend;
use vsss_protocol::wire::ControllerSlot;
use vsss_spec::{AngularVelocity, MatchConfig, MatchState, RobotAction, serialization};

#[derive(Clone, Debug)]
struct FakeClock(Rc<Cell<u64>>);

impl FakeClock {
    fn new() -> Self {
        Self(Rc::new(Cell::new(0)))
    }

    fn set(&self, now_ns: u64) {
        self.0.set(now_ns);
    }
}

impl Clock for FakeClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

fn fixture() -> (MatchConfig, MatchState) {
    let config =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_config.json"))
            .expect("valid config");
    let state = serialization::from_json(include_str!("../../../tests/golden/m1_match_state.json"))
        .expect("valid state");
    (config, state)
}

fn moving(speed: f32) -> [RobotAction; 3] {
    [RobotAction::wheel_velocity(AngularVelocity(speed), AngularVelocity(speed)); 3]
}

#[test]
fn fake_clock_controls_boundary_and_fallback() {
    let (config, initial) = fixture();
    let backend = RapierBackend::new(config.clone(), initial).expect("valid backend");
    let clock = FakeClock::new();
    let mut machine =
        MatchMachine::new(backend, clock.clone(), config, FallbackPolicy::Zero).expect("machine");
    let target_tick = machine.start().expect("reset").tick;
    let deadline = machine.deadline_ns();

    machine
        .submit(ControllerSlot::Blue, target_tick, moving(5.0))
        .expect("blue action");
    clock.set(deadline - 1);
    assert!(machine.advance_if_due().expect("before deadline").is_none());

    clock.set(deadline);
    let advance = machine
        .advance_if_due()
        .expect("advance")
        .expect("deadline reached");
    assert_eq!(advance.blue, TickDecision::Accepted);
    assert_eq!(advance.yellow, TickDecision::DeadlineFallback);
    assert_eq!(advance.state.tick, target_tick + 4);
    assert_eq!(advance.next_deadline_ns, deadline * 2);
}

#[test]
fn rejects_late_wrong_tick_and_out_of_range_action() {
    let (config, initial) = fixture();
    let backend = RapierBackend::new(config.clone(), initial).expect("valid backend");
    let clock = FakeClock::new();
    let mut machine = MatchMachine::new(backend, clock.clone(), config, FallbackPolicy::RepeatLast)
        .expect("machine");
    let target_tick = machine.start().expect("reset").tick;

    assert!(
        machine
            .submit(ControllerSlot::Blue, target_tick + 1, moving(1.0))
            .is_err()
    );
    assert!(
        machine
            .submit(ControllerSlot::Blue, target_tick, moving(1000.0))
            .is_err()
    );
    clock.set(machine.deadline_ns() + 1);
    assert!(
        machine
            .submit(ControllerSlot::Blue, target_tick, moving(1.0))
            .is_err()
    );
}

#[test]
fn sessions_enforce_slot_sequence_and_lease() {
    let mut sessions = SessionRegistry::default();
    sessions
        .register(
            ControllerIdentity {
                routing_id: b"blue-route".to_vec(),
                name: "blue".to_owned(),
            },
            ControllerSlot::Blue,
            100,
        )
        .expect("register blue");

    sessions
        .accept(b"blue-route", ControllerSlot::Blue, 1, 120)
        .expect("first message");
    assert_eq!(
        sessions.accept(b"blue-route", ControllerSlot::Blue, 1, 130),
        Err(SessionError::StaleSequence {
            received: 1,
            last: 1
        })
    );
    assert_eq!(
        sessions.accept(b"blue-route", ControllerSlot::Yellow, 2, 130),
        Err(SessionError::WrongSlot)
    );

    assert_eq!(
        sessions.adjudicate_leases(221, 100),
        LeaseAdjudication::YellowWinsByForfeit
    );
    assert_eq!(
        sessions.get(b"blue-route").expect("session").state,
        SessionState::Disconnected
    );
}
