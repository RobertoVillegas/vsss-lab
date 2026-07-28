//! Replay artifact integrity tests.

use std::{cell::Cell, rc::Rc};

use tempfile::tempdir;
use vsss_match_server::{
    Clock, FallbackPolicy, MatchArtifact, MatchMachine, MatchMetadata, MatchOutcome,
};
use vsss_physics_rapier::RapierBackend;
use vsss_spec::{MatchConfig, MatchState, serialization};

#[derive(Clone)]
struct FakeClock(Rc<Cell<u64>>);

impl Clock for FakeClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

#[test]
fn artifact_records_canonical_state_decisions_and_result() {
    let config: MatchConfig =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_config.json"))
            .expect("config");
    let initial: MatchState =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_state.json"))
            .expect("state");
    let backend = RapierBackend::new(config.clone(), initial).expect("backend");
    let clock = FakeClock(Rc::new(Cell::new(0)));
    let mut machine =
        MatchMachine::new(backend, clock.clone(), config.clone(), FallbackPolicy::Zero)
            .expect("machine");
    machine.start().expect("start");
    clock.0.set(machine.deadline_ns());
    let advance = machine.advance_if_due().expect("advance").expect("due");

    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("match.jsonl");
    let metadata = MatchMetadata {
        match_id: "artifact-test".to_owned(),
        config,
        blue_controller: "blue".to_owned(),
        yellow_controller: "yellow".to_owned(),
        protocol_version: 1,
        build_revision: "test".to_owned(),
    };
    let mut artifact = MatchArtifact::create(&path, &metadata).expect("create");
    artifact.record(&advance).expect("record");
    let digest = artifact.finish(MatchOutcome::Completed).expect("finish");

    let replay = std::fs::read_to_string(path).expect("read replay");
    assert_eq!(replay.lines().count(), 3);
    assert!(replay.contains("\"state_checksum\""));
    assert!(replay.contains("\"deadline_fallback\""));
    assert_eq!(digest.len(), 64);
}
