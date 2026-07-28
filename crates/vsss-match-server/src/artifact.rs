//! Auditable match replay and result artifacts.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use vsss_physics_api::checksum_state;
use vsss_spec::{MatchConfig, MatchState};

use crate::{Advance, TickDecision};

/// Immutable metadata written before any replay tick.
#[derive(Clone, Debug, Serialize)]
pub struct MatchMetadata {
    /// Stable textual match ID.
    pub match_id: String,
    /// Canonical effective configuration.
    pub config: MatchConfig,
    /// Blue controller manifest/name.
    pub blue_controller: String,
    /// Yellow controller manifest/name.
    pub yellow_controller: String,
    /// Wire protocol version.
    pub protocol_version: u32,
    /// Build revision or explicit dirty marker.
    pub build_revision: String,
}

/// Terminal adjudicated outcome.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchOutcome {
    /// Match reached its configured duration.
    Completed,
    /// Blue won after Yellow forfeited.
    BlueForfeitWin,
    /// Yellow won after Blue forfeited.
    YellowForfeitWin,
    /// Neither controller retained its lease.
    DoubleForfeit,
}

#[derive(Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum ReplayRecord<'a> {
    Header {
        metadata: &'a MatchMetadata,
    },
    Tick {
        state: &'a MatchState,
        state_checksum: u64,
        blue_decision: &'a str,
        yellow_decision: &'a str,
        next_deadline_ns: u64,
    },
    Result {
        score_blue: u16,
        score_yellow: u16,
        outcome: MatchOutcome,
    },
}

/// Artifact I/O or lifecycle failure.
#[derive(Debug, Error)]
pub enum ArtifactError {
    /// Filesystem operation failed.
    #[error("artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization failed.
    #[error("artifact serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    /// Result was requested before any tick was recorded.
    #[error("cannot finish an artifact without a recorded state")]
    MissingState,
}

/// Streaming JSONL recorder with a SHA-256 checksum over exact replay bytes.
pub struct MatchArtifact {
    path: PathBuf,
    writer: BufWriter<File>,
    digest: Sha256,
    last_state: Option<MatchState>,
}

impl MatchArtifact {
    /// Create a replay and write its immutable header.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination or header cannot be written.
    pub fn create(path: impl AsRef<Path>, metadata: &MatchMetadata) -> Result<Self, ArtifactError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&path)?;
        let mut artifact = Self {
            path,
            writer: BufWriter::new(file),
            digest: Sha256::new(),
            last_state: None,
        };
        artifact.write(&ReplayRecord::Header { metadata })?;
        Ok(artifact)
    }

    /// Append one authoritative control-boundary result.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization or writing fails.
    pub fn record(&mut self, advance: &Advance) -> Result<(), ArtifactError> {
        self.write(&ReplayRecord::Tick {
            state: &advance.state,
            state_checksum: checksum_state(&advance.state),
            blue_decision: decision_name(advance.blue),
            yellow_decision: decision_name(advance.yellow),
            next_deadline_ns: advance.next_deadline_ns,
        })?;
        self.last_state = Some(advance.state.clone());
        Ok(())
    }

    /// Write the result, flush the file, and return its SHA-256 checksum.
    ///
    /// # Errors
    ///
    /// Returns an error without a prior state or when writing/flushing fails.
    pub fn finish(mut self, outcome: MatchOutcome) -> Result<String, ArtifactError> {
        let state = self
            .last_state
            .as_ref()
            .ok_or(ArtifactError::MissingState)?;
        let result = ReplayRecord::Result {
            score_blue: state.score_blue,
            score_yellow: state.score_yellow,
            outcome,
        };
        let mut bytes = serde_json::to_vec(&result)?;
        bytes.push(b'\n');
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.digest.update(&bytes);
        Ok(format!("{:x}", self.digest.finalize()))
    }

    /// Destination replay path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn write(&mut self, record: &ReplayRecord<'_>) -> Result<(), ArtifactError> {
        let mut bytes = serde_json::to_vec(record)?;
        bytes.push(b'\n');
        self.writer.write_all(&bytes)?;
        self.digest.update(&bytes);
        Ok(())
    }
}

const fn decision_name(decision: TickDecision) -> &'static str {
    match decision {
        TickDecision::Accepted => "accepted",
        TickDecision::DeadlineFallback => "deadline_fallback",
    }
}
