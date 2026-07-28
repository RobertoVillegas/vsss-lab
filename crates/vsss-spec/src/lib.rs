//! Canonical, backend-independent contracts for VSSS Lab.
//!
//! All public values use explicit SI units. The types in this crate must remain
//! independent from physics engines, Python, ROS, and learning frameworks.

pub mod actions;
pub mod config;
pub mod entities;
pub mod events;
pub mod geometry;
pub mod reflection;
pub mod serialization;
pub mod units;
mod validation;

pub use actions::{ControlMode, RobotAction};
pub use config::{BackendKind, MatchConfig, RandomizationConfig, ResetRules};
pub use entities::{BallState, MatchState, Pose2, RobotId, RobotState, Team, Twist2};
pub use events::EventFlags;
pub use geometry::{BallProperties, FieldGeometry, RobotGeometry, WheelGeometry};
pub use reflection::{FieldDescriptor, FieldKind, TypeDescriptor, canonical_types};
pub use units::{Angle, AngularVelocity, Distance, Force, LinearVelocity, Mass, Seconds, Torque};
pub use validation::{Validate, ValidationError};

/// Current canonical contract version.
pub const SCHEMA_VERSION: u32 = 1;
