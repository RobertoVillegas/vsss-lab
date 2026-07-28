//! Backend-independent robot commands.

use serde::{Deserialize, Serialize};

use crate::{AngularVelocity, LinearVelocity, Validate, ValidationError};

/// Interpretation of the two action channels.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    /// Left and right wheel angular velocity.
    WheelVelocity,
    /// Body linear and angular velocity.
    BodyVelocity,
}

/// A command for one physical robot.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobotAction {
    /// Channel interpretation.
    pub mode: ControlMode,
    /// Left wheel speed, or body linear velocity.
    pub left: f32,
    /// Right wheel speed, or body angular velocity.
    pub right: f32,
}

impl RobotAction {
    /// Creates a wheel-velocity command.
    #[must_use]
    pub const fn wheel_velocity(left: AngularVelocity, right: AngularVelocity) -> Self {
        Self {
            mode: ControlMode::WheelVelocity,
            left: left.get(),
            right: right.get(),
        }
    }

    /// Creates a body-velocity command.
    #[must_use]
    pub const fn body_velocity(linear: LinearVelocity, angular: AngularVelocity) -> Self {
        Self {
            mode: ControlMode::BodyVelocity,
            left: linear.get(),
            right: angular.get(),
        }
    }
}

impl Validate for RobotAction {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.left.is_finite() && self.right.is_finite() {
            Ok(())
        } else {
            Err(ValidationError::new("action", "channels must be finite"))
        }
    }
}
