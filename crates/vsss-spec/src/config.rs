//! Versioned match configuration.

use serde::{Deserialize, Serialize};

use crate::{
    AngularVelocity, BallProperties, FieldGeometry, Force, RobotGeometry, SCHEMA_VERSION, Seconds,
    Validate, ValidationError, WheelGeometry,
};

/// Backend selection without backend-specific dependencies.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Deterministic CPU reference backend.
    CpuReference,
    /// External validation backend.
    External,
}

/// Episode reset policy.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResetRules {
    /// Reset after a goal.
    pub after_goal: bool,
    /// Pause simulated time after a goal.
    pub goal_pause: Seconds,
}

/// Inclusive scalar randomization range.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarRange {
    /// Minimum multiplier.
    pub min: f32,
    /// Maximum multiplier.
    pub max: f32,
}

/// Backend-neutral physical randomization.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizationConfig {
    /// Friction multiplier.
    pub friction: ScalarRange,
    /// Restitution multiplier.
    pub restitution: ScalarRange,
    /// Motor multiplier.
    pub motor_strength: ScalarRange,
}

/// Complete effective match configuration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchConfig {
    /// Contract version.
    pub schema_version: u32,
    /// Field geometry.
    pub field: FieldGeometry,
    /// Robot geometry.
    pub robot: RobotGeometry,
    /// Wheel geometry.
    pub wheel: WheelGeometry,
    /// Ball properties.
    pub ball: BallProperties,
    /// Fixed physics timestep.
    pub timestep: Seconds,
    /// Period between accepted commands.
    pub control_period: Seconds,
    /// Maximum wheel speed.
    pub max_wheel_speed: AngularVelocity,
    /// Maximum actuator force.
    pub max_actuator_force: Force,
    /// Surface friction coefficient.
    pub friction: f32,
    /// Collision restitution coefficient.
    pub restitution: f32,
    /// Match duration.
    pub match_duration: Seconds,
    /// Reset behavior.
    pub reset: ResetRules,
    /// Domain randomization ranges.
    pub randomization: RandomizationConfig,
    /// Reproducibility seed.
    pub seed: u64,
    /// Selected backend family.
    pub backend: BackendKind,
    /// Fixed backend substeps per timestep.
    pub backend_substeps: u16,
}

impl Validate for MatchConfig {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ValidationError::new(
                "schema_version",
                "unsupported version",
            ));
        }
        let positive = [
            ("field.length", self.field.length.get()),
            ("field.width", self.field.width.get()),
            ("robot.length", self.robot.length.get()),
            ("robot.width", self.robot.width.get()),
            ("robot.mass", self.robot.mass.get()),
            ("wheel.radius", self.wheel.radius.get()),
            ("wheel.axle_track", self.wheel.axle_track.get()),
            ("ball.radius", self.ball.radius.get()),
            ("ball.mass", self.ball.mass.get()),
            ("timestep", self.timestep.get()),
            ("match_duration", self.match_duration.get()),
        ];
        for (path, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(ValidationError::new(path, "must be finite and positive"));
            }
        }
        if self.control_period.get() < self.timestep.get() {
            return Err(ValidationError::new(
                "control_period",
                "must not be shorter than timestep",
            ));
        }
        if !(0.0..=1.0).contains(&self.restitution)
            || !self.friction.is_finite()
            || self.friction < 0.0
        {
            return Err(ValidationError::new(
                "friction/restitution",
                "coefficients out of range",
            ));
        }
        if self.backend_substeps == 0 {
            return Err(ValidationError::new("backend_substeps", "must be positive"));
        }
        for range in [
            self.randomization.friction,
            self.randomization.restitution,
            self.randomization.motor_strength,
        ] {
            if !range.min.is_finite() || !range.max.is_finite() || range.min > range.max {
                return Err(ValidationError::new(
                    "randomization",
                    "ranges must be finite and ordered",
                ));
            }
        }
        Ok(())
    }
}
