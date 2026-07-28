//! Canonical match entities and state.

use serde::{Deserialize, Serialize};

use crate::{
    Angle, AngularVelocity, EventFlags, LinearVelocity, SCHEMA_VERSION, Seconds, Validate,
    ValidationError,
};

/// Physical robot identity, independent from policy role.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(u8)]
pub enum RobotId {
    /// Physical robot zero.
    R0,
    /// Physical robot one.
    R1,
    /// Physical robot two.
    R2,
    /// Physical robot three.
    R3,
    /// Physical robot four.
    R4,
    /// Physical robot five.
    R5,
}

/// Competition team.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Team {
    /// Blue team.
    Blue,
    /// Yellow team.
    Yellow,
}

impl Team {
    /// Returns the opposing team.
    #[must_use]
    pub const fn opposing(self) -> Self {
        match self {
            Self::Blue => Self::Yellow,
            Self::Yellow => Self::Blue,
        }
    }
}

/// Planar pose in canonical coordinates.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pose2 {
    /// X position in metres.
    pub x: crate::Distance,
    /// Y position in metres.
    pub y: crate::Distance,
    /// Heading in radians.
    pub theta: Angle,
}

/// Planar velocity.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Twist2 {
    /// X velocity.
    pub vx: LinearVelocity,
    /// Y velocity.
    pub vy: LinearVelocity,
    /// Angular velocity.
    pub omega: AngularVelocity,
}

/// State of one robot.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobotState {
    /// Physical identity.
    pub id: RobotId,
    /// Current team label.
    pub team: Team,
    /// Pose.
    pub pose: Pose2,
    /// Velocity.
    pub twist: Twist2,
    /// Left wheel angular velocity.
    pub wheel_speed_left: AngularVelocity,
    /// Right wheel angular velocity.
    pub wheel_speed_right: AngularVelocity,
    /// Whether commands and collisions are enabled.
    pub enabled: bool,
}

/// State of the ball.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BallState {
    /// X position.
    pub x: crate::Distance,
    /// Y position.
    pub y: crate::Distance,
    /// X velocity.
    pub vx: LinearVelocity,
    /// Y velocity.
    pub vy: LinearVelocity,
    /// Angular velocity.
    pub omega: AngularVelocity,
}

/// Complete canonical state at one tick.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchState {
    /// Contract version.
    pub schema_version: u32,
    /// Fixed-step tick.
    pub tick: u64,
    /// Simulated time.
    pub simulation_time: Seconds,
    /// Blue score.
    pub score_blue: u16,
    /// Yellow score.
    pub score_yellow: u16,
    /// Ball state.
    pub ball: BallState,
    /// Exactly six physical robots.
    pub robots: [RobotState; 6],
    /// Tick events.
    pub events: EventFlags,
}

impl MatchState {
    /// Rotates the field by π and swaps blue/yellow labels and results.
    #[must_use]
    pub fn reflected(&self) -> Self {
        let mut state = self.clone();
        core::mem::swap(&mut state.score_blue, &mut state.score_yellow);
        state.ball.x.0 = -state.ball.x.0;
        state.ball.y.0 = -state.ball.y.0;
        state.ball.vx.0 = -state.ball.vx.0;
        state.ball.vy.0 = -state.ball.vy.0;
        for robot in &mut state.robots {
            robot.team = robot.team.opposing();
            robot.pose.x.0 = -robot.pose.x.0;
            robot.pose.y.0 = -robot.pose.y.0;
            robot.pose.theta = Angle(robot.pose.theta.get() + core::f32::consts::PI).normalized();
            robot.twist.vx.0 = -robot.twist.vx.0;
            robot.twist.vy.0 = -robot.twist.vy.0;
        }
        state.events = state.events.reflected();
        state
    }
}

impl Validate for MatchState {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ValidationError::new(
                "schema_version",
                "unsupported version",
            ));
        }
        if !self.simulation_time.is_finite() || self.simulation_time.get() < 0.0 {
            return Err(ValidationError::new(
                "simulation_time",
                "must be finite and non-negative",
            ));
        }
        let mut seen = [false; 6];
        for robot in self.robots {
            let index = robot.id as usize;
            if seen[index] {
                return Err(ValidationError::new(
                    "robots.id",
                    "robot IDs must be unique",
                ));
            }
            seen[index] = true;
            let values = [
                robot.pose.x.get(),
                robot.pose.y.get(),
                robot.pose.theta.get(),
                robot.twist.vx.get(),
                robot.twist.vy.get(),
                robot.twist.omega.get(),
                robot.wheel_speed_left.get(),
                robot.wheel_speed_right.get(),
            ];
            if !values.into_iter().all(f32::is_finite) {
                return Err(ValidationError::new(
                    "robots",
                    "physical values must be finite",
                ));
            }
        }
        let ball = [
            self.ball.x.get(),
            self.ball.y.get(),
            self.ball.vx.get(),
            self.ball.vy.get(),
            self.ball.omega.get(),
        ];
        if !ball.into_iter().all(f32::is_finite) {
            return Err(ValidationError::new(
                "ball",
                "physical values must be finite",
            ));
        }
        if self.events.0 & !EventFlags::KNOWN_BITS != 0 {
            return Err(ValidationError::new("events", "unknown event bits"));
        }
        Ok(())
    }
}
