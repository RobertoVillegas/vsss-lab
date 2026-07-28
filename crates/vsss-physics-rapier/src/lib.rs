//! Fixed-step `Rapier2D` reference backend.

use rapier2d::prelude::*;
use vsss_physics_api::{PhysicsBackend, PhysicsError};
use vsss_spec::{
    Angle, AngularVelocity, ControlMode, Distance, EventFlags, LinearVelocity, MatchConfig,
    MatchState, RobotAction, Seconds, Validate,
};

/// Headless deterministic M2 physics world.
pub struct RapierBackend {
    config: MatchConfig,
    initial: MatchState,
    state: MatchState,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    robot_handles: [RigidBodyHandle; 6],
    ball_handle: RigidBodyHandle,
}

impl RapierBackend {
    /// Builds a world from validated canonical config and kickoff state.
    ///
    /// # Errors
    ///
    /// Returns an error when config or initial state violates canonical invariants.
    pub fn new(config: MatchConfig, initial: MatchState) -> Result<Self, PhysicsError> {
        config.validate().map_err(PhysicsError::InvalidState)?;
        initial.validate().map_err(PhysicsError::InvalidState)?;
        Ok(Self::build(config, initial.clone(), initial))
    }

    fn build(config: MatchConfig, initial: MatchState, state: MatchState) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let robot_handles = core::array::from_fn(|index| {
            let robot = state.robots[index];
            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(robot.pose.x.get(), robot.pose.y.get()))
                .rotation(robot.pose.theta.get())
                .linear_damping(0.4)
                .angular_damping(0.4)
                .build();
            let handle = bodies.insert(body);
            let collider = ColliderBuilder::cuboid(
                config.robot.length.get() / 2.0,
                config.robot.width.get() / 2.0,
            )
            .density(
                config.robot.mass.get() / (config.robot.length.get() * config.robot.width.get()),
            )
            .friction(config.friction)
            .restitution(config.restitution)
            .build();
            colliders.insert_with_parent(collider, handle, &mut bodies);
            handle
        });
        let ball_body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(state.ball.x.get(), state.ball.y.get()))
            .ccd_enabled(true)
            .linear_damping(0.15)
            .angular_damping(0.1)
            .build();
        let ball_handle = bodies.insert(ball_body);
        colliders.insert_with_parent(
            ColliderBuilder::ball(config.ball.radius.get())
                .density(config.ball.mass.get())
                .friction(config.friction)
                .restitution(config.restitution)
                .build(),
            ball_handle,
            &mut bodies,
        );
        add_walls(&config, &mut colliders);
        let mut result = Self {
            config,
            initial,
            state,
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies,
            colliders,
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            robot_handles,
            ball_handle,
        };
        result.sync_velocities();
        result
    }

    fn sync_velocities(&mut self) {
        for (index, handle) in self.robot_handles.into_iter().enumerate() {
            let robot = self.state.robots[index];
            self.bodies[handle].set_linvel(
                Vector::new(robot.twist.vx.get(), robot.twist.vy.get()),
                false,
            );
            self.bodies[handle].set_angvel(robot.twist.omega.get(), false);
        }
        self.bodies[self.ball_handle].set_linvel(
            Vector::new(self.state.ball.vx.get(), self.state.ball.vy.get()),
            false,
        );
        self.bodies[self.ball_handle].set_angvel(self.state.ball.omega.get(), false);
    }

    fn apply_actions(&mut self, actions: &[RobotAction; 6]) {
        let limit = self.config.max_wheel_speed.get().abs();
        for (index, action) in actions.iter().enumerate() {
            let (left, right, linear, angular) = match action.mode {
                ControlMode::WheelVelocity => {
                    let left = action.left.clamp(-limit, limit);
                    let right = action.right.clamp(-limit, limit);
                    let linear = self.config.wheel.radius.get() * (left + right) / 2.0;
                    let angular = self.config.wheel.radius.get() * (right - left)
                        / self.config.wheel.axle_track.get();
                    (left, right, linear, angular)
                }
                ControlMode::BodyVelocity => (0.0, 0.0, action.left, action.right),
            };
            let body = &mut self.bodies[self.robot_handles[index]];
            let theta = body.rotation().angle();
            body.set_linvel(
                Vector::new(linear * theta.cos(), linear * theta.sin()),
                true,
            );
            body.set_angvel(angular, true);
            self.state.robots[index].wheel_speed_left = AngularVelocity(left);
            self.state.robots[index].wheel_speed_right = AngularVelocity(right);
        }
    }

    fn read_state(&mut self) {
        for (index, handle) in self.robot_handles.into_iter().enumerate() {
            let body = &self.bodies[handle];
            self.state.robots[index].pose.x = Distance(body.translation().x);
            self.state.robots[index].pose.y = Distance(body.translation().y);
            self.state.robots[index].pose.theta = Angle(body.rotation().angle()).normalized();
            self.state.robots[index].twist.vx = LinearVelocity(body.linvel().x);
            self.state.robots[index].twist.vy = LinearVelocity(body.linvel().y);
            self.state.robots[index].twist.omega = AngularVelocity(body.angvel());
        }
        let ball = &self.bodies[self.ball_handle];
        self.state.ball.x = Distance(ball.translation().x);
        self.state.ball.y = Distance(ball.translation().y);
        self.state.ball.vx = LinearVelocity(ball.linvel().x);
        self.state.ball.vy = LinearVelocity(ball.linvel().y);
        self.state.ball.omega = AngularVelocity(ball.angvel());
    }

    fn detect_goal(&mut self) {
        self.state.events = EventFlags::NONE;
        let in_mouth = self.state.ball.y.get().abs() <= self.config.field.goal_width.get() / 2.0;
        let goal_line = self.config.field.length.get() / 2.0;
        if in_mouth && self.state.ball.x.get() > goal_line {
            self.state.score_blue = self.state.score_blue.saturating_add(1);
            self.state.events = EventFlags::GOAL_BLUE;
        } else if in_mouth && self.state.ball.x.get() < -goal_line {
            self.state.score_yellow = self.state.score_yellow.saturating_add(1);
            self.state.events = EventFlags::GOAL_YELLOW;
        }
    }
}

impl PhysicsBackend for RapierBackend {
    fn reset(&mut self) -> Result<MatchState, PhysicsError> {
        let config = self.config.clone();
        let initial = self.initial.clone();
        *self = Self::build(config, initial.clone(), initial);
        Ok(self.state.clone())
    }

    fn step(&mut self, actions: &[RobotAction; 6]) -> Result<MatchState, PhysicsError> {
        self.apply_actions(actions);
        let parameters = IntegrationParameters {
            dt: self.config.timestep.get(),
            num_solver_iterations: self.config.backend_substeps.into(),
            ..IntegrationParameters::default()
        };
        self.pipeline.step(
            Vector::new(0.0, 0.0),
            &parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
        self.state.tick += 1;
        self.state.simulation_time =
            Seconds(self.state.simulation_time.get() + self.config.timestep.get());
        self.read_state();
        self.detect_goal();
        Ok(self.state.clone())
    }

    fn snapshot(&self) -> MatchState {
        self.state.clone()
    }

    fn restore(&mut self, snapshot: &MatchState) -> Result<(), PhysicsError> {
        snapshot.validate().map_err(PhysicsError::InvalidState)?;
        *self = Self::build(self.config.clone(), self.initial.clone(), snapshot.clone());
        Ok(())
    }
}

fn add_walls(config: &MatchConfig, colliders: &mut ColliderSet) {
    let half_l = config.field.length.get() / 2.0;
    let half_w = config.field.width.get() / 2.0;
    let thickness = config.field.wall_thickness.get();
    colliders.insert(
        ColliderBuilder::cuboid(half_l + thickness, thickness / 2.0)
            .translation(Vector::new(0.0, half_w + thickness / 2.0))
            .build(),
    );
    colliders.insert(
        ColliderBuilder::cuboid(half_l + thickness, thickness / 2.0)
            .translation(Vector::new(0.0, -half_w - thickness / 2.0))
            .build(),
    );
    let segment = (config.field.width.get() - config.field.goal_width.get()) / 4.0;
    let offset = config.field.goal_width.get() / 2.0 + segment;
    for x in [-half_l - thickness / 2.0, half_l + thickness / 2.0] {
        for y in [-offset, offset] {
            colliders.insert(
                ColliderBuilder::cuboid(thickness / 2.0, segment)
                    .translation(Vector::new(x, y))
                    .build(),
            );
        }
    }
}
