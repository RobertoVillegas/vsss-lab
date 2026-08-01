//! Native contiguous Python batch API.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};
use vsss_batch::PhysicsBatch;
use vsss_features::{Observation, group_widths, team_observation};
use vsss_physics_api::PhysicsBackend;
use vsss_physics_rapier::RapierBackend;
use vsss_spec::{AngularVelocity, MatchConfig, MatchState, RobotAction, serialization};

/// Number of scalars in one canonical flattened M3 state row.
pub const STATE_WIDTH: usize = 77;

/// Native independent-world simulator.
#[pyclass]
struct BatchSimulator {
    batch: PhysicsBatch<RapierBackend>,
}

#[pymethods]
impl BatchSimulator {
    #[new]
    fn new(config_json: &str, state_json: &str, num_worlds: usize) -> PyResult<Self> {
        if num_worlds == 0 {
            return Err(PyValueError::new_err("num_worlds must be positive"));
        }
        let config: MatchConfig = serialization::from_json(config_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let state: MatchState = serialization::from_json(state_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let worlds = (0..num_worlds)
            .map(|_| RapierBackend::new(config.clone(), state.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            batch: PhysicsBatch::new(worlds),
        })
    }

    #[getter]
    fn num_worlds(&self) -> usize {
        self.batch.len()
    }

    #[staticmethod]
    fn state_width() -> usize {
        STATE_WIDTH
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let states = (0..self.batch.len())
            .map(|index| self.batch.reset_world(index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        rows_to_numpy(py, &states)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = actions.as_array();
        if view.shape() != [self.batch.len(), 6, 2] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 6, 2)",
                self.batch.len()
            )));
        }
        let commands = (0..self.batch.len())
            .map(|world| {
                core::array::from_fn(|robot| {
                    RobotAction::wheel_velocity(
                        AngularVelocity(view[[world, robot, 0]]),
                        AngularVelocity(view[[world, robot, 1]]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let states = py
            .detach(|| self.batch.step(&commands))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        rows_to_numpy(py, &states)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn step_repeated<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray3<'py, f32>,
        repeats: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if repeats == 0 {
            return Err(PyValueError::new_err("repeats must be positive"));
        }
        let commands = self.actions_from_numpy(&actions)?;
        let states = py
            .detach(|| self.batch.step_repeated(&commands, repeats))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        rows_to_numpy(py, &states)
    }

    /// Build every world's team observation in one call.
    ///
    /// The Python path built these with a loop over worlds on each decision, which measured
    /// at almost all of an environment step while the physics itself measured at one per cent.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    fn observations<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        roles: PyReadonlyArray2<'py, f32>,
        field_length: f32,
        field_width: f32,
        match_duration: f32,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        let roles = roles.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let role_stride = vsss_features::TEAM_SIZE * vsss_features::ROLE_WIDTH;
        if roles.len() != worlds * role_stride {
            return Err(PyValueError::new_err("one role row per world is required"));
        }
        let widths = group_widths();
        let mut buffers: Vec<Vec<f32>> = widths.iter().map(|w| vec![0.0f32; worlds * w]).collect();
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        py.detach(|| -> Result<(), String> {
            let (self_buf, rest) = buffers.split_at_mut(1);
            let (ball_buf, rest) = rest.split_at_mut(1);
            let (goal_buf, rest) = rest.split_at_mut(1);
            let (context_buf, rest) = rest.split_at_mut(1);
            let (teammate_buf, opponent_buf) = rest.split_at_mut(1);
            for index in 0..worlds {
                let mut out = Observation {
                    self_features: &mut self_buf[0][index * widths[0]..(index + 1) * widths[0]],
                    ball: &mut ball_buf[0][index * widths[1]..(index + 1) * widths[1]],
                    goals: &mut goal_buf[0][index * widths[2]..(index + 1) * widths[2]],
                    context: &mut context_buf[0][index * widths[3]..(index + 1) * widths[3]],
                    teammates: &mut teammate_buf[0][index * widths[4]..(index + 1) * widths[4]],
                    opponents: &mut opponent_buf[0][index * widths[5]..(index + 1) * widths[5]],
                };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let team = teams[index] as u8;
                team_observation(
                    &states[index],
                    team,
                    field_length,
                    field_width,
                    match_duration,
                    &roles[index * role_stride..(index + 1) * role_stride],
                    &mut out,
                )
                .map_err(|error| format!("{error:?}"))?;
            }
            Ok(())
        })
        .map_err(PyValueError::new_err)?;
        buffers
            .into_iter()
            .zip(widths)
            .map(|(flat, width)| {
                let rows: Vec<Vec<f32>> = flat.chunks(width).map(<[f32]>::to_vec).collect();
                Ok(PyArray2::from_vec2(py, &rows)?)
            })
            .collect()
    }

    fn snapshots(&self) -> PyResult<Vec<String>> {
        (0..self.batch.len())
            .map(|index| {
                serialization::to_json(&self.batch.world(index).snapshot())
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect()
    }

    fn restore(&mut self, index: usize, snapshot_json: &str) -> PyResult<()> {
        if index >= self.batch.len() {
            return Err(PyValueError::new_err("world index out of range"));
        }
        let state = serialization::from_json(snapshot_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.batch
            .world_mut(index)
            .restore(&state)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn restore_state<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
        snapshot_json: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.restore(index, snapshot_json)?;
        Ok(PyArray1::from_vec(
            py,
            flatten_state(&self.batch.world(index).snapshot()),
        ))
    }
}

impl BatchSimulator {
    fn actions_from_numpy(
        &self,
        actions: &PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Vec<[RobotAction; 6]>> {
        let view = actions.as_array();
        if view.shape() != [self.batch.len(), 6, 2] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 6, 2)",
                self.batch.len()
            )));
        }
        Ok((0..self.batch.len())
            .map(|world| {
                core::array::from_fn(|robot| {
                    RobotAction::wheel_velocity(
                        AngularVelocity(view[[world, robot, 0]]),
                        AngularVelocity(view[[world, robot, 1]]),
                    )
                })
            })
            .collect())
    }
}

fn rows_to_numpy<'py>(
    py: Python<'py>,
    states: &[MatchState],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows = states.iter().map(flatten_state).collect::<Vec<_>>();
    Ok(PyArray2::from_vec2(py, &rows)?)
}

#[allow(clippy::cast_precision_loss)]
fn flatten_state(state: &MatchState) -> Vec<f32> {
    let mut row = Vec::with_capacity(STATE_WIDTH);
    row.extend([
        state.schema_version as f32,
        state.tick as f32,
        state.simulation_time.get(),
        f32::from(state.score_blue),
        f32::from(state.score_yellow),
    ]);
    row.extend([
        state.ball.x.get(),
        state.ball.y.get(),
        state.ball.vx.get(),
        state.ball.vy.get(),
        state.ball.omega.get(),
    ]);
    for robot in state.robots {
        row.extend([
            f32::from(robot.id as u8),
            f32::from(robot.team as u8),
            robot.pose.x.get(),
            robot.pose.y.get(),
            robot.pose.theta.get(),
            robot.twist.vx.get(),
            robot.twist.vy.get(),
            robot.twist.omega.get(),
            robot.wheel_speed_left.get(),
            robot.wheel_speed_right.get(),
            f32::from(robot.enabled),
        ]);
    }
    row.push(state.events.0 as f32);
    debug_assert_eq!(row.len(), STATE_WIDTH);
    row
}

/// Native extension module.
#[pymodule]
mod _native {
    #[pymodule_export]
    use super::BatchSimulator;
}
