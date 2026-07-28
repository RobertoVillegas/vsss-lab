//! Native contiguous Python batch API.

use numpy::{PyArray1, PyArray2, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};
use vsss_batch::PhysicsBatch;
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
