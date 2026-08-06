//! Native contiguous Python batch API.

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};
use vsss_batch::PhysicsBatch;
use vsss_features::actions::circular_primitive_wheel_actions;
use vsss_features::baseline::scripted_team_actions;
use vsss_features::contact::{Contacts, Rules as ContactRules, contact_metrics};
use vsss_features::geometry::{Geometry, goal_geometry_metrics};
use vsss_features::roles::{Assignment, HystereticAssigner, Role, assign_roles, role_features};
use vsss_features::scalars::{Field as ScalarField, team_scalars};
use vsss_features::spin::{Spin, Thresholds, idle_spin};
use vsss_features::{Observation, group_widths, team_observation};
use vsss_physics_api::PhysicsBackend;
use vsss_physics_rapier::RapierBackend;
use vsss_spec::{
    AngularVelocity, Distance, LinearVelocity, MatchConfig, MatchState, RobotAction, Team,
    serialization,
};

/// Number of scalars in one canonical flattened M3 state row.
pub const STATE_WIDTH: usize = 77;

/// Native independent-world simulator.
#[pyclass]
struct BatchSimulator {
    batch: PhysicsBatch<RapierBackend>,
    /// One assigner per world, so hysteresis follows a world across decisions and resets with it.
    assigners: Vec<HystereticAssigner>,
}

/// Role features, per-slot change flags, the coverage flag and the joint cost, per world.
type RoleArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f64>>,
);

/// [`RoleArrays`] together with the role names, in slot order, for callers that use them.
type NamedRoleArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f64>>,
    Vec<Vec<&'static str>>,
);

/// One world's previous ball position and its two streak vectors.
type ContactInput = (
    (f64, f64),
    [i64; vsss_features::contact::ALLY_PAIRS],
    [i64; vsss_features::contact::OPPONENT_PAIRS],
);

/// Updated ally streaks, updated opponent streaks, and a summary row, all per world.
type ContactArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
);

/// Per-slot spin flags and the intensity the penalty scales by, one row per world.
type SpinArrays<'py> = (Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<f32>>);

/// Flatten a batch of assignments into the arrays Python reads them back as.
fn assignment_arrays<'py>(
    py: Python<'py>,
    assignments: &[Assignment],
) -> PyResult<RoleArrays<'py>> {
    let stride = vsss_features::TEAM_SIZE * vsss_features::ROLE_WIDTH;
    let mut features = vec![0.0f32; assignments.len() * stride];
    let mut changed = Vec::with_capacity(assignments.len());
    let mut uncovered = Vec::with_capacity(assignments.len());
    let mut cost = Vec::with_capacity(assignments.len());
    for (index, assignment) in assignments.iter().enumerate() {
        role_features(
            assignment,
            &mut features[index * stride..(index + 1) * stride],
        )
        .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        changed.push(
            assignment
                .changed
                .iter()
                .map(|flag| i64::from(*flag))
                .collect::<Vec<_>>(),
        );
        uncovered.push(assignment.uncovered);
        cost.push(assignment.cost);
    }
    let rows: Vec<Vec<f32>> = features.chunks(stride).map(<[f32]>::to_vec).collect();
    Ok((
        PyArray2::from_vec2(py, &rows)?,
        PyArray2::from_vec2(py, &changed)?,
        PyArray1::from_vec(py, uncovered),
        PyArray1::from_vec(py, cost),
    ))
}

/// Decode the role names Python names them by, in slot order.
fn role_names(assignment: &Assignment) -> Vec<&'static str> {
    assignment
        .roles
        .iter()
        .map(|role| match role {
            Role::Attacker => "attacker",
            Role::Support => "support",
            Role::Coverage => "coverage",
        })
        .collect()
}

#[pymethods]
impl BatchSimulator {
    #[new]
    #[pyo3(signature = (
        config_json,
        state_json,
        num_worlds,
        role_switch_penalty = vsss_features::roles::SWITCH_PENALTY,
        role_emergency_margin = vsss_features::roles::EMERGENCY_MARGIN
    ))]
    fn new(
        config_json: &str,
        state_json: &str,
        num_worlds: usize,
        role_switch_penalty: f64,
        role_emergency_margin: f64,
    ) -> PyResult<Self> {
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
            assigners: vec![
                HystereticAssigner::with_hysteresis(
                    role_switch_penalty,
                    role_emergency_margin
                );
                num_worlds
            ],
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

    /// Assign roles for every world, optionally carrying each world's own hysteresis.
    ///
    /// Returns the role features an observation consumes, the per-slot change flags, the
    /// coverage flag and the joint cost, plus the role names for callers that use them.
    ///
    /// `hysteretic` selects between the two calls the environment genuinely needs. The observation
    /// path wants history, so roles do not thrash between decisions. The reward path must not have
    /// it: a shaping potential has to be a function of the state alone, and an assignment that
    /// depends on the previous one is not. The two disagree on about seven per cent of decisions.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    fn team_roles<'py>(
        &mut self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        hysteretic: bool,
    ) -> PyResult<NamedRoleArrays<'py>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let assigners = &mut self.assigners;
        let assignments = py
            .detach(|| -> Result<Vec<Assignment>, String> {
                states
                    .iter()
                    .zip(teams)
                    .zip(assigners)
                    .map(|((state, team), assigner)| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let team = *team as u8;
                        if hysteretic {
                            assigner.assign(state, team)
                        } else {
                            assign_roles(state, team, None)
                        }
                        .map_err(|error| format!("{error:?}"))
                    })
                    .collect()
            })
            .map_err(PyValueError::new_err)?;
        let (features, changed, uncovered, cost) = assignment_arrays(py, &assignments)?;
        let names = assignments.iter().map(role_names).collect();
        Ok((features, changed, uncovered, cost, names))
    }

    /// Turn one team's circular primitive tokens into wheel commands, for every world.
    ///
    /// The reference ran this once per robot per decision through numpy, which the profile put
    /// at almost a fifth of an iteration. Reading the states from the batch keeps the tokens the
    /// only thing that has to cross the boundary.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (teams, tokens, ball_deceleration, strike_clearing_enabled = true, strike_clearing_distance = 0.16))]
    fn circular_wheel_actions<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        tokens: PyReadonlyArray3<'py, f32>,
        ball_deceleration: f64,
        strike_clearing_enabled: bool,
        strike_clearing_distance: f64,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let view = tokens.as_array();
        if view.shape() != [worlds, vsss_features::TEAM_SIZE, 3] {
            return Err(PyValueError::new_err(format!(
                "tokens must have shape ({worlds}, {}, 3)",
                vsss_features::TEAM_SIZE
            )));
        }
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let requests: Vec<[[f32; 3]; vsss_features::TEAM_SIZE]> = (0..worlds)
            .map(|world| {
                core::array::from_fn(|slot| {
                    core::array::from_fn(|channel| view[[world, slot, channel]])
                })
            })
            .collect();
        let wheels = py
            .detach(
                || -> Result<Vec<[[f32; 2]; vsss_features::TEAM_SIZE]>, String> {
                    states
                        .iter()
                        .zip(teams)
                        .zip(&requests)
                        .map(|((state, team), request)| {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            let team = *team as u8;
                            circular_primitive_wheel_actions(
                                state,
                                team,
                                request,
                                ball_deceleration,
                                strike_clearing_enabled,
                                strike_clearing_distance,
                            )
                            .map_err(|error| format!("{error:?}"))
                        })
                        .collect()
                },
            )
            .map_err(PyValueError::new_err)?;
        let nested: Vec<Vec<Vec<f32>>> = wheels
            .iter()
            .map(|world| world.iter().map(|pair| pair.to_vec()).collect())
            .collect();
        Ok(PyArray3::from_vec3(py, &nested)?)
    }

    /// Describe every world's attacking line, the quantity the shaping term is built from.
    ///
    /// Returns one row per world holding the potential followed by the four components it is
    /// assembled from, so the decomposition the reward records stays assertable term by term.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    fn goal_geometry<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        field_length: f64,
        goal_width: f64,
        ball_radius: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let geometry = Geometry {
            field_length,
            goal_width,
            ball_radius,
        };
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let rows = py
            .detach(|| -> Result<Vec<Vec<f64>>, String> {
                states
                    .iter()
                    .zip(teams)
                    .map(|(state, team)| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let team = *team as u8;
                        goal_geometry_metrics(state, team, geometry)
                            .map(|metrics| {
                                vec![
                                    metrics.potential,
                                    metrics.attacker_alignment,
                                    metrics.goal_aperture,
                                    metrics.controllable_proximity,
                                    metrics.attacking_progress,
                                ]
                            })
                            .map_err(|error| format!("{error:?}"))
                    })
                    .collect()
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    /// Flag robots spinning in place across every world.
    ///
    /// Returns the per-slot flags and the intensity the penalty scales by, so the caller keeps
    /// both the count it reports and the magnitude it charges.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn idle_spin<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        actions: PyReadonlyArray3<'py, f32>,
        angular_speed: f64,
        drive: f64,
        speed: f64,
        ball_distance: f64,
    ) -> PyResult<SpinArrays<'py>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let view = actions.as_array();
        if view.shape() != [worlds, vsss_features::TEAM_SIZE, 2] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({worlds}, {}, 2)",
                vsss_features::TEAM_SIZE
            )));
        }
        let thresholds = Thresholds {
            angular_speed,
            drive,
            speed,
            ball_distance,
        };
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let commands: Vec<[[f32; 2]; vsss_features::TEAM_SIZE]> = (0..worlds)
            .map(|world| {
                core::array::from_fn(|slot| core::array::from_fn(|side| view[[world, slot, side]]))
            })
            .collect();
        let spins = py
            .detach(|| -> Result<Vec<Spin>, String> {
                states
                    .iter()
                    .zip(teams)
                    .zip(&commands)
                    .map(|((state, team), action)| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let team = *team as u8;
                        idle_spin(state, team, action, thresholds)
                            .map_err(|error| format!("{error:?}"))
                    })
                    .collect()
            })
            .map_err(PyValueError::new_err)?;
        let flags: Vec<Vec<bool>> = spins.iter().map(|spin| spin.flags.to_vec()).collect();
        let intensity: Vec<Vec<f32>> = spins.iter().map(|spin| spin.intensity.to_vec()).collect();
        Ok((
            PyArray2::from_vec2(py, &flags)?,
            PyArray2::from_vec2(py, &intensity)?,
        ))
    }

    /// Measure sustained contact and deadlock across every world.
    ///
    /// The streaks travel in and out rather than living here, because an episode boundary
    /// resets them and the environment already owns when that happens.
    ///
    /// Returns the updated ally and opponent streaks, then a per-world row holding the ally
    /// penalty, the opponent penalty, the two contact counts, the two deadlock counts and the
    /// escapes, in that order.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn contacts<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        previous_ball: PyReadonlyArray2<'py, f32>,
        ally_streaks: PyReadonlyArray2<'py, i64>,
        opponent_streaks: PyReadonlyArray2<'py, i64>,
        contact_distance: f64,
        grace_steps: i64,
        meaningful_ball_displacement: f64,
        robot: (f64, f64, f64),
    ) -> PyResult<ContactArrays<'py>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let ball_view = previous_ball.as_array();
        let ally_view = ally_streaks.as_array();
        let opponent_view = opponent_streaks.as_array();
        if ball_view.shape() != [worlds, 2]
            || ally_view.shape() != [worlds, vsss_features::contact::ALLY_PAIRS]
            || opponent_view.shape() != [worlds, vsss_features::contact::OPPONENT_PAIRS]
        {
            return Err(PyValueError::new_err(
                "previous ball and streak arrays must have one row per world",
            ));
        }
        let (robot_length, robot_width, ball_radius) = robot;
        let rules = ContactRules {
            contact_distance,
            grace_steps,
            meaningful_ball_displacement,
            robot_length,
            robot_width,
            ball_radius,
        };
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let inputs: Vec<ContactInput> = (0..worlds)
            .map(|world| {
                (
                    (
                        f64::from(ball_view[[world, 0]]),
                        f64::from(ball_view[[world, 1]]),
                    ),
                    core::array::from_fn(|pair| ally_view[[world, pair]]),
                    core::array::from_fn(|pair| opponent_view[[world, pair]]),
                )
            })
            .collect();
        let measured = py
            .detach(|| -> Result<Vec<Contacts>, String> {
                states
                    .iter()
                    .zip(teams)
                    .zip(&inputs)
                    .map(|((state, team), (ball, ally, opponent))| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let team = *team as u8;
                        contact_metrics(state, team, *ball, ally, opponent, rules)
                            .map_err(|error| format!("{error:?}"))
                    })
                    .collect()
            })
            .map_err(PyValueError::new_err)?;

        let ally: Vec<Vec<i64>> = measured
            .iter()
            .map(|found| found.ally_streaks.to_vec())
            .collect();
        let opponent: Vec<Vec<i64>> = measured
            .iter()
            .map(|found| found.opponent_streaks.to_vec())
            .collect();
        #[allow(clippy::cast_precision_loss)] // counts are at most nine per world
        let summary: Vec<Vec<f64>> = measured
            .iter()
            .map(|found| {
                vec![
                    found.ally_penalty,
                    found.opponent_penalty,
                    found.ally_contacts as f64,
                    found.opponent_contacts as f64,
                    found.ally_deadlocks as f64,
                    found.opponent_deadlocks as f64,
                    found.escapes as f64,
                ]
            })
            .collect();
        Ok((
            PyArray2::from_vec2(py, &ally)?,
            PyArray2::from_vec2(py, &opponent)?,
            PyArray2::from_vec2(py, &summary)?,
        ))
    }

    /// Plan the scripted controller's wheel commands for one team in every world.
    ///
    /// `teams` names the scripted side per world, which is the side the learner does not
    /// control. The commands are normalized against the wheel limit, as the reference's are.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    fn scripted_actions<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let wheels = py
            .detach(
                || -> Result<Vec<[[f32; 2]; vsss_features::TEAM_SIZE]>, String> {
                    states
                        .iter()
                        .zip(teams)
                        .map(|(state, team)| {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            let team = *team as u8;
                            scripted_team_actions(state, team).map_err(|error| format!("{error:?}"))
                        })
                        .collect()
                },
            )
            .map_err(PyValueError::new_err)?;
        let nested: Vec<Vec<Vec<f32>>> = wheels
            .iter()
            .map(|world| world.iter().map(|pair| pair.to_vec()).collect())
            .collect();
        Ok(PyArray3::from_vec3(py, &nested)?)
    }

    /// Measure the per-world scalars in one pass.
    ///
    /// Returns a row per world holding the ball-touch flag, the distance to the nearest robot,
    /// the teammate congestion, the distance to the defensive post, the attacker's approach
    /// alignment and the ball's direction of travel, in that order.
    // PyO3 requires argument types by value in an exported signature.
    #[allow(clippy::needless_pass_by_value)]
    fn team_scalars<'py>(
        &self,
        py: Python<'py>,
        teams: PyReadonlyArray1<'py, i64>,
        field: (f64, f64, f64, f64, f64, f64),
        speed_threshold: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let worlds = self.batch.len();
        let teams = teams.as_slice()?;
        if teams.len() != worlds {
            return Err(PyValueError::new_err(
                "one team index per world is required",
            ));
        }
        let (length, goal_width, robot_length, robot_width, ball_radius, teammate_spacing) = field;
        let field = ScalarField {
            length,
            goal_width,
            robot_length,
            robot_width,
            ball_radius,
            teammate_spacing,
        };
        let states: Vec<Vec<f32>> = (0..worlds)
            .map(|index| flatten_state(&self.batch.world(index).snapshot()))
            .collect();
        let rows = py
            .detach(|| -> Result<Vec<Vec<f64>>, String> {
                states
                    .iter()
                    .zip(teams)
                    .map(|(state, team)| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let team = *team as u8;
                        team_scalars(state, team, field, speed_threshold)
                            .map(|found| {
                                vec![
                                    f64::from(u8::from(found.touches_ball)),
                                    found.closest_distance,
                                    found.congestion,
                                    found.defensive_distance,
                                    found.attacker_alignment,
                                    found.ball_direction,
                                ]
                            })
                            .map_err(|error| format!("{error:?}"))
                    })
                    .collect()
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    /// Place the ball on the quadrant's free-ball mark and let play continue.
    ///
    /// Rule 15 resolves an impasse away from both goal areas by repositioning, not by ending
    /// the game. The reference did this by serializing the world to a JSON dictionary, editing
    /// it in Python and parsing it back, inside the hot loop; here the state is edited where it
    /// already lives.
    ///
    /// Returns whether the restart applied and the world's state after it. Inside a goal area
    /// the correct restart is a goal kick, which is not modelled, so nothing is moved and the
    /// caller is told so rather than being given a silently unchanged state.
    #[allow(clippy::too_many_arguments)]
    fn restart_free_ball<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
        mark_x: f32,
        mark_y: f32,
        clearance: f32,
        goal_area_depth: f32,
        goal_area_half_width: f32,
        field_length: f32,
    ) -> PyResult<(bool, Bound<'py, PyArray1<f32>>)> {
        if index >= self.batch.len() {
            return Err(PyValueError::new_err("world index out of range"));
        }
        let mut state = self.batch.world(index).snapshot();
        let ball_x = state.ball.x.0;
        let ball_y = state.ball.y.0;
        if ball_x.abs() >= field_length / 2.0 - goal_area_depth
            && ball_y.abs() <= goal_area_half_width
        {
            let row = flatten_state(&state);
            return Ok((false, PyArray1::from_vec(py, row)));
        }

        // A ball resting exactly on an axis has no side to be placed on, and the reference
        // resolves that by treating zero as positive rather than by leaving the sign undefined.
        let placed_x = mark_x.copysign(if ball_x == 0.0 { 1.0 } else { ball_x });
        let placed_y = mark_y.copysign(if ball_y == 0.0 { 1.0 } else { ball_y });
        state.ball.x = Distance(placed_x);
        state.ball.y = Distance(placed_y);
        state.ball.vx = LinearVelocity(0.0);
        state.ball.vy = LinearVelocity(0.0);
        state.ball.omega = AngularVelocity(0.0);

        for robot in &mut state.robots {
            let away = (robot.pose.x.0 - placed_x).hypot(robot.pose.y.0 - placed_y);
            if away < clearance {
                let own_sign = if robot.team == Team::Blue { -1.0 } else { 1.0 };
                robot.pose.x = Distance(own_sign * mark_x);
                robot.pose.y = Distance(-placed_y);
            }
            robot.twist.vx = LinearVelocity(0.0);
            robot.twist.vy = LinearVelocity(0.0);
            robot.twist.omega = AngularVelocity(0.0);
            robot.wheel_speed_left = AngularVelocity(0.0);
            robot.wheel_speed_right = AngularVelocity(0.0);
        }

        self.batch
            .world_mut(index)
            .restore(&state)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let row = flatten_state(&self.batch.world(index).snapshot());
        Ok((true, PyArray1::from_vec(py, row)))
    }

    /// Forget one world's role history, as an episode boundary does.
    ///
    /// Clearing the history is only half of what a restart means. The reference immediately
    /// assigns again and keeps that result as the new history, so the next decision compares
    /// against the restarted assignment rather than against nothing; this does the same and
    /// returns it, since the caller needs it to rebuild that world's observation.
    fn restart_roles<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
        team: u8,
    ) -> PyResult<NamedRoleArrays<'py>> {
        if index >= self.batch.len() {
            return Err(PyValueError::new_err("world index out of range"));
        }
        let state = flatten_state(&self.batch.world(index).snapshot());
        let assigner = &mut self.assigners[index];
        assigner.reset();
        let assignment = assigner
            .assign(&state, team)
            .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        let (features, changed, uncovered, cost) = assignment_arrays(py, &[assignment])?;
        Ok((
            features,
            changed,
            uncovered,
            cost,
            vec![role_names(&assignment)],
        ))
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
