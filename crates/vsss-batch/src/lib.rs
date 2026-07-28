//! Parallel independent-world batch used by training and evaluation.

use rayon::prelude::*;
use vsss_physics_api::{PhysicsBackend, PhysicsError};
use vsss_spec::{MatchState, RobotAction};

/// A sequential collection of independent backend instances.
pub struct PhysicsBatch<B> {
    worlds: Vec<B>,
}

const PARALLEL_WORLD_THRESHOLD: usize = 32;

impl<B: PhysicsBackend + Send> PhysicsBatch<B> {
    /// Creates a non-empty batch.
    ///
    /// # Panics
    ///
    /// Panics when `worlds` is empty.
    #[must_use]
    pub fn new(worlds: Vec<B>) -> Self {
        assert!(!worlds.is_empty(), "physics batch must not be empty");
        Self { worlds }
    }

    /// Returns the number of worlds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Returns whether the batch has no worlds.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Advances each world once in stable index order.
    ///
    /// # Errors
    ///
    /// Returns the first backend error in stable world order.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one action set is provided per world.
    pub fn step(&mut self, actions: &[[RobotAction; 6]]) -> Result<Vec<MatchState>, PhysicsError> {
        assert_eq!(actions.len(), self.worlds.len(), "one action set per world");
        if self.worlds.len() < PARALLEL_WORLD_THRESHOLD {
            return self
                .worlds
                .iter_mut()
                .zip(actions)
                .map(|(world, action)| world.step(action))
                .collect();
        }
        self.worlds
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(world, action)| world.step(action))
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }

    /// Advances every world multiple steps while scheduling each world once.
    ///
    /// # Errors
    ///
    /// Returns the first backend error in stable world and repeat order.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one action set is provided per world or `repeats`
    /// is zero.
    pub fn step_repeated(
        &mut self,
        actions: &[[RobotAction; 6]],
        repeats: usize,
    ) -> Result<Vec<MatchState>, PhysicsError> {
        assert_eq!(actions.len(), self.worlds.len(), "one action set per world");
        assert!(repeats > 0, "repeats must be positive");
        let advance = |world: &mut B, action: &[RobotAction; 6]| {
            let mut state = world.step(action)?;
            let mut events = state.events;
            for _ in 1..repeats {
                state = world.step(action)?;
                events.0 |= state.events.0;
            }
            state.events = events;
            Ok(state)
        };
        if self.worlds.len() < PARALLEL_WORLD_THRESHOLD {
            return self
                .worlds
                .iter_mut()
                .zip(actions)
                .map(|(world, action)| advance(world, action))
                .collect();
        }
        self.worlds
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(world, action)| advance(world, action))
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }

    /// Resets one world without touching its neighbors.
    ///
    /// # Errors
    ///
    /// Returns the selected backend's reset error.
    pub fn reset_world(&mut self, index: usize) -> Result<MatchState, PhysicsError> {
        self.worlds[index].reset()
    }

    /// Returns one world for inspection.
    #[must_use]
    pub fn world(&self, index: usize) -> &B {
        &self.worlds[index]
    }

    /// Returns one world for mutation.
    #[must_use]
    pub fn world_mut(&mut self, index: usize) -> &mut B {
        &mut self.worlds[index]
    }
}
