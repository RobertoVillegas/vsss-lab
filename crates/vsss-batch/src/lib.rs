//! Sequential independent-world batch used by M2.

use vsss_physics_api::{PhysicsBackend, PhysicsError};
use vsss_spec::{MatchState, RobotAction};

/// A sequential collection of independent backend instances.
pub struct PhysicsBatch<B> {
    worlds: Vec<B>,
}

impl<B: PhysicsBackend> PhysicsBatch<B> {
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
        self.worlds
            .iter_mut()
            .zip(actions)
            .map(|(world, action)| world.step(action))
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
}
