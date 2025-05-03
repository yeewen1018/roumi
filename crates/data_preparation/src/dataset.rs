use crate::sample::Sample;
use anyhow::Result;

/// A `Dataset` provides unified access to data samples.
///
/// Implementations can be either:
/// - In-memory (`InMemoryDataset`) for small datasets that fit in RAM,
/// - Streaming (`IterableDataset`) for large datasets.
///
/// All implementations must be `Send + Sync` to allow for safe
/// sharing across threads.
pub trait Dataset: Send + Sync {
    /// The iterator type produced by `iter()`.
    type Iter<'a>: Iterator<Item = Result<Sample>> + Send + 'a
    where
        Self: 'a;

    /// Creates an iterator over all samples in the dataset.
    fn iter(&self) -> Self::Iter<'_>;

    /// Retrieves a reference to a sample by index, if available.
    fn get(&self, index: usize) -> Option<Result<&Sample>>;

    /// Returns total number of samples, if known.
    fn len(&self) -> Option<usize>;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len().map(|l| l == 0).unwrap_or(true)
    }
}
