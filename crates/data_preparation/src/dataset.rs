use crate::sample::Sample;
use anyhow::Result;

/// A `Dataset` provides unified, thread-safe access to data samples.
///
/// Implementations can be either:
/// - **In-memory** (`InMemoryDataset`) for data that fits in RAM,
/// - **Streaming** (`IterableDataset`) for large or continuous data sources.
pub trait Dataset: Send + Sync {
    /// Creates an iterator over all `Sample`s in the dataset.
    fn iter(&self) -> Box<dyn Iterator<Item = Result<Sample>> + Send + '_>;
}
