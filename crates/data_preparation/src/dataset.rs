use crate::sample::Sample;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

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

/// A dataset that stores all samples in a contiguous memory
/// with atomic-reference counting (`Arc<[Sample]>`).
///
/// This enables cheap cloning, thread-safe sharing, and efficient batching.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    samples: Arc<[Sample]>,
    metadata: HashMap<String, String>,
}

impl InMemoryDataset {
    /// Creates a new in-memory dataset from a vector of samples.
    pub fn new(samples: Vec<Sample>) -> Self {
        Self {
            samples: samples.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds/updates metadata and returns self (enables builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns the value of a metadata field, if it exists.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

impl Dataset for InMemoryDataset {
    type Iter<'a> = std::iter::Map<
        std::iter::Cloned<std::slice::Iter<'a, Sample>>,
        fn(Sample) -> Result<Sample>,
    >;

    fn iter(&self) -> Self::Iter<'_> {
        self.samples.iter().cloned().map(Ok)
    }

    fn get(&self, index: usize) -> Option<Result<&Sample>> {
        self.samples.get(index).map(Ok)
    }

    fn len(&self) -> Option<usize> {
        Some(self.samples.len())
    }
}
