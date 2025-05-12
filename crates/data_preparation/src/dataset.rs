use crate::sample::Sample;
use crate::transform::Transform;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A `Dataset` provides unified, thread-safe access to data samples.
///
/// Implementations can be either:
/// - **In-memory** (`InMemoryDataset`) for data that fits in RAM,
/// - **Streaming** (`IterableDataset`) for large or continuous data sources.
pub trait Dataset: Send + Sync {
    /// Creates an iterator over all `Sample`s in the dataset.
    fn iter(&self) -> Box<dyn Iterator<Item = Result<Sample>> + Send + '_>;
}

/// -------------------------------------------------------------------------------------
/// In-memory Dataset
///
/// Stores all un-processed data (`Raw`) in a reference-counted slice (`Arc<[Raw]>`)
/// and applies an optional `Transform<Raw, Sample>` pipeline to produce `Sample`s.
///
/// - **Raw**: the original data type (e.g., `String` for lines of text,
///            `Vec<u8>` for image bytes, or a custom multimodal struct).
/// - **Sample**: the preprocessed, typed example ready for batching and model input.
/// - **metadata**: user-supplied key/value tags (e.g., "split" -> "train").
///
/// # Example
/// ```ignore
/// use crate::dataset::InMemoryDataset;
/// use crate::transform::Transform;
///
/// let ds = InMemoryDataset::new(vec!["foo", "bar", "baz"])
///     .with_transform(Transform::new())
///     .with_metadata("split", "train");
/// assert_eq!(ds.len(), 3);
/// ```
#[derive(Clone)]
pub struct InMemoryDataset<Raw> {
    raw_data: Arc<[Raw]>,
    transform: Option<Arc<dyn Transform<Raw, Sample> + Send + Sync>>,
    metadata: HashMap<String, String>,
}

impl<Raw> InMemoryDataset<Raw>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates a dataset from an existing collection.
    pub fn new(raw_data: Vec<Raw>) -> Self {
        Self {
            raw_data: Arc::from(raw_data),
            transform: None,
            metadata: HashMap::new(),
        }
    }

    /// Loads a dataset from all files in a directory (e.g., images, JSONL).
    /// Each file is converted to `Raw` using `From<PathBuf>`.
    pub fn from_directory<P>(dir: P) -> Result<Self>
    where
        P: AsRef<Path>,
        Raw: From<PathBuf>,
    {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir.as_ref())? {
            let path = entry?.path();
            if path.is_file() {
                files.push(Raw::from(path));
            }
        }
        Ok(Self::new(files))
    }

    /// Attaches a transform to convert `Raw` -> `Sample`.
    pub fn with_transform<T>(self, transform: T) -> Self
    where
        T: Transform<Raw, Sample> + Send + Sync + 'static,
    {
        Self {
            transform: Some(Arc::new(transform)),
            ..self
        }
    }

    /// Number of raw data stored.
    pub fn len(&self) -> usize {
        self.raw_data.len()
    }

    /// True if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.raw_data.is_empty()
    }

    /// Random-access lookup by index
    pub fn get(&self, index: usize) -> Option<&Raw> {
        self.raw_data.get(index)
    }

    /// Adds metadata (e.g., "source", "split")
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl<Raw> Dataset for InMemoryDataset<Raw>
where
    Raw: Clone + Send + Sync + 'static,
{
    fn iter(&self) -> Box<dyn Iterator<Item = Result<Sample>> + Send + '_> {
        let base = self.raw_data.iter().cloned();
        match &self.transform {
            Some(transform) => {
                let transform = Arc::clone(transform);
                Box::new(base.map(move |raw| transform.apply(raw)))
            }
            None => Box::new(base.map(|_| Err(anyhow!("No transform attached. Use `.with_transform()` to specify how to convert raw data into Sample")))),
        }
    }
}

/// -------------------------------------------------------------------------------------
/// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    // Shared test utilities
    struct Tokenizer;
    impl Transform<String, Sample> for Tokenizer {
        fn apply(&self, text: String) -> Result<Sample> {
            let length = text.len() as i64;
            Ok(Sample::from_single("length", Tensor::from_slice(&[length])))
        }
    }

    // -----------InMemoryDataset Tests ------------------------
    #[test]
    fn test_in_memory_dataset() -> Result<()> {
        let texts = vec!["Hello".to_string(), "world".to_string()];
        let dataset = InMemoryDataset::new(texts)
            .with_transform(Tokenizer)
            .with_metadata("source", "test");

        // Verify iteration
        let samples: Vec<Sample> = dataset.iter().collect::<Result<_>>()?;
        assert_eq!(samples.len(), 2);
        assert_eq!(
            samples[0].get("length")?.int64_value(&[0]),
            "hello".len() as i64
        );

        // Verify random access
        let raw_data = dataset.get(1).unwrap();
        let sample = Tokenizer.apply(raw_data.clone())?;
        assert_eq!(
            sample.get("length")?.int64_value(&[0]),
            "world".len() as i64
        );
        Ok(())
    }
}
