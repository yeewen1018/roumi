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

    /// Random-access lookup by index.
    /// - In-memory datasets return `Ok(Some(&Sample))` or `Ok(None)` if out-of-bounds.
    /// - Streaming datasets always return `Ok(None)`.
    fn get(&self, index: usize) -> Result<Option<&Sample>>;

    /// Returns total number of samples.
    /// - In-memory datasets return `Some(n)`.
    /// - Streaming datasets return `None`.
    fn len(&self) -> Option<usize>;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len().map(|l| l == 0).unwrap_or(true)
    }
}

/// A dataset that stores all samples in a contiguous memory
/// with atomic-reference counting (`Arc<[Sample]>`).
///
/// This enables:
/// - Zero-copy clone: Cloning only bumps the `Arc` counter
/// - Thread-safe sharing: Safe concurrent read access (`Send + Sync`)
/// - Cache-efficient: Samples laid out back-to-back for efficient batching.
///
/// Ideal for datasets that comfortably fit into RAM.
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

    /// Adds/updates metadata and returns the modified dataset.
    /// Enables chaining: `dataset.with_metadata("source", "train")`.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns the value of a metadata field, if it exists.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// **(Opt-in via `boxed-iter` feature)**
    /// Returns a heap-boxed iterator over owned `Sample`s.
    /// The default `iter` uses static dispatch, but use this
    /// when building runtime defined pipelines or passing iterators
    /// as trait objects.
    ///
    /// Cost: one heap allocation per iterator creation.
    #[cfg(feature = "boxed-iter")]
    pub fn iter_boxed(&self) -> Box<dyn Iterator<Item = Result<Sample>> + Send + '_> {
        Box::new(self.samples.iter().cloned().map(Ok))
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

    fn get(&self, index: usize) -> Result<Option<&Sample>> {
        Ok(self.samples.get(index))
    }

    fn len(&self) -> Option<usize> {
        Some(self.samples.len())
    }
}

#[cfg(test)]
mod in_memory_dataset_tests {
    use super::*;
    use crate::sample::Sample;
    use tch::{Kind, Tensor};

    // Helper functions for creating test data
    mod test_utils {
        use super::*;

        // Creates `n` test samples with predictable values
        // - "input_ids": [i], "labels": [i % 2]
        pub fn create_test_samples(n: usize) -> Vec<Sample> {
            (0..n)
                .map(|i| {
                    Sample::from_single(
                        "input_ids",
                        Tensor::from_slice(&[i as i64]).to_kind(Kind::Int64),
                    )
                    .with_feature(
                        "labels",
                        Tensor::from_slice(&[(i as i64) % 2]).to_kind(Kind::Int64),
                    )
                })
                .collect()
        }
    }

    #[test]
    fn test_creation() {
        let samples = test_utils::create_test_samples(3);
        let dataset = InMemoryDataset::new(samples);

        assert_eq!(dataset.len(), Some(3));
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_iteration_and_random_access() -> Result<()> {
        let samples = test_utils::create_test_samples(2);
        let dataset = InMemoryDataset::new(samples);

        // iter
        let mut it = dataset.iter();
        let sample_0 = it.next().unwrap().unwrap();
        let sample_1 = it.next().unwrap().unwrap();
        assert!(it.next().is_none());
        assert_eq!(sample_0.features["input_ids"].int64_value(&[0]), 0);
        assert_eq!(sample_1.features["labels"].int64_value(&[0]), 1);

        // get
        let r = dataset.get(1)?.unwrap();
        assert_eq!(r.features["input_ids"].int64_value(&[0]), 1);
        assert!(dataset.get(2)?.is_none());
        Ok(())
    }

    #[test]
    fn test_metadata_ops() {
        let dataset = InMemoryDataset::new(test_utils::create_test_samples(1))
            .with_metadata("source", "test");

        assert_eq!(dataset.metadata("source"), Some("test"));
        assert!(dataset.metadata("missing").is_none());
    }

    #[test]
    fn test_concurrent_iter() {
        let dataset = Arc::new(InMemoryDataset::new(test_utils::create_test_samples(100)));

        let threads: Vec<_> = (0..4)
            .map(|_| {
                let dataset = dataset.clone();
                std::thread::spawn(move || {
                    for sample in dataset.iter() {
                        let _ = sample.unwrap().features["input_ids"].int64_value(&[0]);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_get() {
        let dataset = Arc::new(InMemoryDataset::new(test_utils::create_test_samples(100)));

        let threads: Vec<_> = (0..4)
            .map(|_| {
                let dataset = dataset.clone();
                std::thread::spawn(move || {
                    for i in 0..100 {
                        let sample = dataset.get(i).unwrap().unwrap();
                        let _ = sample.features["input_ids"].int64_value(&[0]);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use crate::{
        collator::{PaddingCollator, PaddingRule, StackCollator},
        minibatch::MiniBatch,
        sample::Sample,
    };
    use tch::{Device, Kind, Tensor};

    // Helper functions for creating test data
    mod test_utils {
        use super::*;

        // Creates a sample with variable length features for padding tests
        pub fn make_variable_length_sample(id: i64, len1: i64, len2: i64) -> Sample {
            Sample::from_single(
                "f1",
                Tensor::from_slice(&vec![id; len1 as usize]).to_kind(Kind::Int64),
            )
            .with_feature(
                "f2",
                Tensor::from_slice(&vec![id + 1; len2 as usize]).to_kind(Kind::Int64),
            )
        }
    }

    #[test]
    fn test_stack_collator_pipeline() -> anyhow::Result<()> {
        let samples = vec![
            Sample::from_single(
                "input_ids",
                Tensor::from_slice(&[1, 2, 3]).to_kind(Kind::Int64),
            )
            .with_feature("labels", Tensor::from_slice(&[1]).to_kind(Kind::Int64)),
            Sample::from_single(
                "input_ids",
                Tensor::from_slice(&[4, 5, 6]).to_kind(Kind::Int64),
            )
            .with_feature("labels", Tensor::from_slice(&[0]).to_kind(Kind::Int64)),
        ];

        // Create dataset
        let dataset = InMemoryDataset::new(samples);
        assert_eq!(dataset.len(), Some(2));

        // Create batch using stack collator
        let all_samples: Vec<Sample> = dataset.iter().collect::<Result<Vec<_>>>()?;
        let batch = MiniBatch::collate(all_samples, StackCollator)?;

        // Validate batch
        assert_eq!(batch.get("input_ids")?.size(), &[2, 3]);
        assert_eq!(batch.get("labels")?.size(), &[2, 1]);

        let expected_input_ids = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        assert!(batch.get("input_ids")?.equal(&expected_input_ids));

        Ok(())
    }

    #[test]
    fn test_padding_collator_pipeline() -> Result<()> {
        let samples = vec![
            test_utils::make_variable_length_sample(1, 2, 4), // f1 len 2; f2 len 4
            test_utils::make_variable_length_sample(2, 3, 2), // f1 len 3; f2 len 2
            test_utils::make_variable_length_sample(3, 1, 1), // f1 len 1; f2 len 1
        ];

        // Create dataset
        let dataset = InMemoryDataset::new(samples);

        // Configure padding:
        // - f1: pad to max length in batch
        // - f2: pad to fixed length 5 with value -1.0
        let collator = PaddingCollator::new()
            .pad("f1", vec![(0, PaddingRule::MaxLength)], None)
            .pad("f2", vec![(0, PaddingRule::FixedRight(5))], Some(-1.0));

        // Create batch
        let all_samples: Vec<_> = dataset.iter().collect::<Result<_>>()?;
        let batch = MiniBatch::collate(all_samples, collator)?;

        // Validate batch shapes
        assert_eq!(batch.get("f1")?.size(), &[3, 3]); // max(2,3,1)=3
        assert_eq!(batch.get("f2")?.size(), &[3, 5]); // fixed 5

        // Validate padding value (The fourth element in "f2" of sample 1 should be -1.0)
        assert_eq!(batch.get("f2")?.double_value(&[0, 4]), -1.0);

        // Test device transfer
        let cpu = batch.to_device(Device::Cpu);
        let gpu = cpu.to_device(Device::cuda_if_available());
        let cpu2 = gpu.to_device(Device::Cpu);
        assert_eq!(cpu2.get("f1")?.size(), &[3, 3]);

        Ok(())
    }
}
