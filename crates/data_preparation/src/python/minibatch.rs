//! src/python/minibatch.rs
//!
//! This module provides Python bindings for our Rust framework's `MiniBatch`.
//! It is built on top of the PyTensorHandle zero-copy bridge (DLPack) and provides
//! zero-copy access to PyTorch tensors.
//!
//! # Usage
//!
//! ```python
//! for batch in dataloader:
//!     # Zero-copy access to tensors
//!     images = batch['image']     # PyTorch tensor
//!     labels = batch['labels']     # PyTorch tensor
//!
//!     # Use directly with PyTorch models
//!     outputs = model(images)
//!     loss = criterion(outputs, labels)
//! ```

use super::tensor::PyTensorHandle;
use crate::minibatch::MiniBatch;
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use std::collections::HashMap;
use tch::Tensor;

#[pyclass(name = "MiniBatch", unsendable)]
pub struct PyMiniBatch {
    // Map of feature names to zero-copy tensor handles
    tensors: HashMap<String, PyTensorHandle>,
    batch_size: i64,
}

#[pymethods]
impl PyMiniBatch {
    /// Get the batch size
    #[getter]
    fn batch_size(&self) -> i64 {
        self.batch_size
    }

    /// Get list of all feature names in this batch
    #[getter]
    fn features(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Access PyTorch tensors via dict-like syntax: batch['feature_name'].
    fn __getitem__(&self, feature: &str, py: Python) -> PyResult<PyObject> {
        let handle = self.tensors.get(feature).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Feature '{}' not found", feature))
        })?;

        handle.to_pytorch(py)
    }

    /// Check if feature exists: 'feature_name' in batch.
    fn __contains__(&self, feature: &str) -> bool {
        self.tensors.contains_key(feature)
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "MiniBatch(batch_size={}, features={:?})",
            self.batch_size,
            self.features()
        )
    }
}

impl PyMiniBatch {
    /// Create PyMiniBatch from Rust MiniBatch with zero-copy tensor handles.
    pub fn from_minibatch(batch: MiniBatch) -> Result<Self> {
        let batch_size = batch.batch_size()?;

        let tensors = batch
            .tensors
            .into_iter()
            .map(|(feature, tensor)| (feature, PyTensorHandle::new(tensor)))
            .collect();

        Ok(Self {
            tensors,
            batch_size,
        })
    }
}

/// Create a test MiniBatch for testing purposes 
#[pyfunction]
pub fn create_test_batch() -> PyResult<PyMiniBatch> {
    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4])
        .reshape(&[2, 2]);
    let labels = Tensor::from_slice(&[0i64, 1])
        .reshape(&[2, 1]);

    let mut tensors = std::collections::HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids);
    tensors.insert("labels".to_string(), labels);

    let batch = MiniBatch{tensors};

    PyMiniBatch::from_minibatch(batch)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
