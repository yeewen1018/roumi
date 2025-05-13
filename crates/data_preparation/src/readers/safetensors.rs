use crate::dataset::DataSource;
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::{collections::HashMap, fs, iter, path::PathBuf};
use tch::Tensor;

/// Reads safetensors files with the option for loading all tensors at once
/// or streaming tensors one-by-one.
///
/// # Example
/// 1. Load all tensors at once (default)
/// ```ignore
/// let source = SafetensorsSource::new("model.safetensors");
/// let tensor_map = source.load()?;
/// ```
///
/// 2. Stream tensors one-by-one (memory-efficient)
/// let source = SafetensorsSource::new("big_model.safetensors")
///     .load_all_at_once(false);
/// for tensor_result in source.stream()?{
///     let (name, tensor) = tensor_result?; // Process individually
/// }
pub struct SafetensorsSource {
    path: PathBuf,
    // If true, load all tensors immediately.
    // If false, streams one tensor at a time.
    load_all_at_once: bool,
}

impl SafetensorsSource {
    /// Creates a new reader (defaults to loading all tensors at once).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            load_all_at_once: true, // Default behaviour
        }
    }

    /// Configures whether to load all tensors at once or stream them.
    /// Set to `false` for memory-efficient streaming.
    pub fn load_all_at_once(mut self, load_all: bool) -> Self {
        self.load_all_at_once = load_all;
        self
    }

    /// Loads all tensors immediately
    pub fn load(&self) -> Result<HashMap<String, Tensor>> {
        let file_bytes = fs::read(&self.path)
            .with_context(|| format!("Failed to read safetensors file: {}", self.path.display()))?;

        let safetensors = SafeTensors::deserialize(&file_bytes)?;
        let mut tensors = HashMap::new();

        for (tensor_name, tensor_view) in safetensors.tensors() {
            let shape: Vec<i64> = tensor_view
                .shape()
                .iter()
                .map(|&dimension| dimension as i64)
                .collect();
            tensors.insert(
                tensor_name.to_string(),
                Tensor::from_slice(tensor_view.data()).reshape(&shape),
            );
        }
        Ok(tensors)
    }
}

/// !Note: Re-parses the header for each tensor. Small overhead here.
/// Will optimize later if benchmark results show that this is a bottleneck.
impl DataSource<HashMap<String, Tensor>> for SafetensorsSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<HashMap<String, Tensor>>> + Send>> {
        if self.load_all_at_once {
            // Load all tensors at once
            let tensor_map = self.load()?;
            let tensor_values: Vec<_> = tensor_map.values().collect();
            let stacked_map =
                HashMap::from([("all_tensors".to_string(), Tensor::stack(&tensor_values, 0))]);
            Ok(Box::new(iter::once(Ok(stacked_map))))
        } else {
            // Streaming mode
            let file_bytes = fs::read(&self.path)?;
            let safetensors = SafeTensors::deserialize(&file_bytes)?;
            let names = safetensors
                .names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            Ok(Box::new(names.into_iter().map(move |name| {
                // Reconstruct SafeTensors for each item (lightweight operation)
                let st = SafeTensors::deserialize(&file_bytes).unwrap();
                let view = st.tensor(&name)?;
                let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
                Ok(HashMap::from([(
                    name,
                    Tensor::from_slice(view.data()).reshape(&shape),
                )]))
            })))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_safetensors_source_loads_u8_tensor() -> anyhow::Result<()> {
        // 1. craft a minimal .safetensors file
        // raw data: eight u8 values 0..7
        let raw: Vec<u8> = (0u8..8).collect();

        // header JSON (dtype U8 so each byte is one element)
        let header_json = r#"{
            "weight": {
                "dtype": "U8",
                "shape": [8],
                "data_offsets": [0, 8]
            },
            "__metadata__": {}
        }"#;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // layout = [8-byte little-endian len][header JSON][raw bytes]
        let mut file_bytes = Vec::with_capacity(8 + header_bytes.len() + raw.len());
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&raw);

        //  2. write to temp file & load with our source
        let tmp = NamedTempFile::new()?;
        fs::write(tmp.path(), &file_bytes)?;

        let source = SafetensorsSource::new(tmp.path());
        let tensors = source.load()?;
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("weight"));
        let t = &tensors["weight"];
        assert_eq!(t.size(), &[8]); // 8 elements, one dimension

        Ok(())
    }
}
