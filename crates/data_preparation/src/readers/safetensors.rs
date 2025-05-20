use crate::dataset::DataSource;
use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use safetensors::{
    tensor::{Dtype, TensorView},
    SafeTensors,
};
use std::{collections::HashMap, fs, iter, path::PathBuf, sync::Arc};
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
        let mut tensors = HashMap::with_capacity(safetensors.tensors().len());

        for (tensor_name, tensor_view) in safetensors.tensors() {
            tensors.insert(
                tensor_name.to_string(),
                tensor_from_view(&tensor_view)
                    .with_context(|| format!("Failed to convert tensor '{}'", tensor_name))?,
            );
        }
        Ok(tensors)
    }
}

impl DataSource<HashMap<String, Tensor>> for SafetensorsSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<HashMap<String, Tensor>>> + Send>> {
        if self.load_all_at_once {
            // Load all tensors at once
            let tensor_map = self.load()?;
            Ok(Box::new(iter::once(Ok(tensor_map))))
        } else {
            // Streaming mode
            let file_bytes = Arc::new(fs::read(&self.path)?);
            let safetensors = SafeTensors::deserialize(&file_bytes)?;
            let tensor_info = safetensors
                .tensors()
                .into_iter()
                .map(|(name, view)| (name.to_string(), view.shape().to_vec(), view.dtype()))
                .collect::<Vec<_>>();

            let file_bytes_clone = file_bytes.clone();
            Ok(Box::new(tensor_info.into_iter().map(
                move |(name, _shape, _dtype)| {
                    let st = SafeTensors::deserialize(&file_bytes_clone)
                        .map_err(|e| anyhow::anyhow!(
                            "Failed to re-parse safetensors metadata (file may be corrupt or modified concurrently): {e}"
                        ))?;
                    let view = st.tensor(&name)
                        .map_err(|e| anyhow::anyhow!(
                            "Failed to access tensor '{}' during streaming: {}",
                            name,
                            e
                        ))?;
                    tensor_from_view(&view)
                        .map(|tensor| HashMap::from([(name.clone(), tensor)]))
                        .with_context(|| format!("Failed to convert tensor {}", name))
                },
            )))
        }
    }
}

/// Converts a TensorView to a tch::Tensor.
/// Supported dtypes: U8, I8, I16, I32, I64, F32, F64.
/// Unsupported: U32, U64, F16, BF16 (require conversion first).
fn tensor_from_view(view: &TensorView<'_>) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let raw = view.data();
    let tensor = match view.dtype() {
        Dtype::U8 => Tensor::from_slice(raw),
        Dtype::I8 => Tensor::from_slice(cast_slice::<u8, i8>(raw)),
        Dtype::I16 => Tensor::from_slice(cast_slice::<u8, i16>(raw)),
        Dtype::I32 => Tensor::from_slice(cast_slice::<u8, i32>(raw)),
        Dtype::I64 => Tensor::from_slice(cast_slice::<u8, i64>(raw)),
        Dtype::F32 => Tensor::from_slice(cast_slice::<u8, f32>(raw)),
        Dtype::F64 => Tensor::from_slice(cast_slice::<u8, f64>(raw)),
        Dtype::U32 | Dtype::U64 => bail!(
            "Unsigned 32/64-bit tensors are not supported by libtorch. \
             Please convert to signed equivalent (e.g., I32/I64) before loading. \
             Offending tensor shape: {:?}, dtype: {:?}",
            shape,
            view.dtype()
        ),
        Dtype::F16 | Dtype::BF16 => bail!(
            "Half-precision floats (F16/BF16) are not supported. \
             Convert to F32 before loading. \
             Offending tensor shape: {:?}",
            shape
        ),
        other => bail!(
            "Unsupported dtype '{:?}'. Supported dtypes: U8, I8, I16, I32, I64, F32, F64. \
             Tensor shape: {:?}",
            other,
            shape
        ),
    };
    Ok(tensor.reshape(&shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize_to_file;
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

    // Helper function for test mixed dtype tensors
    fn create_test_tensor(dtype: Dtype, shape: &[usize]) -> Result<TensorView<'static>> {
        let total_size: usize = shape.iter().product();
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => cast_slice(&vec![1.0f32; total_size]).to_vec(),
            Dtype::I64 => cast_slice(&vec![42i64; total_size]).to_vec(),
            Dtype::F64 => cast_slice(&vec![3.14f64; total_size]).to_vec(),
            _ => bail!("Unsupported dtype for test"),
        };
        let leaked_bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Ok(TensorView::new(dtype, shape.to_vec(), leaked_bytes)?)
    }

    #[test]
    fn test_safetensors_source_stream_mixed_dtype_tensors() -> Result<()> {
        // 1. Create a test safetensors file with multiple dtypes
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let test_tensors = vec![
            ("float32", create_test_tensor(Dtype::F32, &[2, 3])?),
            ("int64", create_test_tensor(Dtype::I64, &[1, 5])?),
            ("float64", create_test_tensor(Dtype::F64, &[3, 2])?),
        ];

        serialize_to_file(test_tensors.into_iter(), &None, path)?;

        // 2. Test streaming mode
        let source = SafetensorsSource::new(path).load_all_at_once(false);

        let mut stream = source.stream()?;
        let mut found_tensors = 0;

        while let Some(batch) = stream.next() {
            let tensor_map = batch?;
            for (name, tensor) in tensor_map {
                match name.as_str() {
                    "float32" => {
                        assert_eq!(tensor.kind(), tch::Kind::Float);
                        assert_eq!(tensor.size(), vec![2, 3]);
                    }
                    "int64" => {
                        assert_eq!(tensor.kind(), tch::Kind::Int64);
                        assert_eq!(tensor.size(), vec![1, 5]);
                    }
                    "float64" => {
                        assert_eq!(tensor.kind(), tch::Kind::Double);
                        assert_eq!(tensor.size(), vec![3, 2]);
                    }
                    _ => panic!("Unexpected tensor name"),
                }
                found_tensors += 1;
            }
        }

        assert_eq!(found_tensors, 3);
        Ok(())
    }
}
