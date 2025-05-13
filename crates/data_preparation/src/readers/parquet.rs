use crate::dataset::DataSource;
use anyhow::{Context, Result};
use arrow::record_batch::RecordBatch;
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use std::path::PathBuf;

/// Parquet file reader for Arrow RecordBatches.
///
/// # Example
/// ```ignore
/// // Read a file with 1024-row batches, only selecting "image" and "label" columns
/// let source = ParquetSource::new(
///     "data.parquet",
///     1024,
///     Some(vec!["image".into(), "label".into()]),
/// );
///
/// for batch_res in source.stream()? {
///     let batch: RecordBatch = batch_res?;
/// }
/// ```
pub struct ParquetSource {
    path: PathBuf,
    batch_size: usize,
    projection: Vec<String>,
}

impl ParquetSource {
    /// Creates a new Parquet reader.
    ///
    /// # Arguments
    /// - `path`: Path to Parquet file
    /// - `batch_size`: Rows per RecordBatch (optimize for your GPU/CPU cache)
    /// - `projection`: Optional column names to read (empty = all columns)
    pub fn new(
        path: impl Into<PathBuf>,
        batch_size: usize,
        projection: Option<Vec<String>>,
    ) -> Self {
        Self {
            path: path.into(),
            batch_size,
            projection: projection.unwrap_or_default(),
        }
    }
}
/*

impl DataSource<RecordBatch> for ParquetSource {
    /// Stream RecordBatches from the Parquet file.
    ///
    /// # Errors
    /// 1. Yields per-batch errors for corrupt row groups.
    /// 2. Fails immediately if:
    /// - File does not exist or is not a valid Parquet
    /// - Requested columns do not exist
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Failed to open Parquet file: {}", self.path.display()))?;

        // Configure batch size
        let props = ReaderProperties::builder()
            .with_batch_size(self.batch_size)
            .build();

        let mut builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, props)
            .with_context(|| format!("Not a valid Parquet file: {}", self.path.display()))?;

        // Apply column projection if specified (for reading only selected columns)
        if !self.projection.is_empty() {

            let field_indices: Vec<usize> = builder.schema()
                .fields()
                .iter()
                .enumerate()
                // Keep only fields that match our projection list
                .filter(|(_, field_def)|{
                    self.projection.contains(&field_def.name().to_string())
                })
                // Extract the field indices (Parquet needs numeric indices)
                .map(|(field_index, _)|field_index)
                .collect();

            // Create the projection mask
            let projection_mask = ProjectionMask::roots(
                builder.parquet_schema(),
                field_indices
            );
            builder = builder.with_projection(projection_mask);

            // Verify the projection was valid
            let valid_columns: Vec<_> = builder.schema().fields().iter()
                .map(|f| f.name().to_string())
                .collect();
            if field_indices.is_empty() {
                return Err(anyhow!(
                    "Invalid projection columns. Available: {:?}, Requested: {:?}",
                    valid_columns,
                    self.projection
                ));
            }
        }
        Ok(Box::new(builder.build()?.into_iter().map(|batch| Ok(batch?))))
    }
}

*/
impl DataSource<RecordBatch> for ParquetSource {
    /// Stream RecordBatches from the Parquet file.
    ///
    /// # Errors
    /// 1. Yields per-batch errors for corrupt row groups.
    /// 2. Fails immediately if:
    /// - File does not exist or is not a valid Parquet
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Failed to open Parquet file: {}", self.path.display()))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("Not a valid Parquet file: {}", self.path.display()))?;

        // Set batch size
        let builder = builder.with_batch_size(self.batch_size);

        // Apply column projection if specified (for reading only selected columns)
        let builder = if !self.projection.is_empty() {
            let field_indices: Vec<usize> = builder
                .schema()
                .fields()
                .iter()
                .enumerate()
                // Keep only fields that match our projection list
                .filter(|(_, field_def)| self.projection.contains(&field_def.name().to_string()))
                // Extract the field indices (Parquet needs numeric indices)
                .map(|(field_index, _)| field_index)
                .collect();

            // Verify the projection was valid before creating mask
            if field_indices.is_empty() {
                let valid_columns: Vec<_> = builder
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().to_string())
                    .collect();
                return Err(anyhow::anyhow!(
                    "Invalid projection columns. Available: {:?}, Requested: {:?}",
                    valid_columns,
                    self.projection
                ));
            }

            // Create and apply projection mask
            let projection_mask = ProjectionMask::roots(builder.parquet_schema(), field_indices);
            builder.with_projection(projection_mask)
        } else {
            builder
        };
        Ok(Box::new(
            builder.build()?.into_iter().map(|batch| Ok(batch?)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{ArrayRef, Int32Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::{fs::File, sync::Arc};
    use tempfile::NamedTempFile;

    #[test]
    fn test_parquet_file_streaming() {
        // 1. build a tiny RecordBatch
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        // 2. write it to a temporary Parquet file
        let tmp = NamedTempFile::new().unwrap();
        {
            let file = File::create(tmp.path()).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        // 3. read it back through ParquetSource
        let src = ParquetSource::new(tmp.path(), /*batch_size*/ 2, None);
        let rows: usize = src.stream().unwrap().map(|rb| rb.unwrap().num_rows()).sum();

        assert_eq!(rows, 3);
    }

    #[test]
    fn test_parquet_file_missing_error() {
        let src = ParquetSource::new("no_such.parquet", 16, None);
        assert!(src.stream().is_err());
    }
}
