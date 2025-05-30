use crate::dataset::DataSource;
use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{fs::File, io::BufRead, path::PathBuf};

/// A line-by-line JSONL reader that supports both untyped (`Value`) and
/// typed (`T: DeserializeOwned`) parsing.
///
/// # Examples
/// ## 1. Typed Parsing (Rust Structs)
////```ignore
/// #[derive(serde::Deserialize)]
/// struct Example{
///     text: String,
///     label: i64,
/// }
///
/// let source = JsonlSource::new("data.jsonl");
/// for example in source.stream::<Example>()>? {
///     let example = example?;
///     println!("Text: {}", example.text);
/// }
///
/// ## 2. Untyped Parsing  
/// let source = JsonlSource::new("data.jsonl");
/// for value in source.stream_values()?{
///     let value = value?;  // `serde_json::Value`
///     println!("Text: {}", value["text"].as_str().unwrap());
/// }
/// ```
pub struct JsonlSource {
    path: PathBuf,
}

impl JsonlSource {
    /// Creates a new reader for a JSONL file at the given path.
    ///
    /// # Arguments
    /// - `path`: Accepts `String`, `&str`, or `PathBuf`
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Streams lines as Rust types. Prefer this for type-safe workflows.
    ///
    /// # Type Parameter
    /// - `T`: Must implement `serde::Deserialize` (use `#[derive(Deserialize)]`).
    ///
    /// # Errors
    /// - Fails if the file cannot be opened or any line is invalid JSON for `T`.
    /// - Includes line numbers in errors (e.g., "Invalid JSON at line 3").
    pub fn stream<T: DeserializeOwned>(
        &self,
    ) -> Result<Box<dyn Iterator<Item = Result<T>> + Send>> {
        self.stream_impl()
    }

    /// Streams lines as `serde_json::Value`. Use for dynamic data.
    /// - Use with `DataSource<Value>` in generic pipelines.
    /// - Should be able to map directly to Python `dict` via PyO3.
    pub fn stream_values(&self) -> Result<Box<dyn Iterator<Item = Result<Value>> + Send>> {
        self.stream_impl()
    }

    /// Shared implementation for both typed and untyped streaming.
    fn stream_impl<T: DeserializeOwned>(
        &self,
    ) -> Result<Box<dyn Iterator<Item = Result<T>> + Send>> {
        let file = File::open(&self.path)
            .with_context(|| format!("Failed to open {}", self.path.display()))?;
        let reader = std::io::BufReader::new(file);

        let iter = reader.lines().enumerate().filter_map(|(line_num, line)| {
            let line = match line {
                Ok(l) if l.trim().is_empty() => return None, // Skip blanks
                Ok(l) => l,
                Err(e) => return Some(Err(e.into())),
            };
            Some(
                serde_json::from_str::<T>(&line)
                    .with_context(|| format!("Invalid JSON at line {}", line_num + 1)),
            )
        });
        Ok(Box::new(iter))
    }
}

// Implement `DataSource` for `Value` (untyped as default)
impl DataSource<Value> for JsonlSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<Value>> + Send>> {
        self.stream_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_jsonl_source_streams_typed_data() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, r#"{{"id": 1, "text": "foo"}}"#)?;
        writeln!(file, r#"{{"id": 2, "text": "bar"}}"#)?;

        let source = JsonlSource::new(file.path());
        let items: Vec<serde_json::Value> = source.stream()?.collect::<Result<_>>()?;
        assert_eq!(
            items,
            vec![
                json!({"id": 1, "text": "foo"}),
                json!({"id": 2, "text": "bar"})
            ]
        );
        Ok(())
    }
}
