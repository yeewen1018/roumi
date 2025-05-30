use crate::dataset::DataSource;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Reads text files line by line. Skips blank lones.
///
/// # Example
/// ```ignore
/// let source = TxtSource::new("data.txt");
/// for line_result in source.stream()? {
///     let line = line_result?;
/// }
/// ```
pub struct TxtSource {
    path: PathBuf,
}

impl TxtSource {
    /// Creates a new text file reader
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl DataSource<String> for TxtSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
        let file = File::open(&self.path)
            .with_context(|| format!("Failed to open text file: {}", self.path.display()))?;

        let reader = BufReader::new(file);
        let iter = reader
            .lines()
            .enumerate()
            .filter_map(move |(line_num, line)| match line {
                Ok(text) if text.trim().is_empty() => None, // Skip blank lines
                Ok(text) => Some(Ok(text)),
                Err(e) => {
                    Some(Err(e).with_context(|| format!("Error reading line {}", line_num + 1)))
                }
            });
        Ok(Box::new(iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_text_file_streaming() -> Result<()> {
        // Create a temp file with blank lines and comments
        let mut file = NamedTempFile::new()?;
        writeln!(file, "line1")?;
        writeln!(file, "")?; // Blank line (skipped)
        writeln!(file, "  \t")?; // Whitespace-only (skipped)
        writeln!(file, "line2")?;

        let source = TxtSource::new(file.path());
        let lines: Vec<_> = source.stream()?.collect::<Result<_>>()?;
        assert_eq!(lines, vec!["line1".to_string(), "line2".to_string()]);
        Ok(())
    }
}
