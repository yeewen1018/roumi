use crate::dataset::DataSource;
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

/// Streams image file paths from a directory (with optional recursion and extension filtering).
/// Designed for lazy loading to minimize memory usage. So this source provides file paths,
/// rather than loading and storing the image bytes directly.
///   
/// # Example
/// ```ignore
/// let source = ImageDirSource::new(
///     "./data/images",
///     &["jpg", "png"], // Allowed extensions (case-insensitive)
///     true            // Enable recursion
/// )?;
///
/// // Use in a pipeline:
/// let transform = PILToTensor;
/// for path in source.stream()?{
///     let tensor = transform.apply(path?)?;
/// }
/// ```
pub struct ImageDirSource {
    dir_path: PathBuf,
    extensions: Vec<String>,
    recurse: bool,
}

impl ImageDirSource {
    /// Creates a new image directory source.
    ///
    /// # Arguments
    /// - `dir_path`: Directory to scan (or glob pattern if needed).
    /// - `extensions`: File extensions to include (e.g., `["jpg", "png"]`). Case-insensitive.
    /// - `recurse`: If `true`, scans subdirectories recursively.
    pub fn new(dir_path: impl Into<PathBuf>, extensions: &[&str], recurse: bool) -> Self {
        Self {
            dir_path: dir_path.into(),
            extensions: extensions.iter().map(|s| s.to_lowercase()).collect(),
            recurse,
        }
    }
}

/// Returns an iterator over valid image file paths.
impl DataSource<PathBuf> for ImageDirSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<PathBuf>> + Send>> {
        // Early validation: ensure the directory exists and is indeed a directory.
        let dir_metadata = fs::metadata(&self.dir_path)
            .with_context(|| format!("Failed to access directory: {}", self.dir_path.display()))?;
        if !dir_metadata.is_dir() {
            bail!("Path is not a directory: {}", self.dir_path.display());
        }

        let extensions = self.extensions.clone();
        // Build iterator.
        // - recurse = true: traverse all subdirectories, retrieve all images in nested folders.
        // - recurse = false: only scan the top-level directory, skipping subdirectories.
        let path_iter: Box<dyn Iterator<Item = PathBuf> + Send> = if self.recurse {
            Box::new(
                WalkDir::new(&self.dir_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .map(|entry| entry.path().to_path_buf()),
            )
        } else {
            Box::new(
                fs::read_dir(&self.dir_path)?
                    .filter_map(|e| e.ok())
                    .map(|entry| entry.path()),
            )
        };

        // Filter out symlinks, non-files, and unreadable files
        let iter = path_iter
            .filter(|path| {
                if path.is_symlink() {
                    return false;
                }
                path.metadata().map(|m| m.is_file()).unwrap_or(false)
            })
            .filter_map(move |path| {
                // Extension check (case-insensitive)
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase());

                // Verify readability by opening file once.
                if let Some(ext) = ext {
                    if extensions.contains(&ext) {
                        // Verify readability by opening once.
                        return match fs::File::open(&path) {
                            Ok(_) => Some(Ok(path)),
                            Err(e) => Some(Err(e).with_context(|| {
                                format!("File exists but cannot be opened: {}", path.display())
                            })),
                        };
                    }
                }
                None
            });
        Ok(Box::new(iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_image_dir_stream() {
        let dir = tempdir().unwrap();
        let d = dir.path();

        // create dummy image files (zero bytes are fine for this test)
        File::create(d.join("a.JPG")).unwrap();
        File::create(d.join("b.png")).unwrap();
        File::create(d.join("c.jpg")).unwrap();
        File::create(d.join("ignore.txt")).unwrap(); // should be skipped

        let src = ImageDirSource::new(d, &["jpg", "png"], false);

        let files: Vec<_> = src
            .stream()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(files.len(), 3);
        for p in files {
            let ext = p
                .extension()
                .unwrap()
                .to_string_lossy()
                .to_ascii_lowercase();
            assert!(ext == "jpg" || ext == "png");
        }
    }
}
