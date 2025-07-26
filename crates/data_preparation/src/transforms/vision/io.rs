use crate::sample::Sample;
use crate::transforms::Transform;
use anyhow::{Context, Result};
use image::{io::Reader as ImageReader, DynamicImage};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::PathBuf;
use tch::Tensor;

// ============================================================================
// LoadImage - Base image loader
// ============================================================================

/// Loads images from file paths.
///
/// This is the basic image loading transform that reads image files from disk
/// and returns them as `DynamicImage` objects. Uses buffered I/O for efficient file reading.
///
/// # Input/Output
/// - **Input**: `&Path` - File path to image
/// - **Output**: `DynamicImage` - Loaded image ready for processing
///
/// # Example
/// ```ignore
/// let loader = LoadImage::new();
/// let image = loader.apply(Path::new("photo.jpg"))?;
///
/// // Can inspect image properties
/// println!("Image size: {:?}", image.dimensions());
/// println!("Colour type: {:?}", image.color());
/// ```
#[derive(Debug, Clone)]
pub struct LoadImage {
    buffer_size: usize,
}

impl LoadImage {
    /// Creates a new image loader with optimized 8KB read buffer.
    pub fn new() -> Self {
        Self { buffer_size: 8192 }
    }
}

impl Transform<PathBuf, DynamicImage> for LoadImage {
    fn apply(&self, path: PathBuf) -> Result<DynamicImage> {
        // Open the image file
        let file = File::open(&path)
            .with_context(|| format!("Failed to open image: {}", path.display()))?;

        // Get file size for buffer pre-allocation
        let file_size = file.metadata()?.len() as usize;

        // Use buffered reader with configured buffer size
        let mut reader = BufReader::with_capacity(self.buffer_size, file);

        // Pre-allocate buffer based on file size to minimize reallocations
        let mut buffer = Vec::with_capacity(file_size);
        reader
            .read_to_end(&mut buffer)
            .with_context(|| format!("Failed to read image: {}", path.display()))?;

        // Decode image from memory buffer
        let image = ImageReader::new(Cursor::new(buffer))
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("Failed to decode image: {}", path.display()))?;

        Ok(image)
    }
}

// ============================================================================
// LoadImageToSample
// ============================================================================

/// Loads images from file paths and converts them to Samples (ready for model consumption).
///
/// This transform combines image loading and processing into a single operation:
/// 1. Loads image from file path using the `LoadImage` transform
/// 2. Applies vision transforms (resize, normalize, etc.)
/// 3. Creates a `Sample` with standardized feature name `image` and the provided label
///
/// The resulting `Sample` contains:
/// - `"image"`: Transformed image tensor (typically shape `[C, H, W]`)
/// - `"label"`: Label as i64 tensor (shape `[]` - scalar)
///
/// # Type Parameters
/// - `T`: Image transform pipeline that converts `DynamicImage` -> `Tensor`
/// (e.g., resize -> normalize -> to_tensor chain)
///
/// # Example
/// ```ignore
/// // Create vision processing pipeline
/// let vision_pipeline = Resize::new(224, 224, FilterType::Triangle)?
///     .then(ToTensor)
///     .then(Normalize::imagenet());
///
/// // Create the ML-ready loader
/// let full_vision_pipeline = LoadImageToSample::new(vision_pipeline);
///
/// // Use with labeled dataset
/// let paths_and_labels = vec![
///     (Path::new("cat.jpg"), 0),
///     (Path::new("dog.jpg"), 1),
/// ];
///
/// let dataset = InMemoryDataset::new(paths_and_labels)
///     .with_transform(full_vision_pipeline);
///
/// // Each sample will have:
/// // - sample.get("image")? -> tensor shape [3, 224, 224]
/// // - sample.get("label")? -> tensor with the provided label
/// ```
#[derive(Debug, Clone)]
pub struct LoadImageToSample<T> {
    image_loader: LoadImage,
    image_transform: T,
}

impl<T> LoadImageToSample<T> {
    pub fn new(image_transform: T) -> Self {
        Self {
            image_loader: LoadImage::new(),
            image_transform,
        }
    }
}

impl<T> Transform<(PathBuf, usize), Sample> for LoadImageToSample<T>
where
    T: Transform<DynamicImage, Tensor>,
{
    fn apply(&self, (path, label): (PathBuf, usize)) -> Result<Sample> {
        // Load image
        let image = self
            .image_loader
            .apply(path.clone())
            .with_context(|| format!("Failed to load image: {}", path.display()))?;

        // Apply vision transforms: DynamicImage -> Tensor
        let image_tensor = self
            .image_transform
            .apply(image)
            .with_context(|| format!("Failed to apply transforms to image: {}", path.display()))?;

        // Create Sample with standardized feature names
        Ok(Sample::from_single("image", image_tensor)
            .with_feature("label", Tensor::from(label as i64)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::vision::{Normalize, ToTensor};
    use image::{GenericImageView, Rgb, RgbImage};
    use tempfile::NamedTempFile;

    fn create_test_image() -> Result<NamedTempFile> {
        // Create a test image (3x3 RGB)
        let mut test_img = RgbImage::new(3, 3);
        test_img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red
        test_img.put_pixel(1, 1, Rgb([0, 255, 0])); // Green
        test_img.put_pixel(2, 2, Rgb([0, 0, 255])); // Blue

        // Save to temporary file
        let temp_file = NamedTempFile::with_suffix(".png")?;
        test_img.save(temp_file.path())?;
        Ok(temp_file)
    }

    #[test]
    fn test_load_image() -> Result<()> {
        let temp_file = create_test_image()?;

        // Test basic image loading
        let loader = LoadImage::new();
        let loaded_image = loader.apply(temp_file.path().to_path_buf())?;

        // Verify image properties
        assert_eq!(
            loaded_image.dimensions(),
            (3, 3),
            "Image dimensions should match"
        );

        // Verify image content
        let rgb = loaded_image.to_rgb8();
        assert_eq!(
            rgb.get_pixel(0, 0),
            &Rgb([255, 0, 0]),
            "Red pixel should match"
        );
        assert_eq!(
            rgb.get_pixel(1, 1),
            &Rgb([0, 255, 0]),
            "Green pixel should match"
        );
        assert_eq!(
            rgb.get_pixel(2, 2),
            &Rgb([0, 0, 255]),
            "Blue pixel should match"
        );

        Ok(())
    }

    #[test]
    fn test_load_image_to_sample() -> Result<()> {
        let temp_file = create_test_image()?;

        // Create transform pipeline
        let pipeline = ToTensor.then(Normalize::new(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])?);

        // Test the ML-ready loader
        let loader = LoadImageToSample::new(pipeline);
        let test_label = 42;
        let sample = loader.apply((temp_file.path().to_path_buf(), test_label))?;

        // Verify Sample structure
        assert!(sample.get("image").is_ok(), "Should have 'image' feature");
        assert!(sample.get("label").is_ok(), "Should have 'label' feature");

        // Verify image tensor properties
        let image_tensor = sample.get("image")?;
        assert_eq!(
            image_tensor.size(),
            vec![3, 3, 3],
            "Image should be [C, H, W] format"
        );

        // Verify label
        let label_tensor = sample.get("label")?;
        assert_eq!(
            label_tensor.int64_value(&[]),
            test_label as i64,
            "Label should match"
        );

        Ok(())
    }

    #[test]
    fn test_error_handling() -> Result<()> {
        let loader = LoadImage::new();

        // Test LoadImage error handling
        let result = loader.apply(PathBuf::from("nonexistent.jpg"));
        assert!(result.is_err(), "Should error for non-existent file");

        // Test LoadImageToSample error handling
        let pipeline = ToTensor;
        let ml_loader = LoadImageToSample::new(pipeline);
        let result = ml_loader.apply((PathBuf::from("nonexistent.jpg"), 0));
        assert!(result.is_err(), "Should error for non-existent file");

        Ok(())
    }
}
