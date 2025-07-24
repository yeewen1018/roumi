use crate::dataloader::worker_gen_range;
use crate::transforms::Transform;
use anyhow::{ensure, Context, Result};
use image::DynamicImage;
use std::cell::RefCell;
use std::collections::HashMap;
use tch::{Kind, Tensor};

// ===========================================================================
// ColorJitter
// ===========================================================================

/// Randomly changes the brightness, contrast, saturation, and hue of an image.
///
/// # Parameters
/// Each parameter accepts either:
/// - A single `f32`: Creates a symmetric range around the default value
///   - For brightness/contrast/saturation: `[max(0, 1-value), 1+value]`
///   - For hue: `[-value, value]`
/// - A tuple `(min, max)`: Uses the explicit range `[min, max]`
///
/// - `brightness`: How much to jitter brightness (0 = no change)
/// - `contrast`: How much to jitter contrast (0 = no change)
/// - `saturation`: How much to jitter saturation (0 = no change)
/// - `hue`: How much to jitter hue, must be in `[0, 0.5]` for single value
///
/// # Examples
/// ```ignore
/// // Using single values (symmetric ranges)
/// let jitter = ColorJitter::new(0.4, 0.4, 0.4, 0.1)?;
/// // brightness: [0.6, 1.4], contrast: [0.6, 1.4],
/// // saturation: [0.6, 1.4], hue: [-0.1, 0.1]
///
/// // Using explicit ranges
/// let jitter = ColorJitter::new(
///     (0.8, 1.2),  // brightness: exactly [0.8, 1.2]
///     0.0,         // contrast: no change
///     (0.5, 1.5),  // saturation: exactly [0.5, 1.5]
///     (-0.2, 0.3), // hue: exactly [-0.2, 0.3]
/// )?;
///
/// let jittered = jitter.apply(image)?;
/// ```
#[derive(Debug, Clone)]
pub struct ColorJitter {
    brightness: (f32, f32),
    contrast: (f32, f32),
    saturation: (f32, f32),
    hue: (f32, f32),
}

#[derive(Debug, Clone)]
pub enum ColorJitterRange {
    Symmetric(f32),
    Explicit(f32, f32),
}

impl ColorJitterRange {
    fn to_range(self, param_name: &str) -> Result<(f32, f32)> {
        let is_hue = param_name == "hue";

        match self {
            ColorJitterRange::Symmetric(value) => {
                ensure!(
                    value >= 0.0,
                    "{} value must be non-negative, got {}",
                    param_name,
                    value
                );
                if is_hue {
                    ensure!(value <= 0.5, "Hue must be in [0, 0.5], got {}", value);
                    Ok((-value, value))
                } else {
                    Ok((0.0f32.max(1.0 - value), 1.0 + value))
                }
            }
            ColorJitterRange::Explicit(min, max) => {
                ensure!(
                    min <= max,
                    "{}: min({}) must be <= max ({})",
                    param_name,
                    min,
                    max
                );
                if is_hue {
                    ensure!(
                        min >= -0.5 && max <= 0.5,
                        "Hue range must be in [-0.5, 0.5], got [{}, {}]",
                        min,
                        max
                    );
                } else {
                    ensure!(
                        min >= 0.0,
                        "{} range must be non-negative, got min={}",
                        param_name,
                        min
                    );
                }
                Ok((min, max))
            }
        }
    }
}

impl From<f32> for ColorJitterRange {
    fn from(value: f32) -> Self {
        ColorJitterRange::Symmetric(value)
    }
}

impl From<(f32, f32)> for ColorJitterRange {
    fn from(range: (f32, f32)) -> Self {
        ColorJitterRange::Explicit(range.0, range.1)
    }
}

impl ColorJitter {
    pub fn new(
        brightness: impl Into<ColorJitterRange>,
        contrast: impl Into<ColorJitterRange>,
        saturation: impl Into<ColorJitterRange>,
        hue: impl Into<ColorJitterRange>,
    ) -> Result<Self> {
        let brightness = brightness.into().to_range("brightness")?;
        let contrast = contrast.into().to_range("contrast")?;
        let saturation = saturation.into().to_range("saturation")?;
        let hue = hue.into().to_range("hue")?;

        Ok(Self {
            brightness,
            contrast,
            saturation,
            hue,
        })
    }

    fn sample_factor(&self, range: (f32, f32)) -> Result<f32> {
        worker_gen_range(range.0..=range.1)
    }
}

impl Transform<DynamicImage, DynamicImage> for ColorJitter {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        // Sample random factors
        let brightness_factor = self.sample_factor(self.brightness)?;
        let contrast_factor = self.sample_factor(self.contrast)?;
        let saturation_factor = self.sample_factor(self.saturation)?;
        let hue_factor = self.sample_factor(self.hue)?;

        // Early exit if no adjustments needed
        if brightness_factor == 1.0
            && contrast_factor == 1.0
            && saturation_factor == 1.0
            && hue_factor == 0.0
        {
            return Ok(img);
        }

        // Convert to RGB once
        let mut rgb = img.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());

        // Pre-calculate contrast mean if needed (using ITU-R BT.601 luminance weights)
        let gray_mean = if contrast_factor != 1.0 {
            let mut sum = 0.0f32;
            for pixel in rgb.pixels() {
                // Standard ITU-R BT.601 luminance weights for RGB->grayscale conversion
                sum += 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
            }
            sum / (width * height) as f32
        } else {
            0.0
        };

        // Create lookup tables for brightness and contrast (faster than per-pixel calculation)
        let mut brightness_lut = [0u8; 256];
        if brightness_factor != 1.0 {
            for i in 0..256 {
                brightness_lut[i] = (i as f32 * brightness_factor).round().clamp(0.0, 255.0) as u8;
            }
        }

        let mut contrast_lut = [0u8; 256];
        if contrast_factor != 1.0 {
            for i in 0..256 {
                contrast_lut[i] = ((i as f32 - gray_mean) * contrast_factor + gray_mean)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }

        // Determine which transforms to apply and their order
        let mut transform_order = vec![];
        if brightness_factor != 1.0 {
            transform_order.push(0);
        }
        if contrast_factor != 1.0 {
            transform_order.push(1);
        }
        if saturation_factor != 1.0 {
            transform_order.push(2);
        }
        if hue_factor != 0.0 {
            transform_order.push(3);
        }

        // Shuffle the order for random application.
        // This ensures different colour adjustments do not always happen in the same sequence.
        for i in (1..transform_order.len()).rev() {
            let j = worker_gen_range(0..=i)?;
            transform_order.swap(i, j);
        }

        // Apply transforms in shuffled order
        for &transform_idx in &transform_order {
            match transform_idx {
                0 => {
                    // Brightness using LUT
                    for pixel in rgb.pixels_mut() {
                        pixel[0] = brightness_lut[pixel[0] as usize];
                        pixel[1] = brightness_lut[pixel[1] as usize];
                        pixel[2] = brightness_lut[pixel[2] as usize];
                    }
                }
                1 => {
                    // Contrast using LUT
                    for pixel in rgb.pixels_mut() {
                        pixel[0] = contrast_lut[pixel[0] as usize];
                        pixel[1] = contrast_lut[pixel[1] as usize];
                        pixel[2] = contrast_lut[pixel[2] as usize];
                    }
                }
                2 => {
                    // Saturation adjustment using luminance-based desaturation
                    for pixel in rgb.pixels_mut() {
                        // Calculate grayscale value using same luminance weights
                        let gray = 0.299 * pixel[0] as f32
                            + 0.587 * pixel[1] as f32
                            + 0.114 * pixel[2] as f32;

                        pixel[0] = ((pixel[0] as f32 - gray) * saturation_factor + gray)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        pixel[1] = ((pixel[1] as f32 - gray) * saturation_factor + gray)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        pixel[2] = ((pixel[2] as f32 - gray) * saturation_factor + gray)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
                3 => {
                    // Hue adjustment via RGB->HSV->RGB conversion
                    // Only processes pixels with colour (delta > 0), leaving grayscale pixels unchanged
                    for pixel in rgb.pixels_mut() {
                        let (r, g, b) = (
                            pixel[0] as f32 / 255.0,
                            pixel[1] as f32 / 255.0,
                            pixel[2] as f32 / 255.0,
                        );

                        // Convert RGB to HSV
                        let max = r.max(g).max(b);
                        let min = r.min(g).min(b);
                        let delta = max - min;

                        // Calculate hue based on which channel has the maximum value
                        if delta > 0.0 {
                            let value = max;
                            let saturation = delta / max;
                            let mut hue = if max == r {
                                ((g - b) / delta) % 6.0 // Red is dominant
                            } else if max == g {
                                (b - r) / delta + 2.0 // Green is dominant
                            } else {
                                (r - g) / delta + 4.0 // Blue is dominant
                            };
                            hue /= 6.0;

                            // Adjust hue
                            hue = (hue + hue_factor).fract();
                            if hue < 0.0 {
                                hue += 1.0;
                            }

                            // Convert back to RGB
                            let c = saturation * value;
                            let x = c * (1.0 - ((hue * 6.0) % 2.0 - 1.0).abs());
                            let m = value - c;

                            let (r, g, b) = match (hue * 6.0) as u32 {
                                0 => (c, x, 0.0),
                                1 => (x, c, 0.0),
                                2 => (0.0, c, x),
                                3 => (0.0, x, c),
                                4 => (x, 0.0, c),
                                _ => (c, 0.0, x),
                            };

                            pixel[0] = ((r + m) * 255.0).round().clamp(0.0, 255.0) as u8;
                            pixel[1] = ((g + m) * 255.0).round().clamp(0.0, 255.0) as u8;
                            pixel[2] = ((b + m) * 255.0).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(DynamicImage::ImageRgb8(rgb))
    }
}

// ============================================================================
// Normalize
// ============================================================================

thread_local! {
    /// Thread-local cache for normalization tensors.
    /// Each DataLoader worker thread gets its own cache to avoid lock contention.
    /// Key: (num_channels, kind) -> (mean_tensor, std_tensor)
    static NORM_CACHE: RefCell<HashMap<(i64, Kind), (Tensor, Tensor)>> = RefCell::new(HashMap::new());
}

/// Normalizes tensors using channel-wise statistics.
///
/// Caches normalization tensors per thread to avoid repeated allocations
/// when processing many tensors with the same shape and data type.
///
/// # Arguments:
/// - `mean`: per-channel means
/// - `std`: per-channel standard deviation.
/// The dimensions of mean and and std should match the input tensor's
/// number of channels.
///
/// # Mathematical Operation:
/// ```text
/// output[...,c,h,w] = (input[...,c,h,w] - mean[c]) / std[c]
/// ```
///
/// # Example
/// ```ignore
/// let norm = Normalize::imagenet();
/// let normalized = norm.apply(tensor)?;
/// ```
#[derive(Debug)]
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Creates new normalization parameters.
    pub fn new(mean: &[f32], std: &[f32]) -> Result<Self> {
        ensure!(!mean.is_empty(), "Normalization mean cannot be empty");
        ensure!(
            mean.len() == std.len(),
            "The mean and standard deviation for normalization must match in dimension. 
            The dimension of mean is {} but the dimension of std is {}. ",
            mean.len(),
            std.len()
        );

        // Validate std values are not zero to avoid division by zero
        ensure!(
            std.iter().all(|&s| s > 0.0),
            "Standard deviation values must be positive"
        );

        Ok(Self {
            mean: mean.to_vec(),
            std: std.to_vec(),
        })
    }

    /// ImageNet standard normalization (RGB)
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    /// Get or create cached normalization tensors for the given channels and data type.
    fn get_norm_tensors(&self, num_channels: i64, kind: Kind) -> Result<(Tensor, Tensor)> {
        ensure!(
            num_channels as usize == self.mean.len(),
            "Channel count mismatch: input has {} channels but normalization expects {}",
            num_channels,
            self.mean.len()
        );

        NORM_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = (num_channels, kind);

            // Return cached tensors if available
            if let Some((mean_t, std_t)) = cache.get(&key) {
                return Ok((mean_t.shallow_clone(), std_t.shallow_clone()));
            }

            // Create new tensors if not in cache
            let mean_t = Tensor::from_slice(&self.mean)
                .reshape(&[num_channels, 1, 1])
                .to_kind(kind);

            let std_t = Tensor::from_slice(&self.std)
                .reshape(&[num_channels, 1, 1])
                .to_kind(kind);

            // Cache for future use (limit cache size to prevent unbounded growth)
            if cache.len() < 8 {
                cache.insert(key, (mean_t.shallow_clone(), std_t.shallow_clone()));
            }

            Ok((mean_t, std_t))
        })
    }
}

impl Transform<Tensor, Tensor> for Normalize {
    fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        let (num_channels, _height, _width) = tensor
            .size3()
            .context("Input must be 3D tensor [C, H, W]")?;

        // Get cached or create new normalization tensors
        let (mean_t, std_t) = self.get_norm_tensors(num_channels, tensor.kind())?;

        // Apply normalization
        Ok((tensor - mean_t) / std_t)
    }
}

/// Clear the thread-local normalization cache.
///
/// Useful for tests or when switching between very different workloads
/// with different tensor shapes/types.
pub fn clear_normalize_cache() {
    NORM_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::init_worker_rng;
    use image::{Rgb, RgbImage};
    use tch::{Device, Kind, Tensor};

    // Helper function to compare images
    fn image_equals(img1: &DynamicImage, img2: &DynamicImage) -> bool {
        let rgb1 = img1.to_rgb8();
        let rgb2 = img2.to_rgb8();

        if rgb1.dimensions() != rgb2.dimensions() {
            return false;
        }

        rgb1.pixels().zip(rgb2.pixels()).all(|(p1, p2)| p1 == p2)
    }

    #[test]
    fn test_normalize() -> Result<()> {
        let tensor = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
        let norm = Normalize::new(&[1.0; 3], &[1.0; 3])?;

        let normalized = norm.apply(tensor)?;

        // Check each channel's mean separately
        for c in 0..3 {
            let channel_mean = normalized.select(0, c).mean(Kind::Float);
            assert!(channel_mean.double_value(&[]) < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_normalize_caching() -> Result<()> {
        // Clear cache from any previous tests
        clear_normalize_cache();

        let norm = Normalize::imagenet();

        // First call should populate cache
        let tensor1 = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
        let _ = norm.apply(tensor1)?;

        // Verify cache was populated
        NORM_CACHE.with(|cache| {
            assert_eq!(cache.borrow().len(), 1);
        });

        // Second call with same size should use cache
        let tensor2 = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
        let _ = norm.apply(tensor2)?;

        // Cache size should remain the same
        NORM_CACHE.with(|cache| {
            assert_eq!(cache.borrow().len(), 1);
        });

        // Different kind should create new entry
        let tensor3 = Tensor::ones(&[3, 32, 32], (Kind::Double, Device::Cpu));
        let _ = norm.apply(tensor3)?;

        NORM_CACHE.with(|cache| {
            assert_eq!(cache.borrow().len(), 2);
        });

        Ok(())
    }

    #[test]
    fn test_normalize_imagenet_values() -> Result<()> {
        let norm = Normalize::imagenet();

        // Test with ones
        let tensor = Tensor::ones(&[3, 2, 2], (Kind::Float, Device::Cpu));
        let result = norm.apply(tensor)?;

        // Check each channel has correct normalization
        let expected_values = [
            (1.0 - 0.485) / 0.229, // R channel
            (1.0 - 0.456) / 0.224, // G channel
            (1.0 - 0.406) / 0.225, // B channel
        ];

        for (c, &expected) in expected_values.iter().enumerate() {
            let channel = result.select(0, c as i64);
            let actual = channel.double_value(&[0, 0]);
            assert!(
                (actual - expected).abs() < 1e-5,
                "Channel {} mismatch: expected {}, got {}",
                c,
                expected,
                actual
            );
        }

        Ok(())
    }

    #[test]
    fn test_normalize_thread_local_cache() -> Result<()> {
        use std::thread;

        // Each thread should have its own cache
        let norm = Normalize::imagenet();

        let handle = thread::spawn(move || {
            clear_normalize_cache();

            // This thread's cache should start empty
            NORM_CACHE.with(|cache| {
                assert_eq!(cache.borrow().len(), 0);
            });

            // Populate this thread's cache
            let tensor = Tensor::ones(&[3, 16, 16], (Kind::Float, Device::Cpu));
            let _ = norm.apply(tensor).unwrap();

            NORM_CACHE.with(|cache| {
                assert_eq!(cache.borrow().len(), 1);
            });
        });

        handle.join().unwrap();

        // Main thread's cache should be independent
        NORM_CACHE.with(|cache| {
            // Could be 0 or more depending on other tests
            let _ = cache.borrow().len(); // Just verify it doesn't panic
        });

        Ok(())
    }

    #[test]
    fn test_color_jitter() -> Result<()> {
        // Initialize worker RNG for testing
        init_worker_rng(0, 0, 42);

        // Create test image: simple 2x2 with known values
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([100, 100, 100])); // gray
        img.put_pixel(1, 0, Rgb([200, 50, 50])); // reddish
        img.put_pixel(0, 1, Rgb([50, 200, 50])); // greenish
        img.put_pixel(1, 1, Rgb([50, 50, 200])); // bluish
        let test_img = DynamicImage::ImageRgb8(img);

        // Test 1: No change
        let no_change = ColorJitter::new(0.0, 0.0, 0.0, 0.0)?;
        let result = no_change.apply(test_img.clone())?;
        assert!(
            image_equals(&test_img, &result),
            "Zero jitter should not change image"
        );

        // Test 2: Symmetric ranges with randomness check
        let jitter = ColorJitter::new(0.5, 0.5, 0.5, 0.2)?;
        let result1 = jitter.apply(test_img.clone())?;
        let result2 = jitter.apply(test_img.clone())?;
        assert!(
            !image_equals(&result1, &result2),
            "Should produce different results"
        );

        // Test 3: Brightness only - verify pixels change proportionally
        let brightness_only = ColorJitter::new(0.5, 0.0, 0.0, 0.0)?;
        let bright_result = brightness_only.apply(test_img.clone())?;
        let bright_rgb = bright_result.to_rgb8();
        let orig_rgb = test_img.to_rgb8();

        // Check that all pixels changed by same factor
        let factor = bright_rgb.get_pixel(0, 0)[0] as f32 / orig_rgb.get_pixel(0, 0)[0] as f32;
        assert!(
            factor >= 0.5 && factor <= 1.5,
            "Brightness factor out of range"
        );

        // Test 4: Edge case - saturated colors shouldn't overflow
        let edge_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([255, 0, 128])));
        let jitter_max = ColorJitter::new(0.5, 0.5, 0.5, 0.3)?;
        let edge_result = jitter_max.apply(edge_img)?;

        // Verify all pixels are valid (no overflow/underflow)
        let edge_rgb = edge_result.to_rgb8();
        // If we got here without panic, all values are valid u8s
        assert_eq!(edge_rgb.width(), 2);
        assert_eq!(edge_rgb.height(), 2);

        // Test 5: Combined transforms produce varied results
        let combined = ColorJitter::new(0.5, 0.5, 0.5, 0.2)?;
        let mut results = vec![];
        for _ in 0..5 {
            results.push(combined.apply(test_img.clone())?);
        }

        // Check that at least 3 out of 5 are different (allowing for occasional duplicates)
        let mut different_count = 0;
        for i in 1..results.len() {
            if !image_equals(&results[0], &results[i]) {
                different_count += 1;
            }
        }
        assert!(
            different_count >= 3,
            "Combined transforms should produce varied results"
        );

        // Test 6: Grayscale images (edge case for hue)
        let gray_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([128, 128, 128])));
        let hue_shift = ColorJitter::new(0.0, 0.0, 0.0, 0.5)?;
        let gray_hue_result = hue_shift.apply(gray_img.clone())?;
        assert!(
            image_equals(&gray_img, &gray_hue_result),
            "Hue shift shouldn't affect grayscale"
        );

        // Test 7: Invalid parameters
        assert!(
            ColorJitter::new(-0.1, 0.0, 0.0, 0.0).is_err(),
            "Negative brightness"
        );
        assert!(ColorJitter::new(0.0, 0.0, 0.0, 0.6).is_err(), "Hue > 0.5");
        assert!(
            ColorJitter::new((1.5, 0.5), 0.0, 0.0, 0.0).is_err(),
            "Min > Max"
        );

        Ok(())
    }
}
