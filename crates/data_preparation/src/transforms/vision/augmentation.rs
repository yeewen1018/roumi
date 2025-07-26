use crate::dataloader::worker_gen_bool;
use crate::transforms::Transform;
use anyhow::{ensure, Result};
use image::{DynamicImage, ImageBuffer, RgbImage};

// ============================================================================
// RandomHorizontalFlip
// ============================================================================

/// Randomly flips images horizontally during training.
/// Panics if probability `p` is outside [0.0, 1.0] range.
///
/// # Example
/// ```ignore
/// let flip = RandomHorizontalFlip::new(0.5)?; // 50% flip chance
/// let augmented = flip.apply(image)?;
/// ```
#[derive(Debug)]
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    pub fn new(p: f64) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&p),
            "Probability must be in [0.0, 1.0] range (got {})",
            p
        );
        Ok(Self { p })
    }

    /// Flips an RGB8 image horizontally using optimized memory operations
    fn flip_rgb8(img: RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let width_usize = width as usize;
        let height_usize = height as usize;
        let source_pixels = img.into_raw();

        let mut flipped_pixels = Vec::with_capacity(source_pixels.len());

        // Process each row, copying pixels from right to left
        for y in 0..height_usize {
            for x in 0..width_usize {
                // Calculate source index: rightmost pixel of current row
                let src_pixel_idx = (y * width_usize + (width_usize - 1 - x)) * 3;

                // Copy RGB triplet
                flipped_pixels.extend_from_slice(&source_pixels[src_pixel_idx..src_pixel_idx + 3]);
            }
        }

        ImageBuffer::from_raw(width, height, flipped_pixels)
            .expect("Failed to create flipped image buffer")
    }
}

impl Transform<DynamicImage, DynamicImage> for RandomHorizontalFlip {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        let result = match self.p {
            // Fast path: never flip
            0.0 => img,

            // Fast path: always flip
            1.0 => {
                match img {
                    DynamicImage::ImageRgb8(rgb_img) => {
                        DynamicImage::ImageRgb8(Self::flip_rgb8(rgb_img))
                    }
                    _ => img.fliph(), // Fall back to generic implementation
                }
            }

            // Random flip based on probability
            _ => {
                if worker_gen_bool(self.p) {
                    match img {
                        DynamicImage::ImageRgb8(rgb_img) => {
                            DynamicImage::ImageRgb8(Self::flip_rgb8(rgb_img))
                        }
                        _ => img.fliph(), // Fall back to generic implementation
                    }
                } else {
                    img // No flip
                }
            }
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::init_worker_rng;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_random_horizontal_flip() -> Result<()> {
        // Initialize deterministic RNG
        init_worker_rng(0, 0, 42);

        // Create a 2×1 image where left = red, right = blue
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // red
        img.put_pixel(1, 0, Rgb([0, 0, 255])); // blue

        let flip = RandomHorizontalFlip::new(1.0)?; // Always flip
        let flipped = flip.apply(DynamicImage::ImageRgb8(img))?;

        // After flip, left should be blue, right should be red:
        assert_eq!(flipped.as_bytes(), &[0, 0, 255, 255, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_random_horizontal_flip_no_flip() -> Result<()> {
        init_worker_rng(0, 0, 42);

        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // red
        img.put_pixel(1, 0, Rgb([0, 0, 255])); // blue
        let original = DynamicImage::ImageRgb8(img);

        let flip = RandomHorizontalFlip::new(0.0)?; // Never flip
        let result = flip.apply(original.clone())?;

        // Should be unchanged
        assert_eq!(original.as_bytes(), result.as_bytes());
        Ok(())
    }
}
