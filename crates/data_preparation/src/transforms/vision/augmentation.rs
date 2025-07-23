use crate::dataloader::worker_gen_bool;
use crate::transforms::Transform;
use anyhow::{ensure, Result};
use image::DynamicImage;

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
}

impl Transform<DynamicImage, DynamicImage> for RandomHorizontalFlip {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        Ok(if worker_gen_bool(self.p) {
            img.fliph()
        } else {
            img
        })
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
