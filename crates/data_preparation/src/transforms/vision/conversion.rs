use crate::transforms::Transform;
use anyhow::{ensure, Context, Result};
use image::{DynamicImage, GenericImageView};
use tch::{Kind, Tensor};

// ============================================================================
// ToTensor
// ============================================================================

/// Converts an image to a channel-first f32 tensor in [0.0, 1.0] range.
///
/// Channel Handling
/// | Input Format  | Output Shape |
/// |---------------|--------------|
/// | Grayscale (L) | `[1, H, W]`  |
/// | RGB           | `[3, H, W]`  |
/// | RGBA          | `[4, H, W]`  |
/// | Other         | `[3, H, W]`  |
/// Note: *CMYK, BGR, etc. will undergo implicit conversion to RGB.
///       For precise format control, pre-convert your images.
///
/// # Example
/// ```ignore
/// let converter = ToTensor;
/// let tensor = converter.apply(image)?;
/// ```
#[derive(Debug)]
pub struct ToTensor;

impl Transform<DynamicImage, Tensor> for ToTensor {
    fn apply(&self, img: DynamicImage) -> Result<Tensor> {
        let (width, height) = img.dimensions();
        ensure!(
            width > 0 && height > 0,
            "Image dimensions must be positive (got {}x{})",
            width,
            height
        );

        let tensor = match img {
            DynamicImage::ImageLuma8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[1, height as i64, width as i64])
            }
            DynamicImage::ImageRgb8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[3, height as i64, width as i64])
            }
            DynamicImage::ImageRgba8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[4, height as i64, width as i64])
            }
            // Handle all other cases via conversion to RGB
            _ => {
                let rgb = img.to_rgb8();
                Tensor::from_slice(rgb.as_raw()).reshape(&[3, height as i64, width as i64])
            }
        };

        // Normalize to [0,1] range
        tensor
            .to_kind(Kind::Float)
            .f_div_scalar(255.0)
            .context("Failed to normalize tensor values")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use tch::Kind;

    fn test_rgb_image() -> DynamicImage {
        let mut img = RgbImage::new(3, 3);
        for x in 0..3 {
            for y in 0..3 {
                img.put_pixel(x, y, Rgb([(x * 85) as u8, (y * 85) as u8, 128]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_to_tensor() -> Result<()> {
        let img = test_rgb_image();
        let converter = ToTensor;

        let tensor = converter.apply(img)?;
        assert_eq!(tensor.size(), vec![3, 3, 3]); // CHW format
        assert_eq!(tensor.kind(), Kind::Float);

        // Verify normalization to [0,1]
        let min = tensor.f_min()?.double_value(&[]);
        let max = tensor.f_max()?.double_value(&[]);
        assert!(min >= 0.0 && max <= 1.0);
        Ok(())
    }
}
