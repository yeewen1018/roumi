use crate::transforms::Transform;
use anyhow::{ensure, Result};
use image::{imageops::FilterType, DynamicImage};

// ============================================================================
// EnsureRGB
// ============================================================================
/// Ensures that the image is indeed 3-channel RGB
#[derive(Debug, Clone)]
pub struct EnsureRGB;

impl Transform<DynamicImage, DynamicImage> for EnsureRGB {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        Ok(match img {
            DynamicImage::ImageRgb8(_) => img,
            _ => DynamicImage::ImageRgb8(img.to_rgb8()),
        })
    }
}

// ============================================================================
// Resize
// ============================================================================

/// Resizes an image to the specified dimension while preserving
/// the aspect ratio. Users must specify the filter type.
///
/// # Filter Types
/// - `Nearest`: Nearest neighbour, fastest
/// - `Triangle`: Bilinear filter, good all-round default
/// - `CatmullRom`: Bicubic sharpening
/// - `Gaussian`: Blurring/smoothing
/// - `Lanczos3`: Lanczos with window 3, highest quality re-sampling but slowest.
///
/// # Examples
/// ``` ignore
/// # use image::imageops::FilterType;
/// let resize = Resize::new(256, 256, FilterType::Triangle)?;
/// let img = load_image("cat.jpg")?;
/// let resized = resize.apply(img)?;
/// ```
#[derive(Debug)]
pub struct Resize {
    width: u32,
    height: u32,
    filter: FilterType,
}

impl Resize {
    /// Creates a new Resize transform.
    pub fn new(width: u32, height: u32, filter: FilterType) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "Image dimensions must be positive after resizing (got {}x{})",
            width,
            height
        );
        Ok(Self {
            width,
            height,
            filter,
        })
    }
}

impl Transform<DynamicImage, DynamicImage> for Resize {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        Ok(img.resize(self.width, self.height, self.filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, Rgb, RgbImage};

    fn test_gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = (x * 255 / width) as u8;
                let g = (y * 255 / height) as u8;
                let b = 128;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_resize() -> Result<()> {
        let img = test_gradient_image(100, 100);
        let resize = Resize::new(50, 50, FilterType::Nearest)?;
        let resized = resize.apply(img)?;
        assert_eq!(resized.dimensions(), (50, 50));
        Ok(())
    }
}
