use crate::dataloader::worker_gen_range;
use crate::transforms::Transform;
use anyhow::{anyhow, ensure, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Rgb};

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

// ============================================================================
// RandomCrop
// ============================================================================

/// Randomly crops a region from the input image.
///
/// If the input image is smaller than the crop size, users need to enable
/// automatic padding (centers the image) with `with_pad_if_needed(true)`
/// or explicitly configure padding with `with_padding()`. An error occurs otherwise.
///
/// # Arguments:
/// - `size`: Target crop size as (height, width). Can be a single value for square crops.
/// - `padding`: Optional padding to apply before cropping ([`CropPadding`])
/// - `pad_if_needed`: If true, automatically center small images with symmetric padding (default: false)
/// - `fill`: RGB colour values for padded areas (default: [0, 0, 0] black)
/// - `padding_mode`: How to fill padded areas ([`CropPaddingMode`], default: Constant)
///
/// # Example
/// ```ignore
/// // Basic random crop
/// let crop = RandomCrop::new((224, 224))?;
///
/// // Handle small images with automatic padding
/// let crop = RandomCrop::new((300, 300))?
///     .with_pad_if_needed(true);
///
/// // Custom padding with reflection
/// let crop = RandomCrop::new((300, 300))?
///     .with_padding(CropPadding::Uniform(10))
///     .with_padding_mode(CropPaddingMode::Reflect);
/// ```
#[derive(Debug, Clone)]
pub struct RandomCrop {
    size: (u32, u32),
    padding: Option<CropPadding>,
    pad_if_needed: bool,
    fill: [u8; 3],
    padding_mode: CropPaddingMode,
}

/// Padding strategy for crop operations
#[derive(Debug, Clone)]
pub enum CropPadding {
    /// Same padding on all sides
    Uniform(u32),
    /// Horizontal and vertical padding (left/right, top/bottom)
    Symmetric(u32, u32),
    /// Explicit padding: (left, top, right, bottom)
    Explicit(u32, u32, u32, u32),
}

/// How to fill padded areas
#[derive(Debug, Clone)]
pub enum CropPaddingMode {
    /// Fill with constant colour (specified by `fill`)
    Constant,
    /// Extend edge pixels
    Edge,
    /// Mirror pixels at borders
    Reflect,
    /// Mirror pixels with border repetition
    Symmetric,
}

impl RandomCrop {
    pub fn new(size: impl Into<(u32, u32)>) -> Result<Self> {
        let size = size.into();
        ensure!(
            size.0 > 0 && size.1 > 0,
            "Crop size must be positive (got {}x{})",
            size.0,
            size.1
        );
        Ok(Self {
            size,
            padding: None,
            pad_if_needed: false,
            fill: [0, 0, 0],
            padding_mode: CropPaddingMode::Constant,
        })
    }

    /// Add padding before cropping.
    pub fn with_padding(mut self, padding: CropPadding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// Automatically pad images smaller than crop size.
    pub fn with_pad_if_needed(mut self, pad: bool) -> Self {
        self.pad_if_needed = pad;
        self
    }

    /// Set fill colour for padded areas (RGB values).
    pub fn with_fill(mut self, fill: [u8; 3]) -> Self {
        self.fill = fill;
        self
    }

    /// Set padding mode (how to fill padded areas).
    pub fn with_padding_mode(mut self, mode: CropPaddingMode) -> Self {
        self.padding_mode = mode;
        self
    }

    fn apply_padding(&self, img: &DynamicImage, padding: &CropPadding) -> DynamicImage {
        let (width, height) = img.dimensions();

        let (left, top, right, bottom) = match padding {
            CropPadding::Uniform(p) => (*p, *p, *p, *p),
            CropPadding::Symmetric(h, v) => (*h, *v, *h, *v),
            CropPadding::Explicit(l, t, r, b) => (*l, *t, *r, *b),
        };

        let new_width = width + left + right;
        let new_height = height + top + bottom;

        match self.padding_mode {
            CropPaddingMode::Constant => {
                let mut padded = ImageBuffer::from_pixel(new_width, new_height, Rgb(self.fill));

                for (x, y, pixel) in img.to_rgb8().enumerate_pixels() {
                    padded.put_pixel(x + left, y + top, *pixel);
                }

                DynamicImage::ImageRgb8(padded)
            }
            _ => {
                // TODO: Implement Edge, Reflect, Symmetric padding modes
                todo!("Only Constant padding mode is currently implemented")
            }
        }
    }

    fn get_params(&self, img_width: u32, img_height: u32) -> Result<(u32, u32)> {
        let (crop_height, crop_width) = self.size;

        if img_height < crop_height || img_width < crop_width {
            return Err(anyhow!(
                "Required crop size {:?} is larger than input image size ({}, {})",
                self.size,
                img_width,
                img_height
            ));
        }

        if img_width == crop_width && img_height == crop_height {
            return Ok((0, 0));
        }

        let top = if img_height > crop_height {
            worker_gen_range(0..=(img_height - crop_height))?
        } else {
            0
        };

        let left = if img_width > crop_width {
            worker_gen_range(0..=(img_width - crop_width))?
        } else {
            0
        };

        Ok((left, top))
    }
}

impl Transform<DynamicImage, DynamicImage> for RandomCrop {
    fn apply(&self, mut img: DynamicImage) -> Result<DynamicImage> {
        // 1. Apply explicit padding if specified
        if let Some(ref padding) = self.padding {
            img = self.apply_padding(&img, padding);
        }

        let (mut width, mut height) = img.dimensions();

        // 2. Apply automatic padding if needed
        if self.pad_if_needed {
            if width < self.size.1 {
                let padding = CropPadding::Symmetric((self.size.1 - width + 1) / 2, 0);
                img = self.apply_padding(&img, &padding);
                width = img.width();
            }

            if height < self.size.0 {
                let padding = CropPadding::Symmetric(0, (self.size.0 - height + 1) / 2);
                img = self.apply_padding(&img, &padding);
                height = img.height();
            }
        }

        // 3. Get random crop coordinates
        let (left, top) = self.get_params(width, height)?;

        // 4. Crop
        Ok(img.crop_imm(left, top, self.size.1, self.size.0))
    }
}

// ============================================================================
// CenterCrop
// ============================================================================

/// Crops the center of an image to the specified dimensions.
///
/// If the input image is smaller than the crop size in any dimension,
/// it will be padded with the specified fill value before cropping.
///
/// # Arguments
/// - `width`: Target width in pixels
/// - `height`: Target height in pixels
/// - `pad_value`: RGB fill colour for padding (default: [0, 0, 0] black)
///
/// # Example
/// ```ignore
/// // Basic center crop
/// let crop = CenterCrop::new(224, 224, None)?;
///
/// // With white padding for small images
/// let crop = CenterCrop::new(512, 512, Some([225, 255, 255]))?;
/// ```
#[derive(Debug, Clone)]
pub struct CenterCrop {
    width: u32,
    height: u32,
    pad_value: [u8; 3],
}

impl CenterCrop {
    pub fn new(width: u32, height: u32, pad_value: Option<[u8; 3]>) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "Crop dimensions must be positive (got {}x{})",
            width,
            height
        );
        Ok(Self {
            width,
            height,
            pad_value: pad_value.unwrap_or([0, 0, 0]),
        })
    }
}

impl Transform<DynamicImage, DynamicImage> for CenterCrop {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        let (img_width, img_height) = img.dimensions();

        // Direct crop if image is large enough
        if img_width >= self.width && img_height >= self.height {
            let left = (img_width - self.width) / 2;
            let top = (img_height - self.height) / 2;
            return Ok(img.crop_imm(left, top, self.width, self.height));
        }

        // Create padded result buffer
        let mut result = ImageBuffer::from_pixel(self.width, self.height, Rgb(self.pad_value));

        // Calculate positioning
        let paste_x = if img_width < self.width {
            (self.width - img_width) / 2
        } else {
            0
        };

        let paste_y = if img_height < self.height {
            (self.height - img_height) / 2
        } else {
            0
        };

        // Calculate the region to copy from the source image
        let copy_width = img_width.min(self.width);
        let copy_height = img_height.min(self.height);

        let src_x = if img_width > self.width {
            (img_width - self.width) / 2
        } else {
            0
        };

        let src_y = if img_height > self.height {
            (img_height - self.height) / 2
        } else {
            0
        };

        // Copy pixels efficiently
        match img {
            DynamicImage::ImageRgb8(ref rgb_img) => {
                for y in 0..copy_height {
                    for x in 0..copy_width {
                        let pixel = rgb_img.get_pixel(src_x + x, src_y + y);
                        result.put_pixel(paste_x + x, paste_y + y, *pixel);
                    }
                }
            }
            _ => {
                let rgb_img = img.to_rgb8();
                for y in 0..copy_height {
                    for x in 0..copy_width {
                        let pixel = rgb_img.get_pixel(src_x + x, src_y + y);
                        result.put_pixel(paste_x + x, paste_y + y, *pixel);
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::init_worker_rng;
    use image::{DynamicImage, GenericImageView, Rgb, RgbImage};

    // Helper function to compare images
    fn image_equals(img1: &DynamicImage, img2: &DynamicImage) -> bool {
        let rgb1 = img1.to_rgb8();
        let rgb2 = img2.to_rgb8();

        if rgb1.dimensions() != rgb2.dimensions() {
            return false;
        }

        rgb1.pixels().zip(rgb2.pixels()).all(|(p1, p2)| p1 == p2)
    }

    // Helper function to create test image
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

    #[test]
    fn test_center_crop() -> Result<()> {
        let img = test_gradient_image(100, 100);

        // Test basic crop
        let crop = CenterCrop::new(50, 50, None)?;
        let cropped = crop.apply(img.clone())?;
        assert_eq!(cropped.dimensions(), (50, 50));

        // Test crop larger than image (should pad)
        let large_crop = CenterCrop::new(150, 150, Some([255, 0, 0]))?;
        let padded = large_crop.apply(img)?;
        assert_eq!(padded.dimensions(), (150, 150));
        Ok(())
    }

    #[test]
    fn test_random_crop() -> Result<()> {
        init_worker_rng(0, 0, 42);
        let img = test_gradient_image(100, 100);
        let small_img = test_gradient_image(30, 30);

        // Basic functionality
        let crop = RandomCrop::new((50, 50))?;
        assert_eq!(crop.apply(img.clone())?.dimensions(), (50, 50));
        assert_eq!(
            RandomCrop::new((100, 100))?
                .apply(img.clone())?
                .dimensions(),
            (100, 100)
        );

        // Error handling
        assert!(RandomCrop::new((50, 50))?.apply(small_img.clone()).is_err());
        assert!(RandomCrop::new((0, 50)).is_err());

        // Automatic padding
        let auto_pad = RandomCrop::new((50, 50))?.with_pad_if_needed(true);
        assert_eq!(auto_pad.apply(small_img.clone())?.dimensions(), (50, 50));

        // Explicit padding
        let explicit_pad = RandomCrop::new((40, 40))?.with_padding(CropPadding::Uniform(10));
        assert_eq!(
            explicit_pad.apply(small_img.clone())?.dimensions(),
            (40, 40)
        );

        // Randomness verification
        let results: Vec<_> = (0..5)
            .map(|_| {
                crop.apply(img.clone())
                    .unwrap()
                    .to_rgb8()
                    .get_pixel(0, 0)
                    .clone()
            })
            .collect();
        let unique_count = results
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(unique_count > 2, "Should produce varied crops");

        Ok(())
    }

    #[test]
    fn test_random_crop_determinism() -> Result<()> {
        let img = test_gradient_image(100, 100);
        let crop = RandomCrop::new((30, 30))?;

        // Same seed = same result
        init_worker_rng(0, 0, 42);
        let result1 = crop.apply(img.clone())?;
        init_worker_rng(0, 0, 42);
        let result2 = crop.apply(img.clone())?;
        assert!(image_equals(&result1, &result2));

        // Different seed = different result
        init_worker_rng(0, 0, 100);
        let result3 = crop.apply(img)?;
        assert!(!image_equals(&result1, &result3));

        Ok(())
    }

    #[test]
    fn test_random_crop_padding_modes() -> Result<()> {
        let small_img = test_gradient_image(20, 20);

        // Test fill colors
        let red_fill = RandomCrop::new((40, 40))?
            .with_pad_if_needed(true)
            .with_fill([255, 0, 0]);
        let result = red_fill.apply(small_img.clone())?;
        let result_rgb = result.to_rgb8();
        let corners = [result_rgb.get_pixel(0, 0), result_rgb.get_pixel(39, 39)];
        assert!(corners.iter().any(|p| p[0] == 255 && p[1] == 0));

        // Test combined padding strategies
        let complex = RandomCrop::new((60, 60))?
            .with_padding(CropPadding::Uniform(10))
            .with_pad_if_needed(true);
        assert_eq!(complex.apply(small_img)?.dimensions(), (60, 60));

        Ok(())
    }
}
