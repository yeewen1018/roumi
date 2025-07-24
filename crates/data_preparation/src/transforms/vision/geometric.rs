use crate::dataloader::worker_gen_range;
use crate::transforms::Transform;
use anyhow::{anyhow, ensure, Result};
use fast_image_resize as fir;
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

/// Resizes an image to the specified dimension.
///
/// Supports two resize modes:
/// - **Aspect-preserving**: Resize shorter edge to target size (maintains aspect ratio)
/// - **Exact dimensions**: Resize to specified widthxheight (may distort image).
///
/// Users must specify the filter type.
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
///
/// // Aspect-preserving resize: shorter edge becomes 256px
/// let resize = Resize::new(256, FilterType::Triangle)?;
/// // 1000x800 image -> 320x256 image (aspect ratio preserved)
///
/// // Exact dimensions: resize to specified size (may distort)
/// let resize = Resize::new((224, 224), FilterType::Triangle)?;
/// // 1000x800 image -> 224x224 image (aspect ratio may change)
/// ```
#[derive(Debug, Clone)]
pub struct Resize {
    size: Size,
    filter: FilterType,
}

// Resize target specification
#[derive(Debug, Clone)]
pub enum Size {
    /// Resize shorter edge to this size (preserves aspect ratio)
    Single(u32),
    /// Resize to exact dimensions (may change aspect ratio)
    Tuple(u32, u32),
}

impl From<u32> for Size {
    fn from(size: u32) -> Self {
        Size::Single(size)
    }
}

impl From<(u32, u32)> for Size {
    fn from((width, height): (u32, u32)) -> Self {
        Size::Tuple(width, height)
    }
}

impl Resize {
    pub fn new<S: Into<Size>>(size: S, filter: FilterType) -> Result<Self> {
        let size = size.into();
        match &size {
            Size::Single(s) => ensure!(*s > 0, "Size must be positive (got {})", s),
            Size::Tuple(w, h) => ensure!(
                *w > 0 && *h > 0,
                "Dimensions must be positive (got {}x{})",
                w,
                h
            ),
        }
        Ok(Self { size, filter })
    }

    /// Calculate output dimensions based on input size and resize mode.
    fn calculate_output_size(&self, width: u32, height: u32) -> (u32, u32) {
        match self.size {
            Size::Single(target_size) => {
                // Resize shorter edge to target size, scale longer edge proportionally
                let scale = if width < height {
                    target_size as f32 / width as f32
                } else {
                    target_size as f32 / height as f32
                };

                let new_width = (width as f32 * scale).round() as u32;
                let new_height = (height as f32 * scale).round() as u32;
                (new_width, new_height)
            }
            Size::Tuple(w, h) => (w, h),
        }
    }

    /// Converts image::FilterType to fast_image_resize::ResizeAlg
    fn to_fir_algorithm(&self) -> fir::ResizeAlg {
        match self.filter {
            FilterType::Nearest => fir::ResizeAlg::Nearest,
            FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
            FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
            FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
            FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
        }
    }
}

impl Transform<DynamicImage, DynamicImage> for Resize {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        let rgb = img.to_rgb8();
        let (src_width, src_height) = (rgb.width(), rgb.height());

        // Create source image
        let src = fir::images::Image::from_vec_u8(
            src_width,
            src_height,
            rgb.into_raw(),
            fir::PixelType::U8x3,
        )?;

        // Calculate output dimensions based on resize mode
        let (dst_width, dst_height) = self.calculate_output_size(src_width, src_height);

        // Create destination image buffer
        let mut dst = fir::images::Image::new(dst_width, dst_height, src.pixel_type());

        // Configure resizer with selected algorithm
        let mut resizer = fir::Resizer::new();
        let options = fir::ResizeOptions::new().resize_alg(self.to_fir_algorithm());

        // Perform the resize operation
        resizer.resize(&src, &mut dst, &options)?;

        // Convert back to DynamicImage
        let buffer = ImageBuffer::from_raw(dst_width, dst_height, dst.buffer().to_vec())
            .ok_or_else(|| anyhow::anyhow!("Failed to create output image buffer"))?;

        Ok(DynamicImage::ImageRgb8(buffer))
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

// ===========================================================================
// RandomResizedCrop
// ===========================================================================

/// Randomly crops a region from the image and resizes it to the target size.
///
/// # Parameters
/// - `width`: Target width after resizing
/// - `height`: Target height after resizing
/// - `scale`: Range of the cropped area relative to the original image area (e.g., (0.08, 1.0) means 8% to 100%)
/// - `ratio`: Range of aspect ratios for the crop (e.g., (0.75, 1.333) means 3:4 to 4:3)
/// - `filter`: Interpolation method for resizing (default: Triangle/Bilinear)
///
/// # Example
/// ```ignore
/// // Basic usage with defaults
/// let crop = RandomResizedCrop::new(224, 224)?;
///
/// // Custom scale and ratio ranges
/// let crop = RandomResizedCrop::new(224, 224)?
///     .with_scale((0.08, 1.0))?
///     .with_ratio((0.75, 1.333))?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomResizedCrop {
    width: u32,
    height: u32,
    scale_min: f64,
    scale_max: f64,
    ratio_min: f64,
    ratio_max: f64,
    max_attempts: u32,
    filter: FilterType,
}

impl RandomResizedCrop {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "Target dimensions must be positive (got {}x{})",
            width,
            height
        );
        Ok(Self {
            width,
            height,
            scale_min: 0.08,
            scale_max: 1.0,
            ratio_min: 3.0 / 4.0,
            ratio_max: 4.0 / 3.0,
            max_attempts: 10,
            filter: FilterType::Triangle,
        })
    }

    /// Set the scale range for random cropping.
    ///
    /// Scale represents the fraction of the original image area to crop.
    pub fn with_scale(mut self, scale: (f64, f64)) -> Result<Self> {
        let (min, max) = scale;
        ensure!(
            0.0 < min && min <= max && max <= 1.0,
            "Scale range must satisfy 0 < min <= max <= 1 (got [{}, {}])",
            min,
            max
        );
        self.scale_min = min;
        self.scale_max = max;
        Ok(self)
    }

    /// Set the aspect ratio range for random cropping.
    ///
    /// Ratio is width/height of the crop region.
    pub fn with_ratio(mut self, ratio: (f64, f64)) -> Result<Self> {
        let (min, max) = ratio;
        ensure!(
            0.0 < min && min <= max,
            "Ratio range must satisfy 0 < min <= max (got [{}, {}])",
            min,
            max
        );
        self.ratio_min = min;
        self.ratio_max = max;
        Ok(self)
    }

    /// Set the interpolation filter for resizing.
    pub fn with_filter(mut self, filter: FilterType) -> Self {
        self.filter = filter;
        self
    }

    fn get_params(&self, img_width: u32, img_height: u32) -> Result<(u32, u32, u32, u32)> {
        let area = (img_width * img_height) as f64;

        // Try to find a valid random crop
        for _ in 0..self.max_attempts {
            let target_area = area * worker_gen_range(self.scale_min..=self.scale_max)?;
            let aspect_ratio = worker_gen_range(self.ratio_min..=self.ratio_max)?;

            let w = (target_area * aspect_ratio).sqrt().round() as u32;
            let h = (target_area / aspect_ratio).sqrt().round() as u32;

            if w <= img_width && h <= img_height {
                let left = worker_gen_range(0..=(img_width - w))?;
                let top = worker_gen_range(0..=(img_height - h))?;
                return Ok((left, top, w, h));
            }
        }

        // Fallback: center crop with constrained aspect ratio
        let in_ratio = img_width as f64 / img_height as f64;
        let (w, h) = if in_ratio < self.ratio_min {
            (
                img_width,
                (img_width as f64 / self.ratio_min).round() as u32,
            )
        } else if in_ratio > self.ratio_max {
            (
                (img_height as f64 * self.ratio_max).round() as u32,
                img_height,
            )
        } else {
            (img_width, img_height)
        };

        let left = (img_width - w) / 2;
        let top = (img_height - h) / 2;
        Ok((left, top, w, h))
    }

    fn to_fir_algorithm(&self) -> fir::ResizeAlg {
        match self.filter {
            FilterType::Nearest => fir::ResizeAlg::Nearest,
            FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
            FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
            FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
            FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
        }
    }
}

impl Transform<DynamicImage, DynamicImage> for RandomResizedCrop {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        let (img_width, img_height) = img.dimensions();
        let (left, top, crop_width, crop_height) = self.get_params(img_width, img_height)?;

        // Crop the selected region
        let cropped = img.crop_imm(left, top, crop_width, crop_height);

        // Convert to RGB and prepare for fast resize
        let rgb = cropped.to_rgb8();
        let src = fir::images::Image::from_vec_u8(
            crop_width,
            crop_height,
            rgb.into_raw(),
            fir::PixelType::U8x3,
        )?;

        // Resize to target dimensions
        let mut dst = fir::images::Image::new(self.width, self.height, src.pixel_type());
        let mut resizer = fir::Resizer::new();
        let options = fir::ResizeOptions::new().resize_alg(self.to_fir_algorithm());

        resizer.resize(&src, &mut dst, &options)?;

        // Convert back to DynamicImage
        let buffer = ImageBuffer::from_raw(self.width, self.height, dst.buffer().to_vec())
            .ok_or_else(|| anyhow::anyhow!("Failed to create output image buffer"))?;

        Ok(DynamicImage::ImageRgb8(buffer))
    }
}

// ============================================================================
// RandomRotation
// ============================================================================

/// Randomly rotates an image within a specified degree range.
///
/// # Parameters
/// - `degrees`: Range of degrees to select from. If a single number, range will be (-degrees, +degrees)
/// - `interpolation`: Pixel interpolation method (default: Nearest)
/// - `expand`: Whether to expand output to fit entire rotated image (default: false)
/// - `center`: Custom rotation center as (x, y). Default is image center.
/// - `fill`: Pixel fill value for areas outside the original image (default: black)
///
/// # Example
/// ```ignore
/// // Basic ±30 degree rotation
/// let rotation = RandomRotation::new(30.0)?;
///
/// // Custom range with bilinear interpolation
/// let rotation = RandomRotation::with_range(-90.0, 45.0)?
///     .with_interpolation(InterpolationMode::Bilinear)
///     .with_expand(true)
///     .with_fill(vec![255.0, 0.0, 0.0]); // Red fill
/// ```
#[derive(Debug, Clone)]
pub struct RandomRotation {
    min_degrees: f32,
    max_degrees: f32,
    interpolation: InterpolationMode,
    expand: bool,
    center: Option<(f32, f32)>,
    fill: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    /// Nearest neighbour (fastest, lower quality)
    Nearest,
    /// Bilinear interpolation (slower, higher quality)
    Bilinear,
}

impl RandomRotation {
    /// Create rotation with symmetric range [-degrees, +degrees]
    pub fn new(degrees: f32) -> Result<Self> {
        ensure!(
            degrees >= 0.0,
            "Degrees must be non-negative for symmetric range (got {})",
            degrees
        );
        Ok(Self {
            min_degrees: -degrees,
            max_degrees: degrees,
            interpolation: InterpolationMode::Nearest,
            expand: false,
            center: None,
            fill: vec![0.0],
        })
    }

    /// Create rotation with custom range [min_degrees, max_degrees]
    pub fn with_range(min_degrees: f32, max_degrees: f32) -> Result<Self> {
        ensure!(
            min_degrees <= max_degrees,
            "Min degrees must be <= max degrees (got [{}, {}])",
            min_degrees,
            max_degrees
        );
        Ok(Self {
            min_degrees,
            max_degrees,
            interpolation: InterpolationMode::Nearest,
            expand: false,
            center: None,
            fill: vec![0.0],
        })
    }

    /// Set interpolation method for pixel sampling.
    pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
        self.interpolation = mode;
        self
    }

    /// Set whether to expand output canvas to fit the entire rotated image.
    pub fn with_expand(mut self, expand: bool) -> Self {
        self.expand = expand;
        self
    }

    /// Set center of rotation (default is image center)
    pub fn with_center(mut self, center: (f32, f32)) -> Self {
        self.center = Some(center);
        self
    }

    /// Set fill value for areas outside the original image.
    /// Accepts single value (grayscale) or RGB array.
    pub fn with_fill(mut self, fill: impl Into<Vec<f32>>) -> Self {
        self.fill = fill.into();
        self
    }

    fn rotate_image(&self, img: &DynamicImage, angle: f32) -> Result<DynamicImage> {
        let rgb = img.to_rgb8();
        let (width, height) = (rgb.width() as f32, rgb.height() as f32);
        let (center_x, center_y) = self.center.unwrap_or((width / 2.0, height / 2.0));

        // Convert to radians
        let theta = angle.to_radians();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // 1. Calculate output size
        let (new_width, new_height, offset_x, offset_y) = if self.expand {
            // Calculate bounds of rotated corners
            let corners = [
                (0.0 - center_x, 0.0 - center_y),
                (width - center_x, 0.0 - center_y),
                (0.0 - center_x, height - center_y),
                (width - center_x, height - center_y),
            ];

            let mut min_x: f32 = 0.0;
            let mut max_x: f32 = 0.0;
            let mut min_y: f32 = 0.0;
            let mut max_y: f32 = 0.0;

            for (x, y) in corners.iter() {
                let x_rot = x * cos_theta - y * sin_theta;
                let y_rot = x * sin_theta + y * cos_theta;
                min_x = min_x.min(x_rot);
                max_x = max_x.max(x_rot);
                min_y = min_y.min(y_rot);
                max_y = max_y.max(y_rot);
            }

            let new_width = (max_x - min_x).ceil() as u32;
            let new_height = (max_y - min_y).ceil() as u32;
            let offset_x = -min_x + (new_width as f32 - width) / 2.0;
            let offset_y = -min_y + (new_height as f32 - height) / 2.0;

            (new_width, new_height, offset_x, offset_y)
        } else {
            (width as u32, height as u32, 0.0, 0.0)
        };

        // 2. Get fill values
        let fill_rgb = if self.fill.len() == 1 {
            // Single value for all channels
            let val = (self.fill[0].clamp(0.0, 255.0)) as u8;
            [val, val, val]
        } else if self.fill.len() >= 3 {
            // Per-channel values
            [
                (self.fill[0].clamp(0.0, 255.0)) as u8,
                (self.fill[1].clamp(0.0, 255.0)) as u8,
                (self.fill[2].clamp(0.0, 255.0)) as u8,
            ]
        } else {
            return Err(anyhow!(
                "Fill must be single value or have at least 3 values for RGB"
            ));
        };

        // 3. Create output image
        let mut rotated = ImageBuffer::from_pixel(new_width, new_height, Rgb(fill_rgb));

        // Rotation parameters
        let new_center_x = center_x + offset_x;
        let new_center_y = center_y + offset_y;
        let cos_neg = (-theta).cos();
        let sin_neg = (-theta).sin();

        // Inverse mapping: for each output pixel, find corresponding input pixel
        for y in 0..new_height {
            for x in 0..new_width {
                let dx = x as f32 - new_center_x;
                let dy = y as f32 - new_center_y;

                let src_x = dx * cos_neg - dy * sin_neg + center_x;
                let src_y = dx * sin_neg + dy * cos_neg + center_y;

                // Apply interpolation
                match self.interpolation {
                    InterpolationMode::Nearest => {
                        let src_x_round = src_x.round() as i32;
                        let src_y_round = src_y.round() as i32;

                        if src_x_round >= 0
                            && src_x_round < width as i32
                            && src_y_round >= 0
                            && src_y_round < height as i32
                        {
                            let pixel = rgb.get_pixel(src_x_round as u32, src_y_round as u32);
                            rotated.put_pixel(x, y, *pixel);
                        }
                    }
                    InterpolationMode::Bilinear => {
                        if src_x >= 0.0
                            && src_x < width - 1.0
                            && src_y >= 0.0
                            && src_y < height - 1.0
                        {
                            let x0 = src_x.floor() as u32;
                            let y0 = src_y.floor() as u32;
                            let x1 = (x0 + 1).min(rgb.width() - 1);
                            let y1 = (y0 + 1).min(rgb.height() - 1);

                            let fx = src_x - x0 as f32;
                            let fy = src_y - y0 as f32;

                            let p00 = rgb.get_pixel(x0, y0);
                            let p01 = rgb.get_pixel(x0, y1);
                            let p10 = rgb.get_pixel(x1, y0);
                            let p11 = rgb.get_pixel(x1, y1);

                            let interpolated = Rgb([
                                (p00[0] as f32 * (1.0 - fx) * (1.0 - fy)
                                    + p10[0] as f32 * fx * (1.0 - fy)
                                    + p01[0] as f32 * (1.0 - fx) * fy
                                    + p11[0] as f32 * fx * fy)
                                    .round() as u8,
                                (p00[1] as f32 * (1.0 - fx) * (1.0 - fy)
                                    + p10[1] as f32 * fx * (1.0 - fy)
                                    + p01[1] as f32 * (1.0 - fx) * fy
                                    + p11[1] as f32 * fx * fy)
                                    .round() as u8,
                                (p00[2] as f32 * (1.0 - fx) * (1.0 - fy)
                                    + p10[2] as f32 * fx * (1.0 - fy)
                                    + p01[2] as f32 * (1.0 - fx) * fy
                                    + p11[2] as f32 * fx * fy)
                                    .round() as u8,
                            ]);

                            rotated.put_pixel(x, y, interpolated);
                        }
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(rotated))
    }
}

impl Transform<DynamicImage, DynamicImage> for RandomRotation {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        let angle = worker_gen_range(self.min_degrees..=self.max_degrees)?;
        self.rotate_image(&img, angle)
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
        let resize = Resize::new((50, 50), FilterType::Nearest)?;
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
    fn test_random_resized_crop() -> Result<()> {
        init_worker_rng(0, 0, 42);
        let img = test_gradient_image(100, 100);
        let crop = RandomResizedCrop::new(64, 64)?;
        let cropped = crop.apply(img)?;
        assert_eq!(cropped.dimensions(), (64, 64));
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

    #[test]
    fn test_random_rotation() -> Result<()> {
        init_worker_rng(0, 0, 42);
        let img = test_gradient_image(100, 100);

        // Basic functionality and randomness
        let rotation = RandomRotation::new(45.0)?;
        let rotated = rotation.apply(img.clone())?;
        assert_eq!(rotated.dimensions(), (100, 100));

        // Verify different angles produce different results
        let results: Vec<_> = (0..3)
            .map(|_| {
                let rotated = rotation.apply(img.clone()).unwrap();
                let rgb = rotated.to_rgb8();
                *rgb.get_pixel(25, 25)
            })
            .collect();
        let unique_count = results
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(unique_count > 1, "Should produce varied rotations");

        // Expansion test
        let expand_rotation = RandomRotation::with_range(45.0, 45.0)?.with_expand(true);
        let expanded = expand_rotation.apply(img.clone())?;
        assert!(expanded.width() > 100, "Expansion should increase size");

        // Zero rotation preserves image
        let no_rotation = RandomRotation::new(0.0)?;
        assert!(image_equals(&img, &no_rotation.apply(img.clone())?));

        // Fill color test
        let red_fill = RandomRotation::new(45.0)?.with_fill(vec![255.0, 0.0, 0.0]);
        let filled = red_fill.apply(img)?;
        let filled_rgb = filled.to_rgb8();
        let corner = filled_rgb.get_pixel(0, 0);
        assert_eq!(corner.0, [255, 0, 0]);

        Ok(())
    }
}
