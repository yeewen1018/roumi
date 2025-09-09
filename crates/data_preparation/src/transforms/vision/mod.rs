//! src/transforms/vision/mod.rs
//!
//! Vision transforms for image preprocessing and augmentation.
//!
//! # Module Organization
//!
//! The vision transforms are organized into focused modules based on their primary function:
//!
//! ```text
//! transforms/vision/
//! ├── geometric.rs     → Spatial transformations (resize, crop, rotate)
//! ├── photometric.rs   → Color and appearance (brightness, contrast, normalize)  
//! ├── conversion.rs    → Format conversions (image → tensor)
//! ├── augmentation.rs  → Pure data augmentation (flip, noise, etc.)
//! └── io.rs           → Image loading utilities
//! ```
//!
//! # Quick Start
//!
//! All transforms are re-exported at the module level for convenient access:
//!
//! ```ignore
//! use crate::transforms::Transform;
//! use crate::transforms::vision::{Resize, ToTensor, Normalize};
//! use image::imageops::FilterType;
//!
//! // Create a typical training pipeline
//! let pipeline = Resize::new((256, 256), FilterType::Lanczos3)?
//!     .then(ToTensor)
//!     .then(Normalize::imagenet());
//! ```

pub mod augmentation;
pub mod conversion;
pub mod geometric;
pub mod io;
pub mod photometric;

pub use augmentation::RandomHorizontalFlip;
pub use conversion::ToTensor;
pub use geometric::{CenterCrop, EnsureRGB, RandomCrop, RandomResizedCrop, RandomRotation, Resize};
pub use io::{LoadImage, LoadImageToSample};
pub use photometric::{ColorJitter, Normalize};
