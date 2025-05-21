use crate::{sample::Sample, transforms::Transform};
use anyhow::{Context, Result};
use image::DynamicImage;

/// Raw input container for vision + text multimodal data
#[derive(Clone)]
pub struct RawMultimodalData {
    pub image: DynamicImage,
    pub text: String,
}

/// Multimodal pipeline
///
/// Runs a vision transform pipeline on `raw.image` and a text
/// transform pipeline on `raw.text`, then merges their Samples
/// into one.
///
/// # Example
/// ```ignore
/// let multimodal = MultimodalPipeline::new(vision_pipeline, text_pipeline);
/// let sample = multimodal.apply(RawMultimodalData { image, text })?;
/// ```
#[derive(Debug, Clone)]
pub struct MultimodalPipeline<VisionTransform, TextTransform> {
    vision_pipeline: VisionTransform,
    text_pipeline: TextTransform,
}

impl<V, T> MultimodalPipeline<V, T> {
    /// Combine a vision and a text pipeline into one multimodal transform.
    pub fn new(vision_pipeline: V, text_pipeline: T) -> Self {
        Self {
            vision_pipeline,
            text_pipeline,
        }
    }
}

impl<V, T> Transform<RawMultimodalData, Sample> for MultimodalPipeline<V, T>
where
    V: Transform<DynamicImage, Sample>,
    T: Transform<String, Sample>,
{
    fn apply(&self, raw: RawMultimodalData) -> Result<Sample> {
        let vision = self
            .vision_pipeline
            .apply(raw.image)
            .context("Vision processing failed")?;
        let text = self
            .text_pipeline
            .apply(raw.text)
            .context("Text processing failed")?;
        vision.merge(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{Dataset, InMemoryDataset},
        transforms::{text::Tokenize, vision::*, ToSample},
    };
    use anyhow::Result;
    use image::{imageops::FilterType, RgbImage};
    use tch::Kind;
    use tokenizers::Tokenizer;

    // Test RGB image (2x2 pixels)
    fn test_image() -> image::DynamicImage {
        let buf = vec![
            10u8, 20, 30, 40, 50, 60, // Pixel 1, Pixel 2
            70u8, 80, 90, 100, 110, 120, // Pixel 3, Pixel 4
        ];
        image::DynamicImage::ImageRgb8(RgbImage::from_raw(2, 2, buf).unwrap())
    }

    #[test]
    fn test_multimodal_pipeline() -> Result<()> {
        let vision_pipelineline = Resize::new(2, 2, FilterType::Nearest)?
            .then(ToTensor)
            .then(Normalize::imagenet())
            .then(ToSample::new("pixels"));

        let tokenizer = Tokenizer::from_file("src/tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let text_pipelineline = Tokenize::new(tokenizer);

        // Create multimodal pipeline
        let multimodal = MultimodalPipeline::new(vision_pipelineline, text_pipelineline);

        // Create test dataset
        let dataset = InMemoryDataset::new(vec![
            RawMultimodalData {
                image: test_image(),
                text: "red square".into(),
            },
            RawMultimodalData {
                image: test_image(),
                text: "blue circle".into(),
            },
        ])
        .with_transform(multimodal);

        // Verify processing
        for sample in dataset.iter() {
            let sample = sample?;

            // Check vision output
            let pixels = sample.get("pixels")?;
            assert_eq!(pixels.size(), &[3, 2, 2]); // CHW format
            assert_eq!(pixels.kind(), Kind::Float);

            // Check text output
            let input_ids = sample.get("input_ids")?;
            assert!(input_ids.size()[0] > 0); // At least 1 token
            assert_eq!(input_ids.kind(), Kind::Int64);
        }

        Ok(())
    }
}
