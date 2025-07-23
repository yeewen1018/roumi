use crate::transforms::Transform;
use anyhow::{ensure, Context, Result};
use tch::Tensor;

// ============================================================================
// Normalize
// ============================================================================

/// Normalizes tensors using channel-wise statistics.
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
}

impl Transform<Tensor, Tensor> for Normalize {
    fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        let (num_channels, _height, _width) = tensor
            .size3()
            .context("Input must be 3D tensor [C, H, W]")?;

        ensure!(
            num_channels as usize == self.mean.len(),
            "Channel count mismatch: input has {} channels but normalization expects {} ",
            num_channels,
            self.mean.len()
        );

        let mean_t = Tensor::from_slice(&self.mean)
            .reshape(&[num_channels, 1, 1])
            .to_kind(tensor.kind());

        let std_t = Tensor::from_slice(&self.std)
            .reshape(&[num_channels, 1, 1])
            .to_kind(tensor.kind());

        Ok((tensor - mean_t) / std_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

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
}
