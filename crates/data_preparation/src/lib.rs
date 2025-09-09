pub mod collator;
pub mod dataloader;
pub mod dataset;
pub mod minibatch;
pub mod readers;
pub mod sample;
pub mod sampler;
pub mod transforms;

pub use collator::StackCollator;
pub use dataloader::{DataLoader, DataLoaderConfig};
pub use dataset::Dataset;
pub use dataset::InMemoryDataset;
pub use minibatch::MiniBatch;
pub use sample::Sample;
pub use sampler::Sampler;
pub use transforms::Transform;


#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn roumi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add core classes to root module
    m.add_class::<python::tensor::PyTensorHandle>()?;
    m.add_class::<python::minibatch::PyMiniBatch>()?;
    python::create_submodules(m)?;
    Ok(())
}
