use pyo3::prelude::*;

pub mod minibatch;
pub mod tensor;


pub fn create_submodules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils_module = PyModule::new(m.py(), "utils")?;
    utils_module.add_function(wrap_pyfunction!(minibatch::create_test_batch, &utils_module)?)?;
    
    m.add_submodule(&utils_module)?;
    
    // Register with sys.modules for proper import support
    m.py().import("sys")?.getattr("modules")?.set_item("roumi.utils", &utils_module)?;
    
    Ok(())
}