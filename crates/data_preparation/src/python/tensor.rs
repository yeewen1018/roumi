//! src/python/tensor.rs
//!
//! # Zero-Copy Tensor Bridge
//!
//! This module implements zero-copy tensor sharing between Rust and Python using the DLPack protocol.
//!
//! # Overview
//!
//! ```text
//!   Rust Side                    Bridge                     Python Side
//!   ┌─────────────┐             ┌─────────────┐            ┌─────────────┐
//!   │ tch::Tensor │────────────▶│ DLPack      │───────────▶│ PyTorch     │
//!   │ @0x1234     │             │ Protocol    │            │ Tensor      │
//!   │             │             │             │            │ @0x1234     │
//!   └─────────────┘             └─────────────┘            └─────────────┘
//!                                      │                           
//!                               ┌─────────────┐                   
//!                               │ SAME MEMORY │                   
//!                               │ NO COPYING  │                   
//!                               └─────────────┘                   
//! ```
//!
//! # Why This Architecture?
//!
//! **The Problem**: Direct conversion between Rust and Python tensors requires expensive copying.
//! **The Solution**: DLPack provides a standard way for ML frameworks to share tensor memory directly.
//! **Our Implementation**: PyTensorHandle manages the conversion while ensuring memory safety.
//!
//! DLPack is a specification that defines a C-struct describing tensors in memory that can be understood
//! across ML frameworks. So while it works with TensorFlow, JAX, and others, we focus on PyTorch integration.
//! Adapting to other frameworks would simply require calling their respective `from_dlpack()` functions.
//!
//! # How It Works
//!
//! ## 1. PyTensorHandle: Our Rust-to-DLPack Adapter
//!
//! ```rust
//! pub struct PyTensorHandle {
//!     inner: Arc<Tensor>,        // Keeps original tensor alive
//!     shape: Vec<i64>,          // Cached tensor dimensions  
//!     dtype: String,      // Cached type info
//!     device: String,     // Cached device info
//! }
//! ```
//!
//! **Purpose**: Converts Rust tensors into DLPack format while managing lifetimes.
//! **Metadata**: Shape and strides tell DLPack how to interpret the raw memory - we preserve
//! the original tensor's layout (typically row-major) to avoid expensive reorganization.
//!
//! ## 2. Python Capsules: Safe Transport Mechanism  
//!
//! PyTorch's `torch.from_dlpack()` API requires capsules - Python's standard way to safely
//! pass C pointers between modules. The capsule acts as a "sealed envelope" containing our
//! DLPack data with automatic cleanup callbacks.
//!
//! # Memory Safety: Reference Counting Lifecycle
//!
//! ```text
//! Creation → Handle Wrap → DLPack Context → Python Bridge → Rust Cleanup → Python Cleanup
//!
//! ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │ tch::Tensor │────▶│ PyTensorHandle  │────▶│ PyTorch Tensor  │────▶│       ❌        │
//! │ @0x1234     │     │ @0x1234         │     │ @0x1234         │     │   (freed)       │
//! └─────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
//!      │                       │                       │                       │
//!    Arc count=1           Arc count=3             Arc count=3            Arc count=0
//!    Rust owns             Handle + Context        Python owns            Memory freed                                                                       
//! ```
//!
//! **Key Safety Property**: Memory at `@0x1234` survives Rust cleanup and remains valid
//! for Python until all references are dropped, preventing use-after-free errors.
//!
//! # Integration with PyMiniBatch
//!
//! This module is an **internal implementation detail** used by `PyMiniBatch`:
//!
//! ```python
//! # What users see:
//! for batch in dataloader:
//!     images = batch['image']    # ← Zero-copy PyTorch tensor
//!
//! # What happens internally:
//! # PyMiniBatch calls PyTensorHandle.to_pytorch() behind the scenes
//! ```
//!
//! # Setup Requirements
//!
//! **Critical**: Set this environment variable:
//! ```bash
//! export LIBTORCH_USE_PYTORCH=1
//! ```
//! This ensures `tch` and PyTorch use compatible memory backends for zero-copy sharing.

use anyhow::{anyhow, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tch::{Device, Kind, Tensor};

// DLPack Foreign Function Interface (FFI) bindings
use dlpackrs::ffi::*;
use dlpackrs::ffi::{DLDataType, DLDevice, DLManagedTensor, DLTensor};
use std::ffi::c_void;

// ============================================================================
// Module Cache
// ============================================================================

/// Thread-safe cache for Python modules
static MODULE_CACHE: RwLock<Option<HashMap<String, PyObject>>> = RwLock::new(None);

/// Get cached PyTorch module, importing only once per process.
///
/// This is called on every tensor conversion from PyCapsule to PyTorch tensor.
/// Profiling shows import overhead dominates conversion time without caching,
/// while cached imports achieve near-zero overhead for repeated conversions.
pub fn get_torch_module(py: Python) -> PyResult<PyObject> {
    // Fast path: check cache first (read-only lock)
    {
        let cache = MODULE_CACHE.read().unwrap();
        if let Some(ref modules) = *cache {
            if let Some(torch) = modules.get("torch") {
                return Ok(torch.clone_ref(py));
            }
        }
    }

    // Slow path: import and cache (write lock)
    let torch: PyObject = py.import("torch")?.into();
    {
        let mut cache = MODULE_CACHE.write().unwrap();
        if cache.is_none() {
            *cache = Some(HashMap::new());
        }
        cache
            .as_mut()
            .unwrap()
            .insert("torch".to_string(), torch.clone_ref(py));
    }

    Ok(torch)
}

// ============================================================================
// DLPack Memory Management
// ============================================================================

/// Context for DLPack tensor cleanup
///
/// Holds an additional Arc reference to keep tensor memory alive during the
/// DLPack handoff to Python. Also stores shape/strides vectors that DLPack
/// will reference via raw pointers.
#[allow(dead_code)]
struct DLPackContext {
    tensor: Arc<Tensor>,
    shape: Vec<i64>,   // Referenced by DLTensor.shape pointer
    strides: Vec<i64>, // Referenced by DLTensor.strides pointer
}

/// DLPack cleanup callback - automatically called when PyTorch releases the tensor.
///
/// Safety:
/// - The `managed` pointer is guaranteed valid by DLPack contract.
/// - We check for null before dereferencing
/// - We reclaim ownership of Box types we originally created
extern "C" fn dlpack_tensor_deleter(managed: *mut DLManagedTensor) {
    unsafe {
        if managed.is_null() {
            return;
        }

        // Clean up context (decrement Arc reference count)
        let ctx_ptr = (*managed).manager_ctx as *mut DLPackContext;
        if !ctx_ptr.is_null() {
            let _context = Box::from_raw(ctx_ptr);
        }

        let dl_tensor = &(*managed).dl_tensor;

        // Clean up the shape and stride arrays we allocated
        if !dl_tensor.shape.is_null() {
            let shape_box = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                dl_tensor.shape,
                dl_tensor.ndim as usize,
            ));
            drop(shape_box);
        }

        if !dl_tensor.strides.is_null() {
            let strides_box = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                dl_tensor.strides,
                dl_tensor.ndim as usize,
            ));
            drop(strides_box);
        }

        // Clean up the managed tensor
        let _managed = Box::from_raw(managed);
    }
}

/// PyCapsule destructor.
///
/// PyTorch changes the capsule name from "dltensor" to "used_dltensor" after consumption,
/// so we check both names to ensure proper cleanup.
///
/// Safety:
/// - The `capsule` pointer is guaranteed valid by Python's capsule API contract.
/// - We check for null before dereferencing capsule contents.
/// - `PyCapsule_GetPointer` is a standard Python C API function.
/// - We reclaim ownership of our own `DLManagedTensor` that we originally created.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }

    let tensor_ptr = unsafe {
        let ptr = ffi::PyCapsule_GetPointer(capsule, b"dltensor\0".as_ptr() as *const i8);
        if !ptr.is_null() {
            ptr
        } else {
            ffi::PyCapsule_GetPointer(capsule, b"used_dltensor\0".as_ptr() as *const i8)
        }
    };

    if !tensor_ptr.is_null() {
        let managed_tensor = Box::from_raw(tensor_ptr as *mut DLManagedTensor);

        if let Some(deleter) = managed_tensor.deleter {
            let managed_ptr = Box::into_raw(managed_tensor);
            deleter(managed_ptr);
        } else {
            drop(managed_tensor);
        }
    }
}

// ============================================================================
// PyTensorHandle
// ============================================================================

/// Rust-to-DLPack adapter that manages zero-copy conversion via DLPackContext.
/// Used internally by PyMiniBatch for dict-like PyTorch tensor access.
#[pyclass(name = "TensorHandle", unsendable)]
#[derive(Clone)]
pub struct PyTensorHandle {
    inner: Arc<Tensor>,
    shape: Vec<i64>,
    dtype: String,
    device: String,
}

// ============================================================================
// PyTensorHandle public interface methods
// ============================================================================
#[pymethods]
impl PyTensorHandle {
    /// Get tensor shape  
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    /// Get tensor data type  
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Get tensor device
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        format!(
            "TensorHandle(shape={:?}, dtype={}, device={})",
            self.shape, self.dtype, self.device
        )
    }

    /// Convert to PyTorch tensor via zero-copy DLPack bridge.
    /// Called internally by PyMiniBatch when users access tensors.
    pub fn to_pytorch(&self, py: Python) -> PyResult<PyObject> {
        self.to_pytorch_impl(py)
    }
}

// ============================================================================
// PyTensorHandle constructor
// ============================================================================
impl PyTensorHandle {
    pub fn new(tensor: Tensor) -> Self {
        let shape = tensor.size();
        let dtype = Self::kind_to_static_str(tensor.kind()).to_string();
        let device = Self::device_to_string(tensor.device());

        Self {
            inner: Arc::new(tensor),
            shape,
            dtype,
            device,
        }
    }

    /// Get reference to underlying tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.inner
    }
}

// ============================================================================
// PyTensorHandle - DLPack Bridge implementation
// ============================================================================

impl PyTensorHandle {
    /// Convert to PyTorch tensor using zero-copy DLPack bridge.
    fn to_pytorch_impl(&self, py: Python) -> PyResult<PyObject> {
        let torch = get_torch_module(py)?;

        // 1. Create DLPack structure from Rust tch::Tensor
        let managed_tensor = self
            .create_dlpack_structure()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 2. Wrap DLPack in a Python capsule for safe transport
        let capsule = self.create_python_capsule(py, managed_tensor)?;

        // 3. Convert capsule to PyTorch tensor using torch.from_dlpack()
        let pytorch_tensor = torch.call_method1(py, "from_dlpack", (capsule,))?;
        Ok(pytorch_tensor.into())
    }

    /// Create DLPack structure from Rust tch::Tensor
    fn create_dlpack_structure(&self) -> Result<DLManagedTensor> {
        let tensor = &self.inner;

        // Ensure tensor is contiguous
        let tensor = if tensor.is_contiguous() {
            tensor.shallow_clone()
        } else {
            tensor.contiguous()
        };

        // Extract tensor metadata
        let data_ptr = tensor.data_ptr() as *mut c_void;
        let shape = tensor.size();
        let stride_elements = tensor.stride();
        let ndim = shape.len() as i32;

        let element_size = self.get_element_size(tensor.kind())?;

        let strides: Vec<i64> = stride_elements.iter().map(|&s| s * element_size).collect();

        // Convert Rust tensor properties to DLPack format
        let dl_device = self.convert_device_to_dlpack(tensor.device())?;
        let dl_dtype = self.convert_dtype_to_dlpack(tensor.kind())?;

        // Create context to keep everything alive
        let context = DLPackContext {
            tensor: Arc::new(tensor),
            shape: shape.clone(),
            strides: strides.clone(),
        };

        let shape_ptr = Box::leak(shape.into_boxed_slice()).as_mut_ptr();
        let strides_ptr = Box::leak(stride_elements.clone().into_boxed_slice()).as_mut_ptr();

        // Build the DLTensor structure
        let dl_tensor = DLTensor {
            data: data_ptr,
            device: dl_device,
            ndim,
            dtype: dl_dtype,
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        };

        let managed_tensor = DLManagedTensor {
            dl_tensor,
            manager_ctx: Box::into_raw(Box::new(context)) as *mut c_void,
            deleter: Some(dlpack_tensor_deleter),
        };

        Ok(managed_tensor)
    }

    /// Get element size in bytes for stride calculation
    fn get_element_size(&self, kind: Kind) -> Result<i64> {
        let size = match kind {
            Kind::Bool | Kind::Uint8 | Kind::Int8 => 1,
            Kind::Int16 | Kind::Half => 2,
            Kind::Int | Kind::Float => 4,
            Kind::Int64 | Kind::Double => 8,
            Kind::ComplexHalf => 4,
            Kind::ComplexFloat => 8,
            Kind::ComplexDouble => 16,
            _ => {
                return Err(anyhow!(
                    "Unsupported tensor type for size calculation: {:?}",
                    kind
                ))
            }
        };
        Ok(size)
    }

    /// Wrap DLPack structure in Python capsule for safe transport to PyTorch.
    ///
    /// Safety:
    /// - `PyCapsule_New` is a standard Python C API function with valid inputs.
    /// - We pass a valid destructor callback and proper null-terminated name.
    /// - `PyObject::from_borrowed_ptr` is safe with non-null capsule from successful creation.
    fn create_python_capsule(
        &self,
        py: Python,
        managed_tensor: DLManagedTensor,
    ) -> PyResult<PyObject> {
        let tensor_ptr = Box::into_raw(Box::new(managed_tensor)) as *mut c_void;

        unsafe {
            let capsule =
                pyo3::ffi::PyCapsule_New(tensor_ptr, b"dltensor\0".as_ptr() as *const i8, None);

            if capsule.is_null() {
                let _managed = Box::from_raw(tensor_ptr as *mut DLManagedTensor);
                return Err(PyRuntimeError::new_err(
                    "Failed to create DLPack capsule - this indicates a memory management issue. ",
                ));
            }

            Ok(PyObject::from_owned_ptr(py, capsule))
        }
    }
}

impl PyTensorHandle {
    /// Convert Rust device to DLPack device format.
    ///
    /// Maps Rust device types to DLPack's standardized device codes.
    fn convert_device_to_dlpack(&self, device: Device) -> Result<DLDevice> {
        match device {
            Device::Cpu => Ok(DLDevice {
                device_type: DLDeviceType_kDLCPU,
                device_id: 0,
            }),
            Device::Cuda(device_id) => Ok(DLDevice {
                device_type: DLDeviceType_kDLCUDA,
                device_id: device_id as i32,
            }),
            Device::Mps => Ok(DLDevice {
                device_type: DLDeviceType_kDLMetal,
                device_id: 0,
            }),
            device => Err(anyhow!(
                "Unsupported device for DLPack: {:?}. \
                Only CPU, CUDA, and MPS are currently supported.",
                device
            )),
        }
    }

    /// Convert Rust Kind to DLPack data type format.
    ///
    /// Each type specifies: code (int/float/bool), bits (precision), lanes (vector width).
    fn convert_dtype_to_dlpack(&self, kind: Kind) -> Result<DLDataType> {
        let (code, bits, lanes) = match kind {
            Kind::Bool => (DLDataTypeCode_kDLUInt, 8, 1),
            Kind::Uint8 => (DLDataTypeCode_kDLUInt, 8, 1),
            Kind::Int8 => (DLDataTypeCode_kDLInt, 8, 1),
            Kind::Int16 => (DLDataTypeCode_kDLInt, 16, 1),
            Kind::Int => (DLDataTypeCode_kDLInt, 32, 1),
            Kind::Int64 => (DLDataTypeCode_kDLInt, 64, 1),
            Kind::Half => (DLDataTypeCode_kDLFloat, 16, 1),
            Kind::Float => (DLDataTypeCode_kDLFloat, 32, 1),
            Kind::Double => (DLDataTypeCode_kDLFloat, 64, 1),
            Kind::ComplexHalf => (DLDataTypeCode_kDLComplex, 32, 1),
            Kind::ComplexFloat => (DLDataTypeCode_kDLComplex, 64, 1),
            Kind::ComplexDouble => (DLDataTypeCode_kDLComplex, 128, 1),
            _ => return Err(anyhow!("Unsupported tensor type: {:?}", kind)),
        };

        Ok(DLDataType {
            code: code as u8,
            bits: bits as u8,
            lanes: lanes as u16,
        })
    }

    /// Convert Device to String
    fn device_to_string(device: Device) -> String {
        match device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{}", id),
            Device::Mps => "mps".to_string(),
            Device::Vulkan => "vulkan".to_string(),
        }
    }

    /// Convert Kind to static string (optimized for performance)
    fn kind_to_static_str(kind: Kind) -> &'static str {
        match kind {
            // Most common types first for branch prediction optimization
            Kind::Bool => "Bool",
            Kind::Uint8 => "Uint8",
            Kind::Int8 => "Int8",
            Kind::Int16 => "Int16",
            Kind::Int => "Int",
            Kind::Int64 => "Int64",
            Kind::Half => "Half",
            Kind::Float => "Float",
            Kind::Double => "Double",
            Kind::ComplexHalf => "ComplexHalf",
            Kind::ComplexFloat => "ComplexFloat",
            Kind::ComplexDouble => "ComplexDouble",
            _ => "Unknown",
        }
    }
}
