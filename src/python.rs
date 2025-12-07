//! Python bindings using PyO3

use half::f16;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{affine_transform_3d_f16, affine_transform_3d_f32, affine_transform_3d_u8};

// =============================================================================
// Helper functions
// =============================================================================

/// Convert any numpy array to a float64 array
fn to_float64_array<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    let numpy = py.import("numpy")?;
    let float64_dtype = numpy.getattr("float64")?;
    
    // Use np.asarray with dtype=float64 for efficient conversion
    // This is a no-op if already float64 and contiguous
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", float64_dtype)?;
    kwargs.set_item("order", "C")?;
    
    let converted = numpy.call_method("asarray", (array,), Some(&kwargs))?;
    converted.extract().map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "Failed to convert matrix to float64 array: {}",
            e
        ))
    })
}

/// Get the dtype name from a numpy array
fn get_dtype_name(array: &Bound<'_, PyAny>) -> PyResult<String> {
    let dtype = array.getattr("dtype")?;
    let name = dtype.getattr("name")?;
    name.extract()
}

// =============================================================================
// Build Info
// =============================================================================

/// Get build and runtime information
///
/// Returns a dictionary with:
/// - version: Package version
/// - simd: Dict of available SIMD features (avx2, avx512, fma, f16c)
/// - parallel: Whether parallel processing is enabled
/// - backend_f32: Which backend will be used for f32
/// - backend_f16: Which backend will be used for f16
/// - backend_u8: Which backend will be used for u8
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);

    // Version
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;

    // SIMD features (runtime detection)
    let simd = PyDict::new(py);

    #[cfg(target_arch = "x86_64")]
    {
        simd.set_item("avx2", is_x86_feature_detected!("avx2"))?;
        simd.set_item("avx512f", is_x86_feature_detected!("avx512f"))?;
        simd.set_item("fma", is_x86_feature_detected!("fma"))?;
        simd.set_item("f16c", is_x86_feature_detected!("f16c"))?;
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        simd.set_item("avx2", false)?;
        simd.set_item("avx512f", false)?;
        simd.set_item("fma", false)?;
        simd.set_item("f16c", false)?;
    }

    info.set_item("simd", simd)?;

    // Parallel feature
    #[cfg(feature = "parallel")]
    info.set_item("parallel", true)?;
    #[cfg(not(feature = "parallel"))]
    info.set_item("parallel", false)?;

    // Determine which backend will be used
    #[cfg(target_arch = "x86_64")]
    {
        // f32 backend
        let backend_f32 = if is_x86_feature_detected!("avx512f") {
            "avx512"
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            "avx2"
        } else {
            "scalar"
        };
        info.set_item("backend_f32", backend_f32)?;

        // f16 backend
        let backend_f16 = if is_x86_feature_detected!("avx512f") {
            "avx512"
        } else if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            "avx2"
        } else {
            "scalar"
        };
        info.set_item("backend_f16", backend_f16)?;

        // u8 backend
        let backend_u8 = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            "avx2"
        } else {
            "scalar"
        };
        info.set_item("backend_u8", backend_u8)?;
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        info.set_item("backend_f32", "scalar")?;
        info.set_item("backend_f16", "scalar")?;
        info.set_item("backend_u8", "scalar")?;
    }

    // Number of threads (if parallel)
    #[cfg(feature = "parallel")]
    {
        info.set_item("num_threads", rayon::current_num_threads())?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        info.set_item("num_threads", 1)?;
    }

    Ok(info)
}

// =============================================================================
// Unified API - Auto-dispatches based on dtype
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation
///
/// Automatically dispatches to the appropriate implementation based on input dtype:
/// - float32: Standard floating point (~1.5 Gvoxels/s)
/// - float16: Half precision, 2x less memory
/// - uint8: 2.2x faster (~3.3 Gvoxels/s), 4x less memory
///
/// Args:
///     input: 3D numpy array (float32, float16, or uint8)
///     matrix: 4x4 homogeneous transformation matrix (any numeric dtype)
///     output_shape: Optional output shape (z, y, x). If None, uses input shape.
///     cval: Constant value for out-of-bounds (default: 0.0)
///     order: Interpolation order (only 1 is supported)
///
/// The matrix format is:
///     [[m00, m01, m02, tz],
///      [m10, m11, m12, ty],
///      [m20, m21, m22, tx],
///      [0,   0,   0,   1 ]]
///
/// Returns:
///     Transformed 3D array (same dtype as input)
#[pyfunction]
#[pyo3(signature = (input, matrix, output_shape=None, cval=0.0, order=1))]
fn affine_transform<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    matrix: &Bound<'py, PyAny>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
    order: i32,
) -> PyResult<Py<PyAny>> {
    if order != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only order=1 (trilinear interpolation) is supported",
        ));
    }

    // Convert matrix to float64 (accepts any numeric dtype)
    let matrix = to_float64_array(py, matrix)?;

    let mat_shape = matrix.shape();
    if mat_shape != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 4x4 homogeneous transformation matrix",
        ));
    }

    // Get input dtype and dispatch
    let dtype_name = get_dtype_name(input)?;
    
    match dtype_name.as_str() {
        "float32" => {
            let input: PyReadonlyArray3<'py, f32> = input.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract float32 array: {}", e
                ))
            })?;
            let input_array = input.as_array();
            let matrix_array = matrix.as_array();
            let output = affine_transform_3d_f32(&input_array, &matrix_array, output_shape, cval);
            Ok(output.into_pyarray(py).into_any().unbind())
        }
        "float16" => {
            let input: PyReadonlyArray3<'py, u16> = input.getattr("view")?
                .call1((py.import("numpy")?.getattr("uint16")?,))?
                .extract()
                .map_err(|e| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "Failed to view float16 as uint16: {}", e
                    ))
                })?;
            
            let input_shape = input.shape();
            let (d, h, w) = (input_shape[0], input_shape[1], input_shape[2]);
            let input_slice = input.as_slice()?;

            // Reinterpret u16 as f16
            let input_f16: &[f16] = unsafe {
                std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
            };

            let input_array = ndarray::ArrayView3::from_shape((d, h, w), input_f16)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

            let matrix_array = matrix.as_array();
            let output = affine_transform_3d_f16(&input_array, &matrix_array, output_shape, cval);

            // Reinterpret f16 output as u16 for numpy, then view as float16
            let out_shape = output.dim();
            let output_u16: Array3<u16> = unsafe {
                let ptr = output.as_ptr() as *const u16;
                let slice = std::slice::from_raw_parts(ptr, output.len());
                Array3::from_shape_vec(out_shape, slice.to_vec()).unwrap()
            };

            let result = output_u16.into_pyarray(py);
            let float16_result = result.call_method1("view", (py.import("numpy")?.getattr("float16")?,))?;
            Ok(float16_result.unbind())
        }
        "uint8" => {
            let input: PyReadonlyArray3<'py, u8> = input.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract uint8 array: {}", e
                ))
            })?;
            let input_array = input.as_array();
            let matrix_array = matrix.as_array();
            let cval_u8 = cval.round().clamp(0.0, 255.0) as u8;
            let output = affine_transform_3d_u8(&input_array, &matrix_array, output_shape, cval_u8);
            Ok(output.into_pyarray(py).into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype '{}'. Supported dtypes: float32, float16, uint8",
            dtype_name
        ))),
    }
}

// =============================================================================
// Type-specific functions (for explicit control)
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation (f32)
///
/// Use affine_transform() for automatic dtype dispatch.
#[pyfunction]
#[pyo3(signature = (input, matrix, output_shape=None, cval=0.0, order=1))]
fn affine_transform_f32<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, f32>,
    matrix: &Bound<'py, PyAny>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
    order: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if order != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only order=1 (trilinear interpolation) is supported",
        ));
    }

    let matrix = to_float64_array(py, matrix)?;

    let shape = matrix.shape();
    if shape != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 4x4 homogeneous transformation matrix",
        ));
    }

    let input_array = input.as_array();
    let matrix_array = matrix.as_array();
    let output = affine_transform_3d_f32(&input_array, &matrix_array, output_shape, cval);

    Ok(output.into_pyarray(py))
}

/// Apply 3D affine transformation for f16 arrays
///
/// Use affine_transform() for automatic dtype dispatch.
/// Note: Input must be viewed as uint16 (numpy float16 bit representation).
#[pyfunction]
#[pyo3(signature = (input, matrix, output_shape=None, cval=0.0, order=1))]
fn affine_transform_f16<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u16>, // numpy float16 stored as u16
    matrix: &Bound<'py, PyAny>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
    order: i32,
) -> PyResult<Bound<'py, PyArray3<u16>>> {
    if order != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only order=1 (trilinear interpolation) is supported",
        ));
    }

    let matrix = to_float64_array(py, matrix)?;

    let mat_shape = matrix.shape();
    if mat_shape != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 4x4 homogeneous transformation matrix",
        ));
    }

    let input_shape = input.shape();
    let (d, h, w) = (input_shape[0], input_shape[1], input_shape[2]);
    let input_slice = input.as_slice()?;

    // Reinterpret u16 as f16
    let input_f16: &[f16] = unsafe {
        std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
    };

    let input_array = ndarray::ArrayView3::from_shape((d, h, w), input_f16)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    let matrix_array = matrix.as_array();
    let output = affine_transform_3d_f16(&input_array, &matrix_array, output_shape, cval);

    // Reinterpret f16 output as u16 for numpy
    let out_shape = output.dim();
    let output_u16: Array3<u16> = unsafe {
        let ptr = output.as_ptr() as *const u16;
        let slice = std::slice::from_raw_parts(ptr, output.len());
        Array3::from_shape_vec(out_shape, slice.to_vec()).unwrap()
    };

    Ok(output_u16.into_pyarray(py))
}

/// Apply 3D affine transformation for u8 arrays
///
/// Use affine_transform() for automatic dtype dispatch.
#[pyfunction]
#[pyo3(signature = (input, matrix, output_shape=None, cval=0))]
fn affine_transform_u8<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u8>,
    matrix: &Bound<'py, PyAny>,
    output_shape: Option<(usize, usize, usize)>,
    cval: u8,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let matrix = to_float64_array(py, matrix)?;

    let shape = matrix.shape();
    if shape != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 4x4 homogeneous transformation matrix",
        ));
    }

    let input_array = input.as_array();
    let matrix_array = matrix.as_array();
    let output = affine_transform_3d_u8(&input_array, &matrix_array, output_shape, cval);

    Ok(output.into_pyarray(py))
}

// =============================================================================
// Module registration
// =============================================================================

/// Fast 3D affine transformations with trilinear interpolation using AVX2/AVX512 SIMD
///
/// Main function:
/// - affine_transform(): Auto-dispatches based on input dtype (float32, float16, uint8)
///
/// Type-specific functions (for explicit control):
/// - affine_transform_f32(): Standard floating point
/// - affine_transform_f16(): Half precision (input as uint16 view)
/// - affine_transform_u8(): Unsigned 8-bit integer
///
/// All functions accept a 4x4 homogeneous transformation matrix:
///     [[m00, m01, m02, tz],
///      [m10, m11, m12, ty],
///      [m20, m21, m22, tx],
///      [0,   0,   0,   1 ]]
#[pymodule]
fn affiners(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main unified API
    m.add_function(wrap_pyfunction!(affine_transform, m)?)?;
    
    // Type-specific functions for explicit control
    m.add_function(wrap_pyfunction!(affine_transform_f32, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_f16, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_u8, m)?)?;
    
    // Utilities
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
