//! Python bindings using PyO3

use half::f16;
use ndarray::Array3;
use numpy::{
    IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    affine_transform_3d_f16, affine_transform_3d_f16_into, affine_transform_3d_f32,
    affine_transform_3d_f32_into, affine_transform_3d_u8, affine_transform_3d_u8_into,
    apply_warp_3d_f16, apply_warp_3d_f32, apply_warp_3d_u8, set_scalar_fallback_allowed,
    upsample_warp_field_2x,
};

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
///         Ignored if output is provided.
///     cval: Constant value for out-of-bounds (default: 0.0)
///     order: Interpolation order (only 1 is supported)
///     output: Optional pre-allocated output array (same dtype as input).
///         If provided, the result will be written directly into this array,
///         avoiding memory allocation. The shape of this array determines
///         the output shape (output_shape parameter is ignored).
///
/// The matrix format is:
///     [[m00, m01, m02, tz],
///      [m10, m11, m12, ty],
///      [m20, m21, m22, tx],
///      [0,   0,   0,   1 ]]
///
/// Returns:
///     Transformed 3D array (same dtype as input). If output was provided,
///     returns the same array.
#[pyfunction]
#[pyo3(signature = (input, matrix, output_shape=None, cval=0.0, order=1, output=None))]
fn affine_transform<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    matrix: &Bound<'py, PyAny>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
    order: i32,
    output: Option<&Bound<'py, PyAny>>,
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

    // Validate output dtype matches input if provided
    if let Some(out) = &output {
        let out_dtype = get_dtype_name(out)?;
        if out_dtype != dtype_name {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Output dtype '{}' does not match input dtype '{}'",
                out_dtype, dtype_name
            )));
        }
    }

    match dtype_name.as_str() {
        "float32" => {
            let input: PyReadonlyArray3<'py, f32> = input.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract float32 array: {}",
                    e
                ))
            })?;
            let input_array = input.as_array();
            let matrix_array = matrix.as_array();

            if let Some(out) = output {
                // Use pre-allocated output
                let out_array: &Bound<'py, PyArray3<f32>> = out.downcast().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Output must be a contiguous float32 numpy array",
                    )
                })?;
                let mut out_rw: PyReadwriteArray3<'py, f32> = out_array.readwrite();
                let mut out_view = out_rw.as_array_mut();
                affine_transform_3d_f32_into(&input_array, &matrix_array, &mut out_view, cval);
                drop(out_rw);
                Ok(out.clone().unbind())
            } else {
                let result =
                    affine_transform_3d_f32(&input_array, &matrix_array, output_shape, cval);
                Ok(result.into_pyarray(py).into_any().unbind())
            }
        }
        "float16" => {
            let input: PyReadonlyArray3<'py, u16> = input
                .getattr("view")?
                .call1((py.import("numpy")?.getattr("uint16")?,))?
                .extract()
                .map_err(|e| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "Failed to view float16 as uint16: {}",
                        e
                    ))
                })?;

            let input_shape = input.shape();
            let (d, h, w) = (input_shape[0], input_shape[1], input_shape[2]);
            let input_slice = input.as_slice()?;

            // Reinterpret u16 as f16
            let input_f16: &[f16] = unsafe {
                std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
            };

            let input_array =
                ndarray::ArrayView3::from_shape((d, h, w), input_f16).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e))
                })?;

            let matrix_array = matrix.as_array();

            if let Some(out) = output {
                // View as uint16 to get mutable access, then reinterpret as f16
                let out_view_any = out.getattr("view")?;
                let out_u16_any =
                    out_view_any.call1((py.import("numpy")?.getattr("uint16")?,))?;
                let out_u16: &Bound<'py, PyArray3<u16>> = out_u16_any.downcast().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Output must be a contiguous float16 numpy array",
                    )
                })?;
                let mut out_rw: PyReadwriteArray3<'py, u16> = out_u16.readwrite();

                let out_shape = out_rw.shape();
                let (od, oh, ow) = (out_shape[0], out_shape[1], out_shape[2]);
                let out_slice = out_rw.as_slice_mut()?;

                // Reinterpret u16 as f16
                let out_f16: &mut [f16] = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_slice.as_mut_ptr() as *mut f16,
                        out_slice.len(),
                    )
                };

                let mut out_view =
                    ndarray::ArrayViewMut3::from_shape((od, oh, ow), out_f16).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e))
                    })?;

                affine_transform_3d_f16_into(&input_array, &matrix_array, &mut out_view, cval);
                drop(out_rw);
                Ok(out.clone().unbind())
            } else {
                let result =
                    affine_transform_3d_f16(&input_array, &matrix_array, output_shape, cval);

                // Reinterpret f16 output as u16 for numpy, then view as float16
                let out_shape = result.dim();
                let output_u16: Array3<u16> = unsafe {
                    let ptr = result.as_ptr() as *const u16;
                    let slice = std::slice::from_raw_parts(ptr, result.len());
                    Array3::from_shape_vec(out_shape, slice.to_vec()).unwrap()
                };

                let numpy_result = output_u16.into_pyarray(py);
                let float16_result = numpy_result
                    .call_method1("view", (py.import("numpy")?.getattr("float16")?,))?;
                Ok(float16_result.unbind())
            }
        }
        "uint8" => {
            let input: PyReadonlyArray3<'py, u8> = input.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract uint8 array: {}",
                    e
                ))
            })?;
            let input_array = input.as_array();
            let matrix_array = matrix.as_array();
            let cval_u8 = cval.round().clamp(0.0, 255.0) as u8;

            if let Some(out) = output {
                // Use pre-allocated output
                let out_array: &Bound<'py, PyArray3<u8>> = out.downcast().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Output must be a contiguous uint8 numpy array",
                    )
                })?;
                let mut out_rw: PyReadwriteArray3<'py, u8> = out_array.readwrite();
                let mut out_view = out_rw.as_array_mut();
                affine_transform_3d_u8_into(&input_array, &matrix_array, &mut out_view, cval_u8);
                drop(out_rw);
                Ok(out.clone().unbind())
            } else {
                let result =
                    affine_transform_3d_u8(&input_array, &matrix_array, output_shape, cval_u8);
                Ok(result.into_pyarray(py).into_any().unbind())
            }
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
// Warp field functions
// =============================================================================

use numpy::PyReadonlyArray4;

/// Apply a 3D warp field to an image using trilinear interpolation
///
/// This is a CPU implementation equivalent to dexpv2's CUDA `apply_warp` function.
/// Automatically dispatches to the appropriate implementation based on input dtype.
///
/// Args:
///     image: 3D numpy array (float32, float16, or uint8) - the image to be warped
///     warp_field: 4D numpy array (float32) with shape (channels, d, h, w)
///         - Channel 0: Z displacement (dz)
///         - Channel 1: Y displacement (dy)
///         - Channel 2: X displacement (dx)
///         - Channel 3+ (optional): Score or other data (ignored)
///     cval: Constant value for out-of-bounds coordinates (default: 0.0)
///     upsample: Whether to upsample the warp field by 2x before warping (default: True)
///               This matches dexpv2's default behavior.
///
/// Returns:
///     Warped 3D array (same dtype as input) with same shape as input image
///
/// Notes:
///     The warping uses the convention: src_coord = out_coord - displacement
///     This matches dexpv2's CUDA implementation.
///
/// Example:
///     >>> import numpy as np
///     >>> from affiners import apply_warp
///     >>> image = np.random.rand(100, 100, 100).astype(np.float32)
///     >>> warp_field = np.zeros((4, 10, 10, 10), dtype=np.float32)  # identity
///     >>> warped = apply_warp(image, warp_field)
#[pyfunction]
#[pyo3(signature = (image, warp_field, cval=0.0, upsample=true))]
fn apply_warp<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    warp_field: PyReadonlyArray4<'py, f32>,
    cval: f64,
    upsample: bool,
) -> PyResult<Py<PyAny>> {
    let wf_input = warp_field.as_array();

    if wf_input.shape()[0] < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Warp field must have at least 3 channels (dz, dy, dx), got {}",
            wf_input.shape()[0]
        )));
    }

    // Get input dtype and dispatch
    let dtype_name = get_dtype_name(image)?;

    match dtype_name.as_str() {
        "float32" => {
            let image_arr: PyReadonlyArray3<'py, f32> = image.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract float32 array: {}",
                    e
                ))
            })?;
            let image_array = image_arr.as_array();
            let cval_f32 = cval as f32;

            let output = if upsample {
                let upsampled = upsample_warp_field_2x(&wf_input);
                apply_warp_3d_f32(&image_array, &upsampled.view(), cval_f32)
            } else {
                apply_warp_3d_f32(&image_array, &wf_input, cval_f32)
            };

            Ok(output.into_pyarray(py).into_any().unbind())
        }
        "float16" => {
            let image_arr: PyReadonlyArray3<'py, u16> = image
                .getattr("view")?
                .call1((py.import("numpy")?.getattr("uint16")?,))?
                .extract()
                .map_err(|e| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "Failed to view float16 as uint16: {}",
                        e
                    ))
                })?;

            let input_shape = image_arr.shape();
            let (d, h, w) = (input_shape[0], input_shape[1], input_shape[2]);
            let input_slice = image_arr.as_slice()?;

            // Reinterpret u16 as f16
            let input_f16: &[f16] = unsafe {
                std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
            };

            let image_array =
                ndarray::ArrayView3::from_shape((d, h, w), input_f16).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e))
                })?;

            let cval_f16 = f16::from_f64(cval);

            let output = if upsample {
                let upsampled = upsample_warp_field_2x(&wf_input);
                apply_warp_3d_f16(&image_array, &upsampled.view(), cval_f16)
            } else {
                apply_warp_3d_f16(&image_array, &wf_input, cval_f16)
            };

            // Reinterpret f16 output as u16 for numpy, then view as float16
            let out_shape = output.dim();
            let output_u16: Array3<u16> = unsafe {
                let ptr = output.as_ptr() as *const u16;
                let slice = std::slice::from_raw_parts(ptr, output.len());
                Array3::from_shape_vec(out_shape, slice.to_vec()).unwrap()
            };

            let result = output_u16.into_pyarray(py);
            let float16_result =
                result.call_method1("view", (py.import("numpy")?.getattr("float16")?,))?;
            Ok(float16_result.unbind())
        }
        "uint8" => {
            let image_arr: PyReadonlyArray3<'py, u8> = image.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to extract uint8 array: {}",
                    e
                ))
            })?;
            let image_array = image_arr.as_array();
            let cval_u8 = cval.round().clamp(0.0, 255.0) as u8;

            let output = if upsample {
                let upsampled = upsample_warp_field_2x(&wf_input);
                apply_warp_3d_u8(&image_array, &upsampled.view(), cval_u8)
            } else {
                apply_warp_3d_u8(&image_array, &wf_input, cval_u8)
            };

            Ok(output.into_pyarray(py).into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype '{}'. Supported dtypes: float32, float16, uint8",
            dtype_name
        ))),
    }
}

// =============================================================================
// Scalar Fallback Control
// =============================================================================

/// Set whether scalar fallback is allowed (internal use)
///
/// When set to False, any attempt to use scalar fallback will raise a RuntimeError.
/// This is useful for ensuring SIMD code paths are being used.
///
/// Use the `disable_scalar_fallback()` context manager instead of calling this directly.
#[pyfunction]
fn _set_scalar_fallback_allowed(allowed: bool) {
    set_scalar_fallback_allowed(allowed);
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

    // Warp field functions
    m.add_function(wrap_pyfunction!(apply_warp, m)?)?;

    // Utilities
    m.add_function(wrap_pyfunction!(build_info, m)?)?;

    // Internal functions for scalar fallback control
    m.add_function(wrap_pyfunction!(_set_scalar_fallback_allowed, m)?)?;
    Ok(())
}
