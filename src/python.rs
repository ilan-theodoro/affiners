//! Python bindings using PyO3

use half::f16;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    AffineMatrix3D, affine_transform_3d_f16, affine_transform_3d_f32, affine_transform_3d_u8,
};

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
// F32 - Main API
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation (f32)
///
/// Args:
///     input: 3D numpy array (f32)
///     matrix: 3x3 transformation matrix
///     offset: Translation vector [z, y, x]
///     cval: Constant value for out-of-bounds (default: 0.0)
///     order: Interpolation order (only 1 is supported)
///
/// Returns:
///     Transformed 3D array
#[pyfunction]
#[pyo3(signature = (input, matrix, offset=None, cval=0.0, order=1))]
fn affine_transform<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, f32>,
    matrix: PyReadonlyArray2<'py, f64>,
    offset: Option<Vec<f64>>,
    cval: f64,
    order: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if order != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only order=1 (trilinear interpolation) is supported",
        ));
    }

    let matrix_slice = matrix.as_slice()?;
    if matrix_slice.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 3x3",
        ));
    }

    let affine_matrix = AffineMatrix3D::new([
        [matrix_slice[0], matrix_slice[1], matrix_slice[2]],
        [matrix_slice[3], matrix_slice[4], matrix_slice[5]],
        [matrix_slice[6], matrix_slice[7], matrix_slice[8]],
    ]);

    let shift = match offset {
        Some(v) if v.len() == 3 => [v[0], v[1], v[2]],
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Offset must have 3 elements",
            ));
        }
        None => [0.0, 0.0, 0.0],
    };

    let input_array = input.as_array();
    let output = affine_transform_3d_f32(&input_array, &affine_matrix, &shift, cval);

    Ok(output.into_pyarray(py))
}

// =============================================================================
// F16 - Half precision
// =============================================================================

/// Apply 3D affine transformation for f16 arrays
///
/// Note: numpy float16 is stored as u16 bits, same as half::f16
#[pyfunction]
#[pyo3(signature = (input, matrix, offset=None, cval=0.0, order=1))]
fn affine_transform_f16<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u16>, // numpy float16 stored as u16
    matrix: PyReadonlyArray2<'py, f64>,
    offset: Option<Vec<f64>>,
    cval: f64,
    order: i32,
) -> PyResult<Bound<'py, PyArray3<u16>>> {
    if order != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only order=1 (trilinear interpolation) is supported",
        ));
    }

    let matrix_slice = matrix.as_slice()?;
    if matrix_slice.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 3x3",
        ));
    }

    let affine_matrix = AffineMatrix3D::new([
        [matrix_slice[0], matrix_slice[1], matrix_slice[2]],
        [matrix_slice[3], matrix_slice[4], matrix_slice[5]],
        [matrix_slice[6], matrix_slice[7], matrix_slice[8]],
    ]);

    let shift = match offset {
        Some(v) if v.len() == 3 => [v[0], v[1], v[2]],
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Offset must have 3 elements",
            ));
        }
        None => [0.0, 0.0, 0.0],
    };

    let shape = input.shape();
    let (d, h, w) = (shape[0], shape[1], shape[2]);
    let input_slice = input.as_slice()?;

    // Reinterpret u16 as f16
    let input_f16: &[f16] = unsafe {
        std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
    };

    let input_array = ndarray::ArrayView3::from_shape((d, h, w), input_f16)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    let output = affine_transform_3d_f16(&input_array, &affine_matrix, &shift, cval);

    // Reinterpret f16 output as u16 for numpy
    let output_u16: Array3<u16> = unsafe {
        let ptr = output.as_ptr() as *const u16;
        let slice = std::slice::from_raw_parts(ptr, output.len());
        Array3::from_shape_vec((d, h, w), slice.to_vec()).unwrap()
    };

    Ok(output_u16.into_pyarray(py))
}

// =============================================================================
// U8 - 2.2x faster, 4x less memory
// =============================================================================

/// Apply 3D affine transformation for u8 arrays
///
/// 2.2x faster than f32 due to 4x less memory traffic.
/// Ideal for image data (microscopy, CT, MRI).
///
/// Args:
///     input: 3D numpy array (uint8)
///     matrix: 3x3 transformation matrix
///     offset: Translation vector [z, y, x]
///     cval: Constant value for out-of-bounds (default: 0)
///
/// Returns:
///     Transformed 3D array (uint8)
#[pyfunction]
#[pyo3(signature = (input, matrix, offset=None, cval=0))]
fn affine_transform_u8<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u8>,
    matrix: PyReadonlyArray2<'py, f64>,
    offset: Option<Vec<f64>>,
    cval: u8,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let matrix_slice = matrix.as_slice()?;
    if matrix_slice.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Matrix must be 3x3",
        ));
    }

    let affine_matrix = AffineMatrix3D::new([
        [matrix_slice[0], matrix_slice[1], matrix_slice[2]],
        [matrix_slice[3], matrix_slice[4], matrix_slice[5]],
        [matrix_slice[6], matrix_slice[7], matrix_slice[8]],
    ]);

    let shift = match offset {
        Some(v) if v.len() == 3 => [v[0], v[1], v[2]],
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Offset must have 3 elements",
            ));
        }
        None => [0.0, 0.0, 0.0],
    };

    let input_array = input.as_array();
    let output = affine_transform_3d_u8(&input_array, &affine_matrix, &shift, cval);

    Ok(output.into_pyarray(py))
}

// =============================================================================
// Module registration
// =============================================================================

/// Fast 3D affine transformations with trilinear interpolation using AVX2/AVX512 SIMD
///
/// Supported data types:
/// - f32: affine_transform() - Standard floating point
/// - f16: affine_transform_f16() - Half precision, 2x less memory
/// - u8: affine_transform_u8() - 2.2x faster, 4x less memory
#[pymodule]
fn affiners(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine_transform, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_f16, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_u8, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
