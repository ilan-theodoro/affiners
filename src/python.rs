//! Python bindings using PyO3

use half::f16;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::{affine_transform_3d_f16, affine_transform_3d_f32, affine_transform_3d_f64, AffineMatrix3D};

/// Python wrapper for affine_transform_3d for f32 arrays
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

    // Convert matrix to AffineMatrix3D
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

    // Get offset/shift
    let shift = match offset {
        Some(v) => {
            if v.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Offset must have 3 elements",
                ));
            }
            [v[0], v[1], v[2]]
        }
        None => [0.0, 0.0, 0.0],
    };

    // Convert input to ndarray
    let input_array = input.as_array();

    // Apply transformation
    let output = affine_transform_3d_f32(&input_array, &affine_matrix, &shift, cval);

    // Convert back to numpy
    Ok(output.into_pyarray(py))
}

/// Python wrapper for affine_transform_3d for f64 arrays
#[pyfunction]
#[pyo3(signature = (input, matrix, offset=None, cval=0.0, order=1))]
fn affine_transform_f64<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, f64>,
    matrix: PyReadonlyArray2<'py, f64>,
    offset: Option<Vec<f64>>,
    cval: f64,
    order: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
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
        Some(v) => {
            if v.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Offset must have 3 elements",
                ));
            }
            [v[0], v[1], v[2]]
        }
        None => [0.0, 0.0, 0.0],
    };

    let input_array = input.as_array();
    let output = affine_transform_3d_f64(&input_array, &affine_matrix, &shift, cval);

    Ok(output.into_pyarray(py))
}

/// Python wrapper for affine_transform_3d for f16 arrays
/// Note: numpy float16 is stored as u16 bits, same as half::f16
#[pyfunction]
#[pyo3(signature = (input, matrix, offset=None, cval=0.0, order=1))]
fn affine_transform_f16<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u16>,  // numpy float16 stored as u16
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
        Some(v) => {
            if v.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Offset must have 3 elements",
                ));
            }
            [v[0], v[1], v[2]]
        }
        None => [0.0, 0.0, 0.0],
    };

    // Get input dimensions and data
    let shape = input.shape();
    let (d, h, w) = (shape[0], shape[1], shape[2]);
    let input_slice = input.as_slice()?;

    // Reinterpret u16 as f16 (safe because they have the same memory layout)
    let input_f16: &[f16] = unsafe {
        std::slice::from_raw_parts(input_slice.as_ptr() as *const f16, input_slice.len())
    };

    // Create ndarray view from the f16 slice
    let input_array = ndarray::ArrayView3::from_shape((d, h, w), input_f16)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    // Apply transformation
    let output = affine_transform_3d_f16(&input_array, &affine_matrix, &shift, cval);

    // Reinterpret f16 output as u16 for numpy
    let output_u16: Array3<u16> = unsafe {
        let ptr = output.as_ptr() as *const u16;
        let slice = std::slice::from_raw_parts(ptr, output.len());
        Array3::from_shape_vec((d, h, w), slice.to_vec()).unwrap()
    };

    Ok(output_u16.into_pyarray(py))
}

/// Fast 3D trilinear interpolation using AVX2 SIMD
#[pymodule]
fn interp3d_avx2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine_transform, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_f64, m)?)?;
    m.add_function(wrap_pyfunction!(affine_transform_f16, m)?)?;
    Ok(())
}
