//! Fast 3D trilinear interpolation using AVX2 SIMD instructions
//!
//! This crate provides high-performance 3D interpolation functions optimized
//! for modern x86-64 processors with AVX2 support. It is a Rust port of the
//! scipy.ndimage interpolation routines, focused on order=1 (trilinear) interpolation.
//!
//! # Features
//!
//! - **AVX2 SIMD**: Processes 4 f64 or 8 f32 values per iteration
//! - **Parallel execution**: Uses rayon for multi-threaded processing
//! - **ndarray integration**: Works directly with ndarray arrays
//!
//! # Example
//!
//! ```rust
//! use ndarray::Array3;
//! use interp3d_avx2::{affine_transform_3d_f32, AffineMatrix3D};
//!
//! // Create a 3D volume
//! let input = Array3::<f32>::zeros((100, 100, 100));
//!
//! // Define an affine transformation (identity + translation)
//! let matrix = AffineMatrix3D::identity();
//! let shift = [10.0, 20.0, 30.0];
//!
//! // Apply the transformation
//! let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);
//! ```

pub mod scalar;
pub mod simd;

#[cfg(feature = "python")]
mod python;

pub use half::f16;
use ndarray::{Array3, ArrayView3};

/// 3x3 affine transformation matrix (row-major)
///
/// The matrix transforms coordinates as:
/// ```text
/// [z']   [m00 m01 m02] [z]   [shift_z]
/// [y'] = [m10 m11 m12] [y] + [shift_y]
/// [x']   [m20 m21 m22] [x]   [shift_x]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AffineMatrix3D {
    pub m: [[f64; 3]; 3],
}

impl AffineMatrix3D {
    /// Create a new affine matrix from a 3x3 array
    #[inline]
    pub fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// Create an identity matrix
    #[inline]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create a scaling matrix
    #[inline]
    pub fn scale(sz: f64, sy: f64, sx: f64) -> Self {
        Self {
            m: [[sz, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sx]],
        }
    }

    /// Create a rotation matrix around the Z axis (in radians)
    #[inline]
    pub fn rotate_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }

    /// Create a rotation matrix around the Y axis (in radians)
    #[inline]
    pub fn rotate_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }

    /// Create a rotation matrix around the X axis (in radians)
    #[inline]
    pub fn rotate_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Get the flat representation for SIMD operations
    #[inline]
    pub fn as_flat(&self) -> [f64; 9] {
        [
            self.m[0][0],
            self.m[0][1],
            self.m[0][2],
            self.m[1][0],
            self.m[1][1],
            self.m[1][2],
            self.m[2][0],
            self.m[2][1],
            self.m[2][2],
        ]
    }

    /// Get the flat representation as f32 for SIMD operations
    #[inline]
    pub fn as_flat_f32(&self) -> [f32; 9] {
        [
            self.m[0][0] as f32,
            self.m[0][1] as f32,
            self.m[0][2] as f32,
            self.m[1][0] as f32,
            self.m[1][1] as f32,
            self.m[1][2] as f32,
            self.m[2][0] as f32,
            self.m[2][1] as f32,
            self.m[2][2] as f32,
        ]
    }
}

impl Default for AffineMatrix3D {
    fn default() -> Self {
        Self::identity()
    }
}

/// Trait for types that support trilinear interpolation
pub trait Interpolate: Copy + Send + Sync + Default + 'static {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Interpolate for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Interpolate for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Apply a 3D affine transformation with trilinear interpolation (f32)
///
/// # Arguments
///
/// * `input` - Input 3D array view
/// * `matrix` - 3x3 transformation matrix
/// * `shift` - Translation vector [z, y, x]
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Returns
///
/// Transformed 3D array with the same shape as input
#[inline]
pub fn affine_transform_3d_f32(
    input: &ArrayView3<f32>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) -> Array3<f32> {
    let shape = input.dim();
    let mut output = Array3::from_elem(shape, cval as f32);

    #[cfg(target_arch = "x86_64")]
    {
        // Check if we can use AVX512 (preferred)
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                simd::avx512::trilinear_3d_f32_avx512(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }

        // Fallback to AVX2
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd::avx2::trilinear_3d_f32_avx2(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }
    }

    // Scalar fallback
    scalar::trilinear_3d_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

/// Apply a 3D affine transformation with trilinear interpolation (f64)
///
/// # Arguments
///
/// * `input` - Input 3D array view
/// * `matrix` - 3x3 transformation matrix
/// * `shift` - Translation vector [z, y, x]
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Returns
///
/// Transformed 3D array with the same shape as input
#[inline]
pub fn affine_transform_3d_f64(
    input: &ArrayView3<f64>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) -> Array3<f64> {
    let shape = input.dim();
    let mut output = Array3::from_elem(shape, cval);

    #[cfg(target_arch = "x86_64")]
    {
        // Check if we can use AVX512 (preferred)
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                simd::avx512::trilinear_3d_f64_avx512(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }

        // Fallback to AVX2
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd::avx2::trilinear_3d_f64_avx2(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }
    }

    // Scalar fallback
    scalar::trilinear_3d_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

/// Apply a 3D affine transformation with trilinear interpolation (f16/half precision)
///
/// # Arguments
///
/// * `input` - Input 3D array view (f16)
/// * `matrix` - 3x3 transformation matrix
/// * `shift` - Translation vector [z, y, x]
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Returns
///
/// Transformed 3D array with the same shape as input
///
/// # Note
///
/// Computation is performed in f32 for accuracy, with f16↔f32 conversion
/// using F16C instructions when available.
#[inline]
pub fn affine_transform_3d_f16(
    input: &ArrayView3<f16>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) -> Array3<f16> {
    let shape = input.dim();
    let mut output = Array3::from_elem(shape, f16::from_f64(cval));

    #[cfg(target_arch = "x86_64")]
    {
        // Check if we can use AVX512 (preferred)
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                simd::avx512::trilinear_3d_f16_avx512(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }

        // Fallback to AVX2 + F16C
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
            unsafe {
                simd::avx2::trilinear_3d_f16_avx2(input, &mut output.view_mut(), matrix, shift, cval);
            }
            return output;
        }
    }

    // Scalar fallback - convert to f32, process, convert back
    scalar::trilinear_3d_f16_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

/// Generic 3D affine transformation (dispatches based on type)
#[inline]
pub fn affine_transform_3d<T: Interpolate>(
    input: &ArrayView3<T>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) -> Array3<T> {
    let shape = input.dim();
    let mut output = Array3::from_elem(shape, T::from_f64(cval));
    scalar::trilinear_3d_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

/// Apply map_coordinates with trilinear interpolation (f32)
///
/// # Arguments
///
/// * `input` - Input 3D array view
/// * `z_coords` - Z coordinates (same shape as desired output)
/// * `y_coords` - Y coordinates (same shape as desired output)
/// * `x_coords` - X coordinates (same shape as desired output)
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Returns
///
/// Interpolated values at the given coordinates
pub fn map_coordinates_3d_f32(
    input: &ArrayView3<f32>,
    z_coords: &[f64],
    y_coords: &[f64],
    x_coords: &[f64],
    cval: f64,
) -> Vec<f32> {
    assert_eq!(z_coords.len(), y_coords.len());
    assert_eq!(y_coords.len(), x_coords.len());

    let (d, h, w) = input.dim();
    let input_slice = input.as_slice().expect("Input must be contiguous");

    z_coords
        .iter()
        .zip(y_coords.iter())
        .zip(x_coords.iter())
        .map(|((&z, &y), &x)| scalar::trilinear_interp_f32(input_slice, d, h, w, z, y, x, cval as f32))
        .collect()
}

/// Apply map_coordinates with trilinear interpolation (f64)
pub fn map_coordinates_3d_f64(
    input: &ArrayView3<f64>,
    z_coords: &[f64],
    y_coords: &[f64],
    x_coords: &[f64],
    cval: f64,
) -> Vec<f64> {
    assert_eq!(z_coords.len(), y_coords.len());
    assert_eq!(y_coords.len(), x_coords.len());

    let (d, h, w) = input.dim();
    let input_slice = input.as_slice().expect("Input must be contiguous");

    z_coords
        .iter()
        .zip(y_coords.iter())
        .zip(x_coords.iter())
        .map(|((&z, &y), &x)| scalar::trilinear_interp_f64(input_slice, d, h, w, z, y, x, cval))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_identity_transform_f32() {
        let input = Array3::from_shape_fn((10, 10, 10), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

        // Interior points should match exactly (edges may differ due to boundary)
        for z in 1..9 {
            for y in 1..9 {
                for x in 1..9 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_identity_transform_f64() {
        let input = Array3::from_shape_fn((10, 10, 10), |(z, y, x)| (z * 100 + y * 10 + x) as f64);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        let output = affine_transform_3d_f64(&input.view(), &matrix, &shift, 0.0);

        // Interior points should match exactly
        for z in 1..9 {
            for y in 1..9 {
                for x in 1..9 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_translation() {
        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z + y + x) as f32);
        let matrix = AffineMatrix3D::identity();
        let shift = [1.0, 1.0, 1.0]; // Shift by 1 in each dimension

        let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

        // Point at (5,5,5) should now sample from (6,6,6) in input
        // Due to trilinear interpolation, check interior
        for z in 2..17 {
            for y in 2..17 {
                for x in 2..17 {
                    let expected = input[[z + 1, y + 1, x + 1]];
                    assert_relative_eq!(output[[z, y, x]], expected, epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_affine_matrix_constructors() {
        let identity = AffineMatrix3D::identity();
        assert_eq!(identity.m[0][0], 1.0);
        assert_eq!(identity.m[1][1], 1.0);
        assert_eq!(identity.m[2][2], 1.0);

        let scale = AffineMatrix3D::scale(2.0, 3.0, 4.0);
        assert_eq!(scale.m[0][0], 2.0);
        assert_eq!(scale.m[1][1], 3.0);
        assert_eq!(scale.m[2][2], 4.0);
    }
}
