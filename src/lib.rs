// Allow unsafe operations in unsafe functions (Rust 2024 lint)
#![allow(unsafe_op_in_unsafe_fn)]

//! Fast 3D trilinear interpolation using AVX2/AVX512 SIMD instructions
//!
//! This crate provides high-performance 3D interpolation functions optimized
//! for modern x86-64 processors. Supports:
//! - f32: Standard floating point
//! - f16: Half precision (reduced memory)
//! - u8: 2.2x faster than f32, 4x less memory
//!
//! # Example
//!
//! ```rust
//! use ndarray::{Array2, Array3};
//! use affiners::affine_transform_3d_f32;
//!
//! let input = Array3::<f32>::zeros((100, 100, 100));
//! // 4x4 homogeneous transformation matrix (identity with translation [10, 20, 30])
//! let matrix = Array2::from_shape_vec((4, 4), vec![
//!     1.0, 0.0, 0.0, 10.0,
//!     0.0, 1.0, 0.0, 20.0,
//!     0.0, 0.0, 1.0, 30.0,
//!     0.0, 0.0, 0.0, 1.0,
//! ]).unwrap();
//!
//! // Same shape as input
//! let output = affine_transform_3d_f32(&input.view(), &matrix.view(), None, 0.0);
//!
//! // Custom output shape
//! let output = affine_transform_3d_f32(&input.view(), &matrix.view(), Some((50, 50, 50)), 0.0);
//! ```

pub mod scalar;
pub mod simd;

#[cfg(feature = "python")]
mod python;

pub use half::f16;
use ndarray::{Array3, ArrayView2, ArrayView3};

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
    #[inline]
    pub fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// Decompose a 4x4 homogeneous transformation matrix into a 3x3 affine matrix and shift vector.
    ///
    /// The homogeneous matrix format is:
    /// ```text
    /// [m00 m01 m02 tz]
    /// [m10 m11 m12 ty]
    /// [m20 m21 m22 tx]
    /// [0   0   0   1 ]
    /// ```
    ///
    /// Returns `(AffineMatrix3D, [shift_z, shift_y, shift_x])`.
    ///
    /// # Panics
    ///
    /// Panics if the input matrix is not 4x4.
    #[inline]
    pub fn from_homogeneous(matrix: &ArrayView2<f64>) -> (Self, [f64; 3]) {
        let shape = matrix.dim();
        assert!(
            shape == (4, 4),
            "Homogeneous matrix must be 4x4, got {:?}",
            shape
        );

        let affine = Self {
            m: [
                [matrix[[0, 0]], matrix[[0, 1]], matrix[[0, 2]]],
                [matrix[[1, 0]], matrix[[1, 1]], matrix[[1, 2]]],
                [matrix[[2, 0]], matrix[[2, 1]], matrix[[2, 2]]],
            ],
        };

        let shift = [matrix[[0, 3]], matrix[[1, 3]], matrix[[2, 3]]];

        (affine, shift)
    }

    #[inline]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    #[inline]
    pub fn scale(sz: f64, sy: f64, sx: f64) -> Self {
        Self {
            m: [[sz, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sx]],
        }
    }

    #[inline]
    pub fn rotate_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }

    #[inline]
    pub fn rotate_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }

    #[inline]
    pub fn rotate_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

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

// =============================================================================
// F32 - Main workhorse
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation (f32)
///
/// # Arguments
///
/// * `input` - 3D input array
/// * `homogeneous_matrix` - 4x4 homogeneous transformation matrix
/// * `output_shape` - Optional output shape (z, y, x). If None, uses input shape.
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Example
///
/// ```rust
/// use ndarray::{Array2, Array3};
/// use affiners::affine_transform_3d_f32;
///
/// let input = Array3::<f32>::zeros((100, 100, 100));
/// // Identity transform with translation [10, 20, 30]
/// let matrix = Array2::from_shape_vec((4, 4), vec![
///     1.0, 0.0, 0.0, 10.0,
///     0.0, 1.0, 0.0, 20.0,
///     0.0, 0.0, 1.0, 30.0,
///     0.0, 0.0, 0.0, 1.0,
/// ]).unwrap();
///
/// // Same shape as input
/// let output = affine_transform_3d_f32(&input.view(), &matrix.view(), None, 0.0);
///
/// // Custom output shape
/// let output = affine_transform_3d_f32(&input.view(), &matrix.view(), Some((50, 50, 50)), 0.0);
/// ```
#[inline]
pub fn affine_transform_3d_f32(
    input: &ArrayView3<f32>,
    homogeneous_matrix: &ArrayView2<f64>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
) -> Array3<f32> {
    let (matrix, shift) = AffineMatrix3D::from_homogeneous(homogeneous_matrix);
    let shape = output_shape.unwrap_or_else(|| input.dim());
    affine_transform_3d_f32_internal(input, &matrix, &shift, shape, cval)
}

/// Internal implementation that takes decomposed matrix and shift
#[inline]
fn affine_transform_3d_f32_internal(
    input: &ArrayView3<f32>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    output_shape: (usize, usize, usize),
    cval: f64,
) -> Array3<f32> {
    let mut output = Array3::from_elem(output_shape, cval as f32);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                simd::avx512::trilinear_3d_f32_avx512(
                    input,
                    &mut output.view_mut(),
                    matrix,
                    shift,
                    cval,
                );
            }
            return output;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd::avx2::trilinear_3d_f32_avx2(
                    input,
                    &mut output.view_mut(),
                    matrix,
                    shift,
                    cval,
                );
            }
            return output;
        }
    }

    scalar::trilinear_3d_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

// =============================================================================
// F16 - Half precision
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation (f16)
///
/// Computation is performed in f32 for accuracy, with f16↔f32 conversion.
///
/// # Arguments
///
/// * `input` - 3D input array (f16)
/// * `homogeneous_matrix` - 4x4 homogeneous transformation matrix
/// * `output_shape` - Optional output shape (z, y, x). If None, uses input shape.
/// * `cval` - Constant value for out-of-bounds coordinates
#[inline]
pub fn affine_transform_3d_f16(
    input: &ArrayView3<f16>,
    homogeneous_matrix: &ArrayView2<f64>,
    output_shape: Option<(usize, usize, usize)>,
    cval: f64,
) -> Array3<f16> {
    let (matrix, shift) = AffineMatrix3D::from_homogeneous(homogeneous_matrix);
    let shape = output_shape.unwrap_or_else(|| input.dim());
    affine_transform_3d_f16_internal(input, &matrix, &shift, shape, cval)
}

/// Internal implementation that takes decomposed matrix and shift
#[inline]
fn affine_transform_3d_f16_internal(
    input: &ArrayView3<f16>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    output_shape: (usize, usize, usize),
    cval: f64,
) -> Array3<f16> {
    let mut output = Array3::from_elem(output_shape, f16::from_f64(cval));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                simd::avx512::trilinear_3d_f16_avx512(
                    input,
                    &mut output.view_mut(),
                    matrix,
                    shift,
                    cval,
                );
            }
            return output;
        }

        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            unsafe {
                simd::avx2::trilinear_3d_f16_avx2(
                    input,
                    &mut output.view_mut(),
                    matrix,
                    shift,
                    cval,
                );
            }
            return output;
        }
    }

    scalar::trilinear_3d_f16_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

// =============================================================================
// U8 - 2.2x faster than f32, 4x less memory
// =============================================================================

/// Apply 3D affine transformation with trilinear interpolation (u8)
///
/// # Performance
/// - 2.2x faster than f32 due to 4x less memory traffic
/// - 4x less memory consumption
///
/// Ideal for image data (microscopy, CT, MRI).
///
/// # Arguments
///
/// * `input` - 3D input array (u8)
/// * `homogeneous_matrix` - 4x4 homogeneous transformation matrix
/// * `output_shape` - Optional output shape (z, y, x). If None, uses input shape.
/// * `cval` - Constant value for out-of-bounds coordinates
#[inline]
pub fn affine_transform_3d_u8(
    input: &ArrayView3<u8>,
    homogeneous_matrix: &ArrayView2<f64>,
    output_shape: Option<(usize, usize, usize)>,
    cval: u8,
) -> Array3<u8> {
    let (matrix, shift) = AffineMatrix3D::from_homogeneous(homogeneous_matrix);
    let shape = output_shape.unwrap_or_else(|| input.dim());
    affine_transform_3d_u8_internal(input, &matrix, &shift, shape, cval)
}

/// Internal implementation that takes decomposed matrix and shift
#[inline]
fn affine_transform_3d_u8_internal(
    input: &ArrayView3<u8>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    output_shape: (usize, usize, usize),
    cval: u8,
) -> Array3<u8> {
    let mut output = Array3::from_elem(output_shape, cval);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd::avx2::trilinear_3d_u8_avx2(
                    input,
                    &mut output.view_mut(),
                    matrix,
                    shift,
                    cval,
                );
            }
            return output;
        }
    }

    scalar::trilinear_3d_u8_scalar(input, &mut output.view_mut(), matrix, shift, cval);
    output
}

// =============================================================================
// Utilities
// =============================================================================

/// Apply map_coordinates with trilinear interpolation (f32)
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
        .map(|((&z, &y), &x)| {
            scalar::trilinear_interp_f32(input_slice, d, h, w, z, y, x, cval as f32)
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array2, Array3};

    /// Create a 4x4 identity homogeneous matrix
    fn identity_homogeneous() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap()
    }

    /// Create a 4x4 homogeneous matrix with translation
    fn translation_homogeneous(tz: f64, ty: f64, tx: f64) -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, tz, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tx, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_identity_transform_f32() {
        let input = Array3::from_shape_fn((10, 10, 10), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let matrix = identity_homogeneous();

        let output = affine_transform_3d_f32(&input.view(), &matrix.view(), None, 0.0);

        for z in 1..9 {
            for y in 1..9 {
                for x in 1..9 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_translation() {
        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z + y + x) as f32);
        let matrix = translation_homogeneous(1.0, 1.0, 1.0);

        let output = affine_transform_3d_f32(&input.view(), &matrix.view(), None, 0.0);

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
    fn test_custom_output_shape() {
        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z + y + x) as f32);
        let matrix = identity_homogeneous();

        // Smaller output
        let output = affine_transform_3d_f32(&input.view(), &matrix.view(), Some((10, 10, 10)), 0.0);
        assert_eq!(output.dim(), (10, 10, 10));

        // Larger output (edges will be cval)
        let output =
            affine_transform_3d_f32(&input.view(), &matrix.view(), Some((30, 30, 30)), -1.0);
        assert_eq!(output.dim(), (30, 30, 30));
        // Check that out-of-bounds values are cval
        assert_eq!(output[[25, 25, 25]], -1.0);
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

    #[test]
    fn test_from_homogeneous() {
        let homogeneous = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0, 20.0, 7.0, 8.0, 9.0, 30.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();

        let (affine, shift) = AffineMatrix3D::from_homogeneous(&homogeneous.view());

        assert_eq!(affine.m[0][0], 1.0);
        assert_eq!(affine.m[0][1], 2.0);
        assert_eq!(affine.m[0][2], 3.0);
        assert_eq!(affine.m[1][0], 4.0);
        assert_eq!(affine.m[1][1], 5.0);
        assert_eq!(affine.m[1][2], 6.0);
        assert_eq!(affine.m[2][0], 7.0);
        assert_eq!(affine.m[2][1], 8.0);
        assert_eq!(affine.m[2][2], 9.0);

        assert_eq!(shift[0], 10.0);
        assert_eq!(shift[1], 20.0);
        assert_eq!(shift[2], 30.0);
    }
}
