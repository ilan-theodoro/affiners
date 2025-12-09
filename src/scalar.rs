//! Scalar (non-SIMD) implementations of interpolation functions
//!
//! These serve as fallbacks when AVX2/AVX512 is not available and as reference
//! implementations for testing.

use crate::{AffineMatrix3D, Interpolate};
use half::f16;
use ndarray::{ArrayView3, ArrayViewMut3};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Trilinear interpolation at a single point (f32)
///
/// Used by map_coordinates for point-by-point interpolation.
#[inline]
pub fn trilinear_interp_f32(
    data: &[f32],
    d: usize,
    h: usize,
    w: usize,
    z: f64,
    y: f64,
    x: f64,
    cval: f32,
) -> f32 {
    let z0 = z.floor() as isize;
    let y0 = y.floor() as isize;
    let x0 = x.floor() as isize;

    // Check bounds - allow last index (will clamp +1 indices)
    if x0 < 0 || x0 >= w as isize || y0 < 0 || y0 >= h as isize || z0 < 0 || z0 >= d as isize {
        return cval;
    }

    let z0u = z0 as usize;
    let y0u = y0 as usize;
    let x0u = x0 as usize;

    // Clamp +1 indices to handle boundary
    let z1u = (z0u + 1).min(d - 1);
    let y1u = (y0u + 1).min(h - 1);
    let x1u = (x0u + 1).min(w - 1);

    let fz = (z - z.floor()) as f32;
    let fy = (y - y.floor()) as f32;
    let fx = (x - x.floor()) as f32;

    let stride_z = h * w;
    let stride_y = w;

    let idx000 = z0u * stride_z + y0u * stride_y + x0u;
    let idx001 = z0u * stride_z + y0u * stride_y + x1u;
    let idx010 = z0u * stride_z + y1u * stride_y + x0u;
    let idx011 = z0u * stride_z + y1u * stride_y + x1u;
    let idx100 = z1u * stride_z + y0u * stride_y + x0u;
    let idx101 = z1u * stride_z + y0u * stride_y + x1u;
    let idx110 = z1u * stride_z + y1u * stride_y + x0u;
    let idx111 = z1u * stride_z + y1u * stride_y + x1u;

    let v000 = data[idx000];
    let v001 = data[idx001];
    let v010 = data[idx010];
    let v011 = data[idx011];
    let v100 = data[idx100];
    let v101 = data[idx101];
    let v110 = data[idx110];
    let v111 = data[idx111];

    let one_fx = 1.0 - fx;
    let one_fy = 1.0 - fy;
    let one_fz = 1.0 - fz;

    v000 * one_fx * one_fy * one_fz
        + v001 * fx * one_fy * one_fz
        + v010 * one_fx * fy * one_fz
        + v011 * fx * fy * one_fz
        + v100 * one_fx * one_fy * fz
        + v101 * fx * one_fy * fz
        + v110 * one_fx * fy * fz
        + v111 * fx * fy * fz
}

/// Generic scalar 3D trilinear affine transform implementation
///
/// Works for any type implementing the Interpolate trait.
pub fn trilinear_3d_scalar<T: Interpolate>(
    input: &ArrayView3<T>,
    output: &mut ArrayViewMut3<T>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let m = &matrix.m;
    let m00 = m[0][0];
    let m01 = m[0][1];
    let m02 = m[0][2];
    let m10 = m[1][0];
    let m11 = m[1][1];
    let m12 = m[1][2];
    let m20 = m[2][0];
    let m21 = m[2][1];
    let m22 = m[2][2];

    let shift_z = shift[0];
    let shift_y = shift[1];
    let shift_x = shift[2];

    let stride_z = h * w;
    let stride_y = w;

    #[cfg(feature = "parallel")]
    {
        let chunk_size = oh * ow;
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                let oz_f = oz as f64;
                for oy in 0..oh {
                    let oy_f = oy as f64;
                    let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                    let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                    let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                    for ox in 0..ow {
                        let ox_f = ox as f64;
                        let z_src = m02 * ox_f + base_z;
                        let y_src = m12 * ox_f + base_y;
                        let x_src = m22 * ox_f + base_x;

                        let z0 = z_src.floor() as isize;
                        let y0 = y_src.floor() as isize;
                        let x0 = x_src.floor() as isize;

                        let out_idx = oy * ow + ox;

                        if x0 >= 0
                            && x0 < w as isize
                            && y0 >= 0
                            && y0 < h as isize
                            && z0 >= 0
                            && z0 < d as isize
                        {
                            let z0u = z0 as usize;
                            let y0u = y0 as usize;
                            let x0u = x0 as usize;

                            // Clamp +1 indices to handle boundary
                            let z1u = (z0u + 1).min(d - 1);
                            let y1u = (y0u + 1).min(h - 1);
                            let x1u = (x0u + 1).min(w - 1);

                            let fz = z_src - z_src.floor();
                            let fy = y_src - y_src.floor();
                            let fx = x_src - x_src.floor();

                            let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                            let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                            let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                            let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                            let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                            let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                            let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                            let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                            let v000 = input_slice[idx000].to_f64();
                            let v001 = input_slice[idx001].to_f64();
                            let v010 = input_slice[idx010].to_f64();
                            let v011 = input_slice[idx011].to_f64();
                            let v100 = input_slice[idx100].to_f64();
                            let v101 = input_slice[idx101].to_f64();
                            let v110 = input_slice[idx110].to_f64();
                            let v111 = input_slice[idx111].to_f64();

                            let one_fx = 1.0 - fx;
                            let one_fy = 1.0 - fy;
                            let one_fz = 1.0 - fz;

                            let result = v000 * one_fx * one_fy * one_fz
                                + v001 * fx * one_fy * one_fz
                                + v010 * one_fx * fy * one_fz
                                + v011 * fx * fy * one_fz
                                + v100 * one_fx * one_fy * fz
                                + v101 * fx * one_fy * fz
                                + v110 * one_fx * fy * fz
                                + v111 * fx * fy * fz;

                            slice_z[out_idx] = T::from_f64(result);
                        } else {
                            slice_z[out_idx] = T::from_f64(cval);
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for oz in 0..output.dim().0 {
            let oz_f = oz as f64;
            for oy in 0..oh {
                let oy_f = oy as f64;
                let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                for ox in 0..ow {
                    let ox_f = ox as f64;
                    let z_src = m02 * ox_f + base_z;
                    let y_src = m12 * ox_f + base_y;
                    let x_src = m22 * ox_f + base_x;

                    let z0 = z_src.floor() as isize;
                    let y0 = y_src.floor() as isize;
                    let x0 = x_src.floor() as isize;

                    let out_idx = oz * oh * ow + oy * ow + ox;

                    if x0 >= 0
                        && x0 < w as isize
                        && y0 >= 0
                        && y0 < h as isize
                        && z0 >= 0
                        && z0 < d as isize
                    {
                        let z0u = z0 as usize;
                        let y0u = y0 as usize;
                        let x0u = x0 as usize;

                        // Clamp +1 indices to handle boundary
                        let z1u = (z0u + 1).min(d - 1);
                        let y1u = (y0u + 1).min(h - 1);
                        let x1u = (x0u + 1).min(w - 1);

                        let fz = z_src - z_src.floor();
                        let fy = y_src - y_src.floor();
                        let fx = x_src - x_src.floor();

                        let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                        let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                        let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                        let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                        let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                        let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                        let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                        let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                        let v000 = input_slice[idx000].to_f64();
                        let v001 = input_slice[idx001].to_f64();
                        let v010 = input_slice[idx010].to_f64();
                        let v011 = input_slice[idx011].to_f64();
                        let v100 = input_slice[idx100].to_f64();
                        let v101 = input_slice[idx101].to_f64();
                        let v110 = input_slice[idx110].to_f64();
                        let v111 = input_slice[idx111].to_f64();

                        let one_fx = 1.0 - fx;
                        let one_fy = 1.0 - fy;
                        let one_fz = 1.0 - fz;

                        let result = v000 * one_fx * one_fy * one_fz
                            + v001 * fx * one_fy * one_fz
                            + v010 * one_fx * fy * one_fz
                            + v011 * fx * fy * one_fz
                            + v100 * one_fx * one_fy * fz
                            + v101 * fx * one_fy * fz
                            + v110 * one_fx * fy * fz
                            + v111 * fx * fy * fz;

                        output_slice[out_idx] = T::from_f64(result);
                    } else {
                        output_slice[out_idx] = T::from_f64(cval);
                    }
                }
            }
        }
    }
}

/// Scalar 3D trilinear affine transform implementation for f16
///
/// Computation is done in f32 for accuracy, with f16 I/O.
pub fn trilinear_3d_f16_scalar(
    input: &ArrayView3<f16>,
    output: &mut ArrayViewMut3<f16>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let m = &matrix.m;
    let m00 = m[0][0] as f32;
    let m01 = m[0][1] as f32;
    let m02 = m[0][2] as f32;
    let m10 = m[1][0] as f32;
    let m11 = m[1][1] as f32;
    let m12 = m[1][2] as f32;
    let m20 = m[2][0] as f32;
    let m21 = m[2][1] as f32;
    let m22 = m[2][2] as f32;

    let shift_z = shift[0] as f32;
    let shift_y = shift[1] as f32;
    let shift_x = shift[2] as f32;
    let cval_f16 = f16::from_f64(cval);

    let stride_z = h * w;
    let stride_y = w;

    #[cfg(feature = "parallel")]
    {
        let chunk_size = oh * ow;
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                let oz_f = oz as f32;
                for oy in 0..oh {
                    let oy_f = oy as f32;
                    let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                    let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                    let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                    for ox in 0..ow {
                        let ox_f = ox as f32;
                        let z_src = m02 * ox_f + base_z;
                        let y_src = m12 * ox_f + base_y;
                        let x_src = m22 * ox_f + base_x;

                        let z0 = z_src.floor() as i32;
                        let y0 = y_src.floor() as i32;
                        let x0 = x_src.floor() as i32;

                        let out_idx = oy * ow + ox;

                        if x0 >= 0
                            && x0 < w as i32
                            && y0 >= 0
                            && y0 < h as i32
                            && z0 >= 0
                            && z0 < d as i32
                        {
                            let z0u = z0 as usize;
                            let y0u = y0 as usize;
                            let x0u = x0 as usize;

                            // Clamp +1 indices to handle boundary
                            let z1u = (z0u + 1).min(d - 1);
                            let y1u = (y0u + 1).min(h - 1);
                            let x1u = (x0u + 1).min(w - 1);

                            let fz = z_src - z_src.floor();
                            let fy = y_src - y_src.floor();
                            let fx = x_src - x_src.floor();

                            let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                            let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                            let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                            let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                            let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                            let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                            let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                            let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                            let v000 = input_slice[idx000].to_f32();
                            let v001 = input_slice[idx001].to_f32();
                            let v010 = input_slice[idx010].to_f32();
                            let v011 = input_slice[idx011].to_f32();
                            let v100 = input_slice[idx100].to_f32();
                            let v101 = input_slice[idx101].to_f32();
                            let v110 = input_slice[idx110].to_f32();
                            let v111 = input_slice[idx111].to_f32();

                            let one_fx = 1.0 - fx;
                            let one_fy = 1.0 - fy;
                            let one_fz = 1.0 - fz;

                            let result = v000 * one_fx * one_fy * one_fz
                                + v001 * fx * one_fy * one_fz
                                + v010 * one_fx * fy * one_fz
                                + v011 * fx * fy * one_fz
                                + v100 * one_fx * one_fy * fz
                                + v101 * fx * one_fy * fz
                                + v110 * one_fx * fy * fz
                                + v111 * fx * fy * fz;

                            slice_z[out_idx] = f16::from_f32(result);
                        } else {
                            slice_z[out_idx] = cval_f16;
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for oz in 0..output.dim().0 {
            let oz_f = oz as f32;
            for oy in 0..oh {
                let oy_f = oy as f32;
                let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                for ox in 0..ow {
                    let ox_f = ox as f32;
                    let z_src = m02 * ox_f + base_z;
                    let y_src = m12 * ox_f + base_y;
                    let x_src = m22 * ox_f + base_x;

                    let z0 = z_src.floor() as i32;
                    let y0 = y_src.floor() as i32;
                    let x0 = x_src.floor() as i32;

                    let out_idx = oz * oh * ow + oy * ow + ox;

                    if x0 >= 0
                        && x0 < w as i32
                        && y0 >= 0
                        && y0 < h as i32
                        && z0 >= 0
                        && z0 < d as i32
                    {
                        let z0u = z0 as usize;
                        let y0u = y0 as usize;
                        let x0u = x0 as usize;

                        // Clamp +1 indices to handle boundary
                        let z1u = (z0u + 1).min(d - 1);
                        let y1u = (y0u + 1).min(h - 1);
                        let x1u = (x0u + 1).min(w - 1);

                        let fz = z_src - z_src.floor();
                        let fy = y_src - y_src.floor();
                        let fx = x_src - x_src.floor();

                        let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                        let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                        let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                        let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                        let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                        let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                        let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                        let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                        let v000 = input_slice[idx000].to_f32();
                        let v001 = input_slice[idx001].to_f32();
                        let v010 = input_slice[idx010].to_f32();
                        let v011 = input_slice[idx011].to_f32();
                        let v100 = input_slice[idx100].to_f32();
                        let v101 = input_slice[idx101].to_f32();
                        let v110 = input_slice[idx110].to_f32();
                        let v111 = input_slice[idx111].to_f32();

                        let one_fx = 1.0 - fx;
                        let one_fy = 1.0 - fy;
                        let one_fz = 1.0 - fz;

                        let result = v000 * one_fx * one_fy * one_fz
                            + v001 * fx * one_fy * one_fz
                            + v010 * one_fx * fy * one_fz
                            + v011 * fx * fy * one_fz
                            + v100 * one_fx * one_fy * fz
                            + v101 * fx * one_fy * fz
                            + v110 * one_fx * fy * fz
                            + v111 * fx * fy * fz;

                        output_slice[out_idx] = f16::from_f32(result);
                    } else {
                        output_slice[out_idx] = cval_f16;
                    }
                }
            }
        }
    }
}

/// Scalar 3D trilinear affine transform implementation for u8
///
/// Computation is done in f32 for accuracy, with u8 I/O.
pub fn trilinear_3d_u8_scalar(
    input: &ArrayView3<u8>,
    output: &mut ArrayViewMut3<u8>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: u8,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let m = &matrix.m;
    let m00 = m[0][0] as f32;
    let m01 = m[0][1] as f32;
    let m02 = m[0][2] as f32;
    let m10 = m[1][0] as f32;
    let m11 = m[1][1] as f32;
    let m12 = m[1][2] as f32;
    let m20 = m[2][0] as f32;
    let m21 = m[2][1] as f32;
    let m22 = m[2][2] as f32;

    let shift_z = shift[0] as f32;
    let shift_y = shift[1] as f32;
    let shift_x = shift[2] as f32;

    let stride_z = h * w;
    let stride_y = w;

    #[cfg(feature = "parallel")]
    {
        let chunk_size = oh * ow;
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                let oz_f = oz as f32;
                for oy in 0..oh {
                    let oy_f = oy as f32;
                    let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                    let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                    let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                    for ox in 0..ow {
                        let ox_f = ox as f32;
                        let z_src = m02 * ox_f + base_z;
                        let y_src = m12 * ox_f + base_y;
                        let x_src = m22 * ox_f + base_x;

                        let z0 = z_src.floor() as i32;
                        let y0 = y_src.floor() as i32;
                        let x0 = x_src.floor() as i32;

                        let out_idx = oy * ow + ox;

                        if x0 >= 0
                            && x0 < w as i32
                            && y0 >= 0
                            && y0 < h as i32
                            && z0 >= 0
                            && z0 < d as i32
                        {
                            let z0u = z0 as usize;
                            let y0u = y0 as usize;
                            let x0u = x0 as usize;

                            // Clamp +1 indices to handle boundary
                            let z1u = (z0u + 1).min(d - 1);
                            let y1u = (y0u + 1).min(h - 1);
                            let x1u = (x0u + 1).min(w - 1);

                            let fz = z_src - z_src.floor();
                            let fy = y_src - y_src.floor();
                            let fx = x_src - x_src.floor();

                            let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                            let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                            let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                            let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                            let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                            let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                            let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                            let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                            let v000 = input_slice[idx000] as f32;
                            let v001 = input_slice[idx001] as f32;
                            let v010 = input_slice[idx010] as f32;
                            let v011 = input_slice[idx011] as f32;
                            let v100 = input_slice[idx100] as f32;
                            let v101 = input_slice[idx101] as f32;
                            let v110 = input_slice[idx110] as f32;
                            let v111 = input_slice[idx111] as f32;

                            let one_fx = 1.0 - fx;
                            let one_fy = 1.0 - fy;
                            let one_fz = 1.0 - fz;

                            let result = v000 * one_fx * one_fy * one_fz
                                + v001 * fx * one_fy * one_fz
                                + v010 * one_fx * fy * one_fz
                                + v011 * fx * fy * one_fz
                                + v100 * one_fx * one_fy * fz
                                + v101 * fx * one_fy * fz
                                + v110 * one_fx * fy * fz
                                + v111 * fx * fy * fz;

                            slice_z[out_idx] = result.clamp(0.0, 255.0) as u8;
                        } else {
                            slice_z[out_idx] = cval;
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for oz in 0..output.dim().0 {
            let oz_f = oz as f32;
            for oy in 0..oh {
                let oy_f = oy as f32;
                let base_z = m00 * oz_f + m01 * oy_f + shift_z;
                let base_y = m10 * oz_f + m11 * oy_f + shift_y;
                let base_x = m20 * oz_f + m21 * oy_f + shift_x;

                for ox in 0..ow {
                    let ox_f = ox as f32;
                    let z_src = m02 * ox_f + base_z;
                    let y_src = m12 * ox_f + base_y;
                    let x_src = m22 * ox_f + base_x;

                    let z0 = z_src.floor() as i32;
                    let y0 = y_src.floor() as i32;
                    let x0 = x_src.floor() as i32;

                    let out_idx = oz * oh * ow + oy * ow + ox;

                    if x0 >= 0
                        && x0 < w as i32
                        && y0 >= 0
                        && y0 < h as i32
                        && z0 >= 0
                        && z0 < d as i32
                    {
                        let z0u = z0 as usize;
                        let y0u = y0 as usize;
                        let x0u = x0 as usize;

                        // Clamp +1 indices to handle boundary
                        let z1u = (z0u + 1).min(d - 1);
                        let y1u = (y0u + 1).min(h - 1);
                        let x1u = (x0u + 1).min(w - 1);

                        let fz = z_src - z_src.floor();
                        let fy = y_src - y_src.floor();
                        let fx = x_src - x_src.floor();

                        let idx000 = z0u * stride_z + y0u * stride_y + x0u;
                        let idx001 = z0u * stride_z + y0u * stride_y + x1u;
                        let idx010 = z0u * stride_z + y1u * stride_y + x0u;
                        let idx011 = z0u * stride_z + y1u * stride_y + x1u;
                        let idx100 = z1u * stride_z + y0u * stride_y + x0u;
                        let idx101 = z1u * stride_z + y0u * stride_y + x1u;
                        let idx110 = z1u * stride_z + y1u * stride_y + x0u;
                        let idx111 = z1u * stride_z + y1u * stride_y + x1u;

                        let v000 = input_slice[idx000] as f32;
                        let v001 = input_slice[idx001] as f32;
                        let v010 = input_slice[idx010] as f32;
                        let v011 = input_slice[idx011] as f32;
                        let v100 = input_slice[idx100] as f32;
                        let v101 = input_slice[idx101] as f32;
                        let v110 = input_slice[idx110] as f32;
                        let v111 = input_slice[idx111] as f32;

                        let one_fx = 1.0 - fx;
                        let one_fy = 1.0 - fy;
                        let one_fz = 1.0 - fz;

                        let result = v000 * one_fx * one_fy * one_fz
                            + v001 * fx * one_fy * one_fz
                            + v010 * one_fx * fy * one_fz
                            + v011 * fx * fy * one_fz
                            + v100 * one_fx * one_fy * fz
                            + v101 * fx * one_fy * fz
                            + v110 * one_fx * fy * fz
                            + v111 * fx * fy * fz;

                        output_slice[out_idx] = result.clamp(0.0, 255.0) as u8;
                    } else {
                        output_slice[out_idx] = cval;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Warp Field Scalar Implementations
// =============================================================================

use ndarray::ArrayView4;

/// Scalar implementation of apply_warp for f32
#[allow(clippy::too_many_arguments)]
pub fn apply_warp_3d_f32_scalar(
    image: &ArrayView3<f32>,
    warp_field: &ArrayView4<f32>,
    output: &mut ArrayViewMut3<f32>,
    cval: f32,
) {
    let (img_d, img_h, img_w) = image.dim();
    let (_wf_channels, wf_d, wf_h, wf_w) = warp_field.dim();

    // CUDA texture coordinate convention with normalize_coords=True:
    // u = pixel / img_size maps to texture position u * wf_size
    // CUDA linear filtering then applies a -0.5 offset before flooring.
    let scale_z = wf_d as f32 / img_d as f32;
    let scale_y = wf_h as f32 / img_h as f32;
    let scale_x = wf_w as f32 / img_w as f32;

    let image_slice = image.as_slice().expect("Image must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    // Extract warp field channels as contiguous slices
    let wf_dz = warp_field.slice(ndarray::s![0, .., .., ..]);
    let wf_dy = warp_field.slice(ndarray::s![1, .., .., ..]);
    let wf_dx = warp_field.slice(ndarray::s![2, .., .., ..]);

    let wf_dz_slice = wf_dz.as_slice().expect("Warp field must be C-contiguous");
    let wf_dy_slice = wf_dy.as_slice().expect("Warp field must be C-contiguous");
    let wf_dx_slice = wf_dx.as_slice().expect("Warp field must be C-contiguous");

    let wf_stride_z = wf_h * wf_w;
    let wf_stride_y = wf_w;
    let img_stride_z = img_h * img_w;
    let img_stride_y = img_w;

    for oz in 0..img_d {
        for oy in 0..img_h {
            for ox in 0..img_w {
                // Map output coords to warp field space with CUDA -0.5 offset
                let wf_z = oz as f32 * scale_z - 0.5;
                let wf_y = oy as f32 * scale_y - 0.5;
                let wf_x = ox as f32 * scale_x - 0.5;

                // Trilinearly interpolate warp field to get displacement
                let (dz, dy, dx) = trilinear_interp_warp_field_f32(
                    wf_dz_slice,
                    wf_dy_slice,
                    wf_dx_slice,
                    wf_d,
                    wf_h,
                    wf_w,
                    wf_stride_z,
                    wf_stride_y,
                    wf_z,
                    wf_y,
                    wf_x,
                );

                // Compute source coordinates (subtract displacement)
                let src_z = oz as f32 - dz;
                let src_y = oy as f32 - dy;
                let src_x = ox as f32 - dx;

                // Trilinearly interpolate input image at source coords
                let value = trilinear_interp_image_warp_f32(
                    image_slice,
                    img_d,
                    img_h,
                    img_w,
                    img_stride_z,
                    img_stride_y,
                    src_z,
                    src_y,
                    src_x,
                    cval,
                );

                output_slice[oz * img_stride_z + oy * img_stride_y + ox] = value;
            }
        }
    }
}

/// Scalar implementation of apply_warp for f16
#[allow(clippy::too_many_arguments)]
pub fn apply_warp_3d_f16_scalar(
    image: &ArrayView3<f16>,
    warp_field: &ArrayView4<f32>, // Warp field is always f32
    output: &mut ArrayViewMut3<f16>,
    cval: f16,
) {
    let (img_d, img_h, img_w) = image.dim();
    let (_wf_channels, wf_d, wf_h, wf_w) = warp_field.dim();

    let scale_z = wf_d as f32 / img_d as f32;
    let scale_y = wf_h as f32 / img_h as f32;
    let scale_x = wf_w as f32 / img_w as f32;

    let image_slice = image.as_slice().expect("Image must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let wf_dz = warp_field.slice(ndarray::s![0, .., .., ..]);
    let wf_dy = warp_field.slice(ndarray::s![1, .., .., ..]);
    let wf_dx = warp_field.slice(ndarray::s![2, .., .., ..]);

    let wf_dz_slice = wf_dz.as_slice().expect("Warp field must be C-contiguous");
    let wf_dy_slice = wf_dy.as_slice().expect("Warp field must be C-contiguous");
    let wf_dx_slice = wf_dx.as_slice().expect("Warp field must be C-contiguous");

    let wf_stride_z = wf_h * wf_w;
    let wf_stride_y = wf_w;
    let img_stride_z = img_h * img_w;
    let img_stride_y = img_w;
    let cval_f32 = cval.to_f32();

    for oz in 0..img_d {
        for oy in 0..img_h {
            for ox in 0..img_w {
                let wf_z = oz as f32 * scale_z - 0.5;
                let wf_y = oy as f32 * scale_y - 0.5;
                let wf_x = ox as f32 * scale_x - 0.5;

                let (dz, dy, dx) = trilinear_interp_warp_field_f32(
                    wf_dz_slice,
                    wf_dy_slice,
                    wf_dx_slice,
                    wf_d,
                    wf_h,
                    wf_w,
                    wf_stride_z,
                    wf_stride_y,
                    wf_z,
                    wf_y,
                    wf_x,
                );

                let src_z = oz as f32 - dz;
                let src_y = oy as f32 - dy;
                let src_x = ox as f32 - dx;

                let value = trilinear_interp_image_warp_f16(
                    image_slice,
                    img_d,
                    img_h,
                    img_w,
                    img_stride_z,
                    img_stride_y,
                    src_z,
                    src_y,
                    src_x,
                    cval_f32,
                );

                output_slice[oz * img_stride_z + oy * img_stride_y + ox] = f16::from_f32(value);
            }
        }
    }
}

/// Scalar implementation of apply_warp for u8
#[allow(clippy::too_many_arguments)]
pub fn apply_warp_3d_u8_scalar(
    image: &ArrayView3<u8>,
    warp_field: &ArrayView4<f32>, // Warp field is always f32
    output: &mut ArrayViewMut3<u8>,
    cval: u8,
) {
    let (img_d, img_h, img_w) = image.dim();
    let (_wf_channels, wf_d, wf_h, wf_w) = warp_field.dim();

    let scale_z = wf_d as f32 / img_d as f32;
    let scale_y = wf_h as f32 / img_h as f32;
    let scale_x = wf_w as f32 / img_w as f32;

    let image_slice = image.as_slice().expect("Image must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let wf_dz = warp_field.slice(ndarray::s![0, .., .., ..]);
    let wf_dy = warp_field.slice(ndarray::s![1, .., .., ..]);
    let wf_dx = warp_field.slice(ndarray::s![2, .., .., ..]);

    let wf_dz_slice = wf_dz.as_slice().expect("Warp field must be C-contiguous");
    let wf_dy_slice = wf_dy.as_slice().expect("Warp field must be C-contiguous");
    let wf_dx_slice = wf_dx.as_slice().expect("Warp field must be C-contiguous");

    let wf_stride_z = wf_h * wf_w;
    let wf_stride_y = wf_w;
    let img_stride_z = img_h * img_w;
    let img_stride_y = img_w;
    let cval_f32 = cval as f32;

    for oz in 0..img_d {
        for oy in 0..img_h {
            for ox in 0..img_w {
                let wf_z = oz as f32 * scale_z - 0.5;
                let wf_y = oy as f32 * scale_y - 0.5;
                let wf_x = ox as f32 * scale_x - 0.5;

                let (dz, dy, dx) = trilinear_interp_warp_field_f32(
                    wf_dz_slice,
                    wf_dy_slice,
                    wf_dx_slice,
                    wf_d,
                    wf_h,
                    wf_w,
                    wf_stride_z,
                    wf_stride_y,
                    wf_z,
                    wf_y,
                    wf_x,
                );

                let src_z = oz as f32 - dz;
                let src_y = oy as f32 - dy;
                let src_x = ox as f32 - dx;

                let value = trilinear_interp_image_warp_u8(
                    image_slice,
                    img_d,
                    img_h,
                    img_w,
                    img_stride_z,
                    img_stride_y,
                    src_z,
                    src_y,
                    src_x,
                    cval_f32,
                );

                // Clamp and convert back to u8
                output_slice[oz * img_stride_z + oy * img_stride_y + ox] =
                    value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// =============================================================================
// Warp Field Interpolation Helpers
// =============================================================================

/// Trilinearly interpolate the warp field at (z, y, x)
#[inline]
pub fn trilinear_interp_warp_field_f32(
    wf_dz: &[f32],
    wf_dy: &[f32],
    wf_dx: &[f32],
    d: usize,
    h: usize,
    w: usize,
    stride_z: usize,
    stride_y: usize,
    z: f32,
    y: f32,
    x: f32,
) -> (f32, f32, f32) {
    let z0 = z.floor() as i32;
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;

    // Clamp to valid range
    if z0 < 0 || z0 >= d as i32 - 1 || y0 < 0 || y0 >= h as i32 - 1 || x0 < 0 || x0 >= w as i32 - 1
    {
        // At boundary, clamp coordinates
        let z_clamped = z.clamp(0.0, d as f32 - 1.0);
        let y_clamped = y.clamp(0.0, h as f32 - 1.0);
        let x_clamped = x.clamp(0.0, w as f32 - 1.0);

        let z0 = z_clamped.floor() as usize;
        let y0 = y_clamped.floor() as usize;
        let x0 = x_clamped.floor() as usize;
        let z1 = (z0 + 1).min(d - 1);
        let y1 = (y0 + 1).min(h - 1);
        let x1 = (x0 + 1).min(w - 1);

        let fz = z_clamped - z_clamped.floor();
        let fy = y_clamped - y_clamped.floor();
        let fx = x_clamped - x_clamped.floor();

        return interp_8_neighbors_warp(
            wf_dz, wf_dy, wf_dx, stride_z, stride_y, z0, y0, x0, z1, y1, x1, fz, fy, fx,
        );
    }

    let z0u = z0 as usize;
    let y0u = y0 as usize;
    let x0u = x0 as usize;
    let z1u = z0u + 1;
    let y1u = y0u + 1;
    let x1u = x0u + 1;

    let fz = z - z.floor();
    let fy = y - y.floor();
    let fx = x - x.floor();

    interp_8_neighbors_warp(
        wf_dz, wf_dy, wf_dx, stride_z, stride_y, z0u, y0u, x0u, z1u, y1u, x1u, fz, fy, fx,
    )
}

#[inline]
pub fn interp_8_neighbors_warp(
    wf_dz: &[f32],
    wf_dy: &[f32],
    wf_dx: &[f32],
    stride_z: usize,
    stride_y: usize,
    z0: usize,
    y0: usize,
    x0: usize,
    z1: usize,
    y1: usize,
    x1: usize,
    fz: f32,
    fy: f32,
    fx: f32,
) -> (f32, f32, f32) {
    let idx000 = z0 * stride_z + y0 * stride_y + x0;
    let idx001 = z0 * stride_z + y0 * stride_y + x1;
    let idx010 = z0 * stride_z + y1 * stride_y + x0;
    let idx011 = z0 * stride_z + y1 * stride_y + x1;
    let idx100 = z1 * stride_z + y0 * stride_y + x0;
    let idx101 = z1 * stride_z + y0 * stride_y + x1;
    let idx110 = z1 * stride_z + y1 * stride_y + x0;
    let idx111 = z1 * stride_z + y1 * stride_y + x1;

    let w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    let w001 = fx * (1.0 - fy) * (1.0 - fz);
    let w010 = (1.0 - fx) * fy * (1.0 - fz);
    let w011 = fx * fy * (1.0 - fz);
    let w100 = (1.0 - fx) * (1.0 - fy) * fz;
    let w101 = fx * (1.0 - fy) * fz;
    let w110 = (1.0 - fx) * fy * fz;
    let w111 = fx * fy * fz;

    let dz = wf_dz[idx000] * w000
        + wf_dz[idx001] * w001
        + wf_dz[idx010] * w010
        + wf_dz[idx011] * w011
        + wf_dz[idx100] * w100
        + wf_dz[idx101] * w101
        + wf_dz[idx110] * w110
        + wf_dz[idx111] * w111;

    let dy = wf_dy[idx000] * w000
        + wf_dy[idx001] * w001
        + wf_dy[idx010] * w010
        + wf_dy[idx011] * w011
        + wf_dy[idx100] * w100
        + wf_dy[idx101] * w101
        + wf_dy[idx110] * w110
        + wf_dy[idx111] * w111;

    let dx = wf_dx[idx000] * w000
        + wf_dx[idx001] * w001
        + wf_dx[idx010] * w010
        + wf_dx[idx011] * w011
        + wf_dx[idx100] * w100
        + wf_dx[idx101] * w101
        + wf_dx[idx110] * w110
        + wf_dx[idx111] * w111;

    (dz, dy, dx)
}

/// Trilinearly interpolate the f32 image at (z, y, x) for warp
#[inline]
pub fn trilinear_interp_image_warp_f32(
    image: &[f32],
    d: usize,
    h: usize,
    w: usize,
    stride_z: usize,
    stride_y: usize,
    z: f32,
    y: f32,
    x: f32,
    cval: f32,
) -> f32 {
    // Clamp tiny negative values to 0 (floating point precision issues)
    let z = if z > -1e-5 && z < 0.0 { 0.0 } else { z };
    let y = if y > -1e-5 && y < 0.0 { 0.0 } else { y };
    let x = if x > -1e-5 && x < 0.0 { 0.0 } else { x };

    let z0 = z.floor() as i32;
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;

    // Out of bounds check
    if z0 < 0 || z0 >= d as i32 || y0 < 0 || y0 >= h as i32 || x0 < 0 || x0 >= w as i32 {
        return cval;
    }

    let z0u = z0 as usize;
    let y0u = y0 as usize;
    let x0u = x0 as usize;

    let z1u = (z0u + 1).min(d - 1);
    let y1u = (y0u + 1).min(h - 1);
    let x1u = (x0u + 1).min(w - 1);

    let fz = z - z.floor();
    let fy = y - y.floor();
    let fx = x - x.floor();

    let idx000 = z0u * stride_z + y0u * stride_y + x0u;
    let idx001 = z0u * stride_z + y0u * stride_y + x1u;
    let idx010 = z0u * stride_z + y1u * stride_y + x0u;
    let idx011 = z0u * stride_z + y1u * stride_y + x1u;
    let idx100 = z1u * stride_z + y0u * stride_y + x0u;
    let idx101 = z1u * stride_z + y0u * stride_y + x1u;
    let idx110 = z1u * stride_z + y1u * stride_y + x0u;
    let idx111 = z1u * stride_z + y1u * stride_y + x1u;

    let v000 = image[idx000];
    let v001 = image[idx001];
    let v010 = image[idx010];
    let v011 = image[idx011];
    let v100 = image[idx100];
    let v101 = image[idx101];
    let v110 = image[idx110];
    let v111 = image[idx111];

    v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + v001 * fx * (1.0 - fy) * (1.0 - fz)
        + v010 * (1.0 - fx) * fy * (1.0 - fz)
        + v011 * fx * fy * (1.0 - fz)
        + v100 * (1.0 - fx) * (1.0 - fy) * fz
        + v101 * fx * (1.0 - fy) * fz
        + v110 * (1.0 - fx) * fy * fz
        + v111 * fx * fy * fz
}

/// Trilinearly interpolate the f16 image at (z, y, x) for warp (internally uses f32)
#[inline]
pub fn trilinear_interp_image_warp_f16(
    image: &[f16],
    d: usize,
    h: usize,
    w: usize,
    stride_z: usize,
    stride_y: usize,
    z: f32,
    y: f32,
    x: f32,
    cval: f32,
) -> f32 {
    let z = if z > -1e-5 && z < 0.0 { 0.0 } else { z };
    let y = if y > -1e-5 && y < 0.0 { 0.0 } else { y };
    let x = if x > -1e-5 && x < 0.0 { 0.0 } else { x };

    let z0 = z.floor() as i32;
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;

    if z0 < 0 || z0 >= d as i32 || y0 < 0 || y0 >= h as i32 || x0 < 0 || x0 >= w as i32 {
        return cval;
    }

    let z0u = z0 as usize;
    let y0u = y0 as usize;
    let x0u = x0 as usize;

    let z1u = (z0u + 1).min(d - 1);
    let y1u = (y0u + 1).min(h - 1);
    let x1u = (x0u + 1).min(w - 1);

    let fz = z - z.floor();
    let fy = y - y.floor();
    let fx = x - x.floor();

    let idx000 = z0u * stride_z + y0u * stride_y + x0u;
    let idx001 = z0u * stride_z + y0u * stride_y + x1u;
    let idx010 = z0u * stride_z + y1u * stride_y + x0u;
    let idx011 = z0u * stride_z + y1u * stride_y + x1u;
    let idx100 = z1u * stride_z + y0u * stride_y + x0u;
    let idx101 = z1u * stride_z + y0u * stride_y + x1u;
    let idx110 = z1u * stride_z + y1u * stride_y + x0u;
    let idx111 = z1u * stride_z + y1u * stride_y + x1u;

    let v000 = image[idx000].to_f32();
    let v001 = image[idx001].to_f32();
    let v010 = image[idx010].to_f32();
    let v011 = image[idx011].to_f32();
    let v100 = image[idx100].to_f32();
    let v101 = image[idx101].to_f32();
    let v110 = image[idx110].to_f32();
    let v111 = image[idx111].to_f32();

    v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + v001 * fx * (1.0 - fy) * (1.0 - fz)
        + v010 * (1.0 - fx) * fy * (1.0 - fz)
        + v011 * fx * fy * (1.0 - fz)
        + v100 * (1.0 - fx) * (1.0 - fy) * fz
        + v101 * fx * (1.0 - fy) * fz
        + v110 * (1.0 - fx) * fy * fz
        + v111 * fx * fy * fz
}

/// Trilinearly interpolate the u8 image at (z, y, x) for warp (returns f32 for precision)
#[inline]
pub fn trilinear_interp_image_warp_u8(
    image: &[u8],
    d: usize,
    h: usize,
    w: usize,
    stride_z: usize,
    stride_y: usize,
    z: f32,
    y: f32,
    x: f32,
    cval: f32,
) -> f32 {
    let z = if z > -1e-5 && z < 0.0 { 0.0 } else { z };
    let y = if y > -1e-5 && y < 0.0 { 0.0 } else { y };
    let x = if x > -1e-5 && x < 0.0 { 0.0 } else { x };

    let z0 = z.floor() as i32;
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;

    if z0 < 0 || z0 >= d as i32 || y0 < 0 || y0 >= h as i32 || x0 < 0 || x0 >= w as i32 {
        return cval;
    }

    let z0u = z0 as usize;
    let y0u = y0 as usize;
    let x0u = x0 as usize;

    let z1u = (z0u + 1).min(d - 1);
    let y1u = (y0u + 1).min(h - 1);
    let x1u = (x0u + 1).min(w - 1);

    let fz = z - z.floor();
    let fy = y - y.floor();
    let fx = x - x.floor();

    let idx000 = z0u * stride_z + y0u * stride_y + x0u;
    let idx001 = z0u * stride_z + y0u * stride_y + x1u;
    let idx010 = z0u * stride_z + y1u * stride_y + x0u;
    let idx011 = z0u * stride_z + y1u * stride_y + x1u;
    let idx100 = z1u * stride_z + y0u * stride_y + x0u;
    let idx101 = z1u * stride_z + y0u * stride_y + x1u;
    let idx110 = z1u * stride_z + y1u * stride_y + x0u;
    let idx111 = z1u * stride_z + y1u * stride_y + x1u;

    let v000 = image[idx000] as f32;
    let v001 = image[idx001] as f32;
    let v010 = image[idx010] as f32;
    let v011 = image[idx011] as f32;
    let v100 = image[idx100] as f32;
    let v101 = image[idx101] as f32;
    let v110 = image[idx110] as f32;
    let v111 = image[idx111] as f32;

    v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + v001 * fx * (1.0 - fy) * (1.0 - fz)
        + v010 * (1.0 - fx) * fy * (1.0 - fz)
        + v011 * fx * fy * (1.0 - fz)
        + v100 * (1.0 - fx) * (1.0 - fy) * fz
        + v101 * fx * (1.0 - fy) * fz
        + v110 * (1.0 - fx) * fy * fz
        + v111 * fx * fy * fz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trilinear_interp_center() {
        let mut data = vec![0.0f32; 27];
        data[0] = 0.0;
        data[1] = 1.0;
        data[3] = 2.0;
        data[4] = 3.0;
        data[9] = 4.0;
        data[10] = 5.0;
        data[12] = 6.0;
        data[13] = 7.0;

        let result = trilinear_interp_f32(&data, 3, 3, 3, 0.5, 0.5, 0.5, 0.0);
        let expected = (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0) / 8.0;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_trilinear_interp_out_of_bounds() {
        let data = vec![1.0f32; 27];
        let cval = -999.0;

        assert_eq!(
            trilinear_interp_f32(&data, 3, 3, 3, -1.0, 0.0, 0.0, cval),
            cval
        );
        assert_eq!(
            trilinear_interp_f32(&data, 3, 3, 3, 0.0, -1.0, 0.0, cval),
            cval
        );
        assert_eq!(
            trilinear_interp_f32(&data, 3, 3, 3, 0.0, 0.0, -1.0, cval),
            cval
        );
    }
}
