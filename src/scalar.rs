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
    let m00 = m[0][0]; let m01 = m[0][1]; let m02 = m[0][2];
    let m10 = m[1][0]; let m11 = m[1][1]; let m12 = m[1][2];
    let m20 = m[2][0]; let m21 = m[2][1]; let m22 = m[2][2];

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

                        if x0 >= 0 && x0 < w as isize
                            && y0 >= 0 && y0 < h as isize
                            && z0 >= 0 && z0 < d as isize
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

                    if x0 >= 0 && x0 < w as isize
                        && y0 >= 0 && y0 < h as isize
                        && z0 >= 0 && z0 < d as isize
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
    let m00 = m[0][0] as f32; let m01 = m[0][1] as f32; let m02 = m[0][2] as f32;
    let m10 = m[1][0] as f32; let m11 = m[1][1] as f32; let m12 = m[1][2] as f32;
    let m20 = m[2][0] as f32; let m21 = m[2][1] as f32; let m22 = m[2][2] as f32;

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

                        if x0 >= 0 && x0 < w as i32
                            && y0 >= 0 && y0 < h as i32
                            && z0 >= 0 && z0 < d as i32
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

                    if x0 >= 0 && x0 < w as i32
                        && y0 >= 0 && y0 < h as i32
                        && z0 >= 0 && z0 < d as i32
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
    let m00 = m[0][0] as f32; let m01 = m[0][1] as f32; let m02 = m[0][2] as f32;
    let m10 = m[1][0] as f32; let m11 = m[1][1] as f32; let m12 = m[1][2] as f32;
    let m20 = m[2][0] as f32; let m21 = m[2][1] as f32; let m22 = m[2][2] as f32;

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

                        if x0 >= 0 && x0 < w as i32
                            && y0 >= 0 && y0 < h as i32
                            && z0 >= 0 && z0 < d as i32
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

                    if x0 >= 0 && x0 < w as i32
                        && y0 >= 0 && y0 < h as i32
                        && z0 >= 0 && z0 < d as i32
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trilinear_interp_center() {
        let mut data = vec![0.0f32; 27];
        data[0] = 0.0;  data[1] = 1.0;  data[3] = 2.0;  data[4] = 3.0;
        data[9] = 4.0;  data[10] = 5.0; data[12] = 6.0; data[13] = 7.0;

        let result = trilinear_interp_f32(&data, 3, 3, 3, 0.5, 0.5, 0.5, 0.0);
        let expected = (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0) / 8.0;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_trilinear_interp_out_of_bounds() {
        let data = vec![1.0f32; 27];
        let cval = -999.0;

        assert_eq!(trilinear_interp_f32(&data, 3, 3, 3, -1.0, 0.0, 0.0, cval), cval);
        assert_eq!(trilinear_interp_f32(&data, 3, 3, 3, 0.0, -1.0, 0.0, cval), cval);
        assert_eq!(trilinear_interp_f32(&data, 3, 3, 3, 0.0, 0.0, -1.0, cval), cval);
    }
}
