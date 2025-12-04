//! AVX512-optimized 3D trilinear interpolation
//!
//! This module provides high-performance implementations of 3D interpolation
//! using AVX512 SIMD instructions. Supported data types:
//! - f32: 16 values per iteration (512-bit registers)
//! - f16: 16 values per iteration (with conversion)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::AffineMatrix3D;
use half::f16;
use ndarray::{ArrayView3, ArrayViewMut3};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// =============================================================================
// F32 IMPLEMENTATION
// =============================================================================

/// AVX512 optimized 3D trilinear interpolation for f32
///
/// Processes 16 voxels at a time using AVX512 512-bit registers.
///
/// # Safety
///
/// Requires AVX512F CPU feature. Caller must verify this is available.
#[target_feature(enable = "avx512f")]
pub unsafe fn trilinear_3d_f32_avx512(
    input: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    if d < 2 || h < 2 || w < 2 {
        return;
    }

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let in_stride_z = h * w;
    let in_stride_y = w;

    let m = matrix.as_flat_f32();
    let m00 = m[0];
    let m01 = m[1];
    let m02 = m[2];
    let m10 = m[3];
    let m11 = m[4];
    let m12 = m[5];
    let m20 = m[6];
    let m21 = m[7];
    let m22 = m[8];

    let shift_z = shift[0] as f32;
    let shift_y = shift[1] as f32;
    let shift_x = shift[2] as f32;
    let cval_f32 = cval as f32;

    let chunk_size = oh * ow;

    #[cfg(feature = "parallel")]
    {
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                process_z_slice_f32_avx512(
                    input_slice,
                    slice_z,
                    oz,
                    oh,
                    ow,
                    d,
                    h,
                    w,
                    in_stride_z,
                    in_stride_y,
                    m00,
                    m01,
                    m02,
                    m10,
                    m11,
                    m12,
                    m20,
                    m21,
                    m22,
                    shift_z,
                    shift_y,
                    shift_x,
                    cval_f32,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (oz, slice_z) in output_slice.chunks_mut(chunk_size).enumerate() {
            process_z_slice_f32_avx512(
                input_slice,
                slice_z,
                oz,
                oh,
                ow,
                d,
                h,
                w,
                in_stride_z,
                in_stride_y,
                m00,
                m01,
                m02,
                m10,
                m11,
                m12,
                m20,
                m21,
                m22,
                shift_z,
                shift_y,
                shift_x,
                cval_f32,
            );
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn process_z_slice_f32_avx512(
    input_slice: &[f32],
    slice_z: &mut [f32],
    oz: usize,
    oh: usize,
    ow: usize,
    d: usize,
    h: usize,
    w: usize,
    in_stride_z: usize,
    in_stride_y: usize,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
    shift_z: f32,
    shift_y: f32,
    shift_x: f32,
    cval_f32: f32,
) {
    let one = _mm512_set1_ps(1.0);
    let oz_f = oz as f32;

    for oy in 0..oh {
        let oy_f = oy as f32;

        let base_z = m00 * oz_f + m01 * oy_f + shift_z;
        let base_y = m10 * oz_f + m11 * oy_f + shift_y;
        let base_x = m20 * oz_f + m21 * oy_f + shift_x;

        let row_start = oy * ow;
        let mut ox = 0usize;

        // AVX512 loop - process 16 voxels at a time
        while ox + 15 < ow {
            let vx = _mm512_setr_ps(
                ox as f32,
                (ox + 1) as f32,
                (ox + 2) as f32,
                (ox + 3) as f32,
                (ox + 4) as f32,
                (ox + 5) as f32,
                (ox + 6) as f32,
                (ox + 7) as f32,
                (ox + 8) as f32,
                (ox + 9) as f32,
                (ox + 10) as f32,
                (ox + 11) as f32,
                (ox + 12) as f32,
                (ox + 13) as f32,
                (ox + 14) as f32,
                (ox + 15) as f32,
            );

            let zs = _mm512_fmadd_ps(_mm512_set1_ps(m02), vx, _mm512_set1_ps(base_z));
            let ys = _mm512_fmadd_ps(_mm512_set1_ps(m12), vx, _mm512_set1_ps(base_y));
            let xs = _mm512_fmadd_ps(_mm512_set1_ps(m22), vx, _mm512_set1_ps(base_x));

            let z_floor = _mm512_roundscale_ps::<0x01>(zs);
            let y_floor = _mm512_roundscale_ps::<0x01>(ys);
            let x_floor = _mm512_roundscale_ps::<0x01>(xs);

            let fz = _mm512_sub_ps(zs, z_floor);
            let fy = _mm512_sub_ps(ys, y_floor);
            let fx = _mm512_sub_ps(xs, x_floor);

            let one_minus_fx = _mm512_sub_ps(one, fx);
            let one_minus_fy = _mm512_sub_ps(one, fy);
            let one_minus_fz = _mm512_sub_ps(one, fz);

            let w000 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, one_minus_fy), one_minus_fz);
            let w001 = _mm512_mul_ps(_mm512_mul_ps(fx, one_minus_fy), one_minus_fz);
            let w010 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, fy), one_minus_fz);
            let w011 = _mm512_mul_ps(_mm512_mul_ps(fx, fy), one_minus_fz);
            let w100 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, one_minus_fy), fz);
            let w101 = _mm512_mul_ps(_mm512_mul_ps(fx, one_minus_fy), fz);
            let w110 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, fy), fz);
            let w111 = _mm512_mul_ps(_mm512_mul_ps(fx, fy), fz);

            let zi = _mm512_cvttps_epi32(z_floor);
            let yi = _mm512_cvttps_epi32(y_floor);
            let xi = _mm512_cvttps_epi32(x_floor);

            let mut xi_arr = [0i32; 16];
            let mut yi_arr = [0i32; 16];
            let mut zi_arr = [0i32; 16];
            let mut weights = [[0.0f32; 16]; 8];

            _mm512_storeu_si512(zi_arr.as_mut_ptr() as *mut __m512i, zi);
            _mm512_storeu_si512(yi_arr.as_mut_ptr() as *mut __m512i, yi);
            _mm512_storeu_si512(xi_arr.as_mut_ptr() as *mut __m512i, xi);

            _mm512_storeu_ps(weights[0].as_mut_ptr(), w000);
            _mm512_storeu_ps(weights[1].as_mut_ptr(), w001);
            _mm512_storeu_ps(weights[2].as_mut_ptr(), w010);
            _mm512_storeu_ps(weights[3].as_mut_ptr(), w011);
            _mm512_storeu_ps(weights[4].as_mut_ptr(), w100);
            _mm512_storeu_ps(weights[5].as_mut_ptr(), w101);
            _mm512_storeu_ps(weights[6].as_mut_ptr(), w110);
            _mm512_storeu_ps(weights[7].as_mut_ptr(), w111);

            let mut result = [0.0f32; 16];

            for j in 0..16 {
                let z0 = zi_arr[j];
                let y0 = yi_arr[j];
                let x0 = xi_arr[j];

                if x0 >= 0 && x0 < w as i32 && y0 >= 0 && y0 < h as i32 && z0 >= 0 && z0 < d as i32
                {
                    let z0u = z0 as usize;
                    let y0u = y0 as usize;
                    let x0u = x0 as usize;

                    // Clamp +1 indices to handle boundary
                    let z1u = (z0u + 1).min(d - 1);
                    let y1u = (y0u + 1).min(h - 1);
                    let x1u = (x0u + 1).min(w - 1);

                    let idx000 = z0u * in_stride_z + y0u * in_stride_y + x0u;
                    let idx001 = z0u * in_stride_z + y0u * in_stride_y + x1u;
                    let idx010 = z0u * in_stride_z + y1u * in_stride_y + x0u;
                    let idx011 = z0u * in_stride_z + y1u * in_stride_y + x1u;
                    let idx100 = z1u * in_stride_z + y0u * in_stride_y + x0u;
                    let idx101 = z1u * in_stride_z + y0u * in_stride_y + x1u;
                    let idx110 = z1u * in_stride_z + y1u * in_stride_y + x0u;
                    let idx111 = z1u * in_stride_z + y1u * in_stride_y + x1u;

                    let v000 = input_slice[idx000];
                    let v001 = input_slice[idx001];
                    let v010 = input_slice[idx010];
                    let v011 = input_slice[idx011];
                    let v100 = input_slice[idx100];
                    let v101 = input_slice[idx101];
                    let v110 = input_slice[idx110];
                    let v111 = input_slice[idx111];

                    result[j] = v000 * weights[0][j]
                        + v001 * weights[1][j]
                        + v010 * weights[2][j]
                        + v011 * weights[3][j]
                        + v100 * weights[4][j]
                        + v101 * weights[5][j]
                        + v110 * weights[6][j]
                        + v111 * weights[7][j];
                } else {
                    result[j] = cval_f32;
                }
            }

            let vresult = _mm512_loadu_ps(result.as_ptr());
            _mm512_storeu_ps(slice_z[row_start + ox..].as_mut_ptr(), vresult);

            ox += 16;
        }

        // Scalar cleanup
        while ox < ow {
            let ox_f = ox as f32;
            let z_src = m02 * ox_f + base_z;
            let y_src = m12 * ox_f + base_y;
            let x_src = m22 * ox_f + base_x;

            let z0 = z_src.floor() as i32;
            let y0 = y_src.floor() as i32;
            let x0 = x_src.floor() as i32;

            if x0 >= 0 && x0 < w as i32 && y0 >= 0 && y0 < h as i32 && z0 >= 0 && z0 < d as i32 {
                let z0u = z0 as usize;
                let y0u = y0 as usize;
                let x0u = x0 as usize;

                // Clamp +1 indices to handle boundary
                let z1u = (z0u + 1).min(d - 1);
                let y1u = (y0u + 1).min(h - 1);
                let x1u = (x0u + 1).min(w - 1);

                let fx = x_src - x_src.floor();
                let fy = y_src - y_src.floor();
                let fz = z_src - z_src.floor();

                let idx000 = z0u * in_stride_z + y0u * in_stride_y + x0u;
                let idx001 = z0u * in_stride_z + y0u * in_stride_y + x1u;
                let idx010 = z0u * in_stride_z + y1u * in_stride_y + x0u;
                let idx011 = z0u * in_stride_z + y1u * in_stride_y + x1u;
                let idx100 = z1u * in_stride_z + y0u * in_stride_y + x0u;
                let idx101 = z1u * in_stride_z + y0u * in_stride_y + x1u;
                let idx110 = z1u * in_stride_z + y1u * in_stride_y + x0u;
                let idx111 = z1u * in_stride_z + y1u * in_stride_y + x1u;

                let v000 = input_slice[idx000];
                let v001 = input_slice[idx001];
                let v010 = input_slice[idx010];
                let v011 = input_slice[idx011];
                let v100 = input_slice[idx100];
                let v101 = input_slice[idx101];
                let v110 = input_slice[idx110];
                let v111 = input_slice[idx111];

                slice_z[row_start + ox] = v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
                    + v001 * fx * (1.0 - fy) * (1.0 - fz)
                    + v010 * (1.0 - fx) * fy * (1.0 - fz)
                    + v011 * fx * fy * (1.0 - fz)
                    + v100 * (1.0 - fx) * (1.0 - fy) * fz
                    + v101 * fx * (1.0 - fy) * fz
                    + v110 * (1.0 - fx) * fy * fz
                    + v111 * fx * fy * fz;
            } else {
                slice_z[row_start + ox] = cval_f32;
            }

            ox += 1;
        }
    }
}

// =============================================================================
// F16 IMPLEMENTATION
// =============================================================================

/// AVX512 optimized 3D trilinear interpolation for f16 (half precision)
///
/// Processes 16 voxels at a time. Computation done in f32.
///
/// # Safety
///
/// Requires AVX512F CPU feature.
#[target_feature(enable = "avx512f")]
pub unsafe fn trilinear_3d_f16_avx512(
    input: &ArrayView3<f16>,
    output: &mut ArrayViewMut3<f16>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    if d < 2 || h < 2 || w < 2 {
        return;
    }

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    let in_stride_z = h * w;
    let in_stride_y = w;

    let m = matrix.as_flat_f32();
    let m00 = m[0];
    let m01 = m[1];
    let m02 = m[2];
    let m10 = m[3];
    let m11 = m[4];
    let m12 = m[5];
    let m20 = m[6];
    let m21 = m[7];
    let m22 = m[8];

    let shift_z = shift[0] as f32;
    let shift_y = shift[1] as f32;
    let shift_x = shift[2] as f32;
    let cval_f16 = f16::from_f64(cval);

    let chunk_size = oh * ow;

    #[cfg(feature = "parallel")]
    {
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                process_z_slice_f16_avx512(
                    input_slice,
                    slice_z,
                    oz,
                    oh,
                    ow,
                    d,
                    h,
                    w,
                    in_stride_z,
                    in_stride_y,
                    m00,
                    m01,
                    m02,
                    m10,
                    m11,
                    m12,
                    m20,
                    m21,
                    m22,
                    shift_z,
                    shift_y,
                    shift_x,
                    cval_f16,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (oz, slice_z) in output_slice.chunks_mut(chunk_size).enumerate() {
            process_z_slice_f16_avx512(
                input_slice,
                slice_z,
                oz,
                oh,
                ow,
                d,
                h,
                w,
                in_stride_z,
                in_stride_y,
                m00,
                m01,
                m02,
                m10,
                m11,
                m12,
                m20,
                m21,
                m22,
                shift_z,
                shift_y,
                shift_x,
                cval_f16,
            );
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn process_z_slice_f16_avx512(
    input_slice: &[f16],
    slice_z: &mut [f16],
    oz: usize,
    oh: usize,
    ow: usize,
    d: usize,
    h: usize,
    w: usize,
    in_stride_z: usize,
    in_stride_y: usize,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
    shift_z: f32,
    shift_y: f32,
    shift_x: f32,
    cval_f16: f16,
) {
    let one = _mm512_set1_ps(1.0);
    let oz_f = oz as f32;

    for oy in 0..oh {
        let oy_f = oy as f32;

        let base_z = m00 * oz_f + m01 * oy_f + shift_z;
        let base_y = m10 * oz_f + m11 * oy_f + shift_y;
        let base_x = m20 * oz_f + m21 * oy_f + shift_x;

        let row_start = oy * ow;
        let mut ox = 0usize;

        while ox + 15 < ow {
            let vx = _mm512_setr_ps(
                ox as f32,
                (ox + 1) as f32,
                (ox + 2) as f32,
                (ox + 3) as f32,
                (ox + 4) as f32,
                (ox + 5) as f32,
                (ox + 6) as f32,
                (ox + 7) as f32,
                (ox + 8) as f32,
                (ox + 9) as f32,
                (ox + 10) as f32,
                (ox + 11) as f32,
                (ox + 12) as f32,
                (ox + 13) as f32,
                (ox + 14) as f32,
                (ox + 15) as f32,
            );

            let zs = _mm512_fmadd_ps(_mm512_set1_ps(m02), vx, _mm512_set1_ps(base_z));
            let ys = _mm512_fmadd_ps(_mm512_set1_ps(m12), vx, _mm512_set1_ps(base_y));
            let xs = _mm512_fmadd_ps(_mm512_set1_ps(m22), vx, _mm512_set1_ps(base_x));

            let z_floor = _mm512_roundscale_ps::<0x01>(zs);
            let y_floor = _mm512_roundscale_ps::<0x01>(ys);
            let x_floor = _mm512_roundscale_ps::<0x01>(xs);

            let fz = _mm512_sub_ps(zs, z_floor);
            let fy = _mm512_sub_ps(ys, y_floor);
            let fx = _mm512_sub_ps(xs, x_floor);

            let one_minus_fx = _mm512_sub_ps(one, fx);
            let one_minus_fy = _mm512_sub_ps(one, fy);
            let one_minus_fz = _mm512_sub_ps(one, fz);

            let w000 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, one_minus_fy), one_minus_fz);
            let w001 = _mm512_mul_ps(_mm512_mul_ps(fx, one_minus_fy), one_minus_fz);
            let w010 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, fy), one_minus_fz);
            let w011 = _mm512_mul_ps(_mm512_mul_ps(fx, fy), one_minus_fz);
            let w100 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, one_minus_fy), fz);
            let w101 = _mm512_mul_ps(_mm512_mul_ps(fx, one_minus_fy), fz);
            let w110 = _mm512_mul_ps(_mm512_mul_ps(one_minus_fx, fy), fz);
            let w111 = _mm512_mul_ps(_mm512_mul_ps(fx, fy), fz);

            let zi = _mm512_cvttps_epi32(z_floor);
            let yi = _mm512_cvttps_epi32(y_floor);
            let xi = _mm512_cvttps_epi32(x_floor);

            let mut xi_arr = [0i32; 16];
            let mut yi_arr = [0i32; 16];
            let mut zi_arr = [0i32; 16];
            let mut weights = [[0.0f32; 16]; 8];

            _mm512_storeu_si512(zi_arr.as_mut_ptr() as *mut __m512i, zi);
            _mm512_storeu_si512(yi_arr.as_mut_ptr() as *mut __m512i, yi);
            _mm512_storeu_si512(xi_arr.as_mut_ptr() as *mut __m512i, xi);

            _mm512_storeu_ps(weights[0].as_mut_ptr(), w000);
            _mm512_storeu_ps(weights[1].as_mut_ptr(), w001);
            _mm512_storeu_ps(weights[2].as_mut_ptr(), w010);
            _mm512_storeu_ps(weights[3].as_mut_ptr(), w011);
            _mm512_storeu_ps(weights[4].as_mut_ptr(), w100);
            _mm512_storeu_ps(weights[5].as_mut_ptr(), w101);
            _mm512_storeu_ps(weights[6].as_mut_ptr(), w110);
            _mm512_storeu_ps(weights[7].as_mut_ptr(), w111);

            let mut result = [0.0f32; 16];

            for j in 0..16 {
                let z0 = zi_arr[j];
                let y0 = yi_arr[j];
                let x0 = xi_arr[j];

                if x0 >= 0 && x0 < w as i32 && y0 >= 0 && y0 < h as i32 && z0 >= 0 && z0 < d as i32
                {
                    let z0u = z0 as usize;
                    let y0u = y0 as usize;
                    let x0u = x0 as usize;

                    // Clamp +1 indices to handle boundary
                    let z1u = (z0u + 1).min(d - 1);
                    let y1u = (y0u + 1).min(h - 1);
                    let x1u = (x0u + 1).min(w - 1);

                    let idx000 = z0u * in_stride_z + y0u * in_stride_y + x0u;
                    let idx001 = z0u * in_stride_z + y0u * in_stride_y + x1u;
                    let idx010 = z0u * in_stride_z + y1u * in_stride_y + x0u;
                    let idx011 = z0u * in_stride_z + y1u * in_stride_y + x1u;
                    let idx100 = z1u * in_stride_z + y0u * in_stride_y + x0u;
                    let idx101 = z1u * in_stride_z + y0u * in_stride_y + x1u;
                    let idx110 = z1u * in_stride_z + y1u * in_stride_y + x0u;
                    let idx111 = z1u * in_stride_z + y1u * in_stride_y + x1u;

                    let v000 = input_slice[idx000].to_f32();
                    let v001 = input_slice[idx001].to_f32();
                    let v010 = input_slice[idx010].to_f32();
                    let v011 = input_slice[idx011].to_f32();
                    let v100 = input_slice[idx100].to_f32();
                    let v101 = input_slice[idx101].to_f32();
                    let v110 = input_slice[idx110].to_f32();
                    let v111 = input_slice[idx111].to_f32();

                    result[j] = v000 * weights[0][j]
                        + v001 * weights[1][j]
                        + v010 * weights[2][j]
                        + v011 * weights[3][j]
                        + v100 * weights[4][j]
                        + v101 * weights[5][j]
                        + v110 * weights[6][j]
                        + v111 * weights[7][j];
                } else {
                    result[j] = cval_f16.to_f32();
                }
            }

            let vresult = _mm512_loadu_ps(result.as_ptr());
            let vresult_f16 = _mm512_cvtps_ph::<0>(vresult);
            _mm256_storeu_si256(
                slice_z[row_start + ox..].as_mut_ptr() as *mut __m256i,
                vresult_f16,
            );

            ox += 16;
        }

        // Scalar cleanup
        while ox < ow {
            let ox_f = ox as f32;
            let z_src = m02 * ox_f + base_z;
            let y_src = m12 * ox_f + base_y;
            let x_src = m22 * ox_f + base_x;

            let z0 = z_src.floor() as i32;
            let y0 = y_src.floor() as i32;
            let x0 = x_src.floor() as i32;

            if x0 >= 0 && x0 < w as i32 && y0 >= 0 && y0 < h as i32 && z0 >= 0 && z0 < d as i32 {
                let z0u = z0 as usize;
                let y0u = y0 as usize;
                let x0u = x0 as usize;

                // Clamp +1 indices to handle boundary
                let z1u = (z0u + 1).min(d - 1);
                let y1u = (y0u + 1).min(h - 1);
                let x1u = (x0u + 1).min(w - 1);

                let fx = x_src - x_src.floor();
                let fy = y_src - y_src.floor();
                let fz = z_src - z_src.floor();

                let idx000 = z0u * in_stride_z + y0u * in_stride_y + x0u;
                let idx001 = z0u * in_stride_z + y0u * in_stride_y + x1u;
                let idx010 = z0u * in_stride_z + y1u * in_stride_y + x0u;
                let idx011 = z0u * in_stride_z + y1u * in_stride_y + x1u;
                let idx100 = z1u * in_stride_z + y0u * in_stride_y + x0u;
                let idx101 = z1u * in_stride_z + y0u * in_stride_y + x1u;
                let idx110 = z1u * in_stride_z + y1u * in_stride_y + x0u;
                let idx111 = z1u * in_stride_z + y1u * in_stride_y + x1u;

                let v000 = input_slice[idx000].to_f32();
                let v001 = input_slice[idx001].to_f32();
                let v010 = input_slice[idx010].to_f32();
                let v011 = input_slice[idx011].to_f32();
                let v100 = input_slice[idx100].to_f32();
                let v101 = input_slice[idx101].to_f32();
                let v110 = input_slice[idx110].to_f32();
                let v111 = input_slice[idx111].to_f32();

                let result_f32 = v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
                    + v001 * fx * (1.0 - fy) * (1.0 - fz)
                    + v010 * (1.0 - fx) * fy * (1.0 - fz)
                    + v011 * fx * fy * (1.0 - fz)
                    + v100 * (1.0 - fx) * (1.0 - fy) * fz
                    + v101 * fx * (1.0 - fy) * fz
                    + v110 * (1.0 - fx) * fy * fz
                    + v111 * fx * fy * fz;
                slice_z[row_start + ox] = f16::from_f32(result_f32);
            } else {
                slice_z[row_start + ox] = cval_f16;
            }

            ox += 1;
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_avx512_identity_f32() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX512F not available, skipping test");
            return;
        }

        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let mut output = Array3::from_elem((20, 20, 20), 0.0f32);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        unsafe {
            trilinear_3d_f32_avx512(&input.view(), &mut output.view_mut(), &matrix, &shift, 0.0);
        }

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_avx512_identity_f16() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX512F not available, skipping test");
            return;
        }

        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| {
            f16::from_f32(((z * 100 + y * 10 + x) % 256) as f32 / 255.0)
        });
        let mut output = Array3::from_elem((20, 20, 20), f16::ZERO);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        unsafe {
            trilinear_3d_f16_avx512(&input.view(), &mut output.view_mut(), &matrix, &shift, 0.0);
        }

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    let expected = input[[z, y, x]].to_f32();
                    let actual = output[[z, y, x]].to_f32();
                    assert!((actual - expected).abs() < 1e-2);
                }
            }
        }
    }
}
