//! AVX2-optimized 3D trilinear interpolation
//!
//! This module provides high-performance implementations of 3D interpolation
//! using AVX2 SIMD instructions. It processes:
//! - 4 f64 values per iteration (256-bit registers)
//! - 8 f32 values per iteration (256-bit registers)
//! - 8 f16 values per iteration (with F16C conversion)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::AffineMatrix3D;
use half::f16;
use ndarray::{ArrayView3, ArrayViewMut3};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// AVX2 optimized 3D trilinear interpolation for f32
///
/// Processes 8 voxels at a time using AVX2 256-bit registers.
///
/// # Safety
///
/// Requires AVX2 and FMA CPU features. Caller must verify these are available.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn trilinear_3d_f32_avx2(
    input: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    // Reject tiny volumes
    if d < 2 || h < 2 || w < 2 {
        return;
    }

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    // Input strides (assuming C-contiguous)
    let in_stride_z = h * w;
    let in_stride_y = w;

    // Matrix elements as f32
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

    // Process z-slices in parallel
    let chunk_size = oh * ow;

    #[cfg(feature = "parallel")]
    {
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                process_z_slice_f32(
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
            process_z_slice_f32(
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

#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn process_z_slice_f32(
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
    let one = _mm256_set1_ps(1.0);
    let oz_f = oz as f32;

    for oy in 0..oh {
        let oy_f = oy as f32;

        // Base source coordinates for this row
        let base_z = m00 * oz_f + m01 * oy_f + shift_z;
        let base_y = m10 * oz_f + m11 * oy_f + shift_y;
        let base_x = m20 * oz_f + m21 * oy_f + shift_x;

        let row_start = oy * ow;
        let mut ox = 0usize;

        // AVX2 loop - process 8 voxels at a time
        while ox + 7 < ow {
            // Generate x indices [ox, ox+1, ..., ox+7]
            let vx = _mm256_setr_ps(
                ox as f32,
                (ox + 1) as f32,
                (ox + 2) as f32,
                (ox + 3) as f32,
                (ox + 4) as f32,
                (ox + 5) as f32,
                (ox + 6) as f32,
                (ox + 7) as f32,
            );

            // Compute source coordinates: matrix * [oz, oy, ox] + shift
            let zs = _mm256_fmadd_ps(_mm256_set1_ps(m02), vx, _mm256_set1_ps(base_z));
            let ys = _mm256_fmadd_ps(_mm256_set1_ps(m12), vx, _mm256_set1_ps(base_y));
            let xs = _mm256_fmadd_ps(_mm256_set1_ps(m22), vx, _mm256_set1_ps(base_x));

            // Floor to get integer coordinates
            let z_floor = _mm256_floor_ps(zs);
            let y_floor = _mm256_floor_ps(ys);
            let x_floor = _mm256_floor_ps(xs);

            // Get fractional parts
            let fz = _mm256_sub_ps(zs, z_floor);
            let fy = _mm256_sub_ps(ys, y_floor);
            let fx = _mm256_sub_ps(xs, x_floor);

            // Compute trilinear weights (8 corners)
            let one_minus_fx = _mm256_sub_ps(one, fx);
            let one_minus_fy = _mm256_sub_ps(one, fy);
            let one_minus_fz = _mm256_sub_ps(one, fz);

            let w000 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), one_minus_fz);
            let w001 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), one_minus_fz);
            let w010 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), one_minus_fz);
            let w011 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), one_minus_fz);
            let w100 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), fz);
            let w101 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), fz);
            let w110 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), fz);
            let w111 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), fz);

            // Convert to integers
            let zi = _mm256_cvttps_epi32(z_floor);
            let yi = _mm256_cvttps_epi32(y_floor);
            let xi = _mm256_cvttps_epi32(x_floor);

            // Extract values to arrays
            let mut xi_arr = [0i32; 8];
            let mut yi_arr = [0i32; 8];
            let mut zi_arr = [0i32; 8];
            let mut weights = [[0.0f32; 8]; 8];

            _mm256_storeu_si256(zi_arr.as_mut_ptr() as *mut __m256i, zi);
            _mm256_storeu_si256(yi_arr.as_mut_ptr() as *mut __m256i, yi);
            _mm256_storeu_si256(xi_arr.as_mut_ptr() as *mut __m256i, xi);

            _mm256_storeu_ps(weights[0].as_mut_ptr(), w000);
            _mm256_storeu_ps(weights[1].as_mut_ptr(), w001);
            _mm256_storeu_ps(weights[2].as_mut_ptr(), w010);
            _mm256_storeu_ps(weights[3].as_mut_ptr(), w011);
            _mm256_storeu_ps(weights[4].as_mut_ptr(), w100);
            _mm256_storeu_ps(weights[5].as_mut_ptr(), w101);
            _mm256_storeu_ps(weights[6].as_mut_ptr(), w110);
            _mm256_storeu_ps(weights[7].as_mut_ptr(), w111);

            let mut result = [0.0f32; 8];

            // Process each voxel
            for j in 0..8 {
                let z0 = zi_arr[j];
                let y0 = yi_arr[j];
                let x0 = xi_arr[j];

                // Bounds check
                if x0 >= 0
                    && x0 < (w - 1) as i32
                    && y0 >= 0
                    && y0 < (h - 1) as i32
                    && z0 >= 0
                    && z0 < (d - 1) as i32
                {
                    // Compute indices for 8 corners
                    let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                    // Load 8 corner values
                    let v000 = input_slice[idx000];
                    let v001 = input_slice[idx000 + 1];
                    let v010 = input_slice[idx000 + in_stride_y];
                    let v011 = input_slice[idx000 + in_stride_y + 1];
                    let v100 = input_slice[idx000 + in_stride_z];
                    let v101 = input_slice[idx000 + in_stride_z + 1];
                    let v110 = input_slice[idx000 + in_stride_z + in_stride_y];
                    let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1];

                    // Trilinear interpolation
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

            // Store results
            let vresult = _mm256_loadu_ps(result.as_ptr());
            _mm256_storeu_ps(slice_z[row_start + ox..].as_mut_ptr(), vresult);

            ox += 8;
        }

        // Scalar cleanup for remaining voxels
        while ox < ow {
            let ox_f = ox as f32;
            let z_src = m02 * ox_f + base_z;
            let y_src = m12 * ox_f + base_y;
            let x_src = m22 * ox_f + base_x;

            let z0 = z_src.floor() as i32;
            let y0 = y_src.floor() as i32;
            let x0 = x_src.floor() as i32;

            if x0 >= 0
                && x0 < (w - 1) as i32
                && y0 >= 0
                && y0 < (h - 1) as i32
                && z0 >= 0
                && z0 < (d - 1) as i32
            {
                let fx = x_src - x_src.floor();
                let fy = y_src - y_src.floor();
                let fz = z_src - z_src.floor();

                let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                let v000 = input_slice[idx000];
                let v001 = input_slice[idx000 + 1];
                let v010 = input_slice[idx000 + in_stride_y];
                let v011 = input_slice[idx000 + in_stride_y + 1];
                let v100 = input_slice[idx000 + in_stride_z];
                let v101 = input_slice[idx000 + in_stride_z + 1];
                let v110 = input_slice[idx000 + in_stride_z + in_stride_y];
                let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1];

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

/// AVX2 optimized 3D trilinear interpolation for f64
///
/// Processes 4 voxels at a time using AVX2 256-bit registers.
///
/// # Safety
///
/// Requires AVX2 and FMA CPU features. Caller must verify these are available.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn trilinear_3d_f64_avx2(
    input: &ArrayView3<f64>,
    output: &mut ArrayViewMut3<f64>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    // Reject tiny volumes
    if d < 2 || h < 2 || w < 2 {
        return;
    }

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    // Input strides (assuming C-contiguous)
    let in_stride_z = h * w;
    let in_stride_y = w;

    // Matrix elements
    let m = matrix.as_flat();
    let m00 = m[0];
    let m01 = m[1];
    let m02 = m[2];
    let m10 = m[3];
    let m11 = m[4];
    let m12 = m[5];
    let m20 = m[6];
    let m21 = m[7];
    let m22 = m[8];

    let shift_z = shift[0];
    let shift_y = shift[1];
    let shift_x = shift[2];

    // Process z-slices in parallel
    let chunk_size = oh * ow;

    #[cfg(feature = "parallel")]
    {
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                process_z_slice_f64(
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
                    cval,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (oz, slice_z) in output_slice.chunks_mut(chunk_size).enumerate() {
            process_z_slice_f64(
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
                cval,
            );
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn process_z_slice_f64(
    input_slice: &[f64],
    slice_z: &mut [f64],
    oz: usize,
    oh: usize,
    ow: usize,
    d: usize,
    h: usize,
    w: usize,
    in_stride_z: usize,
    in_stride_y: usize,
    m00: f64,
    m01: f64,
    m02: f64,
    m10: f64,
    m11: f64,
    m12: f64,
    m20: f64,
    m21: f64,
    m22: f64,
    shift_z: f64,
    shift_y: f64,
    shift_x: f64,
    cval: f64,
) {
    let one = _mm256_set1_pd(1.0);
    let oz_f = oz as f64;

    for oy in 0..oh {
        let oy_f = oy as f64;

        let base_z = m00 * oz_f + m01 * oy_f + shift_z;
        let base_y = m10 * oz_f + m11 * oy_f + shift_y;
        let base_x = m20 * oz_f + m21 * oy_f + shift_x;

        let row_start = oy * ow;
        let mut ox = 0usize;

        // AVX2 loop - process 4 voxels at a time
        while ox + 3 < ow {
            let vx = _mm256_setr_pd(
                ox as f64,
                (ox + 1) as f64,
                (ox + 2) as f64,
                (ox + 3) as f64,
            );

            let zs = _mm256_fmadd_pd(_mm256_set1_pd(m02), vx, _mm256_set1_pd(base_z));
            let ys = _mm256_fmadd_pd(_mm256_set1_pd(m12), vx, _mm256_set1_pd(base_y));
            let xs = _mm256_fmadd_pd(_mm256_set1_pd(m22), vx, _mm256_set1_pd(base_x));

            let z_floor = _mm256_floor_pd(zs);
            let y_floor = _mm256_floor_pd(ys);
            let x_floor = _mm256_floor_pd(xs);

            let fz = _mm256_sub_pd(zs, z_floor);
            let fy = _mm256_sub_pd(ys, y_floor);
            let fx = _mm256_sub_pd(xs, x_floor);

            let one_minus_fx = _mm256_sub_pd(one, fx);
            let one_minus_fy = _mm256_sub_pd(one, fy);
            let one_minus_fz = _mm256_sub_pd(one, fz);

            let w000 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, one_minus_fy), one_minus_fz);
            let w001 = _mm256_mul_pd(_mm256_mul_pd(fx, one_minus_fy), one_minus_fz);
            let w010 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, fy), one_minus_fz);
            let w011 = _mm256_mul_pd(_mm256_mul_pd(fx, fy), one_minus_fz);
            let w100 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, one_minus_fy), fz);
            let w101 = _mm256_mul_pd(_mm256_mul_pd(fx, one_minus_fy), fz);
            let w110 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, fy), fz);
            let w111 = _mm256_mul_pd(_mm256_mul_pd(fx, fy), fz);

            // Convert to integers - for f64 we use cvttpd_epi32 which gives 128-bit result
            let zi = _mm256_cvttpd_epi32(z_floor);
            let yi = _mm256_cvttpd_epi32(y_floor);
            let xi = _mm256_cvttpd_epi32(x_floor);

            let mut xi_arr = [0i32; 4];
            let mut yi_arr = [0i32; 4];
            let mut zi_arr = [0i32; 4];
            let mut weights = [[0.0f64; 4]; 8];

            _mm_storeu_si128(zi_arr.as_mut_ptr() as *mut __m128i, zi);
            _mm_storeu_si128(yi_arr.as_mut_ptr() as *mut __m128i, yi);
            _mm_storeu_si128(xi_arr.as_mut_ptr() as *mut __m128i, xi);

            _mm256_storeu_pd(weights[0].as_mut_ptr(), w000);
            _mm256_storeu_pd(weights[1].as_mut_ptr(), w001);
            _mm256_storeu_pd(weights[2].as_mut_ptr(), w010);
            _mm256_storeu_pd(weights[3].as_mut_ptr(), w011);
            _mm256_storeu_pd(weights[4].as_mut_ptr(), w100);
            _mm256_storeu_pd(weights[5].as_mut_ptr(), w101);
            _mm256_storeu_pd(weights[6].as_mut_ptr(), w110);
            _mm256_storeu_pd(weights[7].as_mut_ptr(), w111);

            let mut result = [0.0f64; 4];

            for j in 0..4 {
                let z0 = zi_arr[j];
                let y0 = yi_arr[j];
                let x0 = xi_arr[j];

                if x0 >= 0
                    && x0 < (w - 1) as i32
                    && y0 >= 0
                    && y0 < (h - 1) as i32
                    && z0 >= 0
                    && z0 < (d - 1) as i32
                {
                    let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                    let v000 = input_slice[idx000];
                    let v001 = input_slice[idx000 + 1];
                    let v010 = input_slice[idx000 + in_stride_y];
                    let v011 = input_slice[idx000 + in_stride_y + 1];
                    let v100 = input_slice[idx000 + in_stride_z];
                    let v101 = input_slice[idx000 + in_stride_z + 1];
                    let v110 = input_slice[idx000 + in_stride_z + in_stride_y];
                    let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1];

                    result[j] = v000 * weights[0][j]
                        + v001 * weights[1][j]
                        + v010 * weights[2][j]
                        + v011 * weights[3][j]
                        + v100 * weights[4][j]
                        + v101 * weights[5][j]
                        + v110 * weights[6][j]
                        + v111 * weights[7][j];
                } else {
                    result[j] = cval;
                }
            }

            let vresult = _mm256_loadu_pd(result.as_ptr());
            _mm256_storeu_pd(slice_z[row_start + ox..].as_mut_ptr(), vresult);

            ox += 4;
        }

        // Scalar cleanup
        while ox < ow {
            let ox_f = ox as f64;
            let z_src = m02 * ox_f + base_z;
            let y_src = m12 * ox_f + base_y;
            let x_src = m22 * ox_f + base_x;

            let z0 = z_src.floor() as i32;
            let y0 = y_src.floor() as i32;
            let x0 = x_src.floor() as i32;

            if x0 >= 0
                && x0 < (w - 1) as i32
                && y0 >= 0
                && y0 < (h - 1) as i32
                && z0 >= 0
                && z0 < (d - 1) as i32
            {
                let fx = x_src - x_src.floor();
                let fy = y_src - y_src.floor();
                let fz = z_src - z_src.floor();

                let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                let v000 = input_slice[idx000];
                let v001 = input_slice[idx000 + 1];
                let v010 = input_slice[idx000 + in_stride_y];
                let v011 = input_slice[idx000 + in_stride_y + 1];
                let v100 = input_slice[idx000 + in_stride_z];
                let v101 = input_slice[idx000 + in_stride_z + 1];
                let v110 = input_slice[idx000 + in_stride_z + in_stride_y];
                let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1];

                slice_z[row_start + ox] = v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
                    + v001 * fx * (1.0 - fy) * (1.0 - fz)
                    + v010 * (1.0 - fx) * fy * (1.0 - fz)
                    + v011 * fx * fy * (1.0 - fz)
                    + v100 * (1.0 - fx) * (1.0 - fy) * fz
                    + v101 * fx * (1.0 - fy) * fz
                    + v110 * (1.0 - fx) * fy * fz
                    + v111 * fx * fy * fz;
            } else {
                slice_z[row_start + ox] = cval;
            }

            ox += 1;
        }
    }
}

/// AVX2 optimized 3D trilinear interpolation for f16 (half precision)
///
/// Processes 8 voxels at a time using AVX2 256-bit registers with F16C conversion.
/// Input/output are f16, but computation is done in f32 for accuracy.
///
/// # Safety
///
/// Requires AVX2, FMA, and F16C CPU features. Caller must verify these are available.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn trilinear_3d_f16_avx2(
    input: &ArrayView3<f16>,
    output: &mut ArrayViewMut3<f16>,
    matrix: &AffineMatrix3D,
    shift: &[f64; 3],
    cval: f64,
) {
    let (d, h, w) = input.dim();
    let (_od, oh, ow) = output.dim();

    // Reject tiny volumes
    if d < 2 || h < 2 || w < 2 {
        return;
    }

    let input_slice = input.as_slice().expect("Input must be C-contiguous");
    let output_slice = output.as_slice_mut().expect("Output must be C-contiguous");

    // Input strides (assuming C-contiguous)
    let in_stride_z = h * w;
    let in_stride_y = w;

    // Matrix elements as f32
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

    // Process z-slices in parallel
    let chunk_size = oh * ow;

    #[cfg(feature = "parallel")]
    {
        output_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(oz, slice_z)| {
                process_z_slice_f16(
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
            process_z_slice_f16(
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

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn process_z_slice_f16(
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
    let one = _mm256_set1_ps(1.0);
    let oz_f = oz as f32;

    for oy in 0..oh {
        let oy_f = oy as f32;

        // Base source coordinates for this row
        let base_z = m00 * oz_f + m01 * oy_f + shift_z;
        let base_y = m10 * oz_f + m11 * oy_f + shift_y;
        let base_x = m20 * oz_f + m21 * oy_f + shift_x;

        let row_start = oy * ow;
        let mut ox = 0usize;

        // AVX2 loop - process 8 voxels at a time
        while ox + 7 < ow {
            // Generate x indices [ox, ox+1, ..., ox+7]
            let vx = _mm256_setr_ps(
                ox as f32,
                (ox + 1) as f32,
                (ox + 2) as f32,
                (ox + 3) as f32,
                (ox + 4) as f32,
                (ox + 5) as f32,
                (ox + 6) as f32,
                (ox + 7) as f32,
            );

            // Compute source coordinates: matrix * [oz, oy, ox] + shift
            let zs = _mm256_fmadd_ps(_mm256_set1_ps(m02), vx, _mm256_set1_ps(base_z));
            let ys = _mm256_fmadd_ps(_mm256_set1_ps(m12), vx, _mm256_set1_ps(base_y));
            let xs = _mm256_fmadd_ps(_mm256_set1_ps(m22), vx, _mm256_set1_ps(base_x));

            // Floor to get integer coordinates
            let z_floor = _mm256_floor_ps(zs);
            let y_floor = _mm256_floor_ps(ys);
            let x_floor = _mm256_floor_ps(xs);

            // Get fractional parts
            let fz = _mm256_sub_ps(zs, z_floor);
            let fy = _mm256_sub_ps(ys, y_floor);
            let fx = _mm256_sub_ps(xs, x_floor);

            // Compute trilinear weights (8 corners)
            let one_minus_fx = _mm256_sub_ps(one, fx);
            let one_minus_fy = _mm256_sub_ps(one, fy);
            let one_minus_fz = _mm256_sub_ps(one, fz);

            let w000 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), one_minus_fz);
            let w001 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), one_minus_fz);
            let w010 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), one_minus_fz);
            let w011 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), one_minus_fz);
            let w100 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), fz);
            let w101 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), fz);
            let w110 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), fz);
            let w111 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), fz);

            // Convert to integers
            let zi = _mm256_cvttps_epi32(z_floor);
            let yi = _mm256_cvttps_epi32(y_floor);
            let xi = _mm256_cvttps_epi32(x_floor);

            // Extract values to arrays
            let mut xi_arr = [0i32; 8];
            let mut yi_arr = [0i32; 8];
            let mut zi_arr = [0i32; 8];
            let mut weights = [[0.0f32; 8]; 8];

            _mm256_storeu_si256(zi_arr.as_mut_ptr() as *mut __m256i, zi);
            _mm256_storeu_si256(yi_arr.as_mut_ptr() as *mut __m256i, yi);
            _mm256_storeu_si256(xi_arr.as_mut_ptr() as *mut __m256i, xi);

            _mm256_storeu_ps(weights[0].as_mut_ptr(), w000);
            _mm256_storeu_ps(weights[1].as_mut_ptr(), w001);
            _mm256_storeu_ps(weights[2].as_mut_ptr(), w010);
            _mm256_storeu_ps(weights[3].as_mut_ptr(), w011);
            _mm256_storeu_ps(weights[4].as_mut_ptr(), w100);
            _mm256_storeu_ps(weights[5].as_mut_ptr(), w101);
            _mm256_storeu_ps(weights[6].as_mut_ptr(), w110);
            _mm256_storeu_ps(weights[7].as_mut_ptr(), w111);

            let mut result = [0.0f32; 8];

            // Process each voxel
            for j in 0..8 {
                let z0 = zi_arr[j];
                let y0 = yi_arr[j];
                let x0 = xi_arr[j];

                // Bounds check
                if x0 >= 0
                    && x0 < (w - 1) as i32
                    && y0 >= 0
                    && y0 < (h - 1) as i32
                    && z0 >= 0
                    && z0 < (d - 1) as i32
                {
                    // Compute indices for 8 corners
                    let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                    // Load 8 corner values (f16 -> f32)
                    let v000 = input_slice[idx000].to_f32();
                    let v001 = input_slice[idx000 + 1].to_f32();
                    let v010 = input_slice[idx000 + in_stride_y].to_f32();
                    let v011 = input_slice[idx000 + in_stride_y + 1].to_f32();
                    let v100 = input_slice[idx000 + in_stride_z].to_f32();
                    let v101 = input_slice[idx000 + in_stride_z + 1].to_f32();
                    let v110 = input_slice[idx000 + in_stride_z + in_stride_y].to_f32();
                    let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1].to_f32();

                    // Trilinear interpolation
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

            // Convert f32 results back to f16 using SIMD
            let vresult = _mm256_loadu_ps(result.as_ptr());
            // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0
            let vresult_f16 = _mm256_cvtps_ph::<0>(vresult);
            _mm_storeu_si128(
                slice_z[row_start + ox..].as_mut_ptr() as *mut __m128i,
                vresult_f16,
            );

            ox += 8;
        }

        // Scalar cleanup for remaining voxels
        while ox < ow {
            let ox_f = ox as f32;
            let z_src = m02 * ox_f + base_z;
            let y_src = m12 * ox_f + base_y;
            let x_src = m22 * ox_f + base_x;

            let z0 = z_src.floor() as i32;
            let y0 = y_src.floor() as i32;
            let x0 = x_src.floor() as i32;

            if x0 >= 0
                && x0 < (w - 1) as i32
                && y0 >= 0
                && y0 < (h - 1) as i32
                && z0 >= 0
                && z0 < (d - 1) as i32
            {
                let fx = x_src - x_src.floor();
                let fy = y_src - y_src.floor();
                let fz = z_src - z_src.floor();

                let idx000 = (z0 as usize) * in_stride_z + (y0 as usize) * in_stride_y + (x0 as usize);

                let v000 = input_slice[idx000].to_f32();
                let v001 = input_slice[idx000 + 1].to_f32();
                let v010 = input_slice[idx000 + in_stride_y].to_f32();
                let v011 = input_slice[idx000 + in_stride_y + 1].to_f32();
                let v100 = input_slice[idx000 + in_stride_z].to_f32();
                let v101 = input_slice[idx000 + in_stride_z + 1].to_f32();
                let v110 = input_slice[idx000 + in_stride_z + in_stride_y].to_f32();
                let v111 = input_slice[idx000 + in_stride_z + in_stride_y + 1].to_f32();

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_avx2_identity_f32() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // Skip test if AVX2 not available
        }

        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let mut output = Array3::from_elem((20, 20, 20), 0.0f32);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        unsafe {
            trilinear_3d_f32_avx2(&input.view(), &mut output.view_mut(), &matrix, &shift, 0.0);
        }

        // Interior points should match
        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_avx2_identity_f64() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z * 100 + y * 10 + x) as f64);
        let mut output = Array3::from_elem((20, 20, 20), 0.0f64);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        unsafe {
            trilinear_3d_f64_avx2(&input.view(), &mut output.view_mut(), &matrix, &shift, 0.0);
        }

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_avx2_identity_f16() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") || !is_x86_feature_detected!("f16c") {
            return;
        }

        let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| {
            f16::from_f32(((z * 100 + y * 10 + x) % 256) as f32 / 255.0)
        });
        let mut output = Array3::from_elem((20, 20, 20), f16::ZERO);
        let matrix = AffineMatrix3D::identity();
        let shift = [0.0, 0.0, 0.0];

        unsafe {
            trilinear_3d_f16_avx2(&input.view(), &mut output.view_mut(), &matrix, &shift, 0.0);
        }

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    let expected = input[[z, y, x]].to_f32();
                    let actual = output[[z, y, x]].to_f32();
                    assert!((actual - expected).abs() < 1e-2, "Mismatch at ({z},{y},{x}): expected {expected}, got {actual}");
                }
            }
        }
    }
}
