//! Warp field application using SIMD-optimized interpolation
//!
//! This module provides functions to apply displacement/warp fields to 3D images
//! using trilinear interpolation, similar to dexpv2's CUDA implementation.
//!
//! The warp field format is `(D+1, depth, height, width)` where:
//! - Channel 0: Z displacement (dz)
//! - Channel 1: Y displacement (dy)
//! - Channel 2: X displacement (dx)
//! - Channel 3 (optional): Score/quality
//!
//! The warp field is typically smaller than the image and is interpolated
//! on-the-fly to save memory.

use half::f16;
use ndarray::{Array3, Array4, ArrayView3, ArrayView4};

// =============================================================================
// Warp Field Upsampling (zoom by 2x in spatial dimensions)
// =============================================================================

/// Upsample a 4D warp field by 2x in the spatial dimensions (z, y, x)
/// 
/// This matches scipy.ndimage.zoom with factors=(1, 2, 2, 2), order=1, mode='nearest'
/// 
/// scipy.ndimage.zoom coordinate mapping:
///   in_coord = out_idx * (in_size - 1) / (out_size - 1)
/// where out_size = round(in_size * zoom)
/// 
/// Input shape: (channels, d, h, w)
/// Output shape: (channels, d*2, h*2, w*2)
pub fn upsample_warp_field_2x(warp_field: &ArrayView4<f32>) -> Array4<f32> {
    let (channels, in_d, in_h, in_w) = warp_field.dim();
    let out_d = in_d * 2;
    let out_h = in_h * 2;
    let out_w = in_w * 2;
    
    // Compute scale factors for coordinate mapping
    // in_coord = out_idx * scale where scale = (in_size - 1) / (out_size - 1)
    let scale_z = if out_d > 1 { (in_d - 1) as f32 / (out_d - 1) as f32 } else { 0.0 };
    let scale_y = if out_h > 1 { (in_h - 1) as f32 / (out_h - 1) as f32 } else { 0.0 };
    let scale_x = if out_w > 1 { (in_w - 1) as f32 / (out_w - 1) as f32 } else { 0.0 };
    
    let mut output = Array4::<f32>::zeros((channels, out_d, out_h, out_w));
    
    // For each channel
    for c in 0..channels {
        let input_channel = warp_field.slice(ndarray::s![c, .., .., ..]);
        let input_slice = input_channel.as_slice().expect("Input must be C-contiguous");
        
        let in_stride_z = in_h * in_w;
        let in_stride_y = in_w;
        
        // Parallel over output z slices
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let out_stride_z = out_h * out_w;
            let output_channel = output.slice_mut(ndarray::s![c, .., .., ..]);
            let output_slice = output_channel.into_slice().expect("Output must be C-contiguous");
            
            output_slice.par_chunks_mut(out_stride_z).enumerate().for_each(|(oz, slice_z)| {
                upsample_z_slice(
                    input_slice, slice_z, oz, 
                    in_d, in_h, in_w, out_h, out_w,
                    in_stride_z, in_stride_y,
                    scale_z, scale_y, scale_x,
                );
            });
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            for oz in 0..out_d {
                let iz = oz as f32 * scale_z;
                for oy in 0..out_h {
                    let iy = oy as f32 * scale_y;
                    for ox in 0..out_w {
                        let ix = ox as f32 * scale_x;
                        
                        let value = trilinear_interp_nearest_mode(
                            input_slice, in_d, in_h, in_w, in_stride_z, in_stride_y,
                            iz, iy, ix,
                        );
                        output[[c, oz, oy, ox]] = value;
                    }
                }
            }
        }
    }
    
    output
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn upsample_z_slice(
    input: &[f32],
    output_slice: &mut [f32],
    oz: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    in_stride_z: usize,
    in_stride_y: usize,
    scale_z: f32,
    scale_y: f32,
    scale_x: f32,
) {
    let iz = oz as f32 * scale_z;
    
    for oy in 0..out_h {
        let iy = oy as f32 * scale_y;
        
        for ox in 0..out_w {
            let ix = ox as f32 * scale_x;
            
            let value = trilinear_interp_nearest_mode(
                input, in_d, in_h, in_w, in_stride_z, in_stride_y,
                iz, iy, ix,
            );
            output_slice[oy * out_w + ox] = value;
        }
    }
}

/// Trilinear interpolation with nearest boundary mode
/// Used for upsampling warp fields
#[inline]
fn trilinear_interp_nearest_mode(
    data: &[f32],
    d: usize,
    h: usize,
    w: usize,
    stride_z: usize,
    stride_y: usize,
    z: f32,
    y: f32,
    x: f32,
) -> f32 {
    // Clamp coordinates to valid range (nearest mode)
    let z_clamped = z.clamp(0.0, (d - 1) as f32);
    let y_clamped = y.clamp(0.0, (h - 1) as f32);
    let x_clamped = x.clamp(0.0, (w - 1) as f32);
    
    let z0 = z_clamped.floor() as usize;
    let y0 = y_clamped.floor() as usize;
    let x0 = x_clamped.floor() as usize;
    
    let z1 = (z0 + 1).min(d - 1);
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    
    let fz = z_clamped - z_clamped.floor();
    let fy = y_clamped - y_clamped.floor();
    let fx = x_clamped - x_clamped.floor();
    
    let idx000 = z0 * stride_z + y0 * stride_y + x0;
    let idx001 = z0 * stride_z + y0 * stride_y + x1;
    let idx010 = z0 * stride_z + y1 * stride_y + x0;
    let idx011 = z0 * stride_z + y1 * stride_y + x1;
    let idx100 = z1 * stride_z + y0 * stride_y + x0;
    let idx101 = z1 * stride_z + y0 * stride_y + x1;
    let idx110 = z1 * stride_z + y1 * stride_y + x0;
    let idx111 = z1 * stride_z + y1 * stride_y + x1;
    
    let v000 = data[idx000];
    let v001 = data[idx001];
    let v010 = data[idx010];
    let v011 = data[idx011];
    let v100 = data[idx100];
    let v101 = data[idx101];
    let v110 = data[idx110];
    let v111 = data[idx111];
    
    // Trilinear interpolation
    v000 * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + v001 * fx * (1.0 - fy) * (1.0 - fz)
        + v010 * (1.0 - fx) * fy * (1.0 - fz)
        + v011 * fx * fy * (1.0 - fz)
        + v100 * (1.0 - fx) * (1.0 - fy) * fz
        + v101 * fx * (1.0 - fy) * fz
        + v110 * (1.0 - fx) * fy * fz
        + v111 * fx * fy * fz
}

// =============================================================================
// F32 Implementation
// =============================================================================

/// Apply a 3D warp field to an image using trilinear interpolation (f32)
///
/// This is a CPU implementation equivalent to dexpv2's CUDA `apply_warp` function.
/// The warp field is interpolated on-the-fly (not expanded to full resolution).
///
/// # Arguments
///
/// * `image` - 3D input image to be warped
/// * `warp_field` - 4D warp field with shape `(channels, d, h, w)` where channels >= 3
///   - Channel 0: Z displacement
///   - Channel 1: Y displacement  
///   - Channel 2: X displacement
///   - Additional channels (e.g., score) are ignored
/// * `cval` - Constant value for out-of-bounds coordinates
///
/// # Returns
///
/// Warped image with same shape as input
pub fn apply_warp_3d_f32(
    image: &ArrayView3<f32>,
    warp_field: &ArrayView4<f32>,
    cval: f32,
) -> Array3<f32> {
    let (img_d, img_h, img_w) = image.dim();
    let (wf_channels, _wf_d, _wf_h, _wf_w) = warp_field.dim();

    assert!(
        wf_channels >= 3,
        "Warp field must have at least 3 channels (dz, dy, dx), got {}",
        wf_channels
    );

    let mut output = Array3::from_elem((img_d, img_h, img_w), cval);

    // Dispatch to best available SIMD implementation
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                crate::simd::avx512::apply_warp_3d_f32_avx512(
                    image,
                    warp_field,
                    &mut output.view_mut(),
                    cval,
                );
            }
            return output;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                crate::simd::avx2::apply_warp_3d_f32_avx2(
                    image,
                    warp_field,
                    &mut output.view_mut(),
                    cval,
                );
            }
            return output;
        }
    }

    // Scalar fallback
    crate::scalar::apply_warp_3d_f32_scalar(image, warp_field, &mut output.view_mut(), cval);
    output
}

// =============================================================================
// F16 Implementation
// =============================================================================

/// Apply a 3D warp field to an f16 image using trilinear interpolation
///
/// The warp field is always f32 for precision. The image and output are f16.
pub fn apply_warp_3d_f16(
    image: &ArrayView3<f16>,
    warp_field: &ArrayView4<f32>,
    cval: f16,
) -> Array3<f16> {
    let (img_d, img_h, img_w) = image.dim();
    let (wf_channels, _wf_d, _wf_h, _wf_w) = warp_field.dim();

    assert!(
        wf_channels >= 3,
        "Warp field must have at least 3 channels (dz, dy, dx), got {}",
        wf_channels
    );

    let mut output = Array3::from_elem((img_d, img_h, img_w), cval);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                crate::simd::avx512::apply_warp_3d_f16_avx512(
                    image,
                    warp_field,
                    &mut output.view_mut(),
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
                crate::simd::avx2::apply_warp_3d_f16_avx2(
                    image,
                    warp_field,
                    &mut output.view_mut(),
                    cval,
                );
            }
            return output;
        }
    }

    // Scalar fallback
    crate::scalar::apply_warp_3d_f16_scalar(image, warp_field, &mut output.view_mut(), cval);
    output
}

// =============================================================================
// U8 Implementation
// =============================================================================

/// Apply a 3D warp field to a u8 image using trilinear interpolation
///
/// The warp field is always f32 for precision. The image and output are u8.
pub fn apply_warp_3d_u8(
    image: &ArrayView3<u8>,
    warp_field: &ArrayView4<f32>,
    cval: u8,
) -> Array3<u8> {
    let (img_d, img_h, img_w) = image.dim();
    let (wf_channels, _wf_d, _wf_h, _wf_w) = warp_field.dim();

    assert!(
        wf_channels >= 3,
        "Warp field must have at least 3 channels (dz, dy, dx), got {}",
        wf_channels
    );

    let mut output = Array3::from_elem((img_d, img_h, img_w), cval);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                crate::simd::avx512::apply_warp_3d_u8_avx512(
                    image,
                    warp_field,
                    &mut output.view_mut(),
                    cval,
                );
            }
            return output;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                crate::simd::avx2::apply_warp_3d_u8_avx2(
                    image,
                    warp_field,
                    &mut output.view_mut(),
                    cval,
                );
            }
            return output;
        }
    }

    // Scalar fallback
    crate::scalar::apply_warp_3d_u8_scalar(image, warp_field, &mut output.view_mut(), cval);
    output
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_upsample_warp_field_2x() {
        // Test that upsample_warp_field_2x matches scipy.ndimage.zoom behavior
        // scipy.ndimage.zoom coordinate formula: in_coord = out_idx * (in_size - 1) / (out_size - 1)
        
        // Create a simple 2x2x2 warp field (single channel for simplicity)
        let mut warp_field = Array4::<f32>::zeros((1, 2, 2, 2));
        warp_field[[0, 0, 0, 0]] = 1.0;
        warp_field[[0, 0, 0, 1]] = 2.0;
        warp_field[[0, 0, 1, 0]] = 3.0;
        warp_field[[0, 0, 1, 1]] = 4.0;
        warp_field[[0, 1, 0, 0]] = 5.0;
        warp_field[[0, 1, 0, 1]] = 6.0;
        warp_field[[0, 1, 1, 0]] = 7.0;
        warp_field[[0, 1, 1, 1]] = 8.0;

        let upsampled = upsample_warp_field_2x(&warp_field.view());

        // Expected output shape: (1, 4, 4, 4)
        assert_eq!(upsampled.dim(), (1, 4, 4, 4));

        // Expected corners (from scipy behavior):
        // [0,0,0,0] = 1.0 (maps to input [0,0,0])
        // [0,0,0,3] = 2.0 (maps to input [0,0,1])
        // [0,0,3,0] = 3.0 (maps to input [0,1,0])
        // [0,0,3,3] = 4.0 (maps to input [0,1,1])
        // [0,3,0,0] = 5.0 (maps to input [1,0,0])
        // [0,3,3,3] = 8.0 (maps to input [1,1,1])
        assert!((upsampled[[0, 0, 0, 0]] - 1.0).abs() < 1e-5, "Corner [0,0,0,0] should be 1.0");
        assert!((upsampled[[0, 0, 0, 3]] - 2.0).abs() < 1e-5, "Corner [0,0,0,3] should be 2.0");
        assert!((upsampled[[0, 0, 3, 0]] - 3.0).abs() < 1e-5, "Corner [0,0,3,0] should be 3.0");
        assert!((upsampled[[0, 0, 3, 3]] - 4.0).abs() < 1e-5, "Corner [0,0,3,3] should be 4.0");
        assert!((upsampled[[0, 3, 0, 0]] - 5.0).abs() < 1e-5, "Corner [0,3,0,0] should be 5.0");
        assert!((upsampled[[0, 3, 3, 3]] - 8.0).abs() < 1e-5, "Corner [0,3,3,3] should be 8.0");

        // Check interpolated values
        // [0,0,0,1] maps to in_x = 1/3 = 0.333, interpolating between 1 and 2
        // Expected: 1 + 0.333*(2-1) = 1.333
        let expected_001 = 1.0 + (1.0 / 3.0) * (2.0 - 1.0);
        assert!(
            (upsampled[[0, 0, 0, 1]] - expected_001).abs() < 1e-5,
            "Interpolated [0,0,0,1] should be ~1.333, got {}",
            upsampled[[0, 0, 0, 1]]
        );

        // [0,1,1,1] maps to all coords at 1/3
        // This is a trilinear interpolation of all 8 corners
        let frac = 1.0 / 3.0;
        let expected_111 = (1.0 - frac) * (1.0 - frac) * (1.0 - frac) * 1.0
            + frac * (1.0 - frac) * (1.0 - frac) * 2.0
            + (1.0 - frac) * frac * (1.0 - frac) * 3.0
            + frac * frac * (1.0 - frac) * 4.0
            + (1.0 - frac) * (1.0 - frac) * frac * 5.0
            + frac * (1.0 - frac) * frac * 6.0
            + (1.0 - frac) * frac * frac * 7.0
            + frac * frac * frac * 8.0;
        assert!(
            (upsampled[[0, 1, 1, 1]] - expected_111).abs() < 1e-4,
            "Trilinear [0,1,1,1] should be ~{}, got {}",
            expected_111,
            upsampled[[0, 1, 1, 1]]
        );
    }

    #[test]
    fn test_upsample_preserves_channels() {
        // Test that upsampling preserves all channels
        let mut warp_field = Array4::<f32>::zeros((4, 2, 2, 2));
        for c in 0..4 {
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        warp_field[[c, z, y, x]] = (c * 100 + z * 10 + y * 5 + x) as f32;
                    }
                }
            }
        }

        let upsampled = upsample_warp_field_2x(&warp_field.view());

        // Check corners for each channel
        for c in 0..4 {
            let expected_000 = (c * 100) as f32;
            let expected_333 = (c * 100 + 10 + 5 + 1) as f32;
            assert!(
                (upsampled[[c, 0, 0, 0]] - expected_000).abs() < 1e-5,
                "Channel {} corner [0,0,0] mismatch",
                c
            );
            assert!(
                (upsampled[[c, 3, 3, 3]] - expected_333).abs() < 1e-5,
                "Channel {} corner [3,3,3] mismatch",
                c
            );
        }
    }

    #[test]
    fn test_apply_warp_identity_f32() {
        // Zero warp field should be identity transform
        let image = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let warp_field = Array4::<f32>::zeros((4, 5, 5, 5));

        let output = apply_warp_3d_f32(&image.view(), &warp_field.view(), 0.0);

        // Check that output matches input (within tolerance for interpolation)
        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    let diff = (output[[z, y, x]] - image[[z, y, x]]).abs();
                    assert!(diff < 1.0, "Diff too large at [{}, {}, {}]: {}", z, y, x, diff);
                }
            }
        }
    }

    #[test]
    fn test_apply_warp_translation_f32() {
        let image = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z + y + x) as f32);

        // Warp field with constant displacement (dz=1, dy=1, dx=1)
        let mut warp_field = Array4::<f32>::zeros((4, 5, 5, 5));
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    warp_field[[0, z, y, x]] = 1.0; // dz
                    warp_field[[1, z, y, x]] = 1.0; // dy
                    warp_field[[2, z, y, x]] = 1.0; // dx
                }
            }
        }

        let output = apply_warp_3d_f32(&image.view(), &warp_field.view(), 0.0);

        // At (z, y, x) we should sample from (z-1, y-1, x-1)
        for z in 3..17 {
            for y in 3..17 {
                for x in 3..17 {
                    let expected = image[[z - 1, y - 1, x - 1]];
                    let actual = output[[z, y, x]];
                    let diff = (actual - expected).abs();
                    assert!(
                        diff < 1.0,
                        "Diff too large at [{}, {}, {}]: expected {}, got {}",
                        z, y, x, expected, actual
                    );
                }
            }
        }
    }

    #[test]
    fn test_apply_warp_identity_f16() {
        let image =
            Array3::from_shape_fn((20, 20, 20), |(z, y, x)| f16::from_f32((z * 100 + y * 10 + x) as f32));
        let warp_field = Array4::<f32>::zeros((4, 5, 5, 5));

        let output = apply_warp_3d_f16(&image.view(), &warp_field.view(), f16::ZERO);

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    let diff = (output[[z, y, x]].to_f32() - image[[z, y, x]].to_f32()).abs();
                    assert!(diff < 2.0, "Diff too large at [{}, {}, {}]: {}", z, y, x, diff);
                }
            }
        }
    }

    #[test]
    fn test_apply_warp_identity_u8() {
        let image = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| ((z * 10 + y + x) % 256) as u8);
        let warp_field = Array4::<f32>::zeros((4, 5, 5, 5));

        let output = apply_warp_3d_u8(&image.view(), &warp_field.view(), 0);

        for z in 2..18 {
            for y in 2..18 {
                for x in 2..18 {
                    let diff = (output[[z, y, x]] as i32 - image[[z, y, x]] as i32).abs();
                    assert!(diff <= 1, "Diff too large at [{}, {}, {}]: {}", z, y, x, diff);
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_avx512_match_exactly() {
        // Verify AVX2 and AVX512 produce identical results for f32
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("avx512f")
        {
            println!("Skipping test: AVX2+FMA or AVX512F not available");
            return;
        }

        let image = Array3::from_shape_fn((30, 40, 50), |(z, y, x)| {
            ((z * 1000 + y * 50 + x) % 256) as f32
        });
        let mut warp_field = Array4::<f32>::zeros((4, 8, 10, 12));
        for z in 0..8 {
            for y in 0..10 {
                for x in 0..12 {
                    warp_field[[0, z, y, x]] = (z as f32 - 4.0) * 0.5;
                    warp_field[[1, z, y, x]] = (y as f32 - 5.0) * 0.3;
                    warp_field[[2, z, y, x]] = (x as f32 - 6.0) * 0.4;
                }
            }
        }

        let mut output_avx2 = Array3::from_elem((30, 40, 50), 0.0f32);
        let mut output_avx512 = Array3::from_elem((30, 40, 50), 0.0f32);

        unsafe {
            crate::simd::avx2::apply_warp_3d_f32_avx2(
                &image.view(),
                &warp_field.view(),
                &mut output_avx2.view_mut(),
                0.0,
            );
            crate::simd::avx512::apply_warp_3d_f32_avx512(
                &image.view(),
                &warp_field.view(),
                &mut output_avx512.view_mut(),
                0.0,
            );
        }

        // They should match exactly
        for z in 0..30 {
            for y in 0..40 {
                for x in 0..50 {
                    assert_eq!(
                        output_avx2[[z, y, x]],
                        output_avx512[[z, y, x]],
                        "AVX2 and AVX512 differ at [{}, {}, {}]: {} vs {}",
                        z, y, x,
                        output_avx2[[z, y, x]],
                        output_avx512[[z, y, x]]
                    );
                }
            }
        }
    }

    // =========================================================================
    // Random Flow Field Tests
    // =========================================================================

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Generate a random warp field with smooth variations
    fn generate_random_warp_field(
        rng: &mut impl Rng,
        wf_d: usize,
        wf_h: usize,
        wf_w: usize,
        max_displacement: f32,
    ) -> Array4<f32> {
        let mut warp_field = Array4::<f32>::zeros((4, wf_d, wf_h, wf_w));
        for z in 0..wf_d {
            for y in 0..wf_h {
                for x in 0..wf_w {
                    // Random displacement in range [-max_displacement, max_displacement]
                    warp_field[[0, z, y, x]] = rng.random_range(-max_displacement..max_displacement);
                    warp_field[[1, z, y, x]] = rng.random_range(-max_displacement..max_displacement);
                    warp_field[[2, z, y, x]] = rng.random_range(-max_displacement..max_displacement);
                    warp_field[[3, z, y, x]] = rng.random_range(0.0..1.0); // score
                }
            }
        }
        warp_field
    }

    /// Generate a random image with f32 values
    fn generate_random_image_f32(rng: &mut impl Rng, d: usize, h: usize, w: usize) -> Array3<f32> {
        Array3::from_shape_fn((d, h, w), |_| rng.random_range(0.0..255.0))
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_avx2_vs_scalar_f32() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping test: AVX2+FMA not available");
            return;
        }

        // Test multiple random configurations
        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed * 1000 + 42);

            let img_d = 40 + (seed as usize % 20);
            let img_h = 50 + (seed as usize % 30);
            let img_w = 60 + (seed as usize % 25);
            let wf_d = 4 + (seed as usize % 4);
            let wf_h = 5 + (seed as usize % 5);
            let wf_w = 6 + (seed as usize % 6);

            let image = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 10.0);

            let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), 0.0f32);
            let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), 0.0f32);

            crate::scalar::apply_warp_3d_f32_scalar(
                &image.view(),
                &warp_field.view(),
                &mut output_scalar.view_mut(),
                0.0,
            );

            unsafe {
                crate::simd::avx2::apply_warp_3d_f32_avx2(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx2.view_mut(),
                    0.0,
                );
            }

            // AVX2 should closely match scalar for f32 (small FP differences allowed)
            let mut max_diff = 0.0f32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff = (output_avx2[[z, y, x]] - output_scalar[[z, y, x]]).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }

            assert!(
                max_diff < 0.01,
                "Seed {}: AVX2 vs Scalar max diff {} exceeds tolerance",
                seed,
                max_diff
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_avx512_vs_scalar_f32() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping test: AVX512F not available");
            return;
        }

        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed * 2000 + 99);

            let img_d = 35 + (seed as usize % 20);
            let img_h = 45 + (seed as usize % 30);
            let img_w = 55 + (seed as usize % 25);
            let wf_d = 3 + (seed as usize % 5);
            let wf_h = 4 + (seed as usize % 6);
            let wf_w = 5 + (seed as usize % 7);

            let image = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 12.0);

            let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), 0.0f32);
            let mut output_avx512 = Array3::from_elem((img_d, img_h, img_w), 0.0f32);

            crate::scalar::apply_warp_3d_f32_scalar(
                &image.view(),
                &warp_field.view(),
                &mut output_scalar.view_mut(),
                0.0,
            );

            unsafe {
                crate::simd::avx512::apply_warp_3d_f32_avx512(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx512.view_mut(),
                    0.0,
                );
            }

            let mut max_diff = 0.0f32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff = (output_avx512[[z, y, x]] - output_scalar[[z, y, x]]).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }

            assert!(
                max_diff < 0.01,
                "Seed {}: AVX512 vs Scalar max diff {} exceeds tolerance",
                seed,
                max_diff
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_avx2_vs_avx512_f32() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("avx512f")
        {
            println!("Skipping test: AVX2+FMA or AVX512F not available");
            return;
        }

        for seed in 0..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed * 3000 + 777);

            let img_d = 30 + (seed as usize % 25);
            let img_h = 40 + (seed as usize % 35);
            let img_w = 50 + (seed as usize % 40);
            let wf_d = 5 + (seed as usize % 5);
            let wf_h = 6 + (seed as usize % 6);
            let wf_w = 7 + (seed as usize % 7);

            let image = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 15.0);

            let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), 0.0f32);
            let mut output_avx512 = Array3::from_elem((img_d, img_h, img_w), 0.0f32);

            unsafe {
                crate::simd::avx2::apply_warp_3d_f32_avx2(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx2.view_mut(),
                    0.0,
                );
                crate::simd::avx512::apply_warp_3d_f32_avx512(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx512.view_mut(),
                    0.0,
                );
            }

            // AVX2 and AVX512 should produce very close results
            // Small FP differences may occur due to different SIMD widths and rounding
            let mut max_diff = 0.0f32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff = (output_avx2[[z, y, x]] - output_avx512[[z, y, x]]).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }
            assert!(
                max_diff < 0.01,
                "Seed {}: AVX2 vs AVX512 max diff {} exceeds tolerance",
                seed,
                max_diff
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_f16() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            println!("Skipping test: AVX2+FMA+F16C not available");
            return;
        }

        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed * 4000 + 123);

            let img_d = 25 + (seed as usize % 15);
            let img_h = 30 + (seed as usize % 20);
            let img_w = 35 + (seed as usize % 25);
            let wf_d = 4 + (seed as usize % 3);
            let wf_h = 5 + (seed as usize % 4);
            let wf_w = 6 + (seed as usize % 5);

            let image_f32 = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
            let image_f16 = Array3::from_shape_fn((img_d, img_h, img_w), |(z, y, x)| {
                f16::from_f32(image_f32[[z, y, x]])
            });
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 8.0);

            let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), f16::ZERO);
            let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), f16::ZERO);

            crate::scalar::apply_warp_3d_f16_scalar(
                &image_f16.view(),
                &warp_field.view(),
                &mut output_scalar.view_mut(),
                f16::ZERO,
            );

            unsafe {
                crate::simd::avx2::apply_warp_3d_f16_avx2(
                    &image_f16.view(),
                    &warp_field.view(),
                    &mut output_avx2.view_mut(),
                    f16::ZERO,
                );
            }

            // f16 allows small differences due to precision
            let mut max_diff = 0.0f32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff =
                            (output_avx2[[z, y, x]].to_f32() - output_scalar[[z, y, x]].to_f32())
                                .abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }

            assert!(
                max_diff < 0.5,
                "Seed {}: f16 AVX2 vs Scalar max diff {} exceeds tolerance",
                seed,
                max_diff
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_u8() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping test: AVX2+FMA not available");
            return;
        }

        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed * 5000 + 456);

            let img_d = 20 + (seed as usize % 15);
            let img_h = 25 + (seed as usize % 20);
            let img_w = 30 + (seed as usize % 25);
            let wf_d = 3 + (seed as usize % 4);
            let wf_h = 4 + (seed as usize % 5);
            let wf_w = 5 + (seed as usize % 6);

            let image = Array3::from_shape_fn((img_d, img_h, img_w), |_| {
                rng.random_range(0u8..255u8)
            });
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 6.0);

            let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), 0u8);
            let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), 0u8);

            crate::scalar::apply_warp_3d_u8_scalar(
                &image.view(),
                &warp_field.view(),
                &mut output_scalar.view_mut(),
                0,
            );

            unsafe {
                crate::simd::avx2::apply_warp_3d_u8_avx2(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx2.view_mut(),
                    0,
                );
            }

            // u8 allows small differences due to rounding
            let mut max_diff = 0i32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff =
                            (output_avx2[[z, y, x]] as i32 - output_scalar[[z, y, x]] as i32).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }

            assert!(
                max_diff <= 1,
                "Seed {}: u8 AVX2 vs Scalar max diff {} exceeds tolerance",
                seed,
                max_diff
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_large_displacements() {
        // Test with larger displacements that may cause out-of-bounds sampling
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping test: AVX2+FMA not available");
            return;
        }

        let mut rng = ChaCha8Rng::seed_from_u64(33333);

        let img_d = 50;
        let img_h = 60;
        let img_w = 70;
        let wf_d = 8;
        let wf_h = 10;
        let wf_w = 12;

        let image = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
        // Large displacements that will cause many out-of-bounds samples
        let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 50.0);

        let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), -1.0f32);
        let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), -1.0f32);

        crate::scalar::apply_warp_3d_f32_scalar(
            &image.view(),
            &warp_field.view(),
            &mut output_scalar.view_mut(),
            -1.0, // Use -1 as cval to distinguish from valid values
        );

        unsafe {
            crate::simd::avx2::apply_warp_3d_f32_avx2(
                &image.view(),
                &warp_field.view(),
                &mut output_avx2.view_mut(),
                -1.0,
            );
        }

        let mut max_diff = 0.0f32;
        let mut num_oob = 0;
        for z in 0..img_d {
            for y in 0..img_h {
                for x in 0..img_w {
                    let diff = (output_avx2[[z, y, x]] - output_scalar[[z, y, x]]).abs();
                    max_diff = max_diff.max(diff);
                    if output_scalar[[z, y, x]] == -1.0 {
                        num_oob += 1;
                    }
                }
            }
        }

        println!(
            "Large displacement test: max_diff={}, num_oob={}/{}",
            max_diff,
            num_oob,
            img_d * img_h * img_w
        );

        assert!(
            max_diff < 0.01,
            "Large displacement: AVX2 vs Scalar max diff {} exceeds tolerance",
            max_diff
        );
        // Verify we actually had some out-of-bounds samples
        assert!(
            num_oob > 0,
            "Expected some out-of-bounds samples with large displacements"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_random_flow_field_varying_wf_sizes() {
        // Test with various warp field to image size ratios
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping test: AVX2+FMA not available");
            return;
        }

        let mut rng = ChaCha8Rng::seed_from_u64(44444);

        // Test different ratios: warp field much smaller, similar size, and larger than image
        let configs = [
            (100, 100, 100, 5, 5, 5),    // WF much smaller
            (50, 50, 50, 10, 10, 10),    // WF smaller
            (30, 30, 30, 30, 30, 30),    // WF same size
            (20, 20, 20, 40, 40, 40),    // WF larger
        ];

        for (img_d, img_h, img_w, wf_d, wf_h, wf_w) in configs {
            let image = generate_random_image_f32(&mut rng, img_d, img_h, img_w);
            let warp_field = generate_random_warp_field(&mut rng, wf_d, wf_h, wf_w, 5.0);

            let mut output_scalar = Array3::from_elem((img_d, img_h, img_w), 0.0f32);
            let mut output_avx2 = Array3::from_elem((img_d, img_h, img_w), 0.0f32);

            crate::scalar::apply_warp_3d_f32_scalar(
                &image.view(),
                &warp_field.view(),
                &mut output_scalar.view_mut(),
                0.0,
            );

            unsafe {
                crate::simd::avx2::apply_warp_3d_f32_avx2(
                    &image.view(),
                    &warp_field.view(),
                    &mut output_avx2.view_mut(),
                    0.0,
                );
            }

            let mut max_diff = 0.0f32;
            for z in 0..img_d {
                for y in 0..img_h {
                    for x in 0..img_w {
                        let diff = (output_avx2[[z, y, x]] - output_scalar[[z, y, x]]).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
            }

            assert!(
                max_diff < 0.01,
                "WF size {}x{}x{} on img {}x{}x{}: max diff {} exceeds tolerance",
                wf_d,
                wf_h,
                wf_w,
                img_d,
                img_h,
                img_w,
                max_diff
            );
        }
    }
}
