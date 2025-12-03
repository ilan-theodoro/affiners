//! Compare AVX2 vs AVX512 performance directly

use std::time::Instant;
use ndarray::Array3;
use interp3d_avx2::{simd, AffineMatrix3D, f16};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);

    println!("{}", "=".repeat(60));
    println!("AVX2 vs AVX512 Comparison - {}³ = {} voxels", size, size * size * size);
    println!("{}", "=".repeat(60));
    println!();

    // Shear transformation
    let shear_matrix = AffineMatrix3D::new([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 1.0],
    ]);
    let shift = [0.0, 0.0, 0.0];
    let voxels = (size * size * size) as f64;

    // ========== f32 ==========
    println!("=== f32 ({size}³) ===");
    let input_f32: Array3<f32> = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
        ((z * 1000 + y * 10 + x) % 65536) as f32 / 65535.0
    });
    println!("Memory: {:.2} GB", (size * size * size * 4) as f64 / 1e9);
    println!();

    // AVX512 f32
    if is_x86_feature_detected!("avx512f") {
        let mut output = Array3::from_elem((size, size, size), 0.0f32);
        let start = Instant::now();
        unsafe {
            simd::avx512::trilinear_3d_f32_avx512(
                &input_f32.view(),
                &mut output.view_mut(),
                &shear_matrix,
                &shift,
                0.0,
            );
        }
        let elapsed = start.elapsed();
        let throughput = voxels / elapsed.as_secs_f64() / 1e9;
        println!("AVX512 f32: {:?} ({:.2} Gvoxels/s)", elapsed, throughput);
    }

    // AVX2 f32
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let mut output = Array3::from_elem((size, size, size), 0.0f32);
        let start = Instant::now();
        unsafe {
            simd::avx2::trilinear_3d_f32_avx2(
                &input_f32.view(),
                &mut output.view_mut(),
                &shear_matrix,
                &shift,
                0.0,
            );
        }
        let elapsed = start.elapsed();
        let throughput = voxels / elapsed.as_secs_f64() / 1e9;
        println!("AVX2 f32:   {:?} ({:.2} Gvoxels/s)", elapsed, throughput);
    }
    println!();

    // ========== f16 ==========
    println!("=== f16 ({size}³) ===");
    let input_f16: Array3<f16> = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
        f16::from_f32(((z * 1000 + y * 10 + x) % 65536) as f32 / 65535.0)
    });
    println!("Memory: {:.2} GB", (size * size * size * 2) as f64 / 1e9);
    println!();

    // AVX512 f16
    if is_x86_feature_detected!("avx512f") {
        let mut output = Array3::from_elem((size, size, size), f16::ZERO);
        let start = Instant::now();
        unsafe {
            simd::avx512::trilinear_3d_f16_avx512(
                &input_f16.view(),
                &mut output.view_mut(),
                &shear_matrix,
                &shift,
                0.0,
            );
        }
        let elapsed = start.elapsed();
        let throughput = voxels / elapsed.as_secs_f64() / 1e9;
        println!("AVX512 f16: {:?} ({:.2} Gvoxels/s)", elapsed, throughput);
    }

    // AVX2 f16
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
        let mut output = Array3::from_elem((size, size, size), f16::ZERO);
        let start = Instant::now();
        unsafe {
            simd::avx2::trilinear_3d_f16_avx2(
                &input_f16.view(),
                &mut output.view_mut(),
                &shear_matrix,
                &shift,
                0.0,
            );
        }
        let elapsed = start.elapsed();
        let throughput = voxels / elapsed.as_secs_f64() / 1e9;
        println!("AVX2 f16:   {:?} ({:.2} Gvoxels/s)", elapsed, throughput);
    }
    println!();

    println!("{}", "=".repeat(60));
    println!("Note: AVX512 processes 16 f32 or 16 f16 per iteration");
    println!("      AVX2 processes 8 f32 or 8 f16 per iteration");
}
