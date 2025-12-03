//! Benchmark a large volume with shear transformation

use std::time::Instant;
use ndarray::Array3;
use interp3d_avx2::{affine_transform_3d_f32, affine_transform_3d_f16, AffineMatrix3D, f16};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    println!("Benchmarking {}³ = {} voxels", size, size * size * size);
    println!();

    // Create a shear transformation matrix
    // Shear in XY plane: x' = x + 0.1*y
    let shear_matrix = AffineMatrix3D::new([
        [1.0, 0.0, 0.0],     // z' = z
        [0.0, 1.0, 0.0],     // y' = y
        [0.0, 0.1, 1.0],     // x' = x + 0.1*y
    ]);
    let shift = [0.0, 0.0, 0.0];

    println!("=== f32 ({}³) ===", size);
    {
        // Create input (cold allocation)
        let start = Instant::now();
        let input: Array3<f32> = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
            ((z * 1000 + y * 10 + x) % 65536) as f32 / 65535.0
        });
        println!("Input allocation: {:?}", start.elapsed());

        // Run transformation
        let start = Instant::now();
        let output = affine_transform_3d_f32(&input.view(), &shear_matrix, &shift, 0.0);
        let elapsed = start.elapsed();

        let voxels = (size * size * size) as f64;
        let throughput = voxels / elapsed.as_secs_f64();
        println!("Transform time: {:?}", elapsed);
        println!("Throughput: {:.2} Gvoxels/s", throughput / 1e9);
        println!("Output sample [512,512,512]: {}", output[[512, 512, 512]]);
    }
    println!();

    println!("=== f16 ({}³) ===", size);
    {
        // Create input (cold allocation)
        let start = Instant::now();
        let input: Array3<f16> = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
            f16::from_f32(((z * 1000 + y * 10 + x) % 65536) as f32 / 65535.0)
        });
        println!("Input allocation: {:?}", start.elapsed());

        // Run transformation
        let start = Instant::now();
        let output = affine_transform_3d_f16(&input.view(), &shear_matrix, &shift, 0.0);
        let elapsed = start.elapsed();

        let voxels = (size * size * size) as f64;
        let throughput = voxels / elapsed.as_secs_f64();
        println!("Transform time: {:?}", elapsed);
        println!("Throughput: {:.2} Gvoxels/s", throughput / 1e9);
        println!("Output sample [512,512,512]: {}", output[[512, 512, 512]].to_f32());
    }

    // Memory usage note
    let voxels = size * size * size;
    let f32_gb = (voxels * 4) as f64 / 1e9;
    let f16_gb = (voxels * 2) as f64 / 1e9;
    println!();
    println!("Memory usage for {}³:", size);
    println!("  f32: ~{:.1} GB input + ~{:.1} GB output = ~{:.1} GB", f32_gb, f32_gb, f32_gb * 2.0);
    println!("  f16: ~{:.1} GB input + ~{:.1} GB output = ~{:.1} GB", f16_gb, f16_gb, f16_gb * 2.0);
}
