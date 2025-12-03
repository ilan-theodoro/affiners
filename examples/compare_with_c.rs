//! Example that loads test data, applies transform, and saves output for comparison

use interp3d_avx2::{affine_transform_3d_f32, AffineMatrix3D};
use ndarray::Array3;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn load_f32_array(path: &Path, shape: (usize, usize, usize)) -> Array3<f32> {
    let mut file = File::open(path).expect("Failed to open file");
    let mut buffer = vec![0u8; shape.0 * shape.1 * shape.2 * 4];
    file.read_exact(&mut buffer).expect("Failed to read file");

    let data: Vec<f32> = buffer
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Array3::from_shape_vec(shape, data).expect("Failed to create array")
}

fn save_f32_array(path: &Path, array: &Array3<f32>) {
    let mut file = File::create(path).expect("Failed to create file");
    let bytes: Vec<u8> = array
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    file.write_all(&bytes).expect("Failed to write file");
}

fn main() {
    let test_dir = Path::new("test_data");

    if !test_dir.exists() {
        eprintln!("Test data directory not found. Run test_correctness.py first.");
        std::process::exit(1);
    }

    let size = 32;
    let shape = (size, size, size);

    // Load input
    let input = load_f32_array(&test_dir.join("input_f32.bin"), shape);
    println!("Loaded input: {:?}", input.dim());
    println!("Input range: [{:.6}, {:.6}]",
             input.iter().cloned().fold(f32::INFINITY, f32::min),
             input.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Apply transform with same parameters as Python
    let matrix = AffineMatrix3D::identity();
    let shift = [0.5, 0.5, 0.5];
    let cval = 0.0;

    let rust_output = affine_transform_3d_f32(&input.view(), &matrix, &shift, cval);

    // Save Rust output
    save_f32_array(&test_dir.join("rust_output_f32.bin"), &rust_output);
    println!("Saved Rust output");

    // Load C reference output
    let c_output = load_f32_array(&test_dir.join("interp_output_f32.bin"), shape);

    // Compare
    let mut max_diff: f32 = 0.0;
    let mut mean_diff: f32 = 0.0;
    let mut count = 0;

    // Compare interior (excluding boundary)
    for z in 2..(size - 2) {
        for y in 2..(size - 2) {
            for x in 2..(size - 2) {
                let diff = (rust_output[[z, y, x]] - c_output[[z, y, x]]).abs();
                max_diff = max_diff.max(diff);
                mean_diff += diff;
                count += 1;
            }
        }
    }
    mean_diff /= count as f32;

    println!("\n===== Comparison Results =====");
    println!("Interior region ({} voxels):", count);
    println!("  Max diff:  {:.2e}", max_diff);
    println!("  Mean diff: {:.2e}", mean_diff);

    // Also check some specific values
    println!("\nSample values at (16, 16, 16):");
    println!("  Rust: {:.8}", rust_output[[16, 16, 16]]);
    println!("  C:    {:.8}", c_output[[16, 16, 16]]);
    println!("  Diff: {:.2e}", (rust_output[[16, 16, 16]] - c_output[[16, 16, 16]]).abs());

    println!("\nSample values at (10, 10, 10):");
    println!("  Rust: {:.8}", rust_output[[10, 10, 10]]);
    println!("  C:    {:.8}", c_output[[10, 10, 10]]);
    println!("  Diff: {:.2e}", (rust_output[[10, 10, 10]] - c_output[[10, 10, 10]]).abs());

    // Check boundary (should both be cval=0 for out-of-bounds)
    println!("\nBoundary values at (0, 0, 0):");
    println!("  Rust: {:.8}", rust_output[[0, 0, 0]]);
    println!("  C:    {:.8}", c_output[[0, 0, 0]]);

    println!("\nBoundary values at (31, 31, 31):");
    println!("  Rust: {:.8}", rust_output[[31, 31, 31]]);
    println!("  C:    {:.8}", c_output[[31, 31, 31]]);

    if max_diff < 1e-5 {
        println!("\n✓ SUCCESS: Rust and C implementations produce identical results!");
    } else {
        println!("\n✗ WARNING: Significant differences detected");
    }
}
