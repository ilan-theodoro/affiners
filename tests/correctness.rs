//! Correctness tests comparing Rust output with expected values

use approx::assert_relative_eq;
use interp3d_avx2::{affine_transform_3d_f32, affine_transform_3d_f64, AffineMatrix3D};
use ndarray::Array3;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load binary f32 array from file
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

#[test]
fn test_against_python_reference() {
    let test_dir = Path::new("test_data");
    if !test_dir.exists() {
        println!("Test data directory not found. Run test_correctness.py first.");
        return;
    }

    let size = 32;
    let shape = (size, size, size);

    // Load input and reference output
    let input = load_f32_array(&test_dir.join("input_f32.bin"), shape);
    let interp_reference = load_f32_array(&test_dir.join("interp_output_f32.bin"), shape);

    // Apply Rust transform with same parameters
    let matrix = AffineMatrix3D::identity();
    let shift = [0.5, 0.5, 0.5];
    let cval = 0.0;

    let rust_output = affine_transform_3d_f32(&input.view(), &matrix, &shift, cval);

    // Compare interior region (excluding boundary)
    let mut max_diff: f32 = 0.0;
    let mut mean_diff: f32 = 0.0;
    let mut count = 0;

    for z in 2..(size - 2) {
        for y in 2..(size - 2) {
            for x in 2..(size - 2) {
                let diff = (rust_output[[z, y, x]] - interp_reference[[z, y, x]]).abs();
                max_diff = max_diff.max(diff);
                mean_diff += diff;
                count += 1;
            }
        }
    }
    mean_diff /= count as f32;

    println!("Rust vs interp-simd (C) comparison:");
    println!("  Interior region max diff: {:.2e}", max_diff);
    println!("  Interior region mean diff: {:.2e}", mean_diff);

    // Should be very close (within floating-point tolerance)
    assert!(
        max_diff < 1e-5,
        "Max diff {} exceeds tolerance",
        max_diff
    );
}

#[test]
fn test_identity_preserves_values() {
    // Identity transform with no shift should preserve interior values
    let input = Array3::from_shape_fn((20, 20, 20), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
    let matrix = AffineMatrix3D::identity();
    let shift = [0.0, 0.0, 0.0];

    let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

    // Interior points should match exactly
    for z in 1..19 {
        for y in 1..19 {
            for x in 1..19 {
                assert_relative_eq!(output[[z, y, x]], input[[z, y, x]], epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn test_translation_f32() {
    // Create gradient input
    let input = Array3::from_shape_fn((30, 30, 30), |(z, y, x)| (z + y + x) as f32);
    let matrix = AffineMatrix3D::identity();
    let shift = [1.0, 1.0, 1.0];

    let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

    // Output at (z,y,x) should sample from (z+1, y+1, x+1) in input
    for z in 2..27 {
        for y in 2..27 {
            for x in 2..27 {
                let expected = input[[z + 1, y + 1, x + 1]];
                assert_relative_eq!(output[[z, y, x]], expected, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn test_translation_f64() {
    let input = Array3::from_shape_fn((30, 30, 30), |(z, y, x)| (z + y + x) as f64);
    let matrix = AffineMatrix3D::identity();
    let shift = [1.0, 1.0, 1.0];

    let output = affine_transform_3d_f64(&input.view(), &matrix, &shift, 0.0);

    for z in 2..27 {
        for y in 2..27 {
            for x in 2..27 {
                let expected = input[[z + 1, y + 1, x + 1]];
                assert_relative_eq!(output[[z, y, x]], expected, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_half_pixel_shift() {
    // Test interpolation with 0.5 pixel shift
    // At integer coordinates, should get average of neighbors
    let mut input = Array3::zeros((10, 10, 10));

    // Set up a simple pattern
    input[[2, 2, 2]] = 0.0;
    input[[2, 2, 3]] = 8.0;
    input[[2, 3, 2]] = 0.0;
    input[[2, 3, 3]] = 0.0;
    input[[3, 2, 2]] = 0.0;
    input[[3, 2, 3]] = 0.0;
    input[[3, 3, 2]] = 0.0;
    input[[3, 3, 3]] = 0.0;

    let matrix = AffineMatrix3D::identity();
    let shift = [0.5, 0.5, 0.5];

    let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

    // At (2, 2, 2), we sample from (2.5, 2.5, 2.5)
    // This interpolates between the 8 corners
    // Only corner (2,2,3) = 8.0, all others = 0.0
    // Weight for (2,2,3): (1-0.5) * (1-0.5) * 0.5 = 0.125
    let expected = 8.0 * 0.125;
    assert_relative_eq!(output[[2, 2, 2]], expected, epsilon = 1e-6);
}

#[test]
fn test_scaling() {
    // 2x downscaling should sample every other voxel
    let input = Array3::from_shape_fn((40, 40, 40), |(z, y, x)| ((z * 100 + y * 10 + x) % 256) as f32);

    let matrix = AffineMatrix3D::scale(2.0, 2.0, 2.0);
    let shift = [0.0, 0.0, 0.0];

    let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);

    // Output at (z,y,x) samples from (2z, 2y, 2x) in input
    for z in 1..18 {
        for y in 1..18 {
            for x in 1..18 {
                let expected = input[[z * 2, y * 2, x * 2]];
                assert_relative_eq!(output[[z, y, x]], expected, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn test_cval_for_out_of_bounds() {
    let input = Array3::from_elem((10, 10, 10), 1.0f32);
    let matrix = AffineMatrix3D::identity();
    let shift = [100.0, 0.0, 0.0]; // Large shift puts everything out of bounds
    let cval = -999.0;

    let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, cval);

    // All values should be cval since we're sampling way outside
    for z in 0..10 {
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(output[[z, y, x]], cval as f32);
            }
        }
    }
}
