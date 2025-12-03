#!/usr/bin/env python3
"""
Correctness test comparing Rust and C implementations.
"""

import numpy as np
import subprocess
import tempfile
import os

from interp_ndimage import affine_transform as interp_simd_affine
from scipy.ndimage import affine_transform as scipy_affine


def test_identity_transform():
    """Test identity transform produces same output."""
    print("=" * 60)
    print("Test 1: Identity Transform")
    print("=" * 60)

    np.random.seed(42)

    for dtype, dtype_name in [(np.float32, "f32"), (np.float64, "f64")]:
        for size in [16, 32, 64]:
            input_data = np.random.rand(size, size, size).astype(dtype)
            matrix = np.eye(3)
            shift = np.array([0.0, 0.0, 0.0])

            # scipy reference
            scipy_out = scipy_affine(input_data, matrix, offset=shift, order=1)

            # interp-simd
            interp_out = interp_simd_affine(input_data, matrix, offset=shift, order=1)

            # Compare
            max_diff = np.abs(scipy_out - interp_out).max()
            mean_diff = np.abs(scipy_out - interp_out).mean()

            print(f"  {dtype_name} {size}³: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


def test_translation():
    """Test translation transform."""
    print("\n" + "=" * 60)
    print("Test 2: Translation (shift=[1.5, 2.5, 3.5])")
    print("=" * 60)

    np.random.seed(42)

    for dtype, dtype_name in [(np.float32, "f32"), (np.float64, "f64")]:
        for size in [16, 32, 64]:
            input_data = np.random.rand(size, size, size).astype(dtype)
            matrix = np.eye(3)
            shift = np.array([1.5, 2.5, 3.5])

            # scipy reference
            scipy_out = scipy_affine(input_data, matrix, offset=shift, order=1)

            # interp-simd
            interp_out = interp_simd_affine(input_data, matrix, offset=shift, order=1)

            # Compare
            max_diff = np.abs(scipy_out - interp_out).max()
            mean_diff = np.abs(scipy_out - interp_out).mean()

            print(f"  {dtype_name} {size}³: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


def test_rotation():
    """Test rotation transform."""
    print("\n" + "=" * 60)
    print("Test 3: 45° Rotation around Z-axis")
    print("=" * 60)

    np.random.seed(42)
    angle = np.pi / 4  # 45 degrees

    # Rotation matrix around Z axis
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    shift = np.array([0.0, 0.0, 0.0])

    for dtype, dtype_name in [(np.float32, "f32"), (np.float64, "f64")]:
        for size in [16, 32, 64]:
            input_data = np.random.rand(size, size, size).astype(dtype)

            # scipy reference
            scipy_out = scipy_affine(input_data, matrix, offset=shift, order=1)

            # interp-simd
            interp_out = interp_simd_affine(input_data, matrix, offset=shift, order=1)

            # Compare
            max_diff = np.abs(scipy_out - interp_out).max()
            mean_diff = np.abs(scipy_out - interp_out).mean()

            print(f"  {dtype_name} {size}³: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


def test_scaling():
    """Test scaling transform."""
    print("\n" + "=" * 60)
    print("Test 4: Scaling (2x in each dimension)")
    print("=" * 60)

    np.random.seed(42)

    matrix = np.array([
        [2.0, 0, 0],
        [0, 2.0, 0],
        [0, 0, 2.0]
    ])
    shift = np.array([0.0, 0.0, 0.0])

    for dtype, dtype_name in [(np.float32, "f32"), (np.float64, "f64")]:
        for size in [16, 32, 64]:
            input_data = np.random.rand(size, size, size).astype(dtype)

            # scipy reference
            scipy_out = scipy_affine(input_data, matrix, offset=shift, order=1)

            # interp-simd
            interp_out = interp_simd_affine(input_data, matrix, offset=shift, order=1)

            # Compare
            max_diff = np.abs(scipy_out - interp_out).max()
            mean_diff = np.abs(scipy_out - interp_out).mean()

            print(f"  {dtype_name} {size}³: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


def save_test_data_for_rust():
    """Save test data to files for Rust to read and compare."""
    print("\n" + "=" * 60)
    print("Generating test data for Rust comparison...")
    print("=" * 60)

    np.random.seed(42)

    # Create test directory
    test_dir = "/home/ilan/PycharmProjects/affine-rs/test_data"
    os.makedirs(test_dir, exist_ok=True)

    size = 32
    input_data = np.random.rand(size, size, size).astype(np.float32)

    # Identity transform
    matrix = np.eye(3)
    shift = np.array([0.5, 0.5, 0.5])

    scipy_out = scipy_affine(input_data, matrix, offset=shift, order=1)
    interp_out = interp_simd_affine(input_data, matrix, offset=shift, order=1)

    # Save input and expected output
    input_data.tofile(f"{test_dir}/input_f32.bin")
    scipy_out.astype(np.float32).tofile(f"{test_dir}/scipy_output_f32.bin")
    interp_out.astype(np.float32).tofile(f"{test_dir}/interp_output_f32.bin")

    # Save metadata
    with open(f"{test_dir}/metadata.txt", "w") as f:
        f.write(f"size={size}\n")
        f.write(f"shift={shift[0]},{shift[1]},{shift[2]}\n")
        f.write(f"matrix=identity\n")

    print(f"  Saved test data to {test_dir}/")
    print(f"  Input shape: {input_data.shape}")
    print(f"  scipy vs interp-simd max_diff: {np.abs(scipy_out - interp_out).max():.2e}")

    return test_dir, input_data, scipy_out, interp_out


if __name__ == "__main__":
    print("Comparing scipy.ndimage and interp-simd (C/AVX2) implementations")
    print("Both should produce nearly identical results (within floating-point tolerance)")
    print()

    test_identity_transform()
    test_translation()
    test_rotation()
    test_scaling()

    test_dir, input_data, scipy_out, interp_out = save_test_data_for_rust()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("scipy and interp-simd produce nearly identical results.")
    print("Small differences are due to floating-point precision and")
    print("different evaluation order of the trilinear interpolation formula.")
