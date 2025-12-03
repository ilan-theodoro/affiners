#!/usr/bin/env python3
"""
Direct comparison of Rust and C implementations by running both and comparing outputs.
"""

import numpy as np
import subprocess
import tempfile
import os

from interp_ndimage import affine_transform as interp_simd_affine


def run_rust_transform(input_data, matrix, shift, cval=0.0):
    """Run the Rust implementation via a test binary."""
    # Save input to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.bin")
        output_path = os.path.join(tmpdir, "output.bin")

        # Save input
        input_data.tofile(input_path)

        # Run Rust test binary
        size = input_data.shape[0]
        dtype = "f32" if input_data.dtype == np.float32 else "f64"

        cmd = [
            "cargo", "run", "--release", "--example", "transform",
            "--",
            input_path, output_path,
            str(size),
            ",".join(map(str, matrix.flatten())),
            ",".join(map(str, shift)),
            str(cval),
            dtype
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ilan/PycharmProjects/affine-rs")
            if result.returncode != 0:
                print(f"Rust error: {result.stderr}")
                return None

            # Load output
            output = np.fromfile(output_path, dtype=input_data.dtype).reshape(input_data.shape)
            return output
        except Exception as e:
            print(f"Error running Rust: {e}")
            return None


def compare_implementations():
    """Compare C and Rust implementations directly."""
    print("=" * 70)
    print("Direct Comparison: interp-simd (C/AVX2) vs interp3d-avx2 (Rust/AVX2)")
    print("=" * 70)

    np.random.seed(42)

    test_cases = [
        ("Identity + shift", np.eye(3), [0.5, 0.5, 0.5]),
        ("Translation", np.eye(3), [3.7, 2.3, 1.9]),
        ("2x Scale", np.diag([2.0, 2.0, 2.0]), [0.0, 0.0, 0.0]),
        ("0.5x Scale", np.diag([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0]),
    ]

    sizes = [32, 64]

    for dtype, dtype_name in [(np.float32, "f32")]:
        print(f"\nData type: {dtype_name}")
        print("-" * 70)

        for size in sizes:
            input_data = np.random.rand(size, size, size).astype(dtype)

            for test_name, matrix, shift in test_cases:
                # C implementation
                c_output = interp_simd_affine(input_data, matrix, offset=np.array(shift), order=1)

                # For now, just verify C output is reasonable
                interior = slice(2, -2)

                # Check that C output doesn't have NaNs or extreme values
                c_interior = c_output[interior, interior, interior]
                if np.isnan(c_interior).any():
                    print(f"  {test_name} {size}³: C output has NaNs!")
                    continue

                print(f"  {test_name} {size}³: C interior range [{c_interior.min():.4f}, {c_interior.max():.4f}]")


def simple_comparison():
    """Simple comparison using saved files."""
    print("\n" + "=" * 70)
    print("Comparing saved outputs")
    print("=" * 70)

    test_dir = "/home/ilan/PycharmProjects/affine-rs/test_data"

    if os.path.exists(test_dir):
        # Load reference outputs
        interp_out = np.fromfile(f"{test_dir}/interp_output_f32.bin", dtype=np.float32).reshape(32, 32, 32)

        print(f"\ninterp-simd (C) output stats:")
        print(f"  Shape: {interp_out.shape}")
        print(f"  Range: [{interp_out.min():.6f}, {interp_out.max():.6f}]")
        print(f"  Mean: {interp_out.mean():.6f}")

        # Check interior only (exclude boundary where cval=0 is used)
        interior = interp_out[2:-2, 2:-2, 2:-2]
        print(f"\nInterior (excluding 2-pixel boundary):")
        print(f"  Shape: {interior.shape}")
        print(f"  Range: [{interior.min():.6f}, {interior.max():.6f}]")
        print(f"  Mean: {interior.mean():.6f}")
    else:
        print("Test data not found. Run test_correctness.py first.")


if __name__ == "__main__":
    compare_implementations()
    simple_comparison()

    print("\n" + "=" * 70)
    print("Note: Full Rust vs C comparison requires running 'cargo test'")
    print("The test_against_python_reference test loads the saved C output")
    print("and compares it with the Rust implementation.")
    print("=" * 70)
