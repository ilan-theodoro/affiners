#!/usr/bin/env python3
"""Compare Rust (AVX2, AVX512) vs Python/scipy performance"""

import numpy as np
import time
import sys

# Try to import the Rust library
try:
    import interp3d_avx2
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: interp3d_avx2 not installed. Run: maturin develop --release")

# Try to import scipy
try:
    from scipy.ndimage import affine_transform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed")


def benchmark_scipy(input_data, matrix, offset, cval=0.0):
    """Benchmark scipy.ndimage.affine_transform"""
    start = time.perf_counter()
    output = affine_transform(input_data, matrix, offset=offset, order=1, cval=cval)
    elapsed = time.perf_counter() - start
    return output, elapsed


def benchmark_rust(input_data, matrix, offset, cval=0.0):
    """Benchmark Rust implementation (auto-dispatch to AVX512 or AVX2)"""
    start = time.perf_counter()
    output = interp3d_avx2.affine_transform(input_data, matrix, offset=offset, order=1, cval=cval)
    elapsed = time.perf_counter() - start
    return output, elapsed


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512

    print(f"=" * 60)
    print(f"Benchmarking {size}³ = {size**3:,} voxels")
    print(f"=" * 60)
    print()

    # Create shear transformation matrix
    # x' = x + 0.1*y (shear in XY plane)
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 1.0],
    ], dtype=np.float64)
    offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    voxels = size ** 3

    # ============ f32 benchmarks ============
    print("=== f32 ===")
    print(f"Creating {size}³ f32 input...")
    input_f32 = np.random.rand(size, size, size).astype(np.float32)
    print(f"  Memory: {input_f32.nbytes / 1e9:.2f} GB")
    print()

    # Scipy (f32 -> f64 internally)
    if HAS_SCIPY:
        print("Running scipy.ndimage.affine_transform (order=1)...")
        output_scipy, time_scipy = benchmark_scipy(input_f32, matrix, offset)
        throughput_scipy = voxels / time_scipy / 1e9
        print(f"  Time: {time_scipy:.3f} s")
        print(f"  Throughput: {throughput_scipy:.2f} Gvoxels/s")
        print()

    # Rust (auto-dispatch)
    if HAS_RUST:
        print("Running Rust (AVX512 auto-dispatch)...")
        output_rust, time_rust = benchmark_rust(input_f32, matrix, offset)
        throughput_rust = voxels / time_rust / 1e9
        print(f"  Time: {time_rust:.3f} s")
        print(f"  Throughput: {throughput_rust:.2f} Gvoxels/s")

        if HAS_SCIPY:
            speedup = time_scipy / time_rust
            print(f"  Speedup vs scipy: {speedup:.2f}x")

            # Check correctness
            max_diff = np.max(np.abs(output_rust - output_scipy))
            print(f"  Max diff vs scipy: {max_diff:.2e}")
        print()

    # ============ f16 benchmarks ============
    if HAS_RUST:
        print("=== f16 ===")
        print(f"Creating {size}³ f16 input...")
        input_f16 = input_f32.astype(np.float16)
        print(f"  Memory: {input_f16.nbytes / 1e9:.2f} GB")
        print()

        print("Running Rust f16 (AVX512 auto-dispatch)...")
        output_rust_f16, time_rust_f16 = benchmark_rust(input_f16, matrix, offset)
        throughput_rust_f16 = voxels / time_rust_f16 / 1e9
        print(f"  Time: {time_rust_f16:.3f} s")
        print(f"  Throughput: {throughput_rust_f16:.2f} Gvoxels/s")

        if HAS_SCIPY:
            speedup_f16 = time_scipy / time_rust_f16
            print(f"  Speedup vs scipy f32: {speedup_f16:.2f}x")
        print()

    # ============ Summary ============
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if HAS_SCIPY:
        print(f"  scipy f32:      {time_scipy:.3f}s  ({throughput_scipy:.2f} Gvoxels/s)")
    if HAS_RUST:
        print(f"  Rust AVX512 f32: {time_rust:.3f}s  ({throughput_rust:.2f} Gvoxels/s)")
        print(f"  Rust AVX512 f16: {time_rust_f16:.3f}s  ({throughput_rust_f16:.2f} Gvoxels/s)")
    print()


if __name__ == "__main__":
    main()
