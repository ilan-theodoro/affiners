#!/usr/bin/env python3
"""
Benchmark comparison between interp-simd (Python/C) and interp3d-avx2 (Rust)

This script benchmarks:
1. interp-simd (Python wrapper with C/AVX2 backend)
2. scipy.ndimage.affine_transform for reference
"""

import numpy as np
import time
import sys

# Try to import interp_simd from the local directory
sys.path.insert(0, '/home/ilan/PycharmProjects/interp-c')

try:
    from interp_ndimage import affine_transform as interp_simd_affine
    HAS_INTERP_SIMD = True
except ImportError as e:
    print(f"Warning: interp-simd not available: {e}")
    HAS_INTERP_SIMD = False

try:
    from scipy.ndimage import affine_transform as scipy_affine
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not available")
    HAS_SCIPY = False


def benchmark_func(func, *args, warmup=3, iterations=10, **kwargs):
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times), result


def run_benchmarks():
    """Run benchmarks for different volume sizes."""
    sizes = [32, 64, 128, 256]
    dtypes = [np.float32, np.float64]

    # Identity matrix (with small shift to force interpolation)
    matrix = np.eye(3)
    shift = np.array([0.5, 0.5, 0.5])

    print("=" * 80)
    print("3D Trilinear Interpolation Benchmark (affine_transform)")
    print("=" * 80)
    print()

    for dtype in dtypes:
        dtype_name = "f32" if dtype == np.float32 else "f64"
        print(f"\n{'='*40}")
        print(f"Data type: {dtype_name}")
        print(f"{'='*40}")

        for size in sizes:
            print(f"\nVolume size: {size}x{size}x{size} ({size**3:,} voxels)")
            print("-" * 50)

            # Create test data
            input_data = np.random.rand(size, size, size).astype(dtype)

            # Benchmark scipy
            if HAS_SCIPY:
                mean_time, std_time, _ = benchmark_func(
                    scipy_affine, input_data, matrix, offset=shift, order=1
                )
                throughput = size**3 / mean_time / 1e9
                print(f"scipy.ndimage:    {mean_time*1000:8.3f} ms ± {std_time*1000:.3f} ms  ({throughput:.2f} Gelem/s)")

            # Benchmark interp-simd
            if HAS_INTERP_SIMD:
                mean_time, std_time, _ = benchmark_func(
                    interp_simd_affine, input_data, matrix, offset=shift, order=1
                )
                throughput = size**3 / mean_time / 1e9
                print(f"interp-simd:      {mean_time*1000:8.3f} ms ± {std_time*1000:.3f} ms  ({throughput:.2f} Gelem/s)")

    print("\n" + "=" * 80)
    print("Note: interp3d-avx2 (Rust) benchmarks are run separately via 'cargo bench'")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
