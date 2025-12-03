#!/usr/bin/env python3
"""Compare Rust vs interp-simd (C) performance"""

import numpy as np
import time
import sys

# Import both implementations
import interp3d_avx2  # Rust
from interp_ndimage import affine_transform as c_affine_transform  # C/interp-simd


def benchmark(func, input_data, matrix, offset, name, warmup=1, runs=3):
    """Benchmark a function with warmup and multiple runs"""
    # Warmup
    for _ in range(warmup):
        func(input_data, matrix, offset=offset, order=1, cval=0.0)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        output = func(input_data, matrix, offset=offset, order=1, cval=0.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return output, avg_time


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512

    print("=" * 70)
    print(f"interp-simd (C) vs interp3d-avx2 (Rust) - {size}³ = {size**3:,} voxels")
    print("=" * 70)
    print()

    # Shear transformation
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 1.0],
    ], dtype=np.float64)
    offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    voxels = size ** 3

    # Create input
    print(f"Creating {size}³ f32 input...")
    input_f32 = np.random.rand(size, size, size).astype(np.float32)
    print(f"  Memory: {input_f32.nbytes / 1e9:.2f} GB")
    print()

    # Benchmark interp-simd (C)
    print("Running interp-simd (C AVX2)...")
    output_c, time_c = benchmark(c_affine_transform, input_f32, matrix, offset, "C")
    throughput_c = voxels / time_c / 1e9
    print(f"  Time: {time_c:.4f} s")
    print(f"  Throughput: {throughput_c:.2f} Gvoxels/s")
    print()

    # Benchmark Rust
    print("Running Rust (AVX512/AVX2 auto)...")
    output_rust, time_rust = benchmark(interp3d_avx2.affine_transform, input_f32, matrix, offset, "Rust")
    throughput_rust = voxels / time_rust / 1e9
    print(f"  Time: {time_rust:.4f} s")
    print(f"  Throughput: {throughput_rust:.2f} Gvoxels/s")
    print()

    # Compare correctness
    max_diff = np.max(np.abs(output_rust - output_c))
    print(f"Max difference: {max_diff:.2e}")
    print()

    # Summary
    speedup = time_c / time_rust
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  interp-simd (C):  {time_c:.4f}s  ({throughput_c:.2f} Gvoxels/s)")
    print(f"  Rust:             {time_rust:.4f}s  ({throughput_rust:.2f} Gvoxels/s)")
    print()
    if speedup > 1:
        print(f"  Rust is {speedup:.2f}x FASTER than C")
    else:
        print(f"  C is {1/speedup:.2f}x FASTER than Rust")


if __name__ == "__main__":
    main()
