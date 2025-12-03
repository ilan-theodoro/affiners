#!/usr/bin/env python3
"""Compare Rust vs interp-simd (C) f16 performance"""

import numpy as np
import time
import sys

# Import both implementations
import interp3d_avx2  # Rust
from interp_ndimage import affine_transform as c_affine_transform  # C/interp-simd


def benchmark(func, input_data, matrix, offset, warmup=1, runs=3):
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
    print(f"interp-simd (C) vs Rust - f16 Comparison - {size}³ = {size**3:,} voxels")
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

    # ========== f16 ==========
    print(f"Creating {size}³ f16 input...")
    input_f16 = np.random.rand(size, size, size).astype(np.float16)
    print(f"  Memory: {input_f16.nbytes / 1e9:.2f} GB")
    print()

    # Benchmark interp-simd (C) f16
    print("Running interp-simd (C AVX2) f16...")
    output_c, time_c = benchmark(c_affine_transform, input_f16, matrix, offset)
    throughput_c = voxels / time_c / 1e9
    print(f"  Time: {time_c:.4f} s")
    print(f"  Throughput: {throughput_c:.2f} Gvoxels/s")
    print()

    # Benchmark Rust f16 (need to pass as u16 view)
    print("Running Rust (AVX512/AVX2 auto) f16...")
    def rust_f16_wrapper(input_data, matrix, offset, order, cval):
        # View f16 as u16 for Rust binding
        input_u16 = input_data.view(np.uint16)
        output_u16 = interp3d_avx2.affine_transform_f16(input_u16, matrix, offset=offset, order=order, cval=cval)
        return output_u16.view(np.float16)

    output_rust, time_rust = benchmark(rust_f16_wrapper, input_f16, matrix, offset)
    throughput_rust = voxels / time_rust / 1e9
    print(f"  Time: {time_rust:.4f} s")
    print(f"  Throughput: {throughput_rust:.2f} Gvoxels/s")
    print()

    # Compare correctness
    max_diff = np.max(np.abs(output_rust.astype(np.float32) - output_c.astype(np.float32)))
    print(f"Max difference: {max_diff:.2e}")
    print()

    # ========== f32 for reference ==========
    print(f"--- f32 for reference ---")
    input_f32 = input_f16.astype(np.float32)

    _, time_c_f32 = benchmark(c_affine_transform, input_f32, matrix, offset)
    throughput_c_f32 = voxels / time_c_f32 / 1e9
    print(f"C f32:    {time_c_f32:.4f}s  ({throughput_c_f32:.2f} Gvoxels/s)")

    _, time_rust_f32 = benchmark(interp3d_avx2.affine_transform, input_f32, matrix, offset)
    throughput_rust_f32 = voxels / time_rust_f32 / 1e9
    print(f"Rust f32: {time_rust_f32:.4f}s  ({throughput_rust_f32:.2f} Gvoxels/s)")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  C f16:    {time_c:.4f}s  ({throughput_c:.2f} Gvoxels/s)")
    print(f"  Rust f16: {time_rust:.4f}s  ({throughput_rust:.2f} Gvoxels/s)")
    print()

    speedup = time_c / time_rust
    if speedup > 1:
        print(f"  Rust f16 is {speedup:.2f}x FASTER than C f16")
    else:
        print(f"  C f16 is {1/speedup:.2f}x FASTER than Rust f16")

    print()
    print(f"  f16 speedup over f32:")
    print(f"    C:    {time_c_f32/time_c:.2f}x")
    print(f"    Rust: {time_rust_f32/time_rust:.2f}x")


if __name__ == "__main__":
    main()
