//! Benchmarks for 3D trilinear interpolation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use interp3d_avx2::{affine_transform_3d_f16, affine_transform_3d_f32, affine_transform_3d_f64, f16, scalar, simd, AffineMatrix3D};
use ndarray::Array3;

fn benchmark_affine_transform_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_transform_3d_f32");

    for size in [32, 64, 128, 256].iter() {
        let input = Array3::from_shape_fn((*size, *size, *size), |(z, y, x)| {
            ((z * 100 + y * 10 + x) % 256) as f32 / 255.0
        });

        let matrix = AffineMatrix3D::identity();
        let shift = [0.5, 0.5, 0.5]; // Small shift to ensure interpolation

        let voxels = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(voxels));

        // Auto-dispatch (uses best available: AVX512 > AVX2 > scalar)
        group.bench_with_input(BenchmarkId::new("auto", size), &size, |b, _| {
            b.iter(|| {
                black_box(affine_transform_3d_f32(
                    &input.view(),
                    &matrix,
                    &shift,
                    0.0,
                ))
            })
        });

        // Explicit AVX512
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(BenchmarkId::new("avx512", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), 0.0f32);
                    unsafe {
                        simd::avx512::trilinear_3d_f32_avx512(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Explicit AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), 0.0f32);
                    unsafe {
                        simd::avx2::trilinear_3d_f32_avx2(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                let mut output = Array3::from_elem((*size, *size, *size), 0.0f32);
                scalar::trilinear_3d_scalar(
                    &input.view(),
                    &mut output.view_mut(),
                    &matrix,
                    &shift,
                    0.0,
                );
                black_box(output)
            })
        });
    }

    group.finish();
}

fn benchmark_affine_transform_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_transform_3d_f64");

    for size in [32, 64, 128, 256].iter() {
        let input = Array3::from_shape_fn((*size, *size, *size), |(z, y, x)| {
            ((z * 100 + y * 10 + x) % 256) as f64 / 255.0
        });

        let matrix = AffineMatrix3D::identity();
        let shift = [0.5, 0.5, 0.5];

        let voxels = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(voxels));

        // Auto-dispatch (uses best available: AVX512 > AVX2 > scalar)
        group.bench_with_input(BenchmarkId::new("auto", size), &size, |b, _| {
            b.iter(|| {
                black_box(affine_transform_3d_f64(
                    &input.view(),
                    &matrix,
                    &shift,
                    0.0,
                ))
            })
        });

        // Explicit AVX512
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(BenchmarkId::new("avx512", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), 0.0f64);
                    unsafe {
                        simd::avx512::trilinear_3d_f64_avx512(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Explicit AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), 0.0f64);
                    unsafe {
                        simd::avx2::trilinear_3d_f64_avx2(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                let mut output = Array3::from_elem((*size, *size, *size), 0.0f64);
                scalar::trilinear_3d_scalar(
                    &input.view(),
                    &mut output.view_mut(),
                    &matrix,
                    &shift,
                    0.0,
                );
                black_box(output)
            })
        });
    }

    group.finish();
}

fn benchmark_affine_transform_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_transform_3d_f16");

    for size in [32, 64, 128].iter() {
        let input = Array3::from_shape_fn((*size, *size, *size), |(z, y, x)| {
            f16::from_f32(((z * 100 + y * 10 + x) % 256) as f32 / 255.0)
        });

        let matrix = AffineMatrix3D::identity();
        let shift = [0.5, 0.5, 0.5];

        let voxels = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(voxels));

        // Auto-dispatch (uses best available: AVX512 > AVX2 > scalar)
        group.bench_with_input(BenchmarkId::new("auto", size), &size, |b, _| {
            b.iter(|| {
                black_box(affine_transform_3d_f16(
                    &input.view(),
                    &matrix,
                    &shift,
                    0.0,
                ))
            })
        });

        // Explicit AVX512
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(BenchmarkId::new("avx512", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), f16::ZERO);
                    unsafe {
                        simd::avx512::trilinear_3d_f16_avx512(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Explicit AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
            group.bench_with_input(BenchmarkId::new("avx2", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = Array3::from_elem((*size, *size, *size), f16::ZERO);
                    unsafe {
                        simd::avx2::trilinear_3d_f16_avx2(
                            &input.view(),
                            &mut output.view_mut(),
                            &matrix,
                            &shift,
                            0.0,
                        );
                    }
                    black_box(output)
                })
            });
        }

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                let mut output = Array3::from_elem((*size, *size, *size), f16::ZERO);
                scalar::trilinear_3d_f16_scalar(
                    &input.view(),
                    &mut output.view_mut(),
                    &matrix,
                    &shift,
                    0.0,
                );
                black_box(output)
            })
        });
    }

    group.finish();
}

fn benchmark_rotation_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_3d");

    let size = 128;
    let input = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
        ((z * 100 + y * 10 + x) % 256) as f32 / 255.0
    });

    // 45 degree rotation around Z axis
    let matrix = AffineMatrix3D::rotate_z(std::f64::consts::PI / 4.0);
    let shift = [0.0, 0.0, 0.0];

    let voxels = (size as u64).pow(3);
    group.throughput(Throughput::Elements(voxels));

    group.bench_function("avx2_f32", |b| {
        b.iter(|| {
            black_box(affine_transform_3d_f32(
                &input.view(),
                &matrix,
                &shift,
                0.0,
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_affine_transform_f32,
    benchmark_affine_transform_f64,
    benchmark_affine_transform_f16,
    benchmark_rotation_transform
);
criterion_main!(benches);
