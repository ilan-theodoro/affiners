//! SIMD-optimized interpolation implementations

#[cfg(target_arch = "x86_64")]
pub mod avx2;

// AVX-512 requires Rust 1.89+ (stable) or nightly with feature flags
#[cfg(all(target_arch = "x86_64", has_stable_avx512))]
pub mod avx512;
