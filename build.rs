fn main() {
    // Enable AVX-512 only on Rust 1.89+ where it's stable
    // On older Rust versions, AVX-512 code is excluded and AVX2 fallback is used
    let version = rustc_version();

    if let Some((major, minor)) = version {
        if major > 1 || (major == 1 && minor >= 89) {
            println!("cargo::rustc-cfg=has_stable_avx512");
        }
    }

    // Tell Cargo about our custom cfg to suppress warnings
    println!("cargo::rustc-check-cfg=cfg(has_stable_avx512)");
}

fn rustc_version() -> Option<(u32, u32)> {
    let output = std::process::Command::new(
        std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string()),
    )
    .arg("--version")
    .output()
    .ok()?;

    let version_str = String::from_utf8(output.stdout).ok()?;
    // Parse "rustc 1.89.0 ..." or "rustc 1.95.0-nightly ..."
    let version_part = version_str.split_whitespace().nth(1)?;
    let mut parts = version_part.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.split('-').next()?.parse().ok()?;

    Some((major, minor))
}
