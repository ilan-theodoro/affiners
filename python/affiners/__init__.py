"""
affiners: Fast 3D affine transformations using AVX2/AVX512 SIMD

Supported data types:
- f32: affine_transform() - Standard floating point (~1.5 Gvoxels/s)
- f16: affine_transform_f16() - Half precision, 2x less memory
- u8: affine_transform_u8() - 2.2x faster (~3.3 Gvoxels/s), 4x less memory

Example:
    >>> import affiners
    >>> print(affiners.build_info())
    {'version': '0.1.0', 'simd': {'avx2': True, 'avx512f': True, ...}, ...}
"""

from .affiners import (
    affine_transform,
    affine_transform_f16,
    affine_transform_u8,
    build_info,
)

# Get version and build info
_info = build_info()
__version__ = _info["version"]


def __getattr__(name):
    """Provide build info on attribute access."""
    if name == "__version_info__":
        return _info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "affine_transform",
    "affine_transform_f16",
    "affine_transform_u8",
    "build_info",
]

