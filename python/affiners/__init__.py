"""
affiners: Fast 3D affine transformations using AVX2/AVX512 SIMD

Main function:
- affine_transform(): Auto-dispatches based on input dtype (float32, float16, uint8)

Type-specific functions (for explicit control):
- affine_transform_f32(): Standard floating point (~1.5 Gvoxels/s)
- affine_transform_f16(): Half precision, 2x less memory
- affine_transform_u8(): 2.2x faster (~3.3 Gvoxels/s), 4x less memory

Example:
    >>> import numpy as np
    >>> import affiners
    >>>
    >>> # Works with any supported dtype
    >>> data_f32 = np.random.rand(32, 32, 32).astype(np.float32)
    >>> data_f16 = data_f32.astype(np.float16)
    >>> data_u8 = (data_f32 * 255).astype(np.uint8)
    >>> matrix = np.eye(4)  # Any numeric dtype works
    >>>
    >>> result_f32 = affiners.affine_transform(data_f32, matrix)  # returns float32
    >>> result_f16 = affiners.affine_transform(data_f16, matrix)  # returns float16
    >>> result_u8 = affiners.affine_transform(data_u8, matrix)    # returns uint8
"""

from .affiners import (
    affine_transform,
    affine_transform_f32,
    affine_transform_f16,
    affine_transform_u8,
    apply_warp,
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
    "affine_transform_f32",
    "affine_transform_f16",
    "affine_transform_u8",
    "apply_warp",
    "build_info",
]
