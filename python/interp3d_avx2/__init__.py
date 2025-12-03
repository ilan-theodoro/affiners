"""
interp3d-avx2: Fast 3D trilinear interpolation using AVX2 SIMD

This package provides high-performance 3D interpolation functions optimized
for modern x86-64 processors with AVX2 support.

Example:
    >>> import numpy as np
    >>> from interp3d_avx2 import affine_transform
    >>>
    >>> input_data = np.random.rand(100, 100, 100).astype(np.float32)
    >>> matrix = np.eye(3)
    >>> offset = np.array([1.0, 2.0, 3.0])
    >>> output = affine_transform(input_data, matrix, offset=offset)
"""

from .interp3d_avx2 import affine_transform, affine_transform_f64, affine_transform_f16

__all__ = ["affine_transform", "affine_transform_f64", "affine_transform_f16"]
__version__ = "0.1.0"
