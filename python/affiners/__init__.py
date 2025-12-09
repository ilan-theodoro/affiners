"""
affiners: Fast 3D affine transformations using AVX2/AVX512 SIMD

Main function:
- affine_transform(): Auto-dispatches based on input dtype (float32, float16, uint8)
- zoom(): Convenience function for scaling/zooming 3D arrays

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
    >>>
    >>> # Zoom by 2x
    >>> zoomed = affiners.zoom(data_f32, 2.0)
    >>>
    >>> # Ensure SIMD is being used (raises if scalar fallback is triggered)
    >>> with affiners.disable_scalar_fallback():
    ...     result = affiners.affine_transform(data_f32, matrix)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Sequence

import numpy as np
from numpy.typing import NDArray

from .affiners import (
    _set_scalar_fallback_allowed,
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


@contextmanager
def disable_scalar_fallback() -> Generator[None, None, None]:
    """
    Context manager to disable scalar fallback and ensure SIMD code paths are used.

    When this context manager is active, any operation that would fall back to
    scalar code (due to missing SIMD support) will raise a RuntimeError instead.

    This is useful for:
    - Ensuring performance-critical code paths use SIMD optimizations
    - Debugging to verify which code path is being taken
    - Testing that SIMD implementations are available on the target hardware

    Example:
        >>> import numpy as np
        >>> import affiners
        >>> data = np.random.rand(32, 32, 32).astype(np.float32)
        >>> matrix = np.eye(4)
        >>>
        >>> # This will raise if SIMD is not available
        >>> with affiners.disable_scalar_fallback():
        ...     result = affiners.affine_transform(data, matrix)

    Raises:
        RuntimeError: If an operation attempts to use scalar fallback while
                     this context is active.

    Notes:
        - The flag is thread-local, so it only affects the current thread.
        - Nested contexts are supported; the flag is restored when exiting.
    """
    _set_scalar_fallback_allowed(False)
    try:
        yield
    finally:
        _set_scalar_fallback_allowed(True)


def __getattr__(name):
    """Provide build info on attribute access."""
    if name == "__version_info__":
        return _info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def zoom(
    input: NDArray,
    zoom_factor: float | Sequence[float] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    align_corners: bool = False,
    cval: float = 0.0,
    order: int = 1,
) -> NDArray:
    """
    Zoom (scale) a 3D array by the specified factor(s) or to a target shape.

    Provide exactly one of `zoom_factor` or `output_shape`.

    Args:
        input: 3D numpy array (float32, float16, or uint8)
        zoom_factor: Zoom factor(s). Can be a single number (uniform zoom)
                    or a sequence of 3 numbers (z, y, x zoom factors).
                    Values > 1 enlarge the image, < 1 shrink it.
        output_shape: Target output shape (z, y, x).
        align_corners: If True, corner pixels are aligned (maps [0, N-1] to
                      [0, M-1]). If False (default), uses simple scaling
                      (maps [0, N] to [0, M]).
        cval: Constant value for out-of-bounds (default: 0.0)
        order: Interpolation order (only 1 is supported)

    Returns:
        Zoomed 3D array (same dtype as input)

    Examples:
        >>> import numpy as np
        >>> import affiners
        >>> data = np.random.rand(32, 32, 32).astype(np.float32)
        >>>
        >>> # Zoom by 2x uniformly (enlarge)
        >>> zoomed = affiners.zoom(data, zoom_factor=2.0)  # shape: (64, 64, 64)
        >>>
        >>> # Zoom by 0.5x (shrink)
        >>> shrunk = affiners.zoom(data, zoom_factor=0.5)  # shape: (16, 16, 16)
        >>>
        >>> # Zoom by different factors per axis
        >>> zoomed = affiners.zoom(data, zoom_factor=(2.0, 1.5, 3.0))
        >>>
        >>> # Resample to a specific output shape
        >>> resampled = affiners.zoom(data, output_shape=(64, 48, 96))
        >>>
        >>> # With corner alignment
        >>> zoomed = affiners.zoom(data, zoom_factor=2.0, align_corners=True)
    """
    # Validate that exactly one of zoom_factor or output_shape is provided
    if zoom_factor is None and output_shape is None:
        raise ValueError("Must provide either zoom_factor or output_shape")
    if zoom_factor is not None and output_shape is not None:
        raise ValueError(
            "Cannot provide both zoom_factor and output_shape. "
            "Use zoom_factor to scale by a factor, or output_shape to resample "
            "to a specific size."
        )

    # Validate input
    if input.ndim != 3:
        raise ValueError(f"Input must be 3D, got {input.ndim}D")
    in_shape = input.shape

    # Compute output shape from zoom_factor if needed
    if output_shape is None:
        assert zoom_factor is not None  # for type checker
        if isinstance(zoom_factor, (int, float)):
            zoom_factors: tuple[float, float, float] = (float(zoom_factor),) * 3
        else:
            zf_list = list(zoom_factor)
            if len(zf_list) != 3:
                raise ValueError(
                    f"zoom_factor must be a scalar or length-3 sequence, got {len(zf_list)}"
                )
            zoom_factors = (float(zf_list[0]), float(zf_list[1]), float(zf_list[2]))

        output_shape = (
            int(round(in_shape[0] * zoom_factors[0])),
            int(round(in_shape[1] * zoom_factors[1])),
            int(round(in_shape[2] * zoom_factors[2])),
        )

    # Compute scale factors
    scales: list[float] = []
    for i in range(3):
        N = in_shape[i]
        M = output_shape[i]
        if align_corners:
            # Corner alignment: map [0, M-1] → [0, N-1]
            scale = (N - 1) / (M - 1) if M > 1 else 1.0
        else:
            # Simple scaling: map [0, M] → [0, N]
            scale = N / M
        scales.append(scale)

    # Build affine matrix (no offset needed for either mode)
    matrix = np.array(
        [
            [scales[0], 0, 0, 0],
            [0, scales[1], 0, 0],
            [0, 0, scales[2], 0],
            [0, 0, 0, 1.0],
        ],
        dtype=np.float64,
    )

    return affine_transform(
        input, matrix, output_shape=output_shape, cval=cval, order=order
    )


__all__ = [
    "affine_transform",
    "affine_transform_f32",
    "affine_transform_f16",
    "affine_transform_u8",
    "apply_warp",
    "build_info",
    "disable_scalar_fallback",
    "zoom",
]
