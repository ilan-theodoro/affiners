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
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

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


def zoom(
    input: NDArray,
    zoom_factor: float | Sequence[float],
    output_shape: tuple[int, int, int] | None = None,
    align_corners: bool = False,
    cval: float = 0.0,
    order: int = 1,
) -> NDArray:
    """
    Zoom (scale) a 3D array by the specified factor(s).

    Args:
        input: 3D numpy array (float32, float16, or uint8)
        zoom_factor: Zoom factor(s). Can be a single number (uniform zoom)
                    or a sequence of 3 numbers (z, y, x zoom factors).
                    Values > 1 enlarge the image, < 1 shrink it.
        output_shape: Output shape (z, y, x). If None, computed as
                     input_shape * zoom_factor.
        align_corners: If True, corner pixels are aligned (like PyTorch
                      align_corners=True). Maps the corner pixels of input
                      and output to each other.
                      If False (default), uses pixel-center interpolation
                      (like PyTorch align_corners=False).
        cval: Constant value for out-of-bounds (default: 0.0)
        order: Interpolation order (only 1 is supported)

    Returns:
        Zoomed 3D array (same dtype as input)

    Examples:
        >>> import numpy as np
        >>> import affiners
        >>> data = np.random.rand(32, 32, 32).astype(np.float32)
        >>>
        >>> # Zoom by 2x uniformly
        >>> zoomed = affiners.zoom(data, 2.0)  # shape: (64, 64, 64)
        >>>
        >>> # Zoom by different factors per axis
        >>> zoomed = affiners.zoom(data, (2.0, 1.5, 3.0))
        >>>
        >>> # Zoom with corner alignment (PyTorch-style align_corners=True)
        >>> zoomed = affiners.zoom(data, 2.0, align_corners=True)

    Notes:
        The align_corners flag mimics PyTorch's behavior:

        - align_corners=False (default): Pixel values are interpolated as if
          they are located at pixel centers (0.5, 1.5, 2.5, ...). This treats
          the input as spanning [0, N] and output as spanning [0, M].
          Out-of-bounds coordinates are clamped (matching PyTorch).

        - align_corners=True: Corner pixels are aligned. This treats the input
          as spanning [0, N-1] and output as spanning [0, M-1], mapping corners
          exactly to corners.
    """
    # Handle zoom_factor
    if isinstance(zoom_factor, (int, float)):
        zoom_factors: tuple[float, float, float] = (float(zoom_factor),) * 3
    else:
        zf_list = list(zoom_factor)
        if len(zf_list) != 3:
            raise ValueError(
                f"zoom_factor must be a scalar or length-3 sequence, got {len(zf_list)}"
            )
        zoom_factors = (float(zf_list[0]), float(zf_list[1]), float(zf_list[2]))

    # Validate input
    if input.ndim != 3:
        raise ValueError(f"Input must be 3D, got {input.ndim}D")
    in_shape = input.shape

    # Compute output shape if not provided
    if output_shape is None:
        output_shape = (
            int(round(in_shape[0] * zoom_factors[0])),
            int(round(in_shape[1] * zoom_factors[1])),
            int(round(in_shape[2] * zoom_factors[2])),
        )

    if align_corners:
        # Corner alignment: map [0, M-1] → [0, N-1]
        # For output o, input i = o * (N-1)/(M-1)
        scales: list[float] = []
        offsets: list[float] = []

        for i in range(3):
            N = in_shape[i]
            M = output_shape[i]
            if M == 1:
                scale = 1.0
            else:
                scale = (N - 1) / (M - 1)
            scales.append(scale)
            offsets.append(0.0)

        matrix = np.array(
            [
                [scales[0], 0, 0, offsets[0]],
                [0, scales[1], 0, offsets[1]],
                [0, 0, scales[2], offsets[2]],
                [0, 0, 0, 1.0],
            ],
            dtype=np.float64,
        )

        return affine_transform(
            input, matrix, output_shape=output_shape, cval=cval, order=order
        )
    else:
        # Pixel-center alignment (align_corners=False)
        # PyTorch formula: input = (output + 0.5) * (N/M) - 0.5
        # PyTorch clamps out-of-bounds coordinates to [0, N-1]
        #
        # To simulate clamping, we pad the input with edge replication
        # and adjust coordinates to account for the padding.
        pad_width = 1

        # Pad input with edge replication
        padded_input = np.pad(input, pad_width, mode="edge")

        # Compute scales and offsets for the padded input
        # Original formula: input = (output + 0.5) * (N/M) - 0.5
        # With padding of 1, coordinates shift by 1:
        # padded_input = input + 1 = (output + 0.5) * (N/M) - 0.5 + 1
        #              = output * (N/M) + 0.5 * (N/M) + 0.5
        scales = []
        offsets = []

        for i in range(3):
            N = in_shape[i]
            M = output_shape[i]
            scale = N / M
            # offset = 0.5 * (N/M) - 0.5 + pad_width
            #        = 0.5 * scale - 0.5 + 1
            #        = 0.5 * scale + 0.5
            #        = 0.5 * (scale + 1)
            offset = 0.5 * (scale + 1.0)
            scales.append(scale)
            offsets.append(offset)

        matrix = np.array(
            [
                [scales[0], 0, 0, offsets[0]],
                [0, scales[1], 0, offsets[1]],
                [0, 0, scales[2], offsets[2]],
                [0, 0, 0, 1.0],
            ],
            dtype=np.float64,
        )

        return affine_transform(
            padded_input, matrix, output_shape=output_shape, cval=cval, order=order
        )


__all__ = [
    "affine_transform",
    "affine_transform_f32",
    "affine_transform_f16",
    "affine_transform_u8",
    "apply_warp",
    "build_info",
    "zoom",
]
