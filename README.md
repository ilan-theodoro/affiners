# interp3d-avx2

Fast 3D trilinear interpolation using AVX2 SIMD instructions.

This is a Rust port of the [interp-simd](https://github.com/ilan-theodoro/interp-simd) library,
optimized for 3D interpolation with order=1 (trilinear).

## Features

- **AVX2 SIMD**: Processes 8 f32 or 4 f64 values per iteration
- **Parallel execution**: Uses rayon for multi-threaded processing
- **Python bindings**: Via PyO3 and maturin
- **ndarray integration**: Works directly with ndarray/numpy arrays

## Installation

```bash
pip install interp3d-avx2
```

Or build from source:

```bash
maturin develop --release
```

## Usage

### Python

```python
import numpy as np
from interp3d_avx2 import affine_transform

# Create input data
input_data = np.random.rand(100, 100, 100).astype(np.float32)

# Define transformation
matrix = np.eye(3)
offset = np.array([1.0, 2.0, 3.0])

# Apply transformation
output = affine_transform(input_data, matrix, offset=offset, order=1)
```

### Rust

```rust
use ndarray::Array3;
use interp3d_avx2::{affine_transform_3d_f32, AffineMatrix3D};

let input = Array3::<f32>::zeros((100, 100, 100));
let matrix = AffineMatrix3D::identity();
let shift = [10.0, 20.0, 30.0];
let output = affine_transform_3d_f32(&input.view(), &matrix, &shift, 0.0);
```

## Performance

Benchmarks on a modern x86-64 CPU:

| Size | scipy.ndimage | interp-simd (C) | interp3d-avx2 (Rust) |
|------|---------------|-----------------|---------------------|
| 128³ | 54 ms         | 0.9 ms          | 0.6 ms             |
| 256³ | 436 ms        | 10 ms           | 11 ms              |

## License

BSD-3-Clause
