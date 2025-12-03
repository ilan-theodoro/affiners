# interp3d-avx2

Fast 3D trilinear interpolation using AVX2/AVX512 SIMD instructions.

## Performance

| Data Type | Throughput | Memory/Voxel | Speedup |
|-----------|------------|--------------|---------|
| f32 | 1.5 Gvoxels/s | 8 bytes | 1.0x |
| f16 | 1.2 Gvoxels/s | 4 bytes | - |
| **u8** | **3.3 Gvoxels/s** | **2 bytes** | **2.2x** |

**Use `u8` for image data** (microscopy, CT, MRI) to get 2.2x faster processing with 4x less memory!

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
from interp3d_avx2 import affine_transform, affine_transform_u8

# Define transformation
matrix = np.array([
    [1.0, 0.25, 0.01],
    [0.0, 1.0, 0.0],
    [0.0, -0.02, 1.0],
], dtype=np.float64)
offset = np.array([-10.0, -5.0, 8.0])

# Float32 data
input_f32 = np.random.rand(512, 512, 512).astype(np.float32)
output_f32 = affine_transform(input_f32, matrix, offset=offset)

# uint8 data (2.2x faster!)
input_u8 = np.random.randint(0, 256, (512, 512, 512), dtype=np.uint8)
output_u8 = affine_transform_u8(input_u8, matrix, offset=offset)
```

### Rust

```rust
use ndarray::Array3;
use interp3d_avx2::{affine_transform_3d_f32, affine_transform_3d_u8, AffineMatrix3D};

// Float32
let input_f32 = Array3::<f32>::zeros((100, 100, 100));
let matrix = AffineMatrix3D::identity();
let shift = [10.0, 20.0, 30.0];
let output_f32 = affine_transform_3d_f32(&input_f32.view(), &matrix, &shift, 0.0);

// uint8 (2.2x faster!)
let input_u8 = Array3::<u8>::zeros((100, 100, 100));
let output_u8 = affine_transform_3d_u8(&input_u8.view(), &matrix, &shift, 0);
```

## API Reference

### Python

| Function | Input Type | Description |
|----------|------------|-------------|
| `affine_transform(input, matrix, offset, cval)` | float32 | Standard floating point |
| `affine_transform_f16(input, matrix, offset, cval)` | float16 | Half precision (pass as `.view(np.uint16)`) |
| `affine_transform_u8(input, matrix, offset, cval)` | uint8 | **2.2x faster**, 4x less memory |

### Parameters

- `input`: 3D numpy array (C-contiguous)
- `matrix`: 3x3 transformation matrix (float64)
- `offset`: Translation vector [z, y, x] (optional, default: [0, 0, 0])
- `cval`: Constant value for out-of-bounds (default: 0)

## When to Use Each Type

| Use Case | Recommended Type |
|----------|-----------------|
| Image data (CT, MRI, microscopy) | `u8` - 2.2x faster |
| Scientific floating-point data | `f32` |
| Reduced memory footprint | `f16` or `u8` |
| Maximum precision | `f32` |

## Memory Requirements

| Volume Size | f32 | u8 |
|-------------|-----|-----|
| 512³ | 1.1 GB | 0.3 GB |
| 1024³ | 8.6 GB | 2.1 GB |
| 2048³ | 68.7 GB | 17.2 GB |

## Features

- **AVX2/AVX512 SIMD**: Processes 8-16 values per iteration
- **Multi-threaded**: Uses rayon for parallel z-slice processing
- **Memory efficient**: u8 uses 4x less memory than f32
- **Python bindings**: Via PyO3 and maturin
- **Zero-copy**: Works directly with numpy arrays

## License

BSD-3-Clause
