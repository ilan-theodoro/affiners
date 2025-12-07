# affiners

Fast 3D affine transformations with trilinear interpolation using AVX2/AVX512 SIMD.

## Performance

| Data Type | Throughput | Memory/Voxel | Speedup |
|-----------|------------|--------------|---------|
| f32 | 1.5 Gvoxels/s | 8 bytes | 1.0x |
| f16 | 1.2 Gvoxels/s | 4 bytes | - |
| **u8** | **3.3 Gvoxels/s** | **2 bytes** | **2.2x** |

**Use `u8` for image data** (microscopy, CT, MRI) to get 2.2x faster processing with 4x less memory!

## Performance vs scipy

Benchmark on AMD Ryzen 9 9950X (32 threads):

| Volume | scipy | affiners f32 | affiners u8 | Speedup (f32) | Speedup (u8) |
|--------|-------|--------------|-------------|---------------|--------------|
| 512³ | 890 ms | 89 ms | 40 ms | **10x** | **22x** |
| 1024³ | 7.1 s | 710 ms | 320 ms | **10x** | **22x** |

## Installation

```bash
pip install affine-rs
```

Or build from source:

```bash
pip install .
```

## Usage

### Python

```python
import numpy as np
import affiners

# Define 4x4 homogeneous transformation matrix
# Format: [[m00, m01, m02, tz],
#          [m10, m11, m12, ty],
#          [m20, m21, m22, tx],
#          [0,   0,   0,   1 ]]
matrix = np.array([
    [1.0, 0.25, 0.01, -10.0],
    [0.0, 1.0, 0.0, -5.0],
    [0.0, -0.02, 1.0, 8.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# Float32 data
input_f32 = np.random.rand(512, 512, 512).astype(np.float32)
output_f32 = affiners.affine_transform(input_f32, matrix)

# uint8 data (2.2x faster!)
input_u8 = np.random.randint(0, 256, (512, 512, 512), dtype=np.uint8)
output_u8 = affiners.affine_transform_u8(input_u8, matrix)
```

### Check Build Info

```python
import affiners

print(affiners.__version__)  # '0.1.0'
print(affiners.build_info())
# {'version': '0.1.0', 'simd': {'avx2': True, 'avx512f': True, ...}, 
#  'backend_f32': 'avx512', 'backend_u8': 'avx2', 'num_threads': 32, ...}
```

### Rust

```rust
use ndarray::{Array2, Array3};
use affiners::{affine_transform_3d_f32, affine_transform_3d_u8};

// Create 4x4 homogeneous transformation matrix (identity with translation)
let matrix = Array2::from_shape_vec((4, 4), vec![
    1.0, 0.0, 0.0, 10.0,  // tz = 10
    0.0, 1.0, 0.0, 20.0,  // ty = 20
    0.0, 0.0, 1.0, 30.0,  // tx = 30
    0.0, 0.0, 0.0, 1.0,
]).unwrap();

// Float32
let input_f32 = Array3::<f32>::zeros((100, 100, 100));
let output_f32 = affine_transform_3d_f32(&input_f32.view(), &matrix.view(), 0.0);

// uint8 (2.2x faster!)
let input_u8 = Array3::<u8>::zeros((100, 100, 100));
let output_u8 = affine_transform_3d_u8(&input_u8.view(), &matrix.view(), 0);
```

## API Reference

### Python

| Function | Input Type | Description |
|----------|------------|-------------|
| `affine_transform(input, matrix, cval)` | float32 | Standard floating point |
| `affine_transform_f16(input, matrix, cval)` | float16 | Half precision (pass as `.view(np.uint16)`) |
| `affine_transform_u8(input, matrix, cval)` | uint8 | **2.2x faster**, 4x less memory |
| `build_info()` | - | Get version, SIMD features, and backend info |

### Parameters

- `input`: 3D numpy array (C-contiguous)
- `matrix`: 4x4 homogeneous transformation matrix (float64)
  ```
  [[m00, m01, m02, tz],
   [m10, m11, m12, ty],
   [m20, m21, m22, tx],
   [0,   0,   0,   1 ]]
  ```
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
- **Native compilation**: Optimized for host CPU features (`target-cpu=native`)

## License

BSD-3-Clause
