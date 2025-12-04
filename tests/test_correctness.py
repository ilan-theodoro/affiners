"""
Transitive correctness tests for affiners.

Test hierarchy:
1. f32 == scipy.ndimage.affine_transform (ground truth)
2. f16 ≈ f32 (within tolerance)
3. u8 ≈ f32 (within tolerance)
"""

import numpy as np
import pytest
from scipy import ndimage

import affiners


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def small_volume_f32():
    """Small test volume for quick tests."""
    np.random.seed(42)
    return np.random.rand(32, 32, 32).astype(np.float32)


@pytest.fixture
def medium_volume_f32():
    """Medium test volume for thorough tests."""
    np.random.seed(42)
    return np.random.rand(64, 64, 64).astype(np.float32)


@pytest.fixture
def identity_matrix():
    """Identity transformation matrix."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def shear_matrix():
    """Shear transformation matrix (typical use case)."""
    return np.array([
        [1.0, 0.25, 0.01],
        [0.0, 1.0, 0.0],
        [0.0, -0.02, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def rotation_matrix():
    """Small rotation matrix."""
    angle = 0.1  # ~5.7 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ], dtype=np.float64)


@pytest.fixture
def translation_offset():
    """Translation offset."""
    return np.array([-5.0, 3.0, -2.0], dtype=np.float64)


def compare_interior(result, expected, margin=3, rtol=1e-5, atol=1e-5):
    """Compare interior voxels, excluding boundary margin."""
    r_interior = result[margin:-margin, margin:-margin, margin:-margin]
    e_interior = expected[margin:-margin, margin:-margin, margin:-margin]
    np.testing.assert_allclose(r_interior, e_interior, rtol=rtol, atol=atol)


# =============================================================================
# Level 1: f32 vs scipy (ground truth)
# =============================================================================

class TestF32VsScipy:
    """Test that our f32 implementation matches scipy for interior voxels."""

    def test_identity_transform(self, small_volume_f32, identity_matrix):
        """Identity transform should return input unchanged."""
        offset = np.array([0.0, 0.0, 0.0])
        
        result = affiners.affine_transform(small_volume_f32, identity_matrix, offset=offset)
        expected = ndimage.affine_transform(small_volume_f32, identity_matrix, offset=offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=2)

    def test_translation_only(self, small_volume_f32, identity_matrix, translation_offset):
        """Pure translation should match scipy."""
        result = affiners.affine_transform(small_volume_f32, identity_matrix, offset=translation_offset)
        expected = ndimage.affine_transform(small_volume_f32, identity_matrix, offset=translation_offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=8)

    def test_shear_transform(self, small_volume_f32, shear_matrix, translation_offset):
        """Shear transform should match scipy."""
        result = affiners.affine_transform(small_volume_f32, shear_matrix, offset=translation_offset)
        expected = ndimage.affine_transform(small_volume_f32, shear_matrix, offset=translation_offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=10)

    def test_rotation_transform(self, small_volume_f32, rotation_matrix, translation_offset):
        """Rotation transform should match scipy."""
        result = affiners.affine_transform(small_volume_f32, rotation_matrix, offset=translation_offset)
        expected = ndimage.affine_transform(small_volume_f32, rotation_matrix, offset=translation_offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=8)

    def test_medium_volume(self, medium_volume_f32, shear_matrix, translation_offset):
        """Test on medium-sized volume."""
        result = affiners.affine_transform(medium_volume_f32, shear_matrix, offset=translation_offset)
        expected = ndimage.affine_transform(medium_volume_f32, shear_matrix, offset=translation_offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=15)

    def test_cval_handling(self, small_volume_f32, identity_matrix):
        """Test constant value for out-of-bounds."""
        offset = np.array([100.0, 100.0, 100.0])  # Large offset to go out of bounds
        cval = -999.0
        
        result = affiners.affine_transform(small_volume_f32, identity_matrix, offset=offset, cval=cval)
        expected = ndimage.affine_transform(small_volume_f32, identity_matrix, offset=offset, order=1, cval=cval)
        
        # Both should be all cval since we shifted completely out of bounds
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_no_transform_interior(self, small_volume_f32, identity_matrix):
        """Interior voxels with identity should be exactly the same."""
        offset = np.array([0.0, 0.0, 0.0])
        
        result = affiners.affine_transform(small_volume_f32, identity_matrix, offset=offset)
        
        # Interior should be identical to input
        margin = 1
        np.testing.assert_allclose(
            result[margin:-margin, margin:-margin, margin:-margin],
            small_volume_f32[margin:-margin, margin:-margin, margin:-margin],
            rtol=1e-6, atol=1e-7
        )


# =============================================================================
# Level 2: f16 vs f32 (our implementation)
# =============================================================================

class TestF16VsF32:
    """Test that f16 implementation is close to our f32 implementation."""

    def test_identity_transform(self, small_volume_f32, identity_matrix):
        """Identity transform f16 should be close to f32."""
        offset = np.array([0.0, 0.0, 0.0])
        
        # f32 result (ground truth for this test)
        result_f32 = affiners.affine_transform(small_volume_f32, identity_matrix, offset=offset)
        
        # f16 result - convert input to f16, run, convert back
        input_f16 = small_volume_f32.astype(np.float16)
        result_f16 = affiners.affine_transform_f16(
            input_f16.view(np.uint16), identity_matrix, offset=offset
        ).view(np.float16).astype(np.float32)
        
        # f16 has ~3 decimal digits of precision, allow 1e-2 relative tolerance
        compare_interior(result_f16, result_f32, margin=2, rtol=1e-2, atol=1e-3)

    def test_shear_transform(self, small_volume_f32, shear_matrix, translation_offset):
        """Shear transform f16 should be close to f32."""
        result_f32 = affiners.affine_transform(small_volume_f32, shear_matrix, offset=translation_offset)
        
        input_f16 = small_volume_f32.astype(np.float16)
        result_f16 = affiners.affine_transform_f16(
            input_f16.view(np.uint16), shear_matrix, offset=translation_offset
        ).view(np.float16).astype(np.float32)
        
        compare_interior(result_f16, result_f32, margin=10, rtol=1e-2, atol=1e-3)

    def test_rotation_transform(self, small_volume_f32, rotation_matrix, translation_offset):
        """Rotation transform f16 should be close to f32."""
        result_f32 = affiners.affine_transform(small_volume_f32, rotation_matrix, offset=translation_offset)
        
        input_f16 = small_volume_f32.astype(np.float16)
        result_f16 = affiners.affine_transform_f16(
            input_f16.view(np.uint16), rotation_matrix, offset=translation_offset
        ).view(np.float16).astype(np.float32)
        
        compare_interior(result_f16, result_f32, margin=8, rtol=1e-2, atol=1e-3)

    def test_medium_volume(self, medium_volume_f32, shear_matrix, translation_offset):
        """Test f16 on medium-sized volume."""
        result_f32 = affiners.affine_transform(medium_volume_f32, shear_matrix, offset=translation_offset)
        
        input_f16 = medium_volume_f32.astype(np.float16)
        result_f16 = affiners.affine_transform_f16(
            input_f16.view(np.uint16), shear_matrix, offset=translation_offset
        ).view(np.float16).astype(np.float32)
        
        compare_interior(result_f16, result_f32, margin=15, rtol=1e-2, atol=1e-3)


# =============================================================================
# Level 3: u8 vs f32 (our implementation)
# =============================================================================

class TestU8VsF32:
    """Test that u8 implementation is close to our f32 implementation."""

    @pytest.fixture
    def small_volume_u8(self):
        """Small test volume in u8."""
        np.random.seed(42)
        return np.random.randint(0, 256, (32, 32, 32), dtype=np.uint8)

    @pytest.fixture
    def medium_volume_u8(self):
        """Medium test volume in u8."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64, 64), dtype=np.uint8)

    def _compare_u8(self, result_u8, input_u8, matrix, offset, margin=5):
        """Compare u8 result with f32 reference."""
        input_f32 = input_u8.astype(np.float32)
        result_f32 = affiners.affine_transform(input_f32, matrix, offset=offset)
        result_f32_u8 = np.clip(np.round(result_f32), 0, 255).astype(np.uint8)
        
        # Compare interior only
        result_f32_interior = result_f32_u8[margin:-margin, margin:-margin, margin:-margin]
        result_u8_interior = result_u8[margin:-margin, margin:-margin, margin:-margin]
        
        diff = np.abs(result_u8_interior.astype(np.int16) - result_f32_interior.astype(np.int16))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        assert max_diff <= 1, f"Max diff: {max_diff}"
        assert mean_diff < 0.6, f"Mean diff: {mean_diff}"

    def test_identity_transform(self, small_volume_u8, identity_matrix):
        """Identity transform u8 should be close to f32."""
        offset = np.array([0.0, 0.0, 0.0])
        result_u8 = affiners.affine_transform_u8(small_volume_u8, identity_matrix, offset=offset)
        self._compare_u8(result_u8, small_volume_u8, identity_matrix, offset, margin=2)

    def test_shear_transform(self, small_volume_u8, shear_matrix, translation_offset):
        """Shear transform u8 should be close to f32."""
        result_u8 = affiners.affine_transform_u8(small_volume_u8, shear_matrix, offset=translation_offset)
        self._compare_u8(result_u8, small_volume_u8, shear_matrix, translation_offset, margin=10)

    def test_rotation_transform(self, small_volume_u8, rotation_matrix, translation_offset):
        """Rotation transform u8 should be close to f32."""
        result_u8 = affiners.affine_transform_u8(small_volume_u8, rotation_matrix, offset=translation_offset)
        self._compare_u8(result_u8, small_volume_u8, rotation_matrix, translation_offset, margin=8)

    def test_medium_volume(self, medium_volume_u8, shear_matrix, translation_offset):
        """Test u8 on medium-sized volume."""
        result_u8 = affiners.affine_transform_u8(medium_volume_u8, shear_matrix, offset=translation_offset)
        self._compare_u8(result_u8, medium_volume_u8, shear_matrix, translation_offset, margin=15)

    def test_cval_handling(self, small_volume_u8, identity_matrix):
        """Test constant value for out-of-bounds with u8."""
        offset = np.array([100.0, 100.0, 100.0])  # Large offset
        cval = 128
        
        result = affiners.affine_transform_u8(small_volume_u8, identity_matrix, offset=offset, cval=cval)
        
        # Most of the output should be cval
        assert np.sum(result == cval) > 0.9 * result.size


# =============================================================================
# Build info tests
# =============================================================================

class TestBuildInfo:
    """Test build_info function."""

    def test_build_info_returns_dict(self):
        """build_info should return a dictionary."""
        info = affiners.build_info()
        assert isinstance(info, dict)

    def test_build_info_has_version(self):
        """build_info should contain version."""
        info = affiners.build_info()
        assert "version" in info
        assert info["version"] == affiners.__version__

    def test_build_info_has_simd(self):
        """build_info should contain SIMD feature flags."""
        info = affiners.build_info()
        assert "simd" in info
        assert "avx2" in info["simd"]
        assert "avx512f" in info["simd"]

    def test_build_info_has_backends(self):
        """build_info should contain backend selection."""
        info = affiners.build_info()
        assert "backend_f32" in info
        assert "backend_f16" in info
        assert "backend_u8" in info


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_volume(self, identity_matrix):
        """Test with small 8x8x8 volume."""
        np.random.seed(42)
        input_data = np.random.rand(8, 8, 8).astype(np.float32)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = affiners.affine_transform(input_data, identity_matrix, offset=offset)
        expected = ndimage.affine_transform(input_data, identity_matrix, offset=offset, order=1, cval=0.0)
        
        # Check interior
        compare_interior(result, expected, margin=1)

    def test_asymmetric_volume(self, shear_matrix, translation_offset):
        """Test with non-cubic volume."""
        np.random.seed(42)
        input_data = np.random.rand(16, 32, 64).astype(np.float32)
        
        result = affiners.affine_transform(input_data, shear_matrix, offset=translation_offset)
        expected = ndimage.affine_transform(input_data, shear_matrix, offset=translation_offset, order=1, cval=0.0)
        
        # Check interior with appropriate margin for each dimension
        margin_z, margin_y, margin_x = 5, 10, 15
        r_interior = result[margin_z:-margin_z, margin_y:-margin_y, margin_x:-margin_x]
        e_interior = expected[margin_z:-margin_z, margin_y:-margin_y, margin_x:-margin_x]
        np.testing.assert_allclose(r_interior, e_interior, rtol=1e-5, atol=1e-6)

    def test_zeros_input(self, shear_matrix, translation_offset):
        """Test with all-zeros input."""
        input_data = np.zeros((16, 16, 16), dtype=np.float32)
        
        result = affiners.affine_transform(input_data, shear_matrix, offset=translation_offset)
        
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_ones_input(self, identity_matrix):
        """Test with all-ones input."""
        input_data = np.ones((16, 16, 16), dtype=np.float32)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = affiners.affine_transform(input_data, identity_matrix, offset=offset)
        
        # Interior should be all ones
        margin = 2
        np.testing.assert_allclose(result[margin:-margin, margin:-margin, margin:-margin], 1.0, rtol=1e-5)

    def test_large_values(self, identity_matrix):
        """Test with large float values."""
        np.random.seed(42)
        input_data = (np.random.rand(16, 16, 16) * 1e6).astype(np.float32)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = affiners.affine_transform(input_data, identity_matrix, offset=offset)
        expected = ndimage.affine_transform(input_data, identity_matrix, offset=offset, order=1, cval=0.0)
        
        compare_interior(result, expected, margin=2, rtol=1e-4, atol=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
