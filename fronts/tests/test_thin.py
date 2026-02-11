"""
Tests for the Cornillon-style front thinning algorithm (thin_cc).
"""

import numpy as np
import pytest

from fronts.finding.thin_cc import (
    thin,
    thin_fronts,
    compute_gradient_magnitude,
    MIN_VALID_SST_K,
    MIN_GRADIENT_K,
)


class TestThin:
    """Tests for the core thin() function."""

    def test_basic_vertical_front(self):
        """Test thinning of a vertical front band."""
        # Create SST field with horizontal gradient (cold left, warm right)
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        sst[:, 25:] = 290.0  # 5K jump at column 25

        # Create wide front band (3 pixels wide) at the gradient
        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4  # 3-pixel wide front

        thinned = thin(sst, fronts)

        # Should thin to single-pixel width
        assert np.sum(thinned == 4) < np.sum(fronts == 4)
        # Should have fronts in the middle of the band
        assert np.any(thinned[:, 25] == 4)

    def test_basic_horizontal_front(self):
        """Test thinning of a horizontal front band."""
        # Create SST field with vertical gradient (cold top, warm bottom)
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        sst[25:, :] = 290.0  # 5K jump at row 25

        # Create wide front band (3 pixels wide) at the gradient
        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[24:27, :] = 4  # 3-pixel wide front

        thinned = thin(sst, fronts)

        # Should thin to single-pixel width
        assert np.sum(thinned == 4) < np.sum(fronts == 4)
        # Should have fronts in the middle of the band
        assert np.any(thinned[25, :] == 4)

    def test_diagonal_front(self):
        """Test thinning of a diagonal front."""
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        # Create diagonal gradient
        for i in range(50):
            sst[i, i:] = 290.0

        # Create diagonal front band
        fronts = np.zeros((50, 50), dtype=np.int16)
        for i in range(2, 48):
            fronts[i-1:i+2, i-1:i+2] = 4

        thinned = thin(sst, fronts)

        # Should reduce pixel count
        assert np.sum(thinned == 4) < np.sum(fronts == 4)
        # Should still have some fronts
        assert np.sum(thinned == 4) > 0

    def test_no_fronts_input(self):
        """Test with no fronts in input."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        fronts = np.zeros((50, 50), dtype=np.int16)

        thinned = thin(sst, fronts)

        assert np.sum(thinned == 4) == 0

    def test_nan_handling(self):
        """Test that NaN values are handled properly."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 295.0
        sst[20:30, 20:30] = np.nan  # NaN block in middle

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        thinned = thin(sst, fronts)

        # Should not place fronts where SST is NaN
        nan_mask = np.isnan(sst)
        assert not np.any((thinned == 4) & nan_mask)

    def test_cold_pixels_excluded(self):
        """Test that pixels below min_valid_sst are excluded."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 295.0
        sst[20:30, :] = 265.0  # Below freezing - invalid

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        thinned = thin(sst, fronts)

        # Fronts in the cold region should not be selected based on
        # gradients involving cold pixels
        cold_mask = sst < MIN_VALID_SST_K
        # The algorithm should avoid using gradients from invalid pixels
        assert np.sum(thinned == 4) > 0

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        fronts = np.zeros((60, 60), dtype=np.int16)

        with pytest.raises(ValueError, match="Shape mismatch"):
            thin(sst, fronts)

    def test_custom_front_value(self):
        """Test with custom front value."""
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        sst[:, 25:] = 290.0

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 1  # Use 1 instead of 4

        thinned = thin(sst, fronts, front_value=1)

        assert np.any(thinned == 1)
        assert not np.any(thinned == 4)

    def test_custom_min_gradient(self):
        """Test with custom minimum gradient threshold."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 290.01  # Very small gradient (0.01 K)

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        # With default min_gradient (0.02), should find no fronts
        thinned_default = thin(sst, fronts)

        # With lower threshold, should find fronts
        thinned_low = thin(sst, fronts, min_gradient=0.005)

        assert np.sum(thinned_default == 4) <= np.sum(thinned_low == 4)

    def test_realistic_temperature_range(self):
        """Test with realistic ocean temperature range (272-305 K)."""
        # Create realistic SST field
        np.random.seed(42)
        sst = 285.0 + 5.0 * np.random.randn(100, 100)
        sst = np.clip(sst, 272, 305).astype(np.float32)

        # Add a clear front
        sst[:, 50:] += 3.0
        sst = np.clip(sst, 272, 305)

        fronts = np.zeros((100, 100), dtype=np.int16)
        fronts[:, 48:53] = 4

        thinned = thin(sst, fronts)

        # Should reduce but not eliminate
        original_count = np.sum(fronts == 4)
        thinned_count = np.sum(thinned == 4)
        assert 0 < thinned_count < original_count

    def test_multiple_separate_fronts(self):
        """Test with multiple separate front segments."""
        sst = np.full((100, 100), 285.0, dtype=np.float32)
        sst[:, 25:] = 290.0
        sst[:, 75:] = 295.0

        fronts = np.zeros((100, 100), dtype=np.int16)
        fronts[:, 23:28] = 4  # First front
        fronts[:, 73:78] = 4  # Second front

        thinned = thin(sst, fronts)

        # Should have fronts at both locations
        assert np.any(thinned[:, 23:28] == 4)
        assert np.any(thinned[:, 73:78] == 4)


class TestThinFronts:
    """Tests for the thin_fronts() convenience wrapper."""

    def test_with_median_filter(self):
        """Test with median filtering enabled."""
        # Create noisy SST
        np.random.seed(42)
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        sst[:, 25:] = 290.0
        sst += 0.5 * np.random.randn(50, 50).astype(np.float32)

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        thinned = thin_fronts(sst, fronts, apply_median=True, median_size=5)

        assert np.sum(thinned == 4) > 0
        assert np.sum(thinned == 4) < np.sum(fronts == 4)

    def test_without_median_filter(self):
        """Test with median filtering disabled."""
        sst = np.full((50, 50), 285.0, dtype=np.float32)
        sst[:, 25:] = 290.0

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        thinned = thin_fronts(sst, fronts, apply_median=False)

        assert np.sum(thinned == 4) > 0

    def test_nan_preserved_after_median(self):
        """Test that NaN locations are preserved after median filtering."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 295.0
        sst[20:25, 20:25] = np.nan

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[:, 24:27] = 4

        thinned = thin_fronts(sst, fronts, apply_median=True)

        # NaN region should not have fronts
        nan_mask = np.isnan(sst)
        assert not np.any((thinned == 4) & nan_mask)


class TestComputeGradientMagnitude:
    """Tests for compute_gradient_magnitude()."""

    def test_uniform_field(self):
        """Test that uniform field has zero gradient."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        gradient = compute_gradient_magnitude(sst)

        # Interior should be ~0 (excluding boundaries which are NaN)
        interior = gradient[2:-2, 2:-2]
        assert np.allclose(interior, 0, atol=1e-6)

    def test_horizontal_gradient(self):
        """Test gradient of field with horizontal variation."""
        sst = np.zeros((50, 50), dtype=np.float32) + 285.0
        # Linear gradient in x: 1 K per pixel
        for i in range(50):
            sst[i, :] = 285.0 + i * 1.0

        gradient = compute_gradient_magnitude(sst)

        # Should have gradient ~1 K/pixel in interior
        interior = gradient[2:-2, 2:-2]
        valid = ~np.isnan(interior)
        assert np.allclose(interior[valid], 1.0, atol=0.1)

    def test_vertical_gradient(self):
        """Test gradient of field with vertical variation."""
        sst = np.zeros((50, 50), dtype=np.float32) + 285.0
        # Linear gradient in y: 1 K per pixel
        for j in range(50):
            sst[:, j] = 285.0 + j * 1.0

        gradient = compute_gradient_magnitude(sst)

        # Should have gradient ~1 K/pixel in interior
        interior = gradient[2:-2, 2:-2]
        valid = ~np.isnan(interior)
        assert np.allclose(interior[valid], 1.0, atol=0.1)

    def test_nan_handling(self):
        """Test that NaN values result in NaN gradients."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[25, 25] = np.nan

        gradient = compute_gradient_magnitude(sst)

        # The NaN pixel and its neighbors should be NaN in gradient
        assert np.isnan(gradient[25, 25])
        assert np.isnan(gradient[24, 25])
        assert np.isnan(gradient[26, 25])
        assert np.isnan(gradient[25, 24])
        assert np.isnan(gradient[25, 26])

    def test_cold_pixels_excluded(self):
        """Test that cold pixels are excluded."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[25, 25] = 260.0  # Below min valid

        gradient = compute_gradient_magnitude(sst)

        # The cold pixel should have NaN gradient
        assert np.isnan(gradient[25, 25])


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_array(self):
        """Test with minimum viable array size."""
        sst = np.array([[285, 290, 285],
                        [285, 290, 285],
                        [285, 290, 285]], dtype=np.float32)
        fronts = np.array([[0, 4, 0],
                          [0, 4, 0],
                          [0, 4, 0]], dtype=np.int16)

        thinned = thin(sst, fronts)
        # Should work without error
        assert thinned.shape == (3, 3)

    def test_single_front_pixel(self):
        """Test with single isolated front pixel."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 295.0

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[25, 25] = 4  # Single pixel

        thinned = thin(sst, fronts)

        # Single pixel should be preserved or removed based on gradient
        assert np.sum(thinned == 4) <= 1

    def test_all_fronts(self):
        """Test when entire image is marked as front."""
        sst = np.full((20, 20), 290.0, dtype=np.float32)
        sst[:, 10:] = 295.0

        fronts = np.full((20, 20), 4, dtype=np.int16)

        thinned = thin(sst, fronts)

        # Should still thin
        assert np.sum(thinned == 4) < np.sum(fronts == 4)

    def test_boundary_fronts(self):
        """Test fronts at array boundaries."""
        sst = np.full((50, 50), 290.0, dtype=np.float32)
        sst[:, 25:] = 295.0

        fronts = np.zeros((50, 50), dtype=np.int16)
        fronts[0, :] = 4  # Top edge
        fronts[-1, :] = 4  # Bottom edge
        fronts[:, 0] = 4  # Left edge
        fronts[:, -1] = 4  # Right edge

        # Should not crash on boundary pixels
        thinned = thin(sst, fronts)
        assert thinned.shape == fronts.shape
