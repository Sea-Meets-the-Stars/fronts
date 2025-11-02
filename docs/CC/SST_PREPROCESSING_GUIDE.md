# SST Preprocessing Quick Start Guide

## Overview

This guide shows you how to convert raw Sea Surface Temperature (SST) data into the format required by the `thin()` function.

## The Problem

The `thin()` function expects **MedSST**: median-filtered SST data in **integer counts** (int16), not raw temperature values.

## The Solution

Use the `sst_preprocessing.py` module:

```python
from sst_preprocessing import preprocess_sst
import numpy as np

# Your SST data in Kelvin
sst_kelvin = load_your_sst_data()  # shape: (nx, ny), float32

# One-line preprocessing
med_sst, stats = preprocess_sst(sst_kelvin)

# Done! Ready for thinning
```

---

## Quick Examples

### Example 1: Simple Case (Defaults)

```python
import numpy as np
from sst_preprocessing import preprocess_sst

# Load satellite SST (Kelvin)
sst = np.load('satellite_sst.npy')

# Preprocess with defaults (5x5 median, 0.01K scale)
med_sst, stats = preprocess_sst(sst, verbose=True)

# Use with thin()
from thin_subroutine import thin
thinned = thin("output.nc", timestamp, med_sst, merged_fronts)
```

### Example 2: Custom Parameters

```python
from sst_preprocessing import preprocess_sst

# Preprocess with custom settings
med_sst, stats = preprocess_sst(
    sst_data,
    median_window=7,          # Larger smoothing window
    output_scale=0.005,       # Higher precision (0.005K per count)
    output_offset=270.15,     # Different minimum (-3°C)
    max_sst_out=320.15,       # Different maximum (47°C)
    sst_range=10000,          # Larger count range
    in_kelvin=True,           # Input is in Kelvin
    verbose=True
)
```

### Example 3: Input in Celsius

```python
from sst_preprocessing import preprocess_sst

# If your SST is in Celsius, not Kelvin
sst_celsius = load_sst_celsius()

med_sst, stats = preprocess_sst(
    sst_celsius,
    in_kelvin=False,  # Important: set to False for Celsius input
    verbose=True
)
```

### Example 4: With Custom Fill Values

```python
from sst_preprocessing import preprocess_sst

# If your data uses a specific fill value (not NaN)
sst_with_fills = np.load('sst_data.npy')
sst_with_fills[land_mask] = -999.0  # Your fill value

med_sst, stats = preprocess_sst(
    sst_with_fills,
    fill_value=-999.0,  # Specify your fill value
    verbose=True
)
```

---

## Step-by-Step Approach

If you need more control, use individual functions:

```python
from sst_preprocessing import sst_to_counts, median_filter_sst

# Step 1: Convert to counts
sst_counts, stats = sst_to_counts(
    sst_data,
    output_scale=0.01,
    output_offset=268.15,
    max_sst_out=313.15,
    sst_range=4500,
    in_kelvin=True
)

print(f"Converted {stats['n_valid']} pixels")
print(f"Range: {stats['min_sst']:.2f} - {stats['max_sst']:.2f} K")

# Step 2: Apply median filter
med_sst = median_filter_sst(sst_counts, window_size=5)

# Done!
```

---

## Parameter Guide

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `median_window` | 5 | Size of median filter (5 = 5×5 pixels) |
| `output_scale` | 0.01 | Kelvin per count (precision) |
| `output_offset` | 268.15 | Minimum SST in Kelvin (-5°C) |
| `max_sst_out` | 313.15 | Maximum SST in Kelvin (40°C) |
| `sst_range` | 400 | Maximum count value |
| `in_kelvin` | True | True if input is Kelvin, False if Celsius |
| `fill_value` | np.nan | Value indicating missing data |
| `verbose` | True | Print processing statistics |

### How to Choose Parameters

**output_scale** (precision):
- 0.01 K = 10 counts per degree (standard)
- 0.005 K = 20 counts per degree (high precision)
- 0.02 K = 5 counts per degree (low precision, faster)

**output_offset** (minimum SST):
- Polar regions: 268.15 K (-5°C)
- Tropical only: 273.15 K (0°C)
- Mid-latitudes: 270.15 K (-3°C)

**max_sst_out** (maximum SST):
- Global: 313.15 K (40°C)
- Extended: 323.15 K (50°C) for very warm regions
- Reduced: 303.15 K (30°C) if memory limited

**sst_range** (count range):
```python
# Calculate based on your temperature range:
sst_range = int((max_sst_out - output_offset) / output_scale)

# Example:
# (313.15 - 268.15) / 0.01 = 4500 counts
```

---

## Complete Workflow Example

```python
import numpy as np
from sst_preprocessing import preprocess_sst
from thin_subroutine import thin

# ============================================================
# 1. Load your SST data
# ============================================================
# Replace this with your actual data loading
sst_kelvin = np.load('my_satellite_sst.npy')
print(f"Loaded SST: {sst_kelvin.shape}")

# ============================================================
# 2. Preprocess: Convert and median filter
# ============================================================
print("\nPreprocessing SST...")
med_sst, stats = preprocess_sst(
    sst_kelvin,
    median_window=5,
    output_scale=0.01,
    output_offset=268.15,
    max_sst_out=313.15,
    sst_range=4500,
    verbose=True
)

# ============================================================
# 3. Load or create merged fronts
# ============================================================
# You need fronts from your edge detection algorithm
# (e.g., SIED, Sobel, Canny, etc.)
merged_fronts = your_front_detection(med_sst)  # Returns 0 or 4

# ============================================================
# 4. Run thinning
# ============================================================
print("\nThinning fronts...")
thinned_fronts = thin(
    mt_filename="thinned_output.nc",
    hdate=your_timestamp,
    med_sst=med_sst,
    merged_fronts=merged_fronts,
    debug=1
)

# ============================================================
# 5. Analyze results
# ============================================================
n_merged = np.sum(merged_fronts == 4)
n_thinned = np.sum(thinned_fronts == 4)
reduction = 100 * (1 - n_thinned / n_merged)

print(f"\nResults:")
print(f"  Merged fronts: {n_merged} pixels")
print(f"  Thinned fronts: {n_thinned} pixels")
print(f"  Reduction: {reduction:.1f}%")
```

---

## Troubleshooting

### Issue: "No valid pixels after filtering"

**Cause**: All your data is outside the valid range.

**Solution**: Check your temperature range and adjust parameters:
```python
# Check your data
print(f"Min SST: {np.nanmin(sst_data):.2f} K")
print(f"Max SST: {np.nanmax(sst_data):.2f} K")

# Adjust parameters to cover your range
med_sst, stats = preprocess_sst(
    sst_data,
    output_offset=np.nanmin(sst_data) - 5,  # 5K buffer
    max_sst_out=np.nanmax(sst_data) + 5,    # 5K buffer
    sst_range=20000,  # Large enough for the range
    verbose=True
)
```

### Issue: "WARNING: Many pixels above/below maximum"

**Cause**: Your SST range doesn't match the parameters.

**Solution**: Adjust `output_offset` and `max_sst_out`:
```python
# For colder regions
output_offset = 260.15  # Allow down to -13°C

# For warmer regions  
max_sst_out = 320.15    # Allow up to 47°C
```

### Issue: "Mean absolute error too large"

**Cause**: The median filter smooths data, introducing small errors.

**Expected**: 0.3-0.5 K error is normal with 5×5 median filter.

**Solution**: This is expected behavior. To reduce:
- Use smaller median window (3×3)
- Use higher precision (output_scale=0.005)

---

## Validation

Always validate your preprocessing:

```python
from sst_preprocessing import preprocess_sst, counts_to_sst

# Preprocess
med_sst, stats = preprocess_sst(sst_original)

# Convert back
sst_reconstructed = counts_to_sst(med_sst, to_celsius=False)

# Compare
valid_mask = ~np.isnan(sst_original) & ~np.isnan(sst_reconstructed)
error = np.abs(sst_original[valid_mask] - sst_reconstructed[valid_mask])

print(f"Mean error: {np.mean(error):.4f} K")
print(f"Max error: {np.max(error):.4f} K")
print(f"Std error: {np.std(error):.4f} K")

# Expected: mean < 0.5K, max < 3K
```

---

## Advanced Usage

### Custom Median Filter

If you want different filtering behavior:

```python
from sst_preprocessing import sst_to_counts
from scipy import ndimage

# Convert to counts
sst_counts, _ = sst_to_counts(sst_data)

# Custom filtering (e.g., Gaussian instead of median)
from scipy.ndimage import gaussian_filter
sst_smooth = gaussian_filter(
    sst_counts.astype(float), 
    sigma=2.0
).astype(np.int16)

# Or use rank filter, percentile filter, etc.
```

### Processing in Chunks

For very large datasets:

```python
from sst_preprocessing import preprocess_sst

# Process in tiles
tile_size = 1000
for i in range(0, len_x, tile_size):
    for j in range(0, len_y, tile_size):
        tile = sst_data[i:i+tile_size, j:j+tile_size]
        
        # Process tile
        med_tile, _ = preprocess_sst(tile, verbose=False)
        
        # Store result
        med_sst[i:i+tile_size, j:j+tile_size] = med_tile
```

---

## Key Takeaways

1. **Use `preprocess_sst()` for simplicity** - it handles everything
2. **Set `in_kelvin=False` if your data is in Celsius**
3. **Adjust `output_offset` and `max_sst_out` to match your data range**
4. **Median filtering is essential** - don't skip it
5. **Validate your output** - check the statistics
6. **Values ≤ 8 are invalid** - this is by design

---

## References

- **Full documentation**: See `sst_preprocessing.py` docstrings
- **Thin algorithm**: See `thin_subroutine.py`
- **MedSST explanation**: See `MEDSST_EXPLANATION.md`

---

## Quick Reference Card

```python
# Minimal example
from sst_preprocessing import preprocess_sst
med_sst, stats = preprocess_sst(sst_kelvin)

# With custom parameters
med_sst, stats = preprocess_sst(
    sst_data,
    median_window=5,
    output_scale=0.01,
    output_offset=268.15,
    max_sst_out=313.15,
    in_kelvin=True
)

# Convert back to temperature
from sst_preprocessing import counts_to_sst
sst_celsius = counts_to_sst(med_sst, to_celsius=True)
```

**That's it! You're ready to preprocess SST data for front detection.**
