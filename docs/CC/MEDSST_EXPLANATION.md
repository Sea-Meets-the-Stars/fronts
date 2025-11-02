# Understanding the MedSST Variable

## Quick Answer

**MedSST is NOT raw temperature data!** It's a **scaled and conditioned** version of Sea Surface Temperature (SST) that has been:
1. **Median-filtered** to reduce noise
2. **Scaled to integer counts** in the range 0-255 (or 0-Range where Range is typically 400)
3. **Optimized for the SIED (Sobel-based Image Edge Detection) algorithm**

The range (0,255) you noticed is a **digitized representation** for computational efficiency, not actual temperature values.

---

## Detailed Explanation

### What is MedSST?

**MedSST** = **Med**ian-filtered **SST** (Sea Surface Temperature)

It's the result of a processing pipeline that converts real temperature data into integer "digital counts" suitable for histogram-based edge detection algorithms.

---

## The Processing Pipeline

### Step 1: Original SST Data
- **Input**: Real SST values in Kelvin or Celsius
- **Example**: 273.15 K to 313.15 K (0°C to 40°C)
- **Format**: Real/floating-point numbers

### Step 2: Scaling and Conditioning ("ConditionInput" subroutine)

The SST values are converted to **integer digital counts** using:

```
SST_counts = (SST_in * ScaleInput) + InputOffset2OutputOffset
```

**Purpose**: Convert temperature to a fixed integer range for the histogram algorithm

**Typical Parameters** (from configuration files):
- `InputScaleFactor`: 0.01 (temperature precision: 0.01 K = 0.01°C)
- `InputOffset`: 273.15 (to convert Celsius to Kelvin)
- `OutputScaleFactor`: 0.01
- `OutputOffset`: 268.15 K (lowest allowed SST)
- `MaxSSTOut`: 313.15 K (highest allowed SST)
- `Range`: 400 (or 255 for some configurations)

**Example Conversion**:
- Real SST = 283.15 K (10°C)
- Digital count = (283.15 - 268.15) / 0.01 = 1500 counts
- But if Range = 400, it might be rescaled to fit 0-400

### Step 3: Median Filtering ("median" subroutine)

The integer SST counts are then **median-filtered** with a small window (typically 5×5 pixels):

```fortran
call median(inpict, pict, n)
```

**Purpose**: 
- Reduce noise
- Smooth small-scale variations
- Preserve sharp edges (fronts)
- Make front detection more robust

**Note from code**: 
> "Values in inpict are assumed to range from 1 to Range. Values equal to 
> the fill value, FillValueInt2, for the present configuration are set to 0 
> on input."

### Step 4: Result = MedSST

The final **MedSST** array contains:
- **Valid data**: Integer values from 0 to Range (e.g., 0-255 or 0-400)
- **Invalid data**: Values ≤ 8 indicate land, clouds, or bad data
- **Fill value**: -32768 (FillValueInt2)

---

## Why the Range (0,255)?

### Historical Context
The SIED algorithm uses **histogram-based** analysis, which requires:
1. A fixed range of integer values
2. Fast histogram accumulation
3. Memory-efficient arrays

The choice of 255 (or 400) comes from:
- **8-bit image processing tradition** (0-255 = 256 levels)
- **Computational efficiency** (integer operations faster than floating-point)
- **Memory constraints** (int16 arrays smaller than float32)
- **SIED requirements** (algorithm needs fixed-range histograms)

### From the Configuration File

Looking at the MSG (Meteosat) configuration:
```
OutputScaleFactor: (lowest allowed SST value) '(A60,9.5)'  :0.01
OutputOffset: (highest value of SST accepted) '(A60,f9.4)' :268.15
MaxSSTOut: (SST = scale*Counts+offset) '(A60,f9.4)'        :313.15
```

This means:
- Lowest SST: 268.15 K (-5°C) → count 0
- Highest SST: 313.15 K (40°C) → count 4500
- But with Range = 400: values rescaled to 0-400

---

## Converting Back to Real Temperature

### Formula
```
SST_Kelvin = (MedSST_counts * OutputScaleFactor) + OutputOffset
SST_Celsius = SST_Kelvin - 273.15
```

### Example
If `MedSST[i,j] = 150`:
- SST = (150 × 0.01) + 268.15 = 269.65 K
- SST = 269.65 - 273.15 = -3.5°C

### Important Notes
1. **Don't try to interpret MedSST directly as temperature!**
2. The actual scaling depends on configuration parameters
3. Different datasets may use different Range values

---

## Why Values ≤ 8 Are Invalid

From the code:
```fortran
if (min(MedSST(i,j-1), MedSST(i,j+1)) .gt. 8) then
    diftemp = abs(MedSST(i,j+1) - MedSST(i,j-1))
endif
```

**Reasoning**:
- **8 counts** represents a very low temperature near the minimum valid SST
- At scale 0.01 K: 8 counts = 268.15 + 0.08 = 268.23 K = -4.92°C
- This is near the **freezing point** of seawater (~-2°C)
- Values this low (or lower) are likely:
  - **Land** (set to 0 or fill value)
  - **Ice** (not valid SST)
  - **Clouds** (masked out)
  - **Bad data** (quality control failures)

**Purpose**: The threshold of 8 ensures that temperature gradients are only calculated where valid ocean SST exists.

---

## Data Flow Summary

```
Raw SST (Kelvin/Celsius, float32)
    ↓
[Scale & Condition]
    ↓
SST Counts (0 to Range, int16)
    ↓
[Median Filter 5×5]
    ↓
MedSST (0 to Range, int16)
    ↓ (used with)
Merged Fronts (binary: 0 or 4, int16)
    ↓
[Thin Algorithm]
    ↓
Thinned Fronts (binary: 0 or 4, int16)
```

---

## Key Points for Using the Python Code

### 1. Input Data Preparation
If you have real SST data, you must:
```python
# Assuming SST in Kelvin, scale 0.01
output_offset = 268.15  # K
output_scale = 0.01     # K/count
sst_range = 400         # or 255

# Convert to counts
med_sst = ((sst_kelvin - output_offset) / output_scale).astype(np.int16)

# Clip to valid range
med_sst = np.clip(med_sst, 0, sst_range)

# Set invalid values (land, ice) to 0
med_sst[invalid_mask] = 0
```

### 2. Already Processed Data
If you have **pre-processed MedSST files** from the FORTRAN pipeline:
```python
# Load directly - already in correct format
med_sst = read_netcdf('median_file.nc', 'median_sst')
# Values should already be 0-255 or 0-400, int16
```

### 3. Understanding the Output
The thinning algorithm works on **gradient magnitudes** in the scaled space:
```python
# In the code:
diftemp = abs(med_sst[i,j+1] - med_sst[i,j-1])

# This is gradient in COUNTS, not temperature
# To convert to temperature gradient:
temp_gradient_K = diftemp * output_scale  # in Kelvin/pixel
```

---

## Common Questions

### Q: Why not just use real temperatures?
**A**: Historical reasons + SIED algorithm requirements:
- SIED uses histogram-based analysis (needs fixed integer range)
- Integer operations are faster
- Memory-efficient (int16 vs float32)
- Easier to handle fill values

### Q: Can I modify the code to use real temperatures?
**A**: Yes, but you'd need to:
1. Change all int16 to float32
2. Adjust the invalid data check (≤ 8 → ≤ minimum valid SST)
3. Update NetCDF output format
4. Test extensively

It's easier to keep the integer format.

### Q: What if my data has a different range?
**A**: You can modify the Python code:
```python
# In thin() function, change:
if min(med_sst[i, j-1], med_sst[i, j+1]) > min_valid_value:
```
Where `min_valid_value` is appropriate for your data scaling.

### Q: Why 255 vs 400 for Range?
**A**: 
- **255**: Traditional 8-bit image processing (1 byte per pixel)
- **400**: More precision for SST (0.1 K steps over 40 K range)
- The FORTRAN code uses **Range** as a parameter, so it's configurable

---

## Example: Full Data Pipeline

### From Raw SST to Thinned Fronts

```python
import numpy as np
from thin_subroutine import thin

# 1. Start with real SST data (Kelvin)
sst_kelvin = np.array(...)  # Your satellite SST data

# 2. Convert to scaled counts (example parameters)
output_offset = 268.15  # K
output_scale = 0.01     # K/count
med_sst = ((sst_kelvin - output_offset) / output_scale).astype(np.int16)
med_sst = np.clip(med_sst, 0, 400)

# 3. Apply median filter (simple example - use scipy for production)
from scipy.ndimage import median_filter
med_sst = median_filter(med_sst, size=5).astype(np.int16)

# 4. Get merged fronts from your edge detection algorithm
merged_fronts = your_edge_detection(med_sst)  # Values: 0 or 4

# 5. Run thinning
thinned = thin("output.nc", time_value, med_sst, merged_fronts)

# 6. Convert back to temperature if needed
sst_at_fronts = (med_sst * output_scale) + output_offset  # Kelvin
sst_at_fronts = sst_at_fronts - 273.15  # Celsius
```

---

## References from Code

### From ParameterStatements.f:
```fortran
c     OutputScaleFactor - SST_out = Counts * OutputScaleFactor +
c      OutputOffset. Called 'scale_factor' in CF-1.5.
c     OutputOffset - This is the smallest allowed SST value in Kelvin 
c      on output. Input SST values smaller than this are set to 
c      _FillValue. This is required by the histogramming algorithm 
c      used by SIED which requires numbers go from 0 digital counts 
c      to an upper limit of Range.
c     Range - the range of digital counts used for SST on output. 
c      The range is limited because SIED uses a histogram algorithm 
c      that operates on a fixed range of values from 0 to a maximum 
c      value: Range. I generally use 400 to cover the range of SST 
c      values in Kelvin at 0.1K steps.
```

### From ConditionInput subroutine:
```fortran
c     This subroutine will condition the input. Specifically, it will
c     subtract Offset from all values and then set values less than 
c     zero and larger than Range to 0.
```

### From median subroutine:
```fortran
c     Fast Median algorithm applies an nxn median filter to inpict.
c     The result is in pict. Values in inpict are assumed to range
c     from 1 to Range. Values equal to the fill value, FillValueInt2,
c     for the present configuration are set to 0 on input.
```

---

## Summary

**MedSST** is:
- ✅ **Scaled integer representation** of SST (0 to Range)
- ✅ **Median-filtered** for noise reduction
- ✅ **Optimized for SIED algorithm** (histogram-based)
- ❌ **NOT raw temperature** in Kelvin or Celsius
- ❌ **NOT directly interpretable** without scale/offset parameters

The (0,255) range is an **encoding scheme** for efficient computation, not a limitation on the temperature values it can represent. The actual temperature range is determined by the `OutputOffset` and `OutputScaleFactor` parameters in the configuration.

**To use the Python code**: Ensure your input data is properly scaled to integer counts in the appropriate range, just like the FORTRAN code expects!
