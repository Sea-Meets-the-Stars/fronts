# Bouali Destriping Algorithm Implementation

Implementation of the stripe noise reduction algorithm for ocean satellite imagery from Bouali et al. (2015).

## Overview

This module implements an iterative variational algorithm to remove stripe noise from ocean satellite data (SST, chlorophyll-a, etc.) to improve the detection of ocean fronts and submesoscale features. The algorithm is particularly useful for data from whiskbroom scanners like MODIS and VIIRS.

## Paper Reference

Bouali, M., O. Sato, and P. Polito (2015). "An algorithm to improve the detection of ocean fronts from whiskbroom scanner images." Remote Sensing Letters, 6:12, 942-951. DOI: 10.1080/2150704X.2015.1093187

## Algorithm Principle

The algorithm performs a multi-layer decomposition of the noisy image:
```
f = u₀ + v₀ = u₀ + u₁ + v₁ = ... = Σᵢuᵢ + vₖ
```

Where:
- `f` is the observed image with stripe noise
- `uᵢ` are the clean components
- `vₖ` is the final residual containing primarily stripe noise

The decomposition is achieved through iterative minimization of a variational model that:
1. Preserves edges using a binary mask M
2. Solves a Poisson equation using DCT (Discrete Cosine Transform)
3. Gradually isolates stripe noise in the residual component

## Installation

### Requirements
```python
numpy >= 1.19.0
scipy >= 1.5.0
matplotlib >= 3.3.0  # For visualization
```

### Files
- `bouali_destriping.py` - Main algorithm implementation
- `demo_bouali.py` - Demonstration script with synthetic data
- `README.md` - This documentation

## Usage

### Basic Usage

```python
from bouali_destriping import process_ocean_data
import numpy as np

# For SST data
sst_corrected = process_ocean_data(sst_image, data_type='sst')

# For chlorophyll-a data
chl_corrected = process_ocean_data(chl_image, data_type='chlorophyll')
```

### Advanced Usage

```python
from bouali_destriping import bouali_destriping

# Custom parameters
corrected = bouali_destriping(
    image,
    threshold=0.8,        # Gradient threshold for edge mask
    n_iterations=7,       # Number of iterations
    lambda0=1.0,         # Lagrange multiplier
    apply_filter=True,   # Apply median filter to final residual
    filter_size=3        # Median filter kernel size
)
```

### Gradient Field Correction

For direct gradient field correction (useful for front detection):

```python
from bouali_destriping import correct_gradient_field

grad_x, grad_y, grad_magnitude = correct_gradient_field(
    noisy_image,
    threshold=0.8
)
```

## Key Functions

### `bouali_destriping()`
Main algorithm implementation that performs iterative decomposition.

**Parameters:**
- `image`: Input image with stripe noise (2D array)
- `threshold`: Gradient threshold for edge detection
- `n_iterations`: Number of decomposition iterations (default: 7)
- `lambda0`: Lagrange multiplier (default: 1.0)
- `apply_filter`: Apply low-pass filter to final residual
- `return_components`: Return all decomposition components

**Returns:**
- Corrected image or dictionary with all components

### `process_ocean_data()`
Convenient wrapper for processing ocean satellite data.

**Parameters:**
- `data`: Input ocean data (SST or chlorophyll-a)
- `data_type`: 'sst' or 'chlorophyll'
- `custom_threshold`: Override default threshold
- `verbose`: Print processing information

**Returns:**
- Dictionary with corrected data and diagnostics (NIF, NDF)

### `compute_gradient()`
Compute gradient field components and magnitude.

### `create_edge_mask()`
Create binary mask M for edge preservation.

### `solve_poisson_dct()`
Solve Poisson equation using Discrete Cosine Transform.

## Recommended Thresholds

- **SST data**: 0.8°C/pixel
- **Chlorophyll-a data**: 0.1 mg/m³/pixel (for log-transformed data)

These values can be adjusted based on:
- Noise level in the data
- Strength of ocean features
- Specific application requirements

## Performance Metrics

### NIF (Normalized Improvement Factor)
Measures the reduction of spatial variations in the cross-track direction:
```python
nif = compute_nif(original, corrected)
```
Typical values: 25-45% improvement

### NDF (Normalized Distortion Factor)
Measures information preservation in the along-track direction:
```python
ndf = compute_ndf(original, corrected)
```
Should be ≤ 0.95 to ensure no visible blur

## Example Workflow

```python
import numpy as np
from bouali_destriping import process_ocean_data
import matplotlib.pyplot as plt

# Load your data (example with synthetic data)
sst_data = np.load('sst_swath.npy')  # Your SST data

# Process the data
result = process_ocean_data(
    sst_data,
    data_type='sst',
    verbose=True
)

# Extract results
sst_corrected = result['corrected']
gradient_magnitude = result['gradient_magnitude']
print(f"NIF: {result['nif']:.1f}%")
print(f"NDF: {result['ndf']:.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sst_data, cmap='RdBu_r')
axes[0].set_title('Original SST')
axes[1].imshow(sst_corrected, cmap='RdBu_r')
axes[1].set_title('Corrected SST')
plt.show()
```

## Running the Demo

```bash
python demo_bouali.py
```

This will:
1. Generate synthetic ocean data (SST and chlorophyll)
2. Add realistic stripe noise
3. Apply the destriping algorithm
4. Display comprehensive results and metrics

## Notes

1. **Input Format**: Data should be in swath projection (not map-projected)
2. **Processing Order**: Apply destriping before map projection for best results
3. **Edge Preservation**: The binary mask M ensures sharp fronts are preserved
4. **Iteration Count**: 7 iterations typically provide good results
5. **Memory Usage**: The algorithm stores intermediate components; for very large images, consider processing in tiles

## Limitations

- Designed for unidirectional stripe noise (cross-track direction)
- May require threshold adjustment for extreme noise levels
- Performance depends on the relative strength of noise vs. ocean features

## Algorithm Advantages

- Preserves ocean fronts and edges
- No significant blur introduction
- Fully automated (no manual parameter tuning required)
- Works on both SST and ocean color products
- Fast execution using DCT

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{bouali2015algorithm,
  title={An algorithm to improve the detection of ocean fronts from whiskbroom scanner images},
  author={Bouali, M. and Sato, O. and Polito, P.},
  journal={Remote Sensing Letters},
  volume={6},
  number={12},
  pages={942--951},
  year={2015},
  publisher={Taylor \& Francis},
  doi={10.1080/2150704X.2015.1093187}
}
```
