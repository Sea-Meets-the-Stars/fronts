#!/usr/bin/env python3
"""
Demonstration of the Bouali Destriping Algorithm
================================================

This script demonstrates the use of the Bouali et al. (2015) destriping algorithm
on synthetic ocean data with artificial stripe noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from fronts.preproc.bouali_destriping import (
    bouali_destriping, 
    compute_gradient,
    compute_nif,
    process_ocean_data
)


def create_synthetic_ocean_data(size=(500, 500), data_type='sst'):
    """Create synthetic ocean data with realistic features."""
    
    x = np.linspace(0, 10, size[1])
    y = np.linspace(0, 10, size[0])
    X, Y = np.meshgrid(x, y)
    
    if data_type == 'sst':
        # Create SST field with fronts and eddies
        # Base temperature field
        base_temp = 20 + 5 * np.sin(X/3) * np.cos(Y/3)
        
        # Add a front
        front = 2 * np.tanh((X - 5) / 0.5)
        
        # Add some eddies
        eddy1 = 2 * np.exp(-((X-3)**2 + (Y-3)**2) / 2)
        eddy2 = -1.5 * np.exp(-((X-7)**2 + (Y-7)**2) / 1.5)
        
        # Add small-scale noise
        noise = np.random.randn(*size) * 0.1
        
        data = base_temp + front + eddy1 + eddy2 + noise
        
    elif data_type == 'chlorophyll':
        # Create chlorophyll field (log-normal distribution)
        # Base field
        base_chl = np.exp(0.5 * np.sin(X/2) * np.cos(Y/2))
        
        # Add bloom patches
        bloom1 = 5 * np.exp(-((X-4)**2 + (Y-4)**2) / 3)
        bloom2 = 3 * np.exp(-((X-6)**2 + (Y-8)**2) / 2)
        
        # Add noise
        noise = np.abs(np.random.randn(*size) * 0.05)
        
        data = base_chl + bloom1 + bloom2 + noise
        
    return data


def add_stripe_noise(data, stripe_intensity=0.5, stripe_frequency=20):
    """Add realistic stripe noise to the data."""
    
    rows, cols = data.shape
    
    # Create stripe pattern (varies across track)
    # Multiple frequency components for more realistic stripes
    stripes = np.zeros((1, cols))
    
    # Primary stripe pattern
    freq1 = np.sin(np.linspace(0, stripe_frequency * np.pi, cols))
    # Secondary harmonics
    freq2 = 0.3 * np.sin(np.linspace(0, stripe_frequency * 3 * np.pi, cols))
    freq3 = 0.2 * np.sin(np.linspace(0, stripe_frequency * 5 * np.pi, cols))
    
    stripes[0, :] = freq1 + freq2 + freq3
    
    # Add some variation in stripe intensity along track
    intensity_variation = 1 + 0.2 * np.sin(np.linspace(0, 4*np.pi, rows)).reshape(-1, 1)
    
    # Apply stripes to all rows with slight variations
    stripe_pattern = np.repeat(stripes, rows, axis=0) * intensity_variation
    
    # Scale and add to data
    noisy_data = data + stripe_intensity * stripe_pattern
    
    return noisy_data


def plot_results(original, noisy, corrected, data_type='sst'):
    """Create a comprehensive visualization of the results."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Color map
    if data_type == 'sst':
        cmap = 'RdBu_r'
        units = '°C'
    else:
        cmap = 'viridis'
        units = 'mg/m³'
    
    # Row 1: Images
    im1 = axes[0, 0].imshow(original, cmap=cmap, aspect='auto')
    axes[0, 0].set_title('Original Clean Data')
    axes[0, 0].set_xlabel('Cross-track')
    axes[0, 0].set_ylabel('Along-track')
    plt.colorbar(im1, ax=axes[0, 0], label=units)
    
    im2 = axes[0, 1].imshow(noisy, cmap=cmap, aspect='auto')
    axes[0, 1].set_title('Data with Stripe Noise')
    axes[0, 1].set_xlabel('Cross-track')
    plt.colorbar(im2, ax=axes[0, 1], label=units)
    
    im3 = axes[0, 2].imshow(corrected, cmap=cmap, aspect='auto')
    axes[0, 2].set_title('Corrected Data')
    axes[0, 2].set_xlabel('Cross-track')
    plt.colorbar(im3, ax=axes[0, 2], label=units)
    
    # Row 2: Gradient Magnitude
    _, _, grad_mag_orig = compute_gradient(original)
    _, _, grad_mag_noisy = compute_gradient(noisy)
    _, _, grad_mag_corr = compute_gradient(corrected)
    
    vmax_grad = np.percentile(grad_mag_orig, 95)
    
    im4 = axes[1, 0].imshow(grad_mag_orig, cmap='hot', aspect='auto', vmax=vmax_grad)
    axes[1, 0].set_title('Original Gradient Magnitude')
    axes[1, 0].set_xlabel('Cross-track')
    axes[1, 0].set_ylabel('Along-track')
    plt.colorbar(im4, ax=axes[1, 0], label=f'{units}/pixel')
    
    im5 = axes[1, 1].imshow(grad_mag_noisy, cmap='hot', aspect='auto', vmax=vmax_grad)
    axes[1, 1].set_title('Noisy Gradient Magnitude')
    axes[1, 1].set_xlabel('Cross-track')
    plt.colorbar(im5, ax=axes[1, 1], label=f'{units}/pixel')
    
    im6 = axes[1, 2].imshow(grad_mag_corr, cmap='hot', aspect='auto', vmax=vmax_grad)
    axes[1, 2].set_title('Corrected Gradient Magnitude')
    axes[1, 2].set_xlabel('Cross-track')
    plt.colorbar(im6, ax=axes[1, 2], label=f'{units}/pixel')
    
    # Row 3: Statistics
    # Gradient magnitude CDFs
    axes[2, 0].hist(grad_mag_orig.flatten(), bins=50, alpha=0.5, label='Original', density=True)
    axes[2, 0].hist(grad_mag_noisy.flatten(), bins=50, alpha=0.5, label='Noisy', density=True)
    axes[2, 0].hist(grad_mag_corr.flatten(), bins=50, alpha=0.5, label='Corrected', density=True)
    axes[2, 0].set_xlabel(f'Gradient Magnitude ({units}/pixel)')
    axes[2, 0].set_ylabel('Probability Density')
    axes[2, 0].set_title('Gradient Magnitude Distribution')
    axes[2, 0].legend()
    axes[2, 0].set_xlim([0, np.percentile(grad_mag_orig, 99)])
    
    # Cross-track profile comparison (middle row)
    middle_row = noisy.shape[0] // 2
    axes[2, 1].plot(original[middle_row, :], label='Original', alpha=0.7)
    axes[2, 1].plot(noisy[middle_row, :], label='Noisy', alpha=0.7)
    axes[2, 1].plot(corrected[middle_row, :], label='Corrected', alpha=0.7)
    axes[2, 1].set_xlabel('Cross-track Position')
    axes[2, 1].set_ylabel(units)
    axes[2, 1].set_title(f'Cross-track Profile (Row {middle_row})')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Difference images
    diff_noisy = noisy - original
    diff_corr = corrected - original
    
    axes[2, 2].hist(diff_noisy.flatten(), bins=50, alpha=0.5, label='Noisy - Original', density=True)
    axes[2, 2].hist(diff_corr.flatten(), bins=50, alpha=0.5, label='Corrected - Original', density=True)
    axes[2, 2].set_xlabel(f'Difference ({units})')
    axes[2, 2].set_ylabel('Probability Density')
    axes[2, 2].set_title('Residual Distribution')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Bouali Destriping Algorithm - {data_type.upper()} Data', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def main():
    """Run the demonstration."""
    
    print("=" * 60)
    print("Bouali Destriping Algorithm Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test with SST data
    print("\n1. Processing SST Data")
    print("-" * 30)
    
    # Create synthetic SST data
    sst_clean = create_synthetic_ocean_data(size=(300, 300), data_type='sst')
    print(f"Created synthetic SST field: {sst_clean.shape}")
    print(f"Temperature range: {sst_clean.min():.2f} - {sst_clean.max():.2f}°C")
    
    # Add stripe noise
    sst_noisy = add_stripe_noise(sst_clean, stripe_intensity=0.8, stripe_frequency=25)
    print(f"Added stripe noise with intensity 0.8")
    
    # Process the data
    print("Applying Bouali destriping algorithm...")
    result_sst = process_ocean_data(sst_noisy, data_type='sst', verbose=True)
    sst_corrected = result_sst['corrected']
    
    # Calculate improvement metrics
    rmse_noisy = np.sqrt(np.mean((sst_noisy - sst_clean)**2))
    rmse_corrected = np.sqrt(np.mean((sst_corrected - sst_clean)**2))
    print(f"\nRMSE (noisy): {rmse_noisy:.3f}°C")
    print(f"RMSE (corrected): {rmse_corrected:.3f}°C")
    print(f"Improvement: {(1 - rmse_corrected/rmse_noisy)*100:.1f}%")
    
    # Plot results for SST
    fig_sst = plot_results(sst_clean, sst_noisy, sst_corrected, data_type='sst')
    
    # Test with Chlorophyll data
    print("\n2. Processing Chlorophyll-a Data")
    print("-" * 30)
    
    # Create synthetic chlorophyll data
    chl_clean = create_synthetic_ocean_data(size=(300, 300), data_type='chlorophyll')
    print(f"Created synthetic chlorophyll field: {chl_clean.shape}")
    print(f"Chlorophyll range: {chl_clean.min():.3f} - {chl_clean.max():.3f} mg/m³")
    
    # Add stripe noise
    chl_noisy = add_stripe_noise(chl_clean, stripe_intensity=0.3, stripe_frequency=30)
    print(f"Added stripe noise with intensity 0.3")
    
    # Process the data (note: chlorophyll data will be log-transformed internally)
    print("Applying Bouali destriping algorithm...")
    result_chl = process_ocean_data(chl_noisy, data_type='chlorophyll', verbose=True)
    chl_corrected = result_chl['corrected']
    
    # For display, we need to inverse the log transform
    chl_corrected_linear = 10**chl_corrected - 1e-6
    
    # Calculate improvement metrics
    rmse_noisy_chl = np.sqrt(np.mean((chl_noisy - chl_clean)**2))
    rmse_corrected_chl = np.sqrt(np.mean((chl_corrected_linear - chl_clean)**2))
    print(f"\nRMSE (noisy): {rmse_noisy_chl:.4f} mg/m³")
    print(f"RMSE (corrected): {rmse_corrected_chl:.4f} mg/m³")
    print(f"Improvement: {(1 - rmse_corrected_chl/rmse_noisy_chl)*100:.1f}%")
    
    # Plot results for Chlorophyll
    fig_chl = plot_results(chl_clean, chl_noisy, chl_corrected_linear, data_type='chlorophyll')
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("The algorithm successfully removes stripe noise while")
    print("preserving ocean fronts and other geophysical features.")
    print("=" * 60)
    
    plt.show()
    
    return fig_sst, fig_chl


if __name__ == "__main__":
    fig_sst, fig_chl = main()
