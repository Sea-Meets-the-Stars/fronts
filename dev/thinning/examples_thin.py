#!/usr/bin/env python3
"""
Example usage of the thin_subroutine module for processing ocean fronts.

This script demonstrates various use cases including:
1. Basic thinning with synthetic data
2. Processing real data from files
3. Batch processing multiple images
4. Visualization of results

Author: Converted from FORTRAN by Claude
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the thinning functions
from fronts.finding.thin_cc import thin, write_merged_thinned, print_array2


def example_1_basic_synthetic():
    """
    Example 1: Basic usage with synthetic data.
    
    Creates a simple SST field with a temperature front and demonstrates
    the thinning process.
    """
    print("=" * 70)
    print("Example 1: Basic Thinning with Synthetic Data")
    print("=" * 70)
    
    # Create synthetic SST field (100x100 pixels)
    len_x, len_y = 100, 100
    med_sst = np.zeros((len_x, len_y), dtype=np.int16)
    
    # Create a temperature gradient (warm on left, cold on right)
    for i in range(len_x):
        for j in range(len_y):
            # Add some noise
            noise = np.random.randint(-5, 5)
            temp = 150 - i + noise
            med_sst[i, j] = np.clip(temp, 9, 255)
    
    # Create merged fronts (a wide band representing detected fronts)
    merged_fronts = np.zeros((len_x, len_y), dtype=np.int16)
    
    # Vertical front at column 50, spanning rows 20-80, width 5 pixels
    merged_fronts[48:53, 20:80] = 4
    
    # Horizontal front at row 50, spanning columns 20-80, width 5 pixels  
    merged_fronts[20:80, 48:53] = 4
    
    print(f"\nInput data:")
    print(f"  SST field shape: {med_sst.shape}")
    print(f"  SST range: [{med_sst.min()}, {med_sst.max()}]")
    print(f"  Merged front pixels: {np.sum(merged_fronts == 4)}")
    
    # Run thinning with debug output
    print("\nRunning thinning algorithm...")
    thinned = thin(
        mt_filename="example1_output.nc",
        hdate=1234567890.0,
        med_sst=med_sst,
        merged_fronts=merged_fronts,
        debug=1
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Thinned front pixels: {np.sum(thinned == 4)}")
    reduction = 100 * (1 - np.sum(thinned == 4) / np.sum(merged_fronts == 4))
    print(f"  Reduction: {reduction:.1f}%")
    
    return med_sst, merged_fronts, thinned


def example_2_realistic_front():
    """
    Example 2: Create a more realistic ocean front scenario.
    
    Simulates a meandering temperature front with gradient variations.
    """
    print("\n" + "=" * 70)
    print("Example 2: Realistic Meandering Front")
    print("=" * 70)
    
    len_x, len_y = 200, 200
    
    # Create SST field with meandering front
    x = np.arange(len_x)
    y = np.arange(len_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create a meandering temperature front
    front_position = len_x // 2 + 10 * np.sin(2 * np.pi * y / 50)
    
    med_sst = np.zeros((len_x, len_y), dtype=np.int16)
    for i in range(len_x):
        for j in range(len_y):
            # Distance from front
            dist_from_front = i - front_position[j]
            
            # Temperature decreases across front
            temp = 150 - 2 * dist_from_front
            
            # Add realistic noise
            temp += np.random.normal(0, 3)
            
            # Clip to valid range
            med_sst[i, j] = np.clip(temp, 9, 255)
    
    # Create front detection (pixels near the front)
    merged_fronts = np.zeros((len_x, len_y), dtype=np.int16)
    for j in range(len_y):
        center = int(front_position[j])
        # Wide detection band (±5 pixels)
        if 5 <= center < len_x - 5:
            merged_fronts[center-5:center+6, j] = 4
    
    print(f"\nInput data:")
    print(f"  Image size: {len_x}x{len_y}")
    print(f"  Front type: Meandering")
    print(f"  Merged front pixels: {np.sum(merged_fronts == 4)}")
    
    # Run thinning
    print("\nRunning thinning...")
    thinned = thin(
        mt_filename="example2_output.nc",
        hdate=1234567890.0,
        med_sst=med_sst,
        merged_fronts=merged_fronts,
        debug=0  # Quiet mode
    )
    
    print(f"\nResults:")
    print(f"  Thinned front pixels: {np.sum(thinned == 4)}")
    reduction = 100 * (1 - np.sum(thinned == 4) / np.sum(merged_fronts == 4))
    print(f"  Reduction: {reduction:.1f}%")
    
    return med_sst, merged_fronts, thinned


def example_3_debug_printing():
    """
    Example 3: Demonstrate debug printing utilities.
    
    Shows how to use print_array2 for debugging small regions.
    """
    print("\n" + "=" * 70)
    print("Example 3: Debug Printing")
    print("=" * 70)
    
    # Create small test array
    test_array = np.random.randint(0, 255, (20, 20), dtype=np.int16)
    
    print("\nPrinting subset of array (rows 5-9, cols 5-9):")
    print_array2(
        test_array,
        message="Test Array Subset --- EOS",
        istrt=5, iend=9,
        jstrt=5, jend=9
    )
    
    print("\nPrinting different subset (rows 10-14, cols 10-14):")
    print_array2(
        test_array,
        message="Another Subset --- EOS",
        istrt=10, iend=14,
        jstrt=10, jend=14
    )


def example_4_batch_processing():
    """
    Example 4: Batch processing multiple images.
    
    Demonstrates how to process multiple front images efficiently.
    """
    print("\n" + "=" * 70)
    print("Example 4: Batch Processing")
    print("=" * 70)
    
    n_images = 5
    len_x, len_y = 100, 100
    
    results = []
    
    print(f"\nProcessing {n_images} images...")
    
    for img_num in range(n_images):
        # Create synthetic data for each image
        med_sst = np.random.randint(50, 200, (len_x, len_y), dtype=np.int16)
        
        # Random front pattern
        merged_fronts = np.zeros((len_x, len_y), dtype=np.int16)
        front_col = len_x // 2 + np.random.randint(-10, 10)
        merged_fronts[front_col-2:front_col+3, :] = 4
        
        # Process
        thinned = thin(
            mt_filename=f"batch_output_{img_num:03d}.nc",
            hdate=1234567890.0 + img_num * 3600,
            med_sst=med_sst,
            merged_fronts=merged_fronts,
            debug=0
        )
        
        # Collect statistics
        orig_pixels = np.sum(merged_fronts == 4)
        thin_pixels = np.sum(thinned == 4)
        reduction = 100 * (1 - thin_pixels / orig_pixels) if orig_pixels > 0 else 0
        
        results.append({
            'image': img_num,
            'original': orig_pixels,
            'thinned': thin_pixels,
            'reduction': reduction
        })
        
        print(f"  Image {img_num}: {orig_pixels} → {thin_pixels} pixels "
              f"({reduction:.1f}% reduction)")
    
    # Summary statistics
    avg_reduction = np.mean([r['reduction'] for r in results])
    print(f"\nAverage reduction: {avg_reduction:.1f}%")
    
    return results


def visualize_results(med_sst, merged_fronts, thinned, save_path=None):
    """
    Visualize the thinning results.
    
    Parameters
    ----------
    med_sst : np.ndarray
        Median SST field
    merged_fronts : np.ndarray
        Original merged fronts
    thinned : np.ndarray
        Thinned fronts
    save_path : str, optional
        Path to save figure
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot SST
        im0 = axes[0].imshow(med_sst.T, origin='lower', cmap='RdYlBu_r',
                            aspect='auto')
        axes[0].set_title('Median SST')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im0, ax=axes[0], label='SST (scaled)')
        
        # Plot merged fronts
        im1 = axes[1].imshow(merged_fronts.T, origin='lower', cmap='binary',
                            aspect='auto', vmin=0, vmax=4)
        axes[1].set_title(f'Merged Fronts ({np.sum(merged_fronts == 4)} pixels)')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        
        # Plot thinned fronts  
        im2 = axes[2].imshow(thinned.T, origin='lower', cmap='binary',
                            aspect='auto', vmin=0, vmax=4)
        axes[2].set_title(f'Thinned Fronts ({np.sum(thinned == 4)} pixels)')
        axes[2].set_xlabel('X (pixels)')
        axes[2].set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        print("\nMatplotlib not available for visualization.")
        print("Install with: pip install matplotlib")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("THIN SUBROUTINE - PYTHON CONVERSION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the Python conversion of the FORTRAN")
    print("thin_subroutine for ocean front detection and thinning.")
    
    # Example 1: Basic synthetic data
    med_sst1, merged1, thinned1 = example_1_basic_synthetic()
    
    # Example 2: Realistic front
    med_sst2, merged2, thinned2 = example_2_realistic_front()
    
    # Example 3: Debug printing
    example_3_debug_printing()
    
    # Example 4: Batch processing
    batch_results = example_4_batch_processing()
    
    # Try to visualize if matplotlib available
    print("\n" + "=" * 70)
    print("Visualization")
    print("=" * 70)
    
    try:
        import matplotlib
        print("\nGenerating visualization plots...")
        visualize_results(med_sst1, merged1, thinned1, 
                         save_path='thinning_example1.png')
        visualize_results(med_sst2, merged2, thinned2,
                         save_path='thinning_example2.png')
        print("\nVisualization complete!")
    except ImportError:
        print("\nMatplotlib not available for visualization.")
        print("To enable visualization, install: pip install matplotlib")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nOutput files created:")
    print("  - example1_output.nc")
    print("  - example2_output.nc")
    print("  - batch_output_000.nc through batch_output_004.nc")
    if Path('thinning_example1.png').exists():
        print("  - thinning_example1.png")
        print("  - thinning_example2.png")


if __name__ == "__main__":
    main()
