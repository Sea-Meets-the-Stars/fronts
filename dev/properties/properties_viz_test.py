"""
Test script for the front-weighted property visualization code.

Exercises the I/O loaders in fronts.properties.io and the visualization
methods in fronts.viz.properties using the v1_bin_A global front results
under $OS_OGCM/LLC/Fronts.

Usage:
    conda run -n ocean14 python dev/properties/properties_viz_test.py
"""

import os
import sys
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

from fronts.properties.io import (
    load_metadata,
    load_labeled_array,
    load_geometry_table,
    load_colocation_table,
    merge_geometry_colocation,
    )
from fronts.properties.io import (
        property_file_path,
        load_single_property,
        load_property_arrays,
    )

from fronts.viz.properties import plot_binned_front_map

from IPython import embed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v1')
TIME_STR    = '2012-11-09T12:00:00'
TIMESTAMP   = '2012-11-09T12_00_00'  # filename-safe version
RUN_TAG     = 'v1_bin_A'
VERSION     = '1'

# Properties available in the derived/ directory
PROPERTY_NAMES = ['gradb2', 'relative_vorticity', 'strain_mag']

# Output directory for test plots
OUT_DIR = os.path.join('dev', 'properties', 'test_output')
os.makedirs(OUT_DIR, exist_ok=True)


def separator(title):
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')


# ===================================================================
# Test 1: Atomic I/O loaders
# ===================================================================

def test_atomic_loaders():
    separator('Test 1: Atomic I/O loaders')


    # 1a. Metadata
    metadata = load_metadata(RESULTS_DIR, TIME_STR, RUN_TAG)
    assert 'shape' in metadata, 'metadata missing shape key'
    assert 'num_fronts' in metadata, 'metadata missing num_fronts key'
    print(f'  load_metadata: shape={metadata["shape"]}, '
          f'num_fronts={metadata["num_fronts"]:,}')

    # 1b. Labeled array
    labeled = load_labeled_array(RESULTS_DIR, TIME_STR, RUN_TAG)
    assert labeled.ndim == 2, f'expected 2-D, got {labeled.ndim}-D'
    assert labeled.shape == tuple(metadata['shape']), \
        f'shape mismatch: {labeled.shape} vs {metadata["shape"]}'
    print(f'  load_labeled_array: shape={labeled.shape}, '
          f'dtype={labeled.dtype}, max_label={labeled.max():,}')

    # 1c. Geometry table
    df_geom = load_geometry_table(RESULTS_DIR, TIME_STR, RUN_TAG)
    assert 'label' in df_geom.columns
    assert 'centroid_lat' in df_geom.columns
    print(f'  load_geometry_table: {len(df_geom):,} fronts x '
          f'{len(df_geom.columns)} cols')

    # 1d. Colocation table
    df_coloc = load_colocation_table(RESULTS_DIR, TIME_STR, RUN_TAG)
    assert 'flabel' in df_coloc.columns
    print(f'  load_colocation_table: {len(df_coloc):,} fronts x '
          f'{len(df_coloc.columns)} cols')

    # 1e. Merge
    df_enriched = merge_geometry_colocation(df_geom, df_coloc)
    assert len(df_enriched) == len(df_geom), \
        f'merge lost rows: {len(df_enriched)} vs {len(df_geom)}'
    print(f'  merge_geometry_colocation: {len(df_enriched):,} fronts x '
          f'{len(df_enriched.columns)} cols')

    print('\n  PASSED')
    return metadata, df_enriched


# ===================================================================
# Test 2: Coordinate handling
# ===================================================================

def test_coordinate_handling():
    separator('Test 2: Coordinate handling')

    from fronts.properties.io import (
        load_llc_coords,
        compute_longitude_shift,
        roll_to_pm180,
    )

    # 2a. Load full-resolution coords
    lat, lon = load_llc_coords()
    assert lat.ndim == 2, f'expected 2-D lat, got {lat.ndim}-D'
    assert lat.shape == lon.shape, f'shape mismatch: lat={lat.shape} lon={lon.shape}'
    print(f'  load_llc_coords (full): shape={lat.shape}')

    # 2b. Load with downsampling
    lat_ds, lon_ds = load_llc_coords(downsample_factor=4)
    expected_shape = (lat.shape[0] // 4, lat.shape[1] // 4)
    assert lat_ds.shape == expected_shape, \
        f'downsample shape: {lat_ds.shape} vs expected {expected_shape}'
    print(f'  load_llc_coords (ds=4): shape={lat_ds.shape}')

    # 2c. Longitude shift
    shift = compute_longitude_shift(lon)
    print(f'  compute_longitude_shift: shift={shift}')

    # 2d. Roll
    lon_r, lat_r = roll_to_pm180(lon, lat, shift=shift)
    # After rolling, middle row should start near -180
    mid_row = lon_r.shape[0] // 2
    assert lon_r[mid_row, 0] < -170, \
        f'after roll, first lon in mid row is {lon_r[mid_row, 0]:.1f} (expected < -170)'
    print(f'  roll_to_pm180: lon range [{lon_r[mid_row, 0]:.1f}, '
          f'{lon_r[mid_row, -1]:.1f}]')

    print('\n  PASSED')
    return lat, lon, shift


# ===================================================================
# Test 3: Property-file loading
# ===================================================================

def test_property_loading(shift):
    separator('Test 3: Property-file loading')


    # 3a. Path construction
    p = property_file_path('gradb2', TIMESTAMP, VERSION)
    assert p.exists(), f'file not found: {p}'
    print(f'  property_file_path: {p.name}  (exists={p.exists()})')

    # 3b. Single property
    arr = load_single_property('gradb2', TIMESTAMP, VERSION)
    assert arr.ndim == 2, f'expected 2-D, got {arr.ndim}-D'
    print(f'  load_single_property (gradb2): shape={arr.shape}, '
          f'range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]')

    # 3c. Batch load with downsample + roll
    props = load_property_arrays(
        PROPERTY_NAMES, TIMESTAMP, VERSION,
        downsample_factor=4, shift=shift,
    )
    assert len(props) == len(PROPERTY_NAMES)
    for name, arr in props.items():
        print(f'    {name:25s}  shape={arr.shape}  '
              f'range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]')

    print('\n  PASSED')
    return props


# ===================================================================
# Test 4: Orchestrator — load_global_front_results
# ===================================================================

def test_orchestrator():
    separator('Test 4: Orchestrator — load_global_front_results')

    from fronts.properties.io import load_global_front_results

    result = load_global_front_results(RESULTS_DIR, TIME_STR, RUN_TAG)

    assert set(result.keys()) == {
        'metadata', 'labeled_global', 'df_enriched',
        'lat_global', 'lon_global', 'shift',
    }
    print(f'  metadata: {result["metadata"]["num_fronts"]:,} fronts')
    print(f'  labeled_global: shape={result["labeled_global"].shape}')
    print(f'  df_enriched: {len(result["df_enriched"]):,} rows x '
          f'{len(result["df_enriched"].columns)} cols')
    print(f'  lat_global: shape={result["lat_global"].shape}')
    print(f'  shift: {result["shift"]}')

    print('\n  PASSED')
    return result


# ===================================================================
# Test 5: plot_global_property_map
# ===================================================================

def test_plot_global_property_map(lat, lon, props, shift):
    separator('Test 5: plot_global_property_map')

    from fronts.properties.io import roll_to_pm180
    from fronts.viz.properties import plot_global_property_map

    # Use full-res coords rolled to -180..+180
    lon_r, lat_r = roll_to_pm180(lon, lat, shift=shift)

    # Load a full-res property and roll it too
    from fronts.properties.io import load_single_property
    vort = load_single_property('relative_vorticity', TIMESTAMP, VERSION)
    vort_r = roll_to_pm180(vort, shift=shift)

    # Plot with downsample=8 for speed
    fig = plot_global_property_map(
        lon_r, lat_r, vort_r,
        cmap='RdBu_r',
        symmetric=True,
        clip_pct=2,
        downsample=8,
        title='Relative vorticity (test)',
        clabel=r'$\omega$  (s$^{-1}$)',
    )
    outfile = os.path.join(OUT_DIR, 'test_global_property_map.png')
    fig.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outfile}')

    # Test asymmetric colorbar
    gradb2_r = roll_to_pm180(
        load_single_property('gradb2', TIMESTAMP, VERSION), shift=shift,
    )
    fig2 = plot_global_property_map(
        lon_r, lat_r, gradb2_r,
        cmap='hot_r',
        symmetric=False,
        downsample=8,
        title=r'$|\nabla b|^2$ (test)',
        clabel=r'$|\nabla b|^2$  (s$^{-4}$)',
    )
    outfile2 = os.path.join(OUT_DIR, 'test_global_property_map_gradb2.png')
    fig2.savefig(outfile2, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {outfile2}')

    print('\n  PASSED')


# ===================================================================
# Test 6: plot_binned_front_map
# ===================================================================

def test_plot_binned_front_map(df_enriched):
    separator('Test 6: plot_binned_front_map')


    # Per-front median relative vorticity on a 2-deg binned map
    fig = plot_binned_front_map(
        df_enriched,
        'relative_vorticity_median',
        n_lat_bins=90,
        n_lon_bins=180,
        statistic='mean',
        min_count=2,
        cmap='RdBu_r',
        symmetric=True,
        title='Mean per-front relative vorticity (2-deg bins)',
        clabel=r'$\omega_{median}$  (s$^{-1}$)',
    )
    outfile = os.path.join(OUT_DIR, 'test_binned_front_map_vorticity.png')
    fig.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outfile}')

    # Per-front median gradb2 (asymmetric)
    fig2 = plot_binned_front_map(
        df_enriched,
        'gradb2_median',
        statistic='mean',
        cmap='hot_r',
        symmetric=False,
        title=r'Mean per-front $|\nabla b|^2$ (2-deg bins)',
        clabel=r'$|\nabla b|^2$ median',
    )
    outfile2 = os.path.join(OUT_DIR, 'test_binned_front_map_gradb2.png')
    fig2.savefig(outfile2, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {outfile2}')

    print('\n  PASSED')


# ===================================================================
# Test 7: plot_property_jpdf
# ===================================================================

def test_plot_property_jpdf(df_enriched):
    separator('Test 7: plot_property_jpdf')

    from fronts.viz.properties import plot_property_jpdf

    # JPDF: per-front Rossby number vs per-front |nabla b|^2
    rossby = df_enriched['rossby_number_median'].values
    gradb2 = df_enriched['gradb2_median'].values

    fig = plot_property_jpdf(
        rossby, gradb2,
        n_x_bins=80,
        n_y_bins=60,
        x_range=(-5, 5),
        y_log=True,
        cmap='Reds',
        xlabel=r'Ro$_{median}$ per front',
        ylabel=r'$|\nabla b|^2_{median}$ per front',
        title=r'JPDF: Rossby number vs $|\nabla b|^2$',
        annotations=[
            {'x': 0, 'color': 'k', 'label': 'Ro = 0'},
            {'x': -1, 'color': 'steelblue', 'label': 'Ro = -1'},
            {'x': 1, 'color': 'firebrick', 'label': 'Ro = +1'},
        ],
    )
    outfile = os.path.join(OUT_DIR, 'test_jpdf_rossby_gradb2.png')
    fig.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outfile}')

    # JPDF with linear y-axis: vorticity vs strain
    vort = df_enriched['relative_vorticity_median'].values
    strain = df_enriched['strain_mag_median'].values

    fig2 = plot_property_jpdf(
        vort, strain,
        n_x_bins=80,
        n_y_bins=80,
        y_log=False,
        cmap='Reds',
        xlabel=r'$\omega_{median}$  (s$^{-1}$)',
        ylabel=r'$|\sigma|_{median}$  (s$^{-1}$)',
        title=r'JPDF: vorticity vs strain magnitude',
    )
    outfile2 = os.path.join(OUT_DIR, 'test_jpdf_vort_strain.png')
    fig2.savefig(outfile2, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {outfile2}')

    print('\n  PASSED')


# ===================================================================
# Test 8: plot_multi_timestamp (single timestamp, layout test)
# ===================================================================

def test_plot_multi_timestamp(df_enriched):
    separator('Test 8: plot_multi_timestamp')

    from fronts.viz.properties import plot_binned_front_map, plot_multi_timestamp

    # Use the same timestamp twice to test the multi-panel layout
    common_kwargs = dict(
        df=df_enriched,
        property_col='relative_vorticity_median',
        statistic='mean',
        min_count=2,
        cmap='RdBu_r',
        symmetric=True,
        clabel=r'$\omega$  (s$^{-1}$)',
    )

    fig = plot_multi_timestamp(
        plot_binned_front_map,
        {
            '2012-11-09 (mean)':   {**common_kwargs, 'statistic': 'mean'},
            '2012-11-09 (median)': {**common_kwargs, 'statistic': 'median'},
        },
        ncols=2,
        figsize_per_panel=(10, 6),
        suptitle='Relative vorticity — multi-panel test',
    )
    outfile = os.path.join(OUT_DIR, 'test_multi_timestamp.png')
    fig.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outfile}')

    print('\n  PASSED')


# ===================================================================
# Main
# ===================================================================

def all_tests():
    print('='*70)
    print('  Properties Visualization Test Script')
    print('='*70)
    print(f'  OS_OGCM  = {os.getenv("OS_OGCM")}')
    print(f'  Results  = {RESULTS_DIR}')
    print(f'  Output   = {OUT_DIR}')

    # I/O tests
    metadata, df_enriched = test_atomic_loaders()
    lat, lon, shift = test_coordinate_handling()
    props = test_property_loading(shift)
    result = test_orchestrator()

    # Viz tests
    test_plot_global_property_map(lat, lon, props, shift)
    test_plot_binned_front_map(df_enriched)
    test_plot_property_jpdf(df_enriched)
    test_plot_multi_timestamp(df_enriched)

    separator('ALL TESTS PASSED')
    print(f'\n  Test plots saved to: {OUT_DIR}/')
    print(f'  Files:')
    for f in sorted(os.listdir(OUT_DIR)):
        print(f'    {f}')


if __name__ == '__main__':
    # All tests
    #all_tests()

    # Per-front on log gradb2
    metadata = load_metadata(RESULTS_DIR, TIME_STR, RUN_TAG)
    df_geom = load_geometry_table(RESULTS_DIR, TIME_STR, RUN_TAG)
    df_coloc = load_colocation_table(RESULTS_DIR, TIME_STR, RUN_TAG)

    # Add log10
    df_coloc['log10_gradb2_median'] = np.log10(df_coloc['gradb2_median'].values)

    df_enriched = merge_geometry_colocation(df_geom, df_coloc)

    # Per-front median gradb2 (asymmetric)
    fig2 = plot_binned_front_map(
        df_enriched,
        'log10_gradb2_median',
        statistic='mean',
        cmap='hot_r',
        symmetric=False,
        title=r'Mean per-front $log10 |\nabla b|^2$ (2-deg bins)',
        clabel=r'$log10 |\nabla b|^2$ median',
    )
    plt.show()