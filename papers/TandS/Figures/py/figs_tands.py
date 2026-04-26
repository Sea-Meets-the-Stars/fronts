""" First-draft figures for the T and S paper. """

import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sklabel

from fronts.properties import io as prop_io
from fronts.properties.characteristics import turner_angle
from fronts.viz.properties import plot_property_jpdf
from fronts.finding import pyboa
from fronts.finding.sharpen import global_sharpen_pq
from fronts.finding.despur import prune_short_spurs
from fronts.config import io as config_io

from fronts.viz import defs

mpl.rcParams['font.family'] = 'stixgeneral'

# Paths
ogcm = os.getenv('OS_OGCM')
results_dir = os.path.join(ogcm, 'LLC', 'Fronts', 'group_fronts', 'v2')
derived_dir = os.path.join(ogcm, 'LLC', 'Fronts', 'derived')
coords_file = os.path.join(ogcm, 'LLC', 'Fronts', 'coords',
                           'LLC_coords_lat_lon.nc')
figures_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), '')  # papers/TandS/Figures/

# Timestamp used across figures
TIMESTAMP = '2012-11-09T12_00_00'


def _load_subregion(field_name, row_slice, col_slice,
                    version='2'):
    """Load a sub-region of a derived field via xarray slicing."""
    fpath = os.path.join(
        derived_dir,
        f'LLC4320_{TIMESTAMP}_{field_name}_v{version}.nc')
    with xr.open_dataset(fpath) as ds:
        return ds[field_name].isel(
            y=row_slice, x=col_slice).values.squeeze()


def _load_coords_subregion(row_slice, col_slice):
    """Load lat/lon for a sub-region of the LLC grid."""
    with xr.open_dataset(coords_file) as ds:
        lat = ds['lat'].isel(x=row_slice, y=col_slice).values
        lon = ds['lon'].isel(x=row_slice, y=col_slice).values
    return lat, lon


def _find_subregion(target_lat, target_lon, half_width=60):
    """Return row/col slices for a ~150km box centred on (target_lat, target_lon).

    half_width is in pixels (~1.2 km each → 60 ≈ 72 km half-side ≈ 144 km full).
    """
    with xr.open_dataset(coords_file) as ds:
        lat = ds['lat'].values
        lon = ds['lon'].values
    dist = (lat - target_lat)**2 + (lon - target_lon)**2
    r0, c0 = np.unravel_index(np.nanargmin(dist), dist.shape)
    row_sl = slice(int(r0 - half_width), int(r0 + half_width))
    col_sl = slice(int(c0 - half_width), int(c0 + half_width))
    return row_sl, col_sl


def fig_turner_vs_gradb(
    outfile: str = 'fig_turner_vs_gradb.png',
    timestamp: str = '2012-11-09T12:00:00',
    run_tag: str = 'v2_bin_D',
    n_bins: int = 100,
):
    """2D histogram of Turner angle vs sqrt(gradb2) for individual fronts.

    Uses the mean Turner angle (computed from mean gradient fields)
    and the median of sqrt(gradb2).
    """
    # Load the front properties table
    df = prop_io.load_colocation_table(
        results_dir, timestamp, run_tag)

    # Compute Turner angle from the mean gradient fields
    tu_deg = turner_angle(
        df['gradtheta2_mean'].values,
        df['gradsalt2_mean'].values,
        df['gradrho2_mean'].values,
    )

    # Median of sqrt(gradb2) = median of |grad b|
    #gradb = np.sqrt(df['gradb2_median'].values)
    gradb = np.sqrt(df['gradb2_p90'].values)

    # Build the 2D histogram using the existing JPDF plotter
    fig = plot_property_jpdf(
        tu_deg, gradb,
        n_x_bins=n_bins,
        n_y_bins=n_bins,
        x_range=(-90, 90),
        y_range=(1e-8, 1e-5),
        fontsize=15,
        cfsz=13,
        y_log=True,
        cmap='Blues',
        xlabel=r'Turner angle  $Tu_h$  (deg)',
        ylabel=r'$|\nabla b|$  (s$^{-2}$)',
        title='Turner angle vs. buoyancy gradient  (2012-11-09)',
    )

    # Save
    outpath = os.path.join(figures_dir, outfile)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


def fig_front_definition(
    outfile: str = 'fig_front_definition.png',
    target_lat: float = 35.0,
    target_lon: float = -65.0,
    half_width: int = 60,
    config_file: str = None,
    derived_field: str = 'divergence',
):
    """Four-panel figure illustrating how fronts are defined and measured.

    Parameters
    ----------
    config_file : str, optional
        Path to a finding_config YAML.  Defaults to finding_config_A.yaml.
    derived_field : str, optional
        Derived field for panel (d) background.  Default 'divergence'.
    """
    from skimage import morphology
    from skimage.measure import regionprops

    # ---- Load config (default: config_D) ----
    if config_file is None:
        config_file = config_io.config_filename('D')
    cfg = config_io.load(config_file)
    binary = cfg.get('binary', {})
    props = cfg.get('properties', {})
    window = binary.get('window', 64)
    threshold = binary.get('threshold', 90)
    min_size = binary.get('min_size', 7)
    connectivity = binary.get('connectivity', 2)
    do_thin = binary.get('thin', True)
    do_sharpen = binary.get('sharpen', False)
    do_despur = binary.get('despur', False)
    Lspur = binary.get('Lspur', 10)
    dilation_radius = props.get('dilation_radius', 2)

    # Locate the sub-region and load data
    row_sl, col_sl = _find_subregion(target_lat, target_lon, half_width)
    lat, lon = _load_coords_subregion(row_sl, col_sl)
    gradb2 = _load_subregion('gradb2', row_sl, col_sl)
    panel_d_field = _load_subregion(derived_field, row_sl, col_sl)
    print(f'Sub-region: rows {row_sl}, cols {col_sl}, shape {gradb2.shape}')
    print(f'Config: window={window}, threshold={threshold}, '
          f'thin={do_thin}, sharpen={do_sharpen}, despur={do_despur}')

    # ---- Step 1: threshold ----
    thresh_mask = pyboa.front_thresh(
        gradb2, wndw=window, prcnt=threshold, mode='vectorized')

    # ---- Step 2: sharpen/thin + despur + crop → thinned fronts ----
    front_mask = thresh_mask.copy()
    if do_sharpen:
        front_mask = global_sharpen_pq(front_mask, gradb2,
                                       protect_endpoints=True)
    front_mask = pyboa.cropping(front_mask, min_size=min_size,
                                connectivity=connectivity)
    if do_thin or do_sharpen:
        front_mask = morphology.thin(front_mask)
    if do_despur:
        front_mask = prune_short_spurs(front_mask, Lspur=Lspur)

    # ---- Step 3: label individual fronts ----
    labeled, n_fronts = sklabel(front_mask, connectivity=connectivity,
                                return_num=True)
    print(f'{n_fronts} fronts detected in sub-region')

    # ---- Step 4: dilation mask for property measurement ----
    background = labeled == 0
    dist, nearest_idx = distance_transform_edt(background, return_indices=True)
    dilated = labeled.copy()
    dilated[background] = 0
    expand_mask = background & (dist <= dilation_radius)
    dilated[expand_mask] = labeled[
        nearest_idx[0][expand_mask], nearest_idx[1][expand_mask]]

    # ---- Plot 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # gradb for display — dark where values are large
    gradb = np.sqrt(gradb2)
    gradb_pos = np.where(gradb > 0, gradb, np.nan)
    gradb_norm = LogNorm(
        vmin=np.nanpercentile(gradb_pos, 1),
        vmax=np.nanpercentile(gradb_pos, 99))
    gradb_cmap = defs.cmaps['gradb']

    # (a) gradb2 greyscale + threshold pixels in red
    ax = axes[0, 0]
    ax.pcolormesh(lon, lat, gradb_pos, cmap=gradb_cmap,
                  norm=gradb_norm, rasterized=True)
    # Uniform light-salmon overlay for threshold pixels
    from matplotlib.colors import ListedColormap
    red_cmap = ListedColormap(['lightsalmon'])
    red_overlay = np.ma.masked_where(~thresh_mask, np.ones_like(gradb2))
    ax.pcolormesh(lon, lat, red_overlay, cmap=red_cmap, vmin=0, vmax=1,
                  alpha=0.4, rasterized=True)
    ax.set_title('(a) Threshold pixels', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # (b) gradb2 + thinned fronts in green, with colorbar for gradb
    ax = axes[0, 1]
    im_b = ax.pcolormesh(lon, lat, gradb_pos, cmap=gradb_cmap,
                         norm=gradb_norm, rasterized=True)
    # Uniform light-green overlay for thinned fronts
    green_cmap = ListedColormap(['lightgreen'])
    green_overlay = np.ma.masked_where(~front_mask, np.ones_like(gradb2))
    ax.pcolormesh(lon, lat, green_overlay, cmap=green_cmap, vmin=0, vmax=1,
                  alpha=0.7, rasterized=True)
    cb_b = plt.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    cb_b.set_label(defs.labels['gradb'], fontsize=10)
    ax.set_title('(b) Thinned fronts', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # (c) Labeled fronts in distinct colors
    ax = axes[1, 0]
    labeled_ma = np.ma.masked_where(labeled == 0, labeled.astype(float))
    ax.pcolormesh(lon, lat, gradb_pos, cmap=gradb_cmap,
                  norm=gradb_norm, rasterized=True)
    ax.pcolormesh(lon, lat, labeled_ma, cmap='tab20',
                  alpha=0.9, rasterized=True)
    # Label a handful of the larger fronts with their ID
    props_list = regionprops(labeled)
    props_sorted = sorted(props_list, key=lambda p: p.area, reverse=True)
    for p in props_sorted[:15]:
        cr, cc = p.centroid
        ri, ci = int(round(cr)), int(round(cc))
        if 0 <= ri < lat.shape[0] and 0 <= ci < lat.shape[1]:
            ax.text(lon[ri, ci], lat[ri, ci], str(p.label),
                    fontsize=7, color='white', ha='center', va='center',
                    fontweight='bold')
    ax.set_title('(c) Labeled fronts', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # (d) Derived field + dilated regions in grey, with colorbar
    ax = axes[1, 1]
    field_vals = np.where(np.isfinite(panel_d_field), panel_d_field, np.nan)
    vmax_d = np.nanpercentile(np.abs(field_vals), 99)
    d_cmap = defs.cmaps.get(derived_field, 'RdBu_r')
    im_d = ax.pcolormesh(lon, lat, field_vals, cmap=d_cmap,
                         vmin=-vmax_d, vmax=vmax_d, rasterized=True)
    # Grey overlay for dilation regions (expanded ring only)
    dilation_ring = (dilated > 0) & (labeled == 0)
    ring_overlay = np.ma.masked_where(~dilation_ring, np.ones_like(gradb2))
    ax.pcolormesh(lon, lat, ring_overlay, cmap='Greys', vmin=0, vmax=1,
                  alpha=0.4, rasterized=True)
    # Fronts as thin black lines
    front_overlay = np.ma.masked_where(labeled == 0, np.ones_like(gradb2))
    ax.pcolormesh(lon, lat, front_overlay, cmap='Greys_r', vmin=0, vmax=1,
                  alpha=0.8, rasterized=True)
    cb_d = plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
    cb_d.set_label(derived_field.replace('_', ' ').capitalize(), fontsize=10)
    ax.set_title(f'(d) Dilation region ({derived_field})', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    fig.tight_layout()
    outpath = os.path.join(figures_dir, outfile)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


def fig_tsr_gradb(
    outfile: str = 'fig_tsr_gradb.png',
    target_lat: float = 35.0,
    target_lon: float = -65.0,
    half_width: int = 60,
):
    """Four-panel figure: Theta, Salt, density, and |grad b| in a sub-region."""
    # jmd95 density from the llc4320 preprocessing package
    from dbof.utils.jmd95_xgcm_implementation import jmd95

    # Locate the sub-region and load coords
    row_sl, col_sl = _find_subregion(target_lat, target_lon, half_width)
    lat, lon = _load_coords_subregion(row_sl, col_sl)

    # Load the four fields
    theta = _load_subregion('Theta', row_sl, col_sl)
    salt = _load_subregion('Salt', row_sl, col_sl)
    gradb2 = _load_subregion('gradb2', row_sl, col_sl)
    # Compute density from T and S at the surface (p=0 dbar), offset from 1025
    p_surf = np.zeros_like(theta)
    density = jmd95(salt, theta, p_surf) - defs.RHO_REF
    print(f'Sub-region: rows {row_sl}, cols {col_sl}, shape {theta.shape}')

    # Buoyancy gradient magnitude — extended range for less contrast
    gradb = np.sqrt(gradb2)
    gradb_pos = np.where(gradb > 0, gradb, np.nan)

    # ---- Plot 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    panels = [
        ('Theta', theta, None),
        ('Salt', salt, None),
        ('density', density, None),
        ('gradb', gradb_pos, LogNorm(
            vmin=np.nanpercentile(gradb_pos, 0.5),
            vmax=np.nanpercentile(gradb_pos, 99.5))),
    ]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for ax, (name, field, norm), plabel in zip(
            axes.ravel(), panels, panel_labels):
        cmap = defs.cmaps.get(name, 'viridis')
        if norm is not None:
            im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                               norm=norm, rasterized=True)
        else:
            im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                               rasterized=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(defs.labels.get(name, name), fontsize=10)
        ax.set_title(f'{plabel} {defs.labels.get(name, name)}', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.tight_layout()
    outpath = os.path.join(figures_dir, outfile)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


def main(flg):
    """Main function to generate figures."""
    flg = int(flg)
    if flg == 1:
        fig_turner_vs_gradb()
    elif flg == 2:
        fig_front_definition(derived_field='strain_mag')
    elif flg == 3:
        fig_tsr_gradb()
    else:
        raise ValueError(f"Invalid flag: {flg}")

# Command line execution
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)