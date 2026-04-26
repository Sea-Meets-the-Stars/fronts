"""Global visualization of front-weighted properties.

Provides field-agnostic plotting methods for:
  plot_global_property_map  — Cartopy map of a gridded 2-D field
  plot_binned_front_map     — Cartopy map of per-front property binned by centroid
  plot_property_jpdf        — 2-D histogram / JPDF of two variables
  plot_multi_timestamp      — arrange any of the above across multiple timestamps

All methods accept pre-loaded data (numpy arrays / pandas DataFrames) and
return the matplotlib Figure so callers can further customize or save.

Follows the patterns established in Section 5 of
``fronts/properties/nb/Turner_Angle_Global_Viz.ipynb``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import binned_statistic_2d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from fronts.viz.utils import set_fontsize

# ---------------------------------------------------------------------------
# Global property map (gridded field on Cartopy projection)
# ---------------------------------------------------------------------------

def plot_global_property_map(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    *,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
    clip_pct: float = 2.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    downsample: int = 4,
    mask_seams: bool = True,
    title: Optional[str] = None,
    clabel: Optional[str] = None,
    fontsize: float = None,
    figsize: Tuple[float, float] = (18, 9),
    ax=None,
):
    """Cartopy Robinson-projection map of a gridded 2-D property field.

    Parameters
    ----------
    lon, lat : np.ndarray
        2-D coordinate arrays (must match *data* shape or be the
        full-resolution arrays that will be downsampled together with *data*).
    data : np.ndarray
        2-D property array.
    cmap : str
        Matplotlib colormap name.
    symmetric : bool
        If True, force the colorbar symmetric around zero.
    clip_pct : float
        Percentile clipping for automatic colorbar limits (ignored when
        *vmin*/*vmax* are provided explicitly).
    vmin, vmax : float, optional
        Explicit colorbar limits.  Override *clip_pct* / *symmetric*.
    downsample : int
        Spatial downsample factor for rendering (1 = full resolution).
    mask_seams : bool
        Blank LLC4320 tile-boundary seams where adjacent lat jumps > 5 deg
        or lon jumps > 90 deg.
    title : str, optional
        Plot title.
    clabel : str, optional
        Colorbar label.
    fontsize : float, optional
        Font size for the plot.
    figsize : tuple
        Figure size in inches (only used when *ax* is None).
    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Existing axes to draw on.  If None a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    # Downsample for rendering
    ds = max(1, int(downsample))
    lon_p = lon[::ds, ::ds]
    lat_p = lat[::ds, ::ds]
    data_p = data[::ds, ::ds].copy()  # copy so we can mask in-place

    # Mask LLC4320 tile-boundary seams
    if mask_seams:
        dlat = np.abs(np.diff(lat_p, axis=0))
        dlon = np.abs(np.diff(lon_p, axis=0))
        seam = (dlat > 5) | (dlon > 90)
        seam_mask = np.zeros(data_p.shape, dtype=bool)
        seam_mask[:-1, :] |= seam
        seam_mask[1:, :] |= seam
        data_p[seam_mask] = np.nan

    # Colorbar limits
    if vmin is None or vmax is None:
        lo = float(np.nanpercentile(data_p, clip_pct))
        hi = float(np.nanpercentile(data_p, 100 - clip_pct))
        if symmetric:
            bound = max(abs(lo), abs(hi))
            auto_vmin, auto_vmax = -bound, bound
        else:
            auto_vmin, auto_vmax = lo, hi
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    # Create figure/axes if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': ccrs.Robinson()},
        )
    else:
        fig = ax.get_figure()

    tfm = ccrs.PlateCarree()
    ax.set_global()

    pm = ax.pcolormesh(
        lon_p, lat_p, data_p,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=tfm, shading='auto', rasterized=True,
    )
    plt.colorbar(
        pm, ax=ax, orientation='vertical',
        label=clabel or '',
        fraction=0.025, pad=0.04, shrink=0.85,
    )

    # Map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='k', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='gray', zorder=3)
    ax.gridlines(
        draw_labels=False, linewidth=0.3, color='gray',
        alpha=0.5, linestyle='--',
    )

    if fontsize is not None:
        set_fontsize(ax, fontsize)

    if title is not None:
        ax.set_title(title, fontsize=11, pad=10)

    if created_fig:
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Binned front map (per-front property on Cartopy projection)
# ---------------------------------------------------------------------------

def plot_binned_front_map(
    df: pd.DataFrame,
    property_col: str,
    *,
    lat_col: str = 'centroid_lat',
    lon_col: str = 'centroid_lon',
    n_lat_bins: int = 90,
    n_lon_bins: int = 180,
    statistic: str = 'mean',
    min_count: int = 2,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
    clip_pct: float = 2.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    clabel: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 9),
    ax=None,
):
    """Cartopy map of a per-front property binned by centroid lat/lon.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *lat_col*, *lon_col*, and *property_col*.
    property_col : str
        Column to visualize.
    lat_col, lon_col : str
        Column names for front centroid coordinates.
    n_lat_bins, n_lon_bins : int
        Number of spatial bins (defaults give 2 deg resolution).
    statistic : str
        Binning statistic passed to ``binned_statistic_2d``
        (e.g. ``'mean'``, ``'median'``, ``'count'``).
    min_count : int
        Bins with fewer fronts than this are masked to NaN.
    cmap, symmetric, clip_pct, vmin, vmax : see :func:`plot_global_property_map`.
    title, clabel : str, optional
    figsize : tuple
    ax : GeoAxes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """

    vals = df[property_col].values
    lats = df[lat_col].values
    lons = df[lon_col].values

    # Keep only finite entries
    valid = np.isfinite(vals) & np.isfinite(lats) & np.isfinite(lons)

    # Binned statistic
    grid, lat_edges, lon_edges, _ = binned_statistic_2d(
        lats[valid], lons[valid], vals[valid],
        statistic=statistic,
        bins=[n_lat_bins, n_lon_bins],
        range=[[-90, 90], [-180, 180]],
    )

    # Count mask — hide bins with too few fronts
    cnt, _, _, _ = binned_statistic_2d(
        lats[valid], lons[valid], vals[valid],
        statistic='count',
        bins=[n_lat_bins, n_lon_bins],
        range=[[-90, 90], [-180, 180]],
    )
    grid = np.where(cnt >= min_count, grid, np.nan)

    # Colorbar limits
    if vmin is None or vmax is None:
        lo = float(np.nanpercentile(grid, clip_pct))
        hi = float(np.nanpercentile(grid, 100 - clip_pct))
        if symmetric:
            bound = max(abs(lo), abs(hi))
            auto_vmin, auto_vmax = -bound, bound
        else:
            auto_vmin, auto_vmax = lo, hi
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    # Create figure/axes if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': ccrs.Robinson()},
        )
    else:
        fig = ax.get_figure()

    tfm = ccrs.PlateCarree()
    ax.set_global()

    pm = ax.pcolormesh(
        lon_edges, lat_edges, grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=tfm, shading='auto', rasterized=True,
    )
    plt.colorbar(
        pm, ax=ax, orientation='vertical',
        label=clabel or '',
        fraction=0.025, pad=0.04, shrink=0.85,
    )

    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='k', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='gray', zorder=3)
    ax.gridlines(
        draw_labels=False, linewidth=0.3, color='gray',
        alpha=0.5, linestyle='--',
    )

    if title is not None:
        ax.set_title(title, fontsize=11, pad=8)

    if created_fig:
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Property JPDF (2-D histogram)
# ---------------------------------------------------------------------------

def plot_property_jpdf(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_x_bins: int = 60,
    n_y_bins: int = 60,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    y_log: bool = True,
    y_pct: float = 1.0,
    fontsize: float = None,
    cmap: str = 'Reds',
    clabel: Optional[str] = None,
    cfsz: float = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[float, float] = (10, 7),
    ax=None,
):
    """JPDF of two per-front (or per-pixel) properties.

    Parameters
    ----------
    x_values, y_values : np.ndarray
        1-D arrays of the two variables (NaN/Inf entries are excluded).
    n_x_bins, n_y_bins : int
        Number of histogram bins along each axis.
    x_range, y_range : (float, float), optional
        Explicit bin ranges.  If None, *x_range* defaults to the data
        min/max and *y_range* is derived from percentiles of positive values.
    y_log : bool
        If True, use log-spaced y-bins and a log y-axis.
    y_pct : float
        Percentile clip for auto y-range (e.g. 1 means [P1, P99]).
    cmap : str
        Colormap for the PDF.
    xlabel, ylabel, title : str, optional
    fontsize : float, optional
        Font size for the plot.
    annotations : list of dict, optional
        Each dict may contain:
        - ``x`` (float): x-position for a vertical reference line
        - ``color`` (str): line / text color
        - ``label`` (str): text label placed at ``(x, y_mid)``
    figsize : tuple
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    x = np.asarray(x_values, dtype=np.float64).ravel()
    y = np.asarray(y_values, dtype=np.float64).ravel()

    # Co-mask NaN / Inf / non-positive y (if log)
    valid = np.isfinite(x) & np.isfinite(y)
    if y_log:
        valid &= y > 0
    x, y = x[valid], y[valid]

    # Bin edges
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_range is None:
        y_lo = float(np.percentile(y[y > 0], y_pct)) if y_log else float(np.min(y))
        y_hi = float(np.percentile(y[y > 0], 100 - y_pct)) if y_log else float(np.max(y))
        y_range = (y_lo, y_hi)

    if y_log:
        y_edges = np.logspace(np.log10(y_range[0]), np.log10(y_range[1]),
                              n_y_bins + 1)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    # 2-D histogram
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Normalize: for log-spaced y-bins use d(ln y) widths
    dx = np.diff(x_edges)[:, None]
    if y_log:
        dy = np.diff(np.log(y_edges))[None, :]
    else:
        dy = np.diff(y_edges)[None, :]
    total = counts.sum()
    pdf = counts / (total * dx * dy) if total > 0 else counts

    # Plot
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pmin = pdf[pdf > 0].min() if (pdf > 0).any() else 1e-10
    pm = ax.pcolormesh(
        x_edges, y_edges, pdf.T,
        norm=LogNorm(vmin=pmin, vmax=pdf.max()),
        cmap=cmap, rasterized=True,
    )

    # Colorbar
    cbar = plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    if clabel is not None:
        cbar.set_label(clabel, fontsize=16)
    if cfsz is not None:
        cbar.ax.tick_params(labelsize=cfsz)

    if y_log:
        ax.set_yscale('log')
    ax.set_xlim(x_range)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)

    # Annotations: vertical reference lines + text labels
    if annotations:
        y_mid = np.sqrt(y_edges[0] * y_edges[-1]) if y_log else \
                0.5 * (y_edges[0] + y_edges[-1])
        for ann in annotations:
            x_pos = ann.get('x')
            color = ann.get('color', 'k')
            label = ann.get('label')
            if x_pos is not None:
                ax.axvline(x_pos, color=color, lw=0.9, ls='--', alpha=0.6)
            if label is not None:
                # Place label at x_pos (or 0) and y_mid
                ax.text(
                    x_pos if x_pos is not None else 0, y_mid, label,
                    color=color, fontsize=9,
                    ha='center', va='center', alpha=0.8,
                )

    if fontsize is not None:
        set_fontsize(ax, fontsize)

    if created_fig:
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Multi-timestamp convenience wrapper
# ---------------------------------------------------------------------------

def plot_multi_timestamp(
    plot_fn: Callable,
    data_per_time: Dict[str, Dict[str, Any]],
    *,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (9, 5),
    suptitle: Optional[str] = None,
):
    """Call *plot_fn* for each timestamp and arrange in a grid of subplots.

    Parameters
    ----------
    plot_fn : callable
        One of :func:`plot_global_property_map`, :func:`plot_binned_front_map`,
        or :func:`plot_property_jpdf`.
    data_per_time : dict
        ``{timestamp_label: kwargs_dict}`` where each *kwargs_dict* is
        passed to *plot_fn* (excluding ``ax`` and ``figsize``, which are
        managed here).
    ncols : int
        Maximum columns per row.
    figsize_per_panel : tuple
        ``(width, height)`` per subplot.
    suptitle : str, optional
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    n = len(data_per_time)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fw, fh = figsize_per_panel
    fig_w = fw * ncols
    fig_h = fh * nrows

    # Detect whether plot_fn needs GeoAxes (map functions) or plain Axes
    is_map = plot_fn in (plot_global_property_map, plot_binned_front_map)

    subplot_kw = {'projection': ccrs.Robinson()} if is_map else {}
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        subplot_kw=subplot_kw,
        squeeze=False,
    )

    for idx, (label, kwargs) in enumerate(data_per_time.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        # Override ax and figsize in the kwargs
        kw = {k: v for k, v in kwargs.items() if k not in ('ax', 'figsize')}
        kw['ax'] = ax
        # Append timestamp to title if caller didn't set one
        if 'title' not in kw or kw['title'] is None:
            kw['title'] = label
        plot_fn(**kw)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)

    fig.tight_layout()
    return fig
