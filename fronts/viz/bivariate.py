"""Bivariate maps: fronts coloured by two fields at once.

Generalised from ``fronts/properties/nb/Bivariate_TurnerAngle.ipynb``, which
is hardcoded to a 2x2 grid of |grad b|^2 x Turner angle.  Here the grid is
``n x n`` for any ``n``, and either axis can be any per-front column.

The scheme: **field A drives lightness, field B drives hue.**  Reading a
bivariate map is then two independent questions -- "how dark?" for A and
"which colour?" for B -- rather than one impossible one.

Two things carried over from the notebook because they matter:

* **Quantile edges, not equal-width.**  Front properties are heavy-tailed;
  equal-width bins put nearly every front in one cell and the map goes flat.
* **A physical split beats a quantile split when one exists.**  Turner angle
  divides at 0 (salinity- vs temperature-dominated), not at its median.
  :data:`NATURAL_SPLITS` records the fields that have one, and
  :func:`bin_edges` uses it.

Nothing here needs the app; it is a plain matplotlib module, usable from a
notebook or a script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Fields with a physically meaningful dividing value.  When a field is in
#: here and the section count is even, that value is used as the middle
#: edge and the rest are quantiles either side of it.
NATURAL_SPLITS: dict[str, float] = {
    "turner_angle": 0.0,
    "relative_vorticity": 0.0,
    "rossby_number": 0.0,
    "divergence": 0.0,
    "okubo_weiss": 0.0,
    "strain_n": 0.0,
    "strain_s": 0.0,
    "frontogenesis_tendency": 0.0,
    "frontogenesis_geo": 0.0,
    "frontogenesis_ageo": 0.0,
    "Eta": 0.0,
}

#: Default hues, as RGB at full saturation.  Field B picks along this list.
DEFAULT_HUES: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.25, 0.65),      # blue
    (0.70, 0.13, 0.13),      # red
    (0.15, 0.45, 0.20),      # green
    (0.55, 0.30, 0.65),      # purple
    (0.85, 0.55, 0.10),      # orange
    (0.10, 0.55, 0.60),      # teal
)


@dataclass
class BivariateScheme:
    """A resolved bivariate colour scheme.

    Attributes
    ----------
    colors : numpy.ndarray
        ``(n, n, 3)`` RGB, indexed ``[bin_a, bin_b]``.
    edges_a, edges_b : numpy.ndarray
        ``(n + 1,)`` bin edges for each field.
    name_a, name_b : str
    n : int
    """

    colors: np.ndarray
    edges_a: np.ndarray
    edges_b: np.ndarray
    name_a: str
    name_b: str
    n: int


# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

def bivariate_colormap(n: int, hues=DEFAULT_HUES) -> np.ndarray:
    """An ``(n, n, 3)`` colour grid: lightness for A, hue for B.

    Parameters
    ----------
    n : int
        Sections per field.  ``n = 2`` reproduces the notebook's layout.
    hues : sequence of RGB triples
        Base hue per B-bin.  Cycled if there are more bins than hues.

    Returns
    -------
    numpy.ndarray
        ``(n, n, 3)`` in 0..1, indexed ``[bin_a, bin_b]``.  Low ``bin_a``
        is light, high ``bin_a`` is the saturated hue.
    """
    if n < 2:
        raise ValueError("need at least 2 sections per field")

    grid = np.zeros((n, n, 3))
    # Lightest bin keeps a little colour so the hue is still readable;
    # pure white would make every B-bin identical at low A.
    lightness = np.linspace(0.78, 0.0, n)

    for b in range(n):
        hue = np.asarray(hues[b % len(hues)], dtype=float)
        for a in range(n):
            grid[a, b] = hue + (1.0 - hue) * lightness[a]
    return np.clip(grid, 0.0, 1.0)


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------

def bin_edges(values, n: int, *, field_name: str = "",
              clip_percentile: float = 2.0) -> np.ndarray:
    """Bin edges for one field.

    Quantiles by default.  If *field_name* has an entry in
    :data:`NATURAL_SPLITS` and *n* is even, that value becomes the middle
    edge and quantiles fill in either side -- so a 2-section Turner angle
    map splits at 0 rather than at the median.

    Returns
    -------
    numpy.ndarray
        ``(n + 1,)`` monotonically increasing edges.  Outer edges are
        clipped percentiles so a single outlier cannot flatten the map.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0.0, 1.0, n + 1)

    lo = np.percentile(v, clip_percentile)
    hi = np.percentile(v, 100.0 - clip_percentile)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5

    split = NATURAL_SPLITS.get(field_name)
    if split is not None and n % 2 == 0 and lo < split < hi:
        half = n // 2
        lower = np.quantile(v[v < split], np.linspace(0, 1, half + 1)) \
            if np.any(v < split) else np.linspace(lo, split, half + 1)
        upper = np.quantile(v[v >= split], np.linspace(0, 1, half + 1)) \
            if np.any(v >= split) else np.linspace(split, hi, half + 1)
        edges = np.concatenate([lower[:-1], [split], upper[1:]])
    else:
        edges = np.quantile(v, np.linspace(0, 1, n + 1))

    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = lo, hi

    # Quantiles collide when a field is largely constant; nudge so
    # np.digitize still produces n distinct bins.
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = np.nextafter(edges[k - 1], np.inf)
    return edges


def assign_bins(values, edges) -> np.ndarray:
    """Bin index per value.  ``-1`` where the value is not finite."""
    v = np.asarray(values, dtype=float)
    n = len(edges) - 1
    out = np.full(v.shape, -1, dtype=int)
    good = np.isfinite(v)
    out[good] = np.clip(np.digitize(v[good], edges[1:-1], right=False), 0, n - 1)
    return out


def build_scheme(values_a, values_b, n: int = 2, *,
                 name_a: str = "", name_b: str = "",
                 hues=DEFAULT_HUES, clip_percentile: float = 2.0
                 ) -> BivariateScheme:
    """Resolve colours and edges for a pair of fields."""
    return BivariateScheme(
        colors=bivariate_colormap(n, hues),
        edges_a=bin_edges(values_a, n, field_name=name_a,
                          clip_percentile=clip_percentile),
        edges_b=bin_edges(values_b, n, field_name=name_b,
                          clip_percentile=clip_percentile),
        name_a=name_a, name_b=name_b, n=n,
    )


def colors_for(values_a, values_b, scheme: BivariateScheme):
    """Per-front RGB, plus a mask of the fronts that could be coloured."""
    ia = assign_bins(values_a, scheme.edges_a)
    ib = assign_bins(values_b, scheme.edges_b)
    ok = (ia >= 0) & (ib >= 0)
    rgb = np.zeros((len(ia), 3))
    rgb[ok] = scheme.colors[ia[ok], ib[ok]]
    return rgb, ok, ia, ib


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) < 0.01 or abs(v) >= 1e4:
        return f"{v:.1e}"
    return f"{v:.3g}"


def plot_legend(scheme: BivariateScheme, ax=None, *, figsize=(3.4, 3.4)):
    """The ``n x n`` legend square, with the edge values on the axes."""
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n = scheme.n
    for a in range(n):
        for b in range(n):
            ax.add_patch(plt.Rectangle((b, a), 1, 1,
                                       facecolor=scheme.colors[a, b],
                                       edgecolor="white", linewidth=0.8))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels([_fmt(v) for v in scheme.edges_b], fontsize=7,
                       rotation=45, ha="right")
    ax.set_yticklabels([_fmt(v) for v in scheme.edges_a], fontsize=7)
    ax.set_xlabel(f"{scheme.name_b}  →", fontsize=9, fontweight="bold")
    ax.set_ylabel(f"{scheme.name_a}  →", fontsize=9, fontweight="bold")
    if owns_figure:
        fig.tight_layout()
    return fig, ax


def plot_map(df, values_a, values_b, scheme: BivariateScheme, *,
             lat_col="centroid_lat", lon_col="centroid_lon",
             spatial_bin_deg: float | None = 2.0,
             marker_size: float = 2.0,
             pacific: bool = True,
             land_from=None,
             title: str = "",
             figsize=(15, 7), dpi=110, ax=None):
    """Fronts on a global map, coloured by the bivariate scheme.

    Parameters
    ----------
    df : pandas.DataFrame
        Front table carrying centroid lat/lon.
    values_a, values_b : array-like
        Per-front values for the two fields, aligned with *df*.
    scheme : BivariateScheme
    spatial_bin_deg : float or None
        When set, aggregate fronts into bins this many degrees wide and
        colour each bin by its *dominant* category, which reads far better
        than tens of thousands of overlapping dots.  ``None`` scatters.
    pacific : bool
        Draw on a 0..360 axis with the Pacific centred.
    land_from : tuple or None
        Optional ``(lon, lat, land_mask)`` raster drawn in gray beneath the
        fronts -- the model's own land mask, so it matches the grid.
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    lat = np.asarray(df[lat_col], dtype=float)
    lon = np.asarray(df[lon_col], dtype=float)
    if pacific:
        lon = lon % 360.0

    rgb, ok, ia, ib = colors_for(values_a, values_b, scheme)
    ok = ok & np.isfinite(lat) & np.isfinite(lon)

    if land_from is not None:
        llon, llat, lmask = land_from
        if pacific:
            order = np.argsort(llon % 360.0)
            llon = (llon % 360.0)[order]
            lmask = lmask[:, order]
        ax.pcolormesh(llon, llat, np.where(lmask > 0, 1.0, np.nan),
                      cmap="Greys", vmin=0, vmax=2, shading="nearest",
                      zorder=0)

    if spatial_bin_deg:
        _plot_binned(ax, lon[ok], lat[ok], ia[ok], ib[ok], scheme,
                     spatial_bin_deg, pacific)
    else:
        ax.scatter(lon[ok], lat[ok], c=rgb[ok], s=marker_size,
                   linewidths=0, zorder=2)

    ax.set_xlim((0, 360) if pacific else (-180, 180))
    ax.set_ylim(-80, 80)
    ax.set_aspect("auto")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{scheme.name_a}  ×  {scheme.name_b}", fontsize=11)
    ax.grid(alpha=0.2, linewidth=0.4)
    if owns_figure:
        fig.tight_layout()
    return fig, ax


def figure_bivariate_grid(lon, lat, values_a, values_b, *, n=2,
                          name_a="", name_b="", title="", land=None,
                          figsize=(15.0, 7.0)):
    """The same two-field colouring, over *every* grid cell.

    ``figure_bivariate`` colours one point per front, from the colocation
    table.  This colours a raster instead, so the fronts can be read
    against the field they came out of rather than on their own.  The
    colour scheme is built the same way, so a pair of maps shares a
    legend and is directly comparable.

    Parameters
    ----------
    lon, lat : numpy.ndarray
        1-D cell centres of the display raster.
    values_a, values_b : numpy.ndarray
        ``(len(lat), len(lon))`` rasters of the two fields.
    land : numpy.ndarray, optional
        Same-shaped mask, non-zero where land, drawn in gray underneath.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"field rasters differ: {a.shape} vs {b.shape}")

    scheme = build_scheme(a.ravel(), b.ravel(), n=n,
                          name_a=name_a, name_b=name_b)
    rgb, ok, _, _ = colors_for(a.ravel(), b.ravel(), scheme)

    # RGBA so cells with no data stay transparent rather than black --
    # a zero RGB is a legitimate colour in this scheme.
    rgba = np.zeros((a.size, 4))
    rgba[:, :3] = rgb
    rgba[ok, 3] = 1.0
    rgba = rgba.reshape(a.shape + (4,))

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.0], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])

    extent = (float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]))
    if land is not None:
        ax.imshow(np.where(np.asarray(land) > 0, 1.0, np.nan),
                  extent=extent, origin="lower", cmap="gray_r",
                  vmin=0, vmax=1.6, interpolation="nearest")
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="nearest")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{name_a}  x  {name_b}  -- all grid points",
                 fontsize=11)
    ax.set_aspect("auto")

    plot_legend(scheme, ax=fig.add_subplot(grid[0, 1]))
    return fig, scheme


def _plot_binned(ax, lon, lat, ia, ib, scheme, deg, pacific):
    """Colour each spatial bin by its most common (a, b) category."""
    lon0 = 0.0 if pacific else -180.0
    nx = int(round(360.0 / deg))
    ny = int(round(160.0 / deg))

    ix = np.clip(((lon - lon0) / deg).astype(int), 0, nx - 1)
    iy = np.clip(((lat + 80.0) / deg).astype(int), 0, ny - 1)

    n = scheme.n
    cat = ia * n + ib
    counts = np.zeros((ny, nx, n * n), dtype=np.int32)
    np.add.at(counts, (iy, ix, cat), 1)

    total = counts.sum(axis=2)
    dominant = counts.argmax(axis=2)
    a_idx, b_idx = np.divmod(dominant, n)

    rgba = np.zeros((ny, nx, 4))
    rgba[..., :3] = scheme.colors[a_idx, b_idx]
    rgba[..., 3] = np.where(total > 0, 1.0, 0.0)

    ax.imshow(rgba, origin="lower",
              extent=(lon0, lon0 + 360.0, -80.0, 80.0),
              interpolation="nearest", aspect="auto", zorder=2)


def figure_bivariate(df, values_a, values_b, *, n=2, name_a="", name_b="",
                     spatial_bin_deg=2.0, land_from=None, title="",
                     figsize=(16, 7), dpi=110):
    """Map plus legend in one figure -- the deliverable for the page."""
    import matplotlib.pyplot as plt

    scheme = build_scheme(values_a, values_b, n, name_a=name_a, name_b=name_b)

    # constrained_layout, not tight_layout: an imshow with aspect='auto'
    # beside a square legend is exactly the case tight_layout warns about.
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    plot_map(df, values_a, values_b, scheme, ax=ax_map,
             spatial_bin_deg=spatial_bin_deg, land_from=land_from,
             title=title)
    plot_legend(scheme, ax=ax_leg)
    return fig, scheme
