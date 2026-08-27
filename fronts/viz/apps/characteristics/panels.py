"""The six statistics panels on page 1.

Each panel is drawn by the preprocessing repo's own builders, so the
figures on the page are the same figures the validation notebooks make:

* ``dbof.plotting.pdfs.pdf_panel`` / ``shared_bins``
* ``dbof.plotting.jpdfs.plot_jpdf_occurrence``
* ``dbof.plotting.jpdfs.plot_jpdf_conditional`` /
  ``plot_jpdf_conditional_log``

All of them take a matplotlib axis and draw on it, which is why they drop
straight into a Panel ``Matplotlib`` pane with no adaptation at all.
"""

from __future__ import annotations

import numpy as np

from fronts.viz.apps.characteristics.stats import RegionSamples
from fronts.viz.apps.common.render import MPL_LOCK, new_figure

#: Fields binned in log10 for the PDF -- positive-definite with a huge
#: dynamic range.  Matches the convention in ``dbof.plotting.pdfs``.
LOG_FIELDS = {"gradb2", "gradtheta2", "gradsalt2"}

FIGSIZE = (4.1, 3.0)
DPI = 110


class BackendMissing(RuntimeError):
    """The preprocessing repo's plotting modules are not importable."""


def _backend():
    """Import the dbof plotting modules, with an actionable error."""
    try:
        from dbof.plotting import jpdfs, pdfs
        return jpdfs, pdfs
    except ImportError as exc:
        raise BackendMissing(
            "Could not import dbof.plotting.  `pip install -e` the "
            "llc4320-native-grid-preprocessing repo, or set "
            "LLC4320_PREPROC_SRC to its src/ directory."
        ) from exc


def _blank(message: str, *, figsize=FIGSIZE):
    """A placeholder panel carrying an explanation."""
    fig, ax = new_figure(figsize, DPI)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8,
            wrap=True, transform=ax.transAxes, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout()
    return fig


def clean_values(samples: RegionSamples, field: str) -> np.ndarray:
    """Finite sample values, log10-transformed for the log fields."""
    v = samples.values
    v = v[np.isfinite(v)]
    if field in LOG_FIELDS:
        v = v[v > 0]
        v = np.log10(v)
    return v


def pdf_bins(columns: dict[str, RegionSamples], field: str):
    """Shared bin edges across both columns, so they are comparable."""
    _, pdfs = _backend()
    pools = [clean_values(s, field) for s in columns.values()]
    pools = [p for p in pools if p.size]
    if not pools:
        return None
    return pdfs.shared_bins(pools)


def figure_pdf(samples: RegionSamples, field: str, bins, *, title=""):
    """Panel (a) -- probability density of the selected field."""
    with MPL_LOCK:
        _, pdfs = _backend()
        values = clean_values(samples, field)
        if bins is None or values.size == 0:
            return _blank("no finite samples in this region")

        label = f"log10({field})" if field in LOG_FIELDS else field
        fig, ax = new_figure(FIGSIZE, DPI)
        pdfs.pdf_panel(ax, values, bins, label=label)
        ax.set_title(f"{title}  n={values.size:,}", fontsize=8)
        fig.tight_layout()
        return fig


def figure_jpdf(samples: RegionSamples, *, title=""):
    """Panel (b) -- occurrence joint PDF in the vorticity-strain plane."""
    with MPL_LOCK:
        jpdfs, _ = _backend()
        if not samples.has_kinematics:
            return _blank(_kinematics_message(samples))

        nb = adaptive_bins(samples.zeta_f.size)
        fig, ax = new_figure(FIGSIZE, DPI)
        jpdfs.plot_jpdf_occurrence(ax, samples.zeta_f, samples.sigma_f, bins=nb)
        ax.set_title(f"{title}  n={samples.zeta_f.size:,}  ({nb} bins)", fontsize=8)
        fig.tight_layout()
        return fig


def figure_jpdf_conditional(samples: RegionSamples, field: str, *, title=""):
    """Panel (c) -- the field's conditional mean in that same plane."""
    with MPL_LOCK:
        jpdfs, _ = _backend()
        if not samples.has_kinematics:
            return _blank(_kinematics_message(samples))

        c = samples.values
        if c.size == 0:
            return _blank("no finite samples in this region")

        nb = adaptive_bins(samples.zeta_f.size)
        min_count = 10 if samples.zeta_f.size > 50_000 else 3
        fig, ax = new_figure(FIGSIZE, DPI)
        positive = field in LOG_FIELDS or np.nanmin(c) > 0

        if positive:
            finite = c[np.isfinite(c) & (c > 0)]
            if finite.size == 0:
                return _blank("no positive samples for a log conditional mean")
            vmin, vmax = np.nanpercentile(finite, [5, 95])
            jpdfs.plot_jpdf_conditional_log(
                ax, samples.zeta_f, samples.sigma_f, c,
                bins=nb, min_count=min_count,
                vmin=float(vmin), vmax=float(max(vmax, vmin * 1.01)),
                clabel=f"mean {field}",
            )
        else:
            scale = np.nanpercentile(np.abs(c[np.isfinite(c)]), 98)
            jpdfs.plot_jpdf_conditional(
                ax, samples.zeta_f, samples.sigma_f, c,
                bins=nb, min_count=min_count,
                vlim=float(scale if scale > 0 else 1.0),
                linthresh=float(scale / 100.0 if scale > 0 else 1e-2),
                clabel=f"mean {field}",
            )

        ax.set_title(f"{title}  |  cond. on {field}  ({nb} bins)", fontsize=8)
        fig.tight_layout()
        return fig


def adaptive_bins(n: int) -> int:
    """Bin count for the joint PDFs, scaled to the sample size.

    ``dbof.plotting.jpdfs`` defaults to 175 bins per axis, which is right
    for a global field.  A small region -- and especially the fronts-only
    column, which is a few percent of the cells -- would leave almost every
    bin under ``min_count`` and render blank.  Scaling the grid keeps the
    panel informative instead of empty, and the bin count is reported in
    the panel title so the difference between the two columns is visible.
    """
    if n <= 0:
        return 20
    return int(np.clip(np.sqrt(n) / 3.0, 20, 175))


#: Which channel each role wants, for the blank-panel message.
_ROLE_CHANNELS = {
    "vorticity": "relative_vorticity",
    "strain": "strain_mag",
    "coriolis": "coriolis_f",
}


def _kinematics_message(samples: RegionSamples) -> str:
    """Why panels (b) and (c) are empty, in terms of what to go and build.

    These two do not use the selected field -- they are vorticity against
    strain -- so "nothing here" is otherwise baffling when panel (a) drew
    perfectly well from the same region.
    """
    if samples.missing:
        wanted = ", ".join(_ROLE_CHANNELS.get(r, r) for r in samples.missing)
        if samples.missing_level:
            # A different job to go and do: the subset is built, just not
            # at this level.  Saying "missing from this store" here sent
            # you looking for something that was already there.
            return (
                "joint PDFs are vorticity x strain, not the selected "
                "field.\n\n"
                f"{wanted}\nis in this store, but not at "
                f"{samples.missing_level}.\n"
                "(the panel to the left is unaffected)"
            )
        return (
            "joint PDFs are vorticity x strain, not the selected field.\n\n"
            f"missing from this store: {wanted}\n"
            "(the kinematic subset, at any depth level)"
        )
    return "no samples outside the equatorial band"


def close(fig):
    """No-op kept for symmetry.

    Figures are built outside pyplot's registry, so there is nothing to
    release; the pane holds the only reference and it is garbage
    collected normally.
    """
    return None


# --------------------------------------------------------------------------
# The selected region, as a static figure
# --------------------------------------------------------------------------

def _column_window(used: np.ndarray) -> tuple[int, int]:
    """Start column and width of the smallest window holding every used column.

    The column axis of the stitched LLC grid **wraps**: the last block of
    faces is the neighbour of the first, so a region sitting on that seam
    has cells at both ends of the array.  Its plain bounding box is then
    the entire grid -- all 17280 columns for an 18-degree box -- and the
    window drags in whole unrelated faces, whose rows are lines of
    constant *longitude* rather than latitude.  That is what drew bands of
    unrelated data straight across the figure at every longitude, for the
    handful of regions that happen to sit on the seam.

    Measuring the largest gap and taking its complement gives the window
    that actually contains the region, wrapping when that is shorter.
    """
    idx = np.nonzero(used)[0]
    if idx.size == 0:
        return 0, 0
    n = int(used.size)
    gaps = np.diff(np.concatenate([idx, idx[:1] + n]))
    k = int(np.argmax(gaps))
    return int(idx[(k + 1) % idx.size]), int(n - gaps[k] + 1)


def figure_region_map(provider, date: str, channel: str, box, *,
                      show_fronts: bool = True, field_name: str = "",
                      figsize=(10.0, 7.0)):
    """The selected region at native resolution, with the fronts on it.

    A **static** figure, built once behind *Rebuild*, and deliberately not
    an interactive map.  The interactive maps draw from the display
    pyramid and are budgeted to stay responsive while you navigate; this
    one is the evidence for the numbers beside it, so it reads the same
    native arrays the statistics do.

    That is also why it costs nothing extra.  ``provider.field`` is a
    memory-mapped local file by the time the statistics have run, so
    slicing a window out of it is a local read -- no pyramid level, no
    S3, no cells shipped to the browser.

    Drawn with ``pcolormesh`` over the window's own 2-D ``XC``/``YC``,
    which is exact on a curvilinear grid.  An ``hv.Image`` would have had
    to assume a regular axis per dimension, and on the native grid that is
    the one thing it is not.
    """
    from fronts.viz import field_styles
    from fronts.viz.apps.common.selection import bbox_mask

    with MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        XC, YC = provider.coords(date)
        mask = np.asarray(bbox_mask(XC, YC, box))
        if not mask.any():
            return _blank("no grid cells in this region")

        # The index window the box covers.  A lat/lon box is not a
        # rectangle on this grid, so the window is its bounding box --
        # cyclic in the column axis, which wraps.  See _column_window.
        js = np.nonzero(mask.any(axis=1))[0]
        j0, j1 = int(js[0]), int(js[-1]) + 1
        i0, ni = _column_window(mask.any(axis=0))
        cols = (np.arange(ni) + i0) % mask.shape[1]

        def take(a):
            return np.asarray(a)[j0:j1][:, cols]

        # Only the cells the box actually selects are drawn.  The window
        # is a rectangle in index space and the region is not, so it also
        # holds cells belonging to other faces -- geographically somewhere
        # else entirely.  Painting those was what put stripes across the
        # figure; this is also exactly the set the statistics use, so the
        # picture and the numbers now describe the same cells.
        inside = take(mask)
        # Unwrap rather than leave a 360-degree jump in the middle of the
        # window: pcolormesh reads cell centres to infer edges, and a jump
        # gives one row of enormous, wrong quads.
        lon = take(XC) % 360.0
        if lon.size and float(lon.max() - lon.min()) > 180.0:
            lon = np.where(lon < 180.0, lon + 360.0, lon)
        lat = take(YC)
        values = np.asarray(take(provider.field(date, channel)), dtype=float)

        style = field_styles.get_style(field_name or channel)
        shown = field_styles.apply_transform(values, style)
        shown = np.where(inside, shown, np.nan)
        clim = field_styles.default_clim(shown, style)

        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(shown),
                             cmap=field_styles.resolve_cmap(style.cmap),
                             vmin=clim[0], vmax=clim[1], shading="auto")
        fig.colorbar(mesh, ax=ax,
                     label=getattr(style, "title", field_name or channel))

        if show_fronts:
            try:
                fronts = take(provider.front_binary(date))
                jj, ii = np.nonzero((fronts > 0) & inside)
                if jj.size:
                    ax.scatter(lon[jj, ii], lat[jj, ii], s=1.2,
                               c="#00e5ff", linewidths=0, zorder=3)
            except Exception as exc:                    # noqa: BLE001
                ax.set_title(f"fronts unavailable: {exc}", fontsize=8,
                             loc="right", color="#b71c1c")

        # To the selected cells, not to the window: the window overshoots
        # wherever the region is not a rectangle in index space.
        ax.set_xlim(float(lon[inside].min()), float(lon[inside].max()))
        ax.set_ylim(float(lat[inside].min()), float(lat[inside].max()))

        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.set_title(f"{channel} — {box.label()}   "
                     f"({j1 - j0} x {ni} native cells)", fontsize=10)
        fig.tight_layout()
        return fig
