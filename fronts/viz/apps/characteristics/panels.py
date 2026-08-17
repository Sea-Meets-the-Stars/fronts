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


def _kinematics_message(samples: RegionSamples) -> str:
    if samples.missing:
        return (
            "joint PDFs need "
            + ", ".join(samples.missing)
            + "\nno matching channel in this store"
        )
    return "no samples outside the equatorial band"


def close(fig):
    """No-op kept for symmetry.

    Figures are built outside pyplot's registry, so there is nothing to
    release; the pane holds the only reference and it is garbage
    collected normally.
    """
    return None
