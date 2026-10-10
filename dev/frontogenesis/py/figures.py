""" M3 task 7: the ten figures — shared style, loaders and the public names.

One function per figure, ``fig01_maps`` … ``fig10_term_budget`` (coding
§4.10), each writing ``figs/fig{NN}_*.png`` at 200 dpi in ``validate_figs``'
style and **returning the numbers it drew**.  Driven by ``py/m3_figs.py``.

**Nothing here computes physics.**  Every number comes from the derived
stores (§3.4) or from ``data/m3_closure_summary.json``, so the figures
cannot disagree with task 6's verdict — ``tests/test_figures.py`` asserts
that the values drawn equal the ones in the summary.

The module passes coding §1.3's ~400 lines, so it is split by family as the
prompt allows: :mod:`figures_maps` (1, 3b, 5, 10) and :mod:`figures_stats`
(2, 2b, 3, 4, 6, 7).  This file holds the style, the loaders and the
re-exports, so callers import one place.

**Framing (M3-Q15 (a), 2026-10-10).**  Task 6's gate failed at every `L`, so
these figures carry a methodological finding, not an efficiency.  Figure 2
draws its slopes under an explicit **NOT QUOTED** banner, and no figure
calls anything "diabatic damping" — Figure 2b is the discriminator and it
does not support that reading.
"""

from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import xarray as xr                                            # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / 'figs'
DATA_DIR = HERE.parent / 'data'
SUMMARY = DATA_DIR / 'm3_closure_summary.json'

DPI = 200
L_ALL = (0, 2, 4, 8)
#: the two hours Figure 1 and the map figures use (prompt 4 task 7): the
#: mixed-layer minimum and its opposite phase, both on day 2 of 3
HOUR_DAY = 45          # 2012-07-03 21 UTC = 13 LST, the KPPhbl minimum
HOUR_NIGHT = 33        # 2012-07-03 09 UTC = 01 LST, the KPPhbl maximum
DAY3_PAIRS = tuple(range(62, 69))
BASELINE, BASELINE_CI = 0.981, (0.970, 0.994)
V3B_BAND = (0.954, 1.003)
TEMPORAL = (0.972, 0.020)

COL = dict(two_F='#1f5fa8', subfilter='#f4a259', vertical='#2a9d8f',
           surface_flux='#c8102e', surface_flux_kpp='#e89a9a', residual='black',
           catchall='#888888', measured='#5b2a86', kpp='#2a9d8f', grey='#888888',
           red='#c8102e', day3='#5b2a86')
DIVERGING = 'RdBu_r'

#: every caption carries these three, which Figure 6 must state verbatim
#: (M2 task 6 item 4; prompt 4 task 7)
FORCING_CAVEAT = (
    'Forcing is 6-hourly and linearly interpolated (kinks at 03/09/15/21 UTC), so the diurnal '
    'shortwave is a TRIANGLE peaking at 13 LST, not resolved insolation.')
WIND_CAVEAT = (
    'The day-to-day deepening of the afternoon minimum (13 -> 12 -> 8 m) is a wind trend '
    '(|tau| 0.11 -> 0.06 N m^-2), not noise.')
WINDOW_CAVEAT = '72 h is three diurnal cycles: indicative, not conclusive (planning §7).'
#: M3-Q14 (a), JXP 2026-10-10
SAMPLING_CAVEAT = (
    'Only hourly fields exist, so the sampling bound of task 6 (semi-Lagrangian and Eulerian '
    'DG/Dt differ by 0.67 of measured at L = 0) cannot be tested against a shorter dt.')


# ---------------------------------------------------------------------------
# loaders -- the only two sources a figure may read
# ---------------------------------------------------------------------------
def derived_path(L, data_dir=None):
    return Path(data_dir or DATA_DIR) / f'tile330_derived_L{int(L)}.zarr'


def open_derived(L, data_dir=None):
    """The §3.4 store for one ``L``, lazily."""
    return xr.open_zarr(derived_path(L, data_dir))


def load_summary(path=None):
    """``m3_closure_summary.json`` -- task 6's numbers, which the figures
    may draw but never recompute."""
    p = Path(path or SUMMARY)
    if not p.exists():
        raise FileNotFoundError(f'{p} not found: run m3_closure.py (task 6) first')
    return json.loads(p.read_text())


def front_pool(L, fields, data_dir=None, pairs=None):
    """Pooled front-pixel values of ``fields`` over the pairs requested
    (all 71 by default), plus ``pair`` and ``lst``.  One pass, one pair in
    memory at a time."""
    ds = open_derived(L, data_dir)
    n = ds.sizes['time']
    idx = range(n) if pairs is None else list(pairs)
    out = {k: [] for k in tuple(fields) + ('pair', 'lst')}
    for p in idx:
        h = ds.isel(time=p).load()
        fr = np.asarray(h['front'].values, bool)
        for k in fields:
            v = h[k]
            a = np.asarray(v.values, 'float64')
            out[k].append(a[fr] if a.shape == fr.shape else np.full(int(fr.sum()), np.nan))
        out['pair'].append(np.full(int(fr.sum()), p, dtype='int32'))
        out['lst'].append(np.full(int(fr.sum()), _lst(h), dtype='float64'))
    ds.close()
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def _lst(h, utc_offset=-8.0):
    t = np.asarray(h['time_mid'].values, dtype='datetime64[ns]')
    sec = (t - t.astype('datetime64[D]')) / np.timedelta64(1, 's')
    return float((sec / 3600.0 + utc_offset) % 24.0)


def resolve_pair(L, pair, data_dir=None):
    """The requested pair index, clamped to the store's length.  A no-op on
    the real 71-pair product; it lets the offline tests drive the same
    functions with a six-pair stub instead of forking them."""
    ds = open_derived(L, data_dir)
    n = int(ds.sizes['time'])
    ds.close()
    return int(min(int(pair), n - 1))


def hour(L, pair, data_dir=None):
    """One pair's fields in memory, with the map coordinates."""
    ds = open_derived(L, data_dir)
    h = ds.isel(time=pair).load()
    ds.close()
    return h


# ---------------------------------------------------------------------------
# style helpers
# ---------------------------------------------------------------------------
def save(fig, name, dpi=DPI, fig_dir=None):
    d = Path(fig_dir or FIG_DIR)
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)


def draw_map(ax, X, Y, C, cmap=DIVERGING, norm=None, land=None, **kw):
    """``pcolormesh(XC, YC, ...)`` so face 10 comes out north-up, east-right
    (``i`` runs south, ``j`` east) -- the same convention as
    ``m0_qa_plot.draw_map``."""
    if land is not None:
        ax.pcolormesh(X, Y, np.ma.masked_where(~land, np.ones_like(X)),
                      cmap=matplotlib.colors.ListedColormap(['#b0b0b0']), shading='nearest')
    m = ax.pcolormesh(X, Y, np.ma.masked_invalid(C), cmap=cmap, norm=norm, shading='nearest', **kw)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(Y))))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    return m


def sym_limit(*arrays, pct=99.0):
    """A shared symmetric colour limit for a diverging map set."""
    v = np.concatenate([np.asarray(a, 'float64').ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return float(np.percentile(np.abs(v), pct)) if v.size else 1.0


def caption(fig, text, y=0.005, fontsize=8):
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=fontsize, color='0.3', wrap=True)


def mask_note(L, mask='front & valid (mask_analysis & isfinite, p90 of G at the midpoint)'):
    """Every title or caption must state its ``L`` and its mask (task 7)."""
    return f'L = {L} cells   |   {mask}'


def binned(x, y, bins=12, lo=2.5, hi=97.5, n_boot=200, labels=None, seed=0):
    """Binned ``E[Y|X]`` in x-quantile bins with a block-bootstrap band when
    ``labels`` is given.  Thin wrapper on ``stats`` so the figures and task 6
    use one implementation."""
    import stats as st
    df = st.binned_conditional_mean(x, y, bins=bins, split_sign=False, labels=labels,
                                    n_boot=n_boot, seed=seed)
    return df


# ---------------------------------------------------------------------------
# the public names (coding §4.10)
# ---------------------------------------------------------------------------
from figures_maps import (fig01_maps, fig03b_filter_ladder, fig05_sharpening_time,   # noqa: E402
                          fig10_term_budget)
from figures_stats import (fig02_joint_pdf, fig02b_discriminator, fig03_slope_vs_L,  # noqa: E402
                           fig04_alignment, fig06_diurnal, fig07_offshore)

#: name -> callable, in figure order; ``m3_figs`` drives this
FIGURES = {
    'fig01': fig01_maps, 'fig02': fig02_joint_pdf, 'fig02b': fig02b_discriminator,
    'fig03': fig03_slope_vs_L, 'fig03b': fig03b_filter_ladder, 'fig04': fig04_alignment,
    'fig05': fig05_sharpening_time, 'fig06': fig06_diurnal, 'fig07': fig07_offshore,
    'fig10': fig10_term_budget}
