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

*(added 2026-11-16, prompt 6 task 1 -- M5 consolidation.)*  ``FIGURES`` now
holds every figure of the study -- the ten above, Figures 8 and 9 and the
tracking-quality diagnostic (drawn in :mod:`figures_m4`), and V1-V6 plus V3b
(drawn in :mod:`validate_figs`) -- behind one signature, and
:func:`footer` is the one caption helper: every entry states its ``L`` and
its mask on the page and returns them as keys.  ``py/make_figs.py`` is the
one command; ``m3_figs.py`` still renders ``M3_FIGURES`` only.

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


#: the two masks the M3 figures use, stated once (prompt 6 task 1)
MASK_FRONT = 'front & valid (mask_analysis & isfinite, p90 of G at the midpoint)'
MASK_FINITE = 'all finite cells (land and the filter rim grey)'
MASK_ANALYSIS = 'mask_analysis (the fronts are found on it)'
MASK_SYNTHETIC = 'synthetic'


def mask_note(L, mask=MASK_FRONT):
    """Every title or caption must state its ``L`` and its mask (task 7)."""
    return f'L = {L} cells   |   {mask}'


def _fmt_L(L):
    if L is None:
        return 'L: not applicable (no filter)'
    if isinstance(L, (tuple, list)):
        return 'L = ' + ' / '.join(str(int(v)) for v in L) + ' cells'
    return f'L = {int(L)} cells'


def footer_text(L, mask, config=None, band=None, text='', caveats=()):
    """The uniform caption (prompt 6 task 1): ``L``, the mask, the finding
    config / band where the figure has one, then the figure's own sentence
    and the caveats.  One implementation for every registry entry."""
    head = [_fmt_L(L), f'mask: {mask}']
    if config is not None:
        head.append(f'config {config}')
    if band is not None:
        head.append(f'band {band}')
    tail = ' '.join(t.strip() for t in (text,) + tuple(caveats) if t and t.strip())
    return '   |   '.join(head) + ('.  ' + tail if tail else '.')


def footer(fig, out, L, mask, *, config=None, band=None, text='', caveats=(), y=0.005,
           fontsize=8):
    """Draw :func:`footer_text` on ``fig`` and record what it states in the
    figure's returned dict -- ``out['L']`` and ``out['mask']`` (plus
    ``config`` / ``band`` when given) -- so "every caption states its L and
    mask" is a key a test can assert, not a thing to read off a PNG."""
    txt = footer_text(L, mask, config, band, text, caveats)
    caption(fig, txt, y=y, fontsize=fontsize)
    out['L'] = list(L) if isinstance(L, (tuple, list)) else L
    out['mask'] = mask
    if config is not None:
        out['config'] = config
    if band is not None:
        out['band'] = band
    return txt


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

# ---------------------------------------------------------------------------
# M4's figures and the validation figures (prompt 6 task 1) -- delegates
# ---------------------------------------------------------------------------
# The drawing code stays where it was written: Figures 8, 9 and the
# tracking-quality diagnostic in ``figures_m4`` and V1-V6 in
# ``validate_figs`` (driven by ``validate``'s gate functions).  Coding
# §4.10's "figV1..figV6 in figures.py" is met by the REGISTRY below, not by
# moving code.  The delegates import lazily: ``figures_m4`` imports this
# module, and ``validate`` pulls in the whole science path.
def fig08_front_case(data_dir=None, fig_dir=None, **kw):
    """Figure 8 -- one tracked front, redrawn from ``m4_strength.json``, the
    tracks, the label zarr and the ``L = 0`` store (``figures_m4``)."""
    import figures_m4
    return figures_m4.fig08_from_products(data_dir, fig_dir, **kw)


def fig09_population(data_dir=None, fig_dir=None, **kw):
    """Figure 9 -- the ten tracked fronts, redrawn from ``m4_strength.json``."""
    import figures_m4
    return figures_m4.fig09_from_products(data_dir, fig_dir, **kw)


def fig_tq_tracking_quality(data_dir=None, fig_dir=None, **kw):
    """The tracking-quality diagnostic (M4 criterion 3; not in planning §7's
    numbering), redrawn from ``m4_quality.json``."""
    import figures_m4
    return figures_m4.fig_tq_from_products(data_dir, fig_dir, **kw)


#: the keys of each gate's result that go into the figures summary (the
#: full results carry arrays); a trailing ``*`` is a prefix
_V_KEEP = {
    'figV1': ('series_max_err_ref', 'semilag_err_max_all', 'convergence_order', 'gate'),
    'figV2': ('err_*', 'lin_err*', 'R_from*', 'gate*', 'worst_x*'),
    'figV3': ('slope', 'ci', 'n_front', 'gate', 'gate_form', 'velocities'),
    'figV3b': ('bias', 'stencil_effect', 'velocities'),
    'figV4': ('headline', 'headline_order1', 'headline_order5', 'width_slope'),
    'figV5': ('bias*',),
    'figV6': ('n_*', 'edge_*', 'halo_*', 'offshore_km'),
}


def _grid(data_dir=None):
    import osn_tiles as ot
    return ot.open_grid(None if data_dir is None else Path(data_dir) / 'tile330_grid.zarr',
                        with_face=True)


def _vfig(name, call, data_dir, fig_dir, L=None, mask=MASK_SYNTHETIC):
    """Run one of ``validate``'s gates with ``png=True`` so it draws through
    ``validate_figs`` as it always has, into ``fig_dir``; return the small
    numbers with the registry's ``png`` / ``L`` / ``mask`` keys."""
    import validate as va
    import validate_figs as vf
    keep_dir = vf.FIG_DIR
    vf.FIG_DIR = Path(fig_dir or FIG_DIR)
    try:
        res = call(va)
    finally:
        vf.FIG_DIR = keep_dir
    keys = _V_KEEP[name]
    out = {k: v for k, v in res.items()
           if any(k == p or (p.endswith('*') and k.startswith(p[:-1])) for p in keys)}
    out.update(png=res['png'], L=res.get('L', L), mask=res.get('mask', mask))
    return out


def figV1(data_dir=None, fig_dir=None, **kw):
    """V1 -- Cartesian deformation (synthetic; ``validate.test_cartesian_deformation``)."""
    return _vfig('figV1', lambda va: va.test_cartesian_deformation(png=True, **kw),
                 data_dir, fig_dir)


def figV2(data_dir=None, fig_dir=None, **kw):
    """V2 -- analytic gradients on the tile-330 metric (needs the grid store)."""
    return _vfig('figV2', lambda va: va.test_native_metric(_grid(data_dir), png=True, **kw),
                 data_dir, fig_dir)


def _null(gate_name, data_dir, kw):
    """V3 / V3b report the real-velocity ('llc') half when the M0 stores are
    there -- its slope is Figure 2's baseline -- and the strain half alone
    when they are not; the figure carries both halves whenever it can."""
    def call(va):
        gate = getattr(va, gate_name)
        vel = kw.pop('velocities', None)
        if vel is None:
            try:
                return dict(gate(velocities='llc', png=True, grid_ds=_grid(data_dir), **kw),
                            velocities='llc')
            except FileNotFoundError:
                vel = 'strain'
        return dict(gate(velocities=vel, png=True, **kw), velocities=vel)
    return call


def figV3(data_dir=None, fig_dir=None, **kw):
    """V3 -- the discrete null: the baseline slope 0.981 [0.970, 0.994]."""
    return _vfig('figV3', _null('test_discrete_null', data_dir, kw), data_dir, fig_dir)


def figV3b(data_dir=None, fig_dir=None, **kw):
    """V3b -- the finite-volume null: the model-advection band (a recorded
    bias, not a gate; M1-Q2).  Drawn with the semi-Lagrangian truth beside the
    four finite-volume schemes, as the figure in ``figs/`` always was (M1 task
    6b): the gate's own default omits those reference rows."""
    kw.setdefault('schemes', ('semilag', 'centred', 'os7', 'os7mp', 'dst3'))
    return _vfig('figV3b', _null('test_fv_null', data_dir, kw), data_dir, fig_dir)


def figV4(data_dir=None, fig_dir=None, **kw):
    """V4 -- interpolation bias under a uniform, zero-strain flow (synthetic)."""
    return _vfig('figV4', lambda va: va.test_interpolation_bias(png=True, **kw),
                 data_dir, fig_dir)


def figV5(data_dir=None, fig_dir=None, **kw):
    """V5 -- interpolate ``b``, never ``G``: the half-cell demonstration (synthetic)."""
    return _vfig('figV5', lambda va: va.demo_interp_half_cell(png=True, **kw),
                 data_dir, fig_dir)


def figV6(data_dir=None, fig_dir=None, **kw):
    """V6 -- land halo, offshore cut and tile-edge margin: the mask geometry
    itself (needs the grid and the 2 h raw store).  Its footer and its
    ``mask`` key are written by ``validate_figs.fig_V6``."""
    return _vfig('figV6', lambda va: va.qa_land_halo(_grid(data_dir), png=True, **kw),
                 data_dir, fig_dir)


#: name -> callable, in figure order: planning §7's twelve, the M4
#: tracking-quality diagnostic, then the validation set.  Every entry is
#: ``fn(data_dir=None, fig_dir=None, **kw)`` and returns the numbers it drew
#: with ``png``, ``L`` and ``mask`` keys.  ``make_figs`` drives all of it.
FIGURES = {
    'fig01': fig01_maps, 'fig02': fig02_joint_pdf, 'fig02b': fig02b_discriminator,
    'fig03': fig03_slope_vs_L, 'fig03b': fig03b_filter_ladder, 'fig04': fig04_alignment,
    'fig05': fig05_sharpening_time, 'fig06': fig06_diurnal, 'fig07': fig07_offshore,
    'fig08': fig08_front_case, 'fig09': fig09_population, 'fig10': fig10_term_budget,
    'fig_tq': fig_tq_tracking_quality,
    'figV1': figV1, 'figV2': figV2, 'figV3': figV3, 'figV3b': figV3b, 'figV4': figV4,
    'figV5': figV5, 'figV6': figV6}

#: M3 task 7's ten, which ``m3_figs`` renders into ``m3_figs_summary.json``
#: (task 8's audit) -- unchanged by M5
M3_FIGURES = ('fig01', 'fig02', 'fig02b', 'fig03', 'fig03b', 'fig04', 'fig05', 'fig06',
              'fig07', 'fig10')
