""" ``fronts_from_gradb2`` on a field containing NaN land (coding doc §5,
prompt 2 task 7, criterion 6).

Land is NaN in the OSN fields (planning §5.5, M0 task 3), so a NaN-bearing
``gradb2`` is the normal input to front finding.  No test of that existed
in ``fronts``.  These tests use **finding config D**
(``fronts/finding/configs/finding_config_D.yaml``: window 64, 85th
percentile, ``thresh_mode='pool'``, sharpen + despur; read from the YAML
because the function's defaults are *not* config D), on small synthetic
fields offline and on M0's first hour (``needs_grid``).

How NaN flows through config D, step by step (measured here):

1. ``pyboa.front_thresh`` -- a sliding 64x64 ``nanpercentile`` with
   ``cval=nan``; NaN cells compare ``False`` (``nan > q``), the window's
   percentile is taken over its finite cells only.  NaN-safe; the three
   modes (generic / vectorized / pool) agree bit for bit.
2. ``sharpen.global_sharpen_pq`` -- priority-queue thinning keyed on
   ``gradb2``; only reads ``gradb2`` on foreground pixels, none of which is
   NaN.  Ends with ``morphology.thin``.  NaN-safe.
3. ``pyboa.cropping`` -- ``spur``, ``remove_small_objects``, then
   ``remove_small_holes`` (default ``area_threshold=64``): a NaN island
   smaller than 64 px enclosed by a front is *filled*, so front pixels land
   on NaN.  **Not NaN-safe** (xfail below).
4. final ``morphology.thin`` -- binary, NaN-blind.
5. ``despur.prune_short_spurs`` -- ``skan.Skeleton`` on the skeleton; it
   raises ``ValueError`` on an *empty* skeleton (an all-NaN field, or any
   field where nothing survives cropping).  **Breaks** (xfail below).

Independently of NaN, config D's ``thresh_mode='pool'`` needs an explicit
``n_workers``: ``fronts_from_gradb2``'s default ``n_workers=None`` reaches
``np.array_split(rows, None)`` and raises ``TypeError`` (xfail below).  The
fixes belong in ``fronts`` (see the task-7a log entry); nothing here patches
it, the tests only record the behaviour.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import ndimage
from skimage import morphology

from fronts.finding import pyboa
from fronts.finding.algorithms import fronts_from_gradb2

import osn_tiles as ot

CONFIG_D = (Path(ot.__file__).resolve().parents[3] / 'fronts' / 'finding' / 'configs'
            / 'finding_config_D.yaml')
N_WORKERS = 2          # config D is 'pool' mode: the caller must supply this (see below)

# the two skimage / numpy deprecation notices the fronts chain emits today;
# not what these tests are about
_NOISE = [pytest.mark.filterwarnings('ignore:Parameter `min_size` is deprecated'),
          pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array')]
pytestmark = _NOISE


def config_d() -> dict:
    """The ``binary`` block of config D, as ``fronts/finding/run.py`` passes
    it (``**bparam``)."""
    with open(CONFIG_D) as f:
        return yaml.safe_load(f)['binary']


def find(G, **over):
    """Config D with an explicit ``n_workers`` (the only way it runs)."""
    return fronts_from_gradb2(G, n_workers=N_WORKERS, **{**config_d(), **over})


# ---------------------------------------------------------------------------
# synthetic fields
# ---------------------------------------------------------------------------
NJ, NI = 160, 200
COAST_I = 40            # land is i < COAST_I (a straight coast along j)


def ridge_field(along='j', centre=100.0, width=2.0, noise=0.05, seed=0):
    """A Gaussian ridge in ``gradb2`` (a clear front, ~4 px above the local
    85th percentile) on a weak uniform-noise background; ``along='j'`` puts
    the ridge at ``i = centre`` (parallel to the coast), ``along='i'`` at
    ``j = centre`` (running into the coast)."""
    rng = np.random.default_rng(seed)
    jj, ii = np.mgrid[0:NJ, 0:NI]
    x = ii if along == 'j' else jj
    return np.exp(-((x - centre) / width) ** 2) + noise * rng.random((NJ, NI))


def land_block():
    land = np.zeros((NJ, NI), bool)
    land[:, :COAST_I] = True
    return land


def with_nan(G, land):
    Gn = G.copy()
    Gn[land] = np.nan
    return Gn


@pytest.fixture(scope='module')
def parallel_case():
    """Ridge parallel to the coast, 60 cells offshore: the NaN field, the
    land-free field, and config D's fronts on both."""
    G = ridge_field('j')
    land = land_block()
    Gn = with_nan(G, land)
    return dict(G=G, Gn=Gn, land=land, f_nan=find(Gn), f_free=find(G))


# ---------------------------------------------------------------------------
# config D itself
# ---------------------------------------------------------------------------
def test_config_d_is_not_the_function_defaults():
    """Read the YAML rather than assuming its numbers (coding §2.5)."""
    cfg = config_d()
    assert cfg['window'] == 64 and cfg['threshold'] == 85
    assert cfg['thresh_mode'] == 'pool'
    assert cfg['sharpen'] and cfg['despur'] and cfg['Lspur'] == 10
    assert not cfg['thin'] and not cfg['dilate']
    assert cfg['min_size'] == 7 and cfg['connectivity'] == 2
    # ...and none of window / threshold / thresh_mode / sharpen / despur is
    # the default of fronts_from_gradb2 (40, 90, 'generic', False, False)
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(fronts_from_gradb2).parameters.items()}
    for k in ('window', 'threshold', 'thresh_mode', 'sharpen', 'despur'):
        assert cfg[k] != defaults[k], k


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="config D is thresh_mode='pool'; fronts_from_gradb2's default "
                          "n_workers=None reaches np.array_split(rows, None) in "
                          "pyboa.front_thresh and raises TypeError -- the config cannot be "
                          "run as-is without an explicit n_workers (run.py hard-codes 10)")
def test_config_d_as_is_needs_n_workers():
    G = with_nan(ridge_field('j'), land_block())
    fronts_from_gradb2(G, **config_d())


# ---------------------------------------------------------------------------
# 1. a whole field with a NaN land block
# ---------------------------------------------------------------------------
def test_land_block_runs_shape_dtype_and_no_front_on_nan(parallel_case):
    f = parallel_case['f_nan']
    assert f.shape == (NJ, NI) and f.dtype == np.bool_
    assert f.sum() > 0                                       # the ridge is found
    assert not (f & parallel_case['land']).any()             # no front pixel on NaN
    assert not (f & ~np.isfinite(parallel_case['Gn'])).any()


def test_land_block_fronts_away_from_land_equal_the_land_free_field(parallel_case):
    """Comparison: pixel-for-pixel equality of the two boolean front masks
    on every column >= 10 cells from the coast (``i >= 50``; the ridge is
    at ``i = 100``), and at most 2 differing pixels in the 10 coastal
    columns (the threshold window sees fewer finite cells there).  Measured:
    0 and 0."""
    f, f0 = parallel_case['f_nan'], parallel_case['f_free']
    far = slice(COAST_I + 10, None)
    near = slice(COAST_I, COAST_I + 10)
    assert np.array_equal(f[:, far], f0[:, far])
    assert (f[:, near] ^ f0[:, near]).sum() <= 2
    # the ridge itself: one connected 1-px line the full height of the field
    assert f[:, 95:106].sum() >= NJ - 4                      # sharpen retracts the ends by ~1 px
    # 1 px wide after sharpen + thin (an 8-connected line has 2 px in a row
    # where it steps diagonally, never 3)
    assert f[:, 95:106].sum(axis=1).max() <= 2
    assert f[:, 95:106].sum() <= NJ + 4


def test_threshold_modes_agree_on_nan_and_are_nan_safe(parallel_case):
    """Step 1 in isolation: the three ``front_thresh`` modes give the same
    mask on the NaN field, no NaN cell is flagged, and the 'vectorized'
    mode emits numpy's All-NaN-slice RuntimeWarning for windows entirely on
    land (harmless: ``nan > nan`` is False) while 'generic' does not."""
    Gn, land = parallel_case['Gn'], parallel_case['land']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        tv = pyboa.front_thresh(Gn, wndw=64, prcnt=85, mode='vectorized')
        tp = pyboa.front_thresh(Gn, wndw=64, prcnt=85, mode='pool', n_workers=N_WORKERS)
    tg = pyboa.front_thresh(Gn, wndw=64, prcnt=85, mode='generic')
    assert tv.dtype == np.bool_ and np.array_equal(tv, tp) and np.array_equal(tv, tg)
    assert not (tv & land).any()
    with pytest.warns(RuntimeWarning, match='All-NaN slice'):
        pyboa.front_thresh(Gn[:70, :70], wndw=64, prcnt=85, mode='vectorized')


def test_percentile_window_has_no_coastal_bias():
    """A flat noise field with a NaN coast: the fraction flagged by the 85th
    percentile window is the same (~15%) in the 32 coastal columns as in the
    interior, to 1 point -- the shrinking finite population next to land
    does not manufacture fronts."""
    rng = np.random.default_rng(1)
    G = with_nan(1.0 + 0.05 * rng.random((NJ, NI)), land_block())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        t = pyboa.front_thresh(G, wndw=64, prcnt=85, mode='vectorized')
    coastal = t[:, COAST_I:COAST_I + 32].mean()
    interior = t[:, 100:].mean()
    assert abs(coastal - 0.15) < 0.01 and abs(interior - 0.15) < 0.01
    assert not (t & land_block()).any()


# ---------------------------------------------------------------------------
# 2. a front running into the coast
# ---------------------------------------------------------------------------
def test_front_into_coast_stops_one_cell_short_and_nothing_else_changes():
    """A ridge along ``i`` (row ``j = 80``) that runs into the coast.  With
    NaN land: no front pixel on NaN; the front reaches the second ocean
    column (``i = 41``) -- the skeleton retracts one endpoint pixel at the
    coast, as it does at an open field edge; no spurious front along the
    coast; away from the coast (``i >= 50``) the mask equals the land-free
    one pixel for pixel, and at most 2 pixels differ in ``40 <= i < 50``."""
    G = ridge_field('i', centre=80.0)
    land = land_block()
    f = find(with_nan(G, land))
    f0 = find(G)
    assert not (f & land).any()
    cols = np.where(f.any(axis=0))[0]
    assert cols.min() in (COAST_I, COAST_I + 1)              # reaches the coast (1-px retraction)
    assert cols.max() == NI - 1 or cols.max() >= NI - 3      # ...and the far edge
    assert np.array_equal(f[:, 50:], f0[:, 50:])
    assert (f[:, COAST_I:50] ^ f0[:, COAST_I:50]).sum() <= 2
    # nothing along the coast but the ridge: the first two ocean columns
    # hold front pixels only within +/-3 rows of the ridge
    off_ridge = np.ones(NJ, bool)
    off_ridge[77:84] = False
    assert not f[off_ridge, COAST_I:COAST_I + 2].any()


@pytest.mark.xfail(strict=True,
                   reason="pyboa.cropping ends with remove_small_holes() (area_threshold 64): "
                          "a NaN island < 64 px enclosed by a front is filled and the final "
                          "thin draws the skeleton across it -- 6 front pixels on NaN for a "
                          "6x5 island on the ridge.  Fix belongs in fronts (mask holes that "
                          "are NaN in gradb2, or re-apply isfinite after cropping)")
def test_small_nan_island_enclosed_by_a_front_gets_no_front_pixels():
    G = ridge_field('j')
    island = np.zeros((NJ, NI), bool)
    island[75:81, 98:103] = True                              # 30 px, on the ridge
    f = find(with_nan(G, island))
    assert not (f & island).any()


def test_small_nan_island_beside_a_front_is_fine():
    """The same island 10 cells off the ridge is not enclosed, so nothing
    fills it: 0 front pixels on NaN."""
    G = ridge_field('j')
    island = np.zeros((NJ, NI), bool)
    island[75:81, 110:115] = True
    f = find(with_nan(G, island))
    assert not (f & island).any() and f.sum() > 0


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="an all-NaN field thresholds to all-False; despur.prune_short_spurs "
                          "builds skan.Skeleton on the empty skeleton and skan raises "
                          "ValueError('index pointer size 0 should be 1') -- config D cannot "
                          "return an empty front mask (any field where nothing survives "
                          "cropping, e.g. an all-land tile).  Fix belongs in fronts: return "
                          "early when the skeleton is empty")
def test_all_nan_field_returns_all_false():
    f = find(np.full((NJ, NI), np.nan))
    assert f.dtype == np.bool_ and not f.any()


def test_all_nan_field_without_despur_returns_all_false():
    """Steps 1-4 handle the all-NaN field; only despur (step 5) breaks."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        f = find(np.full((NJ, NI), np.nan), despur=False)
    assert f.shape == (NJ, NI) and f.dtype == np.bool_ and not f.any()


# ---------------------------------------------------------------------------
# 4. caller-side workarounds
# ---------------------------------------------------------------------------
def test_filling_land_is_not_a_safe_workaround(parallel_case):
    """Filling NaN with 0 or with the ocean median before finding changes
    the fronts *away* from land: the filled cells enter the 64x64 percentile
    window and move the local threshold (measured: 15 pixels differ at
    ``i >= 50`` for either fill, 16 in the coastal columns).  NaN is the
    right input; do not fill."""
    G, Gn, land, f = (parallel_case[k] for k in ('G', 'Gn', 'land', 'f_nan'))
    for fill in (0.0, float(np.nanmedian(Gn))):
        Gf = G.copy()
        Gf[land] = fill
        ff = find(Gf)
        assert not (ff & land).any()                         # nothing on land: the ridge is elsewhere
        n_far = (ff[:, 50:] ^ f[:, 50:]).sum()
        assert n_far > 0, 'a harmless fill would make this test wrong'
        assert n_far < 40


def test_masking_the_output_is_the_safe_workaround():
    """Passing NaN and applying ``isfinite(gradb2)`` to the *output* removes
    the island fill and changes nothing else (the mask is already False on
    every other NaN cell).  This is what M4 should do until ``fronts``
    guards ``remove_small_holes``."""
    G = ridge_field('j')
    land = land_block()
    island = np.zeros((NJ, NI), bool)
    island[75:81, 98:103] = True
    Gn = with_nan(G, land | island)
    f = find(Gn)
    assert (f & island).sum() == 6                           # the xfail above, quantified
    fm = f & np.isfinite(Gn)
    assert not (fm & ~np.isfinite(Gn)).any()
    assert np.array_equal(fm[~island], f[~island])
    # the masked result is a 1-px line broken at the island, not a new artefact
    assert fm[:, 95:106].sum(axis=1).max() <= 2
    assert fm[:, 95:106].sum() <= NJ + 4


# ---------------------------------------------------------------------------
# 3. real tile (needs_grid): the repo's grad_b2 on M0's first hour
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_real_tile_hour0_config_d(grid_ds, raw_ds, masks_ds):
    """Config D on the repo's ``grad_b2`` (``calculate_grad_squared_tracer``,
    the front-*finding* field per the operators contract) for hour 0.
    Measured: 11,836 front pixels in 7.3 s (pool, 4 workers); 0 on NaN;
    878 (7.4%) inside the 7-cell halo; 7,885 on the analysis mask.  The
    front density at the first finite cell from land is 0.17 against 0.032
    in the interior (``grad_b2``'s median there is ~70x the interior's) --
    that is the input field, not the finder: the percentile window shows
    no coastal bias on a flat field -- and it is inside the halo."""
    import xarray as xr
    from dbof.preprocessing import calculate_fields as cf
    ds = xr.merge([raw_ds.isel(time=0).expand_dims('face'), grid_ds],
                  compat='override', combine_attrs='override').astype('float64')
    grid = ot.build_xgcm(grid_ds)
    G = cf.grad_b2(ds, grid).compute().values[0]
    ocean = masks_ds['mask_ocean'].values
    halo = masks_ds['mask_halo'].values
    ana = masks_ds['mask_analysis'].values
    fin = np.isfinite(G)
    assert G.shape == (720, 720) and not (fin & ~ocean).any()   # NaN on every land cell
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        f = fronts_from_gradb2(G, n_workers=4, **config_d())
    dt = time.perf_counter() - t
    assert f.shape == G.shape and f.dtype == np.bool_
    n = int(f.sum())
    n_nan = int((f & ~fin).sum())
    n_halo = int((f & ocean & ~halo).sum())
    n_ana = int((f & ana).sum())
    taxi = ndimage.distance_transform_cdt(fin, metric='taxicab')
    dens = {d: float(f[taxi == d].mean()) for d in (1, 2, 3, 5, 7)}
    dens['>=8'] = float(f[taxi >= 8].mean())
    print(f'\nconfig D on hour-0 grad_b2: {n} front pixels in {dt:.1f} s; {n_nan} on NaN; '
          f'{n_halo} ({n_halo / n:.3f}) inside the halo; {n_ana} on the analysis mask; '
          f'front density by taxicab distance from NaN: '
          + ', '.join(f'{k}: {v:.3f}' for k, v in dens.items()))
    assert n_nan == 0
    assert 8_000 < n < 16_000
    assert n_halo / n < 0.15
    assert n_ana > 0.5 * n
    assert dens[1] > dens['>=8']                             # the coastal excess, all in the halo
    assert dens[7] < 2 * dens['>=8']                         # ...and gone by the halo's edge
    assert dt < 120
