""" M3-digging task 4: the offline tracer -- the pipeline on a case with a known answer
(vet report §3, §6 item 1b; prompt M3-digging task 4; M3-Q29 (a), M3D-Q1 (a), M3D-Q3 (a)).

Two passive tracers -- the model's own ``Theta`` and ``Salt`` at hour 0 -- are advected for
the 72 h of the window by the model's hourly ``U``, ``V`` with the flux-form OS7MP step of
``fvadvect`` (V3b's truth: ``form='advective'``, ``integrator='split'``, ``dt_sub = 100 s``,
land filled by the nearest ocean value), and written hour by hour into a store with the raw
store's layout -- ``Theta``/``Salt`` are the tracer, ``U V W Eta KPPhbl oceTAUX oceTAUY`` are
copied -- so ``m3_run.py --raw <store> --chunk none --tag <tag>`` and ``m3_closure.py --tag``
run the budget on it unchanged.  **For the tracer the true budget is exactly**
``DG/Dt = 2F + tau + (OS7MP dissipation)``: no vertical term, no surface flux, no KPP, no
sub-hourly flow.  Its residual is the pipeline's own floor.

Two velocity columns (M3D-Q3 (a)):

* ``linear``    -- the velocity at each sub-step is linear in time between the hourly
                   snapshots (the primary column);
* ``piecewise`` -- the hour's midpoint velocity ``0.5 (U_t + U_{t+1})`` held constant over
                   the hour: exactly the flow the budget's semi-Lagrangian step assumes.
                   With this column the first hour **is** ``fvadvect.fv_advect`` (check 1).

The 7-cell edge margin (outside ``masking.edge_mask``) is reset to the model's own
``Theta``/``Salt``, interpolated in time, after every sub-step (the clamp of M3-Q29); it is
outside every statistic.  Resumable per hour: the state is the store's last hour.

Checks written into ``data/m3_tracer_{vel}_summary.json`` (all required, prompt task 4):
1. hour 0 -> 1 with the piecewise velocity and no clamp equals ``fv_advect`` bit for bit;
2. conservation: the ocean mean and the variance of each tracer and of ``b`` inside
   ``mask_analysis`` every hour (the mean may drift only by what the clamp lets in; the
   variance decays, OS7MP being dissipative);
3. the front population: ``G``'s p50 / p90 / p99 on ``mask_analysis`` at 0, 24, 48, 72 h for
   the tracer and for the model;
4. the store passes ``series_verify.verify_series`` and opens through
   ``inputs.open_inputs(chunk=None)``.

**The pre-declared reading** (:data:`DECLARED`; M3D-Q1 (a), decided 2026-10-31) is applied by
``--report`` to the closure summaries of the two columns beside the product's.

Run (from dev/frontogenesis; each column ~30 min -- 11 s per hour per tracer measured):
    nohup <python> py/m3_tracer.py --vel linear    > data/m3_tracer_linear.nohup 2>&1 &
    nohup <python> py/m3_tracer.py --vel piecewise > data/m3_tracer_piecewise.nohup 2>&1 &
then ``m3_run.py --raw data/tile330_tracer_linear_20120702T00_72h.zarr --chunk none --tag
tile330tracer`` (and ``..._piecewise...`` with ``--tag tile330tracerpc``), ``m3_closure.py
--tag`` for each, and ``m3_tracer.py --report``.
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget as bg                                            # noqa: E402
import fvadvect as fv                                          # noqa: E402
import inputs as inp                                           # noqa: E402
import masking as mk                                           # noqa: E402
import operators as op                                         # noqa: E402
import osn_tiles as ot                                         # noqa: E402
import zarr_series as zs                                       # noqa: E402
from m2_pull import timestamps_72                              # noqa: E402
from m3_run import Progress                                    # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
VELS = ('linear', 'piecewise')
TAG = dict(linear='tile330tracer', piecewise='tile330tracerpc')
TRACERS = ('Theta', 'Salt')
DT = 3600.0
DT_SUB = fv.DT_SUB                             # 100 s
N_SUB = int(np.ceil(DT / DT_SUB))              # 36
CLAMP_CELLS = 7                                # = edge_cells: the margin outside every statistic
CHECK_HOURS = (0, 24, 48, 72)
OUT_JSON = DATA_DIR / 'm3_tracer_summary.json'
OUT_PNG = FIG_DIR / 'm3_tracer.png'

DECLARED = dict(
    source='frontogenesis_prompt_M3_digging.md task 4; M3-Q29 (a) (JXP 2026-10-31), M3D-Q1 (a) '
           'and M3D-Q3 (a) (JXP 2026-10-31); fixed before m3_tracer.py was run',
    design=dict(tracers=list(TRACERS), initial='the model\'s own fields at hour 0 (2012-07-02 00 UTC)',
                scheme='os7mp', form='advective', integrator='split', dt_sub_s=DT_SUB, n_sub=N_SUB,
                land_fill='nearest ocean value, applied once per hour as fv_advect does (land never changes within the hour)',
                clamp=f'cells outside masking.edge_mask({CLAMP_CELLS}) reset to the model\'s '
                      'Theta/Salt interpolated in time, after every sub-step',
                velocity_columns=dict(linear='linear in time between the hourly snapshots (primary)',
                                      piecewise='the hour\'s midpoint velocity held constant'),
                copied='U V W Eta KPPhbl oceTAUX oceTAUY from the raw store, so the pipeline reads '
                       'the store unchanged; the chunk store is NOT used (--chunk none)'),
    true_budget='DG/Dt = 2F + tau + OS7MP dissipation: no V, no S, no KPP, no sub-hourly flow',
    reading=dict(
        L0=dict(analysis_limited='r_tr(0) >= 0.50: the hourly cadence limits the ANALYSIS at the grid '
                                 'scale even on a within-hour-smooth flow',
                data_limited='r_tr(0) <= 0.25: the limit is the model\'s own sub-hourly evolution -- the data',
                between='both, in the proportion the numbers show'),
        L8=dict(not_the_pipeline='r_tr(8) <= 0.25 against the product\'s two-term catch-all 0.685: '
                                 'the filtered-scale failure is not the pipeline'),
        thresholds=dict(hi=0.50, lo=0.25),
        statistic='rms(residual)/rms(measured) on the tracer\'s own front & valid (the catch-all: '
                  'measured - 2F - tau, the chunk terms absent)',
        slopes='quotable only as the pipeline\'s CALIBRATION, labelled "tracer, not the model", '
               'against V3b 0.975 [0.954, 1.003]; never as an efficiency of LLC4320'),
    checks=['hour 0->1 piecewise no-clamp == fv_advect bit for bit',
            'conservation: ocean mean and variance per hour',
            'G percentiles tracer vs model at 0/24/48/72 h',
            'verify_series passes; open_inputs(chunk=None) opens'])


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def store_path(vel) -> Path:
    return DATA_DIR / f'tile330_tracer_{vel}_20120702T00_72h.zarr'


def summary_path(vel) -> Path:
    return DATA_DIR / f'm3_tracer_{vel}_summary.json'


def log_path(vel) -> Path:
    return DATA_DIR / f'm3_tracer_{vel}.log'


def done_path(vel) -> Path:
    return DATA_DIR / f'm3_tracer_{vel}_done.json'


# ---------------------------------------------------------------------------
# the step, sub-step by sub-step (the body of fvadvect.fv_step_arrays)
# ---------------------------------------------------------------------------
def transports(grid_ds, U, V) -> dict:
    """``fvadvect._grid_arrays`` on raw staggered velocities (DataArrays or arrays)."""
    return fv._grid_arrays(grid_ds, U, V)


def substep(bp, T, h, s, scheme='os7mp', form='advective'):
    """One sub-step of length ``h`` on the padded field ``bp`` (in place) with
    the transports ``T``; the sweep order alternates with ``s`` exactly as
    ``fv_step_arrays``'s ``'split'`` integrator does."""
    kX, kY = T['cX'] * h, T['cY'] * h
    for axis in ((1, 0) if s % 2 == 0 else (0, 1)):
        if axis == 1:
            tend = fv._sweep_tendency(bp, T['uT'], kX, 1, T['r_inv'], scheme, form)
        else:
            tend = fv._sweep_tendency(bp, T['vT'], kY, 0, T['r_inv'], scheme, form)
        bp[fv.PAD:-fv.PAD, fv.PAD:-fv.PAD] += h * tend
        fv._repad(bp)
    return bp


def velocity_at(U0, V0, U1, V1, w, vel):
    """The staggered velocity at fraction ``w`` of the hour: linear in time,
    or the hour's midpoint held constant."""
    if vel == 'piecewise':
        return 0.5 * (U0 + U1), 0.5 * (V0 + V1)
    if vel == 'linear':
        return (1.0 - w) * U0 + w * U1, (1.0 - w) * V0 + w * V1
    raise ValueError(f"vel must be one of {VELS}, got {vel!r}")


class Tile:
    """The static things: grid, land fill map, the clamp mask."""

    def __init__(self, grid_ds):
        self.grid_ds = grid_ds
        self.ocean = mk.ocean_mask(grid_ds)
        self.interior = mk.edge_mask(grid_ds, CLAMP_CELLS)
        self.margin = self.ocean & ~self.interior                 # the clamped cells
        idx = ndimage.distance_transform_edt(~self.ocean, return_distances=False, return_indices=True)
        self.fill_j, self.fill_i = idx[0][~self.ocean], idx[1][~self.ocean]

    def fill(self, a):
        """Land cells <- the nearest ocean value (in place on an unpadded array)."""
        a[~self.ocean] = a[self.fill_j, self.fill_i]
        return a


def advect_hour(bp_list, tile, U0, V0, U1, V1, vel, model0=None, model1=None, clamp=True,
                n_sub=N_SUB, dt=DT):
    """Advance the padded tracers one hour in place.  ``model0``/``model1``:
    the model's own fields at the two hours (list aligned with ``bp_list``)
    for the clamp."""
    h = dt / n_sub
    # land <- the nearest ocean value ONCE per hour, as fvadvect.fv_advect does (land cells
    # have r_inv = 0 and never change within the hour; refilling per sub-step would break
    # check 1's bit-identity with V3b's step)
    for bp in bp_list:
        tile.fill(bp[fv.PAD:-fv.PAD, fv.PAD:-fv.PAD])
        fv._repad(bp)
    if vel == 'piecewise':
        T = transports(tile.grid_ds, *velocity_at(U0, V0, U1, V1, 0.5, vel))
    for s in range(n_sub):
        if vel == 'linear':
            T = transports(tile.grid_ds, *velocity_at(U0, V0, U1, V1, (s + 0.5) / n_sub, vel))
        w1 = (s + 1) / n_sub
        for k, bp in enumerate(bp_list):
            substep(bp, T, h, s)
            if clamp and model0 is not None:
                inner = bp[fv.PAD:-fv.PAD, fv.PAD:-fv.PAD]
                m = (1.0 - w1) * model0[k] + w1 * model1[k]
                inner[tile.margin] = m[tile.margin]
                fv._repad(bp)
    return bp_list


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def _raw_hour(raw, t):
    return raw.isel(time=[int(t)]).load()


def hour_dataset(raw, t, tracer: dict) -> xr.Dataset:
    """The raw store's hour ``t`` with ``Theta``/``Salt`` replaced by the tracer
    (float32, land NaN)."""
    ds = _raw_hour(raw, t)
    for name, a in tracer.items():
        da = ds[name]
        ds[name] = da.copy(data=np.asarray(a, 'float32')[None])
        ds[name].attrs.update(da.attrs, tracer='offline passive tracer (m3_tracer.py), NOT the model\'s '
                                                 f'{name}; initial = the model at hour 0')
    return ds


def store_attrs(raw, vel, hours_done, clamp=True) -> dict:
    a = dict(raw.attrs)
    a.update(DECLARED['design'], velocity_column=vel, clamp_applied=bool(clamp),
             tracer='Theta and Salt are OFFLINE PASSIVE TRACERS advected from the model\'s hour 0 by '
                    'fvadvect (OS7MP); U V W Eta KPPhbl oceTAUX oceTAUY copied from the raw store',
             source_raw=str(inp.RAW_ZARR), hours_done=int(hours_done),
             true_budget=DECLARED['true_budget'], **ot._provenance())
    return a


def _model_fields(raw, t):
    h = raw.isel(time=int(t)).load()
    return [np.asarray(h[k].values, 'float64') for k in TRACERS]


def _uv(raw, t):
    h = raw.isel(time=int(t))
    return h['U'].load(), h['V'].load()


def run_column(vel, hours=None, clamp=True, out=None, say=print, dry=False) -> dict:
    """Advect from hour 0 (or resume from the store's last hour) through the
    window, appending each hour.  Returns the done-dict."""
    out = store_path(vel) if out is None else Path(out)
    raw = xr.open_zarr(inp.RAW_ZARR)
    n_hours = raw.sizes['time']
    hours = list(range(n_hours)) if hours is None else list(hours)
    grid_ds = ot.open_grid(inp.GRID_ZARR, with_face=True)
    tile = Tile(grid_ds)
    zs.repair_trailing(out, log=None)
    present = zs.present_times(out)
    want = raw['time'].values.astype('datetime64[s]')
    have = {int(k) for k, w in enumerate(want) if w in set(present.tolist())}
    todo = [t for t in hours if t not in have]
    say(f'{vel}: {len(have)} hour(s) on disk, {len(todo)} to compute -> {out.name}; clamp {clamp}')
    if dry or not todo:
        return dict(status='nothing to do', present=sorted(have), store=str(out))
    # the state: hour 0 from the model, or the store's last hour
    t_state = min(todo)
    if t_state == 0:
        state = _model_fields(raw, 0)
    else:
        if t_state - 1 not in have:
            raise RuntimeError(f'cannot resume at hour {t_state}: hour {t_state - 1} is not in the store')
        prev = xr.open_zarr(out).isel(time=t_state - 1).load()
        state = [np.asarray(prev[k].values, 'float64') for k in TRACERS]
    bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(a), a, np.nan))) for a in state]
    for k, a in enumerate(bp):
        if not np.all(np.isfinite(a)):
            raise RuntimeError(f'{TRACERS[k]}: non-finite values after the land fill')
    enc = None
    walls = []
    t_start = time.time()
    if t_state == 0 and 0 in todo:
        ds0 = hour_dataset(raw, 0, {k: np.where(tile.ocean, a, np.nan) for k, a in zip(TRACERS, state)})
        enc = ot._clean_encoding(ds0, time_chunk=True)
        zs.append_hour(out, ds0, encoding=enc, attrs=store_attrs(raw, vel, 1, clamp))
        say(f'{vel}: hour 0 written (the model\'s own fields)')
        todo = [t for t in todo if t != 0]
        t_state = 1
    for t in todo:
        tw = time.time()
        if t != t_state:
            raise RuntimeError(f'hours must be contiguous: at {t}, expected {t_state}')
        U0, V0 = _uv(raw, t - 1)
        U1, V1 = _uv(raw, t)
        m0, m1 = (_model_fields(raw, t - 1), _model_fields(raw, t)) if clamp else (None, None)
        advect_hour(bp, tile, U0, V0, U1, V1, vel, m0, m1, clamp=clamp)
        fields = {k: np.where(tile.ocean, a[fv.PAD:-fv.PAD, fv.PAD:-fv.PAD], np.nan)
                  for k, a in zip(TRACERS, bp)}
        ds_t = hour_dataset(raw, t, fields)
        if enc is None and not out.exists():
            enc = ot._clean_encoding(ds_t, time_chunk=True)
        zs.append_hour(out, ds_t, encoding=enc, attrs=store_attrs(raw, vel, t + 1, clamp))
        walls.append(time.time() - tw)
        mean_T = float(np.nanmean(fields['Theta'][tile.interior]))
        say(f'{vel}: hour {t:2d} {walls[-1]:5.1f} s  Theta mean {mean_T:.4f} '
            f'sd {float(np.nanstd(fields["Theta"][tile.interior])):.4f}', pair=True)
        t_state = t + 1
    zs.consolidate(out)
    done = dict(status='ok', vel=vel, store=str(out), hours=[hours[0], hours[-1]], n_computed=len(todo),
                wall_s=round(time.time() - t_start, 1),
                wall_per_hour=dict(median=float(np.median(walls)), max=float(np.max(walls))) if walls else None,
                clamp=bool(clamp), pid=os.getpid(), host=socket.gethostname(),
                finished=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    return done


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def check_1(say=print) -> dict:
    """Hour 0 -> 1, piecewise velocity, no clamp, on both tracers, against
    ``fvadvect.fv_advect`` on the midpoint velocity: bit for bit."""
    raw = xr.open_zarr(inp.RAW_ZARR)
    grid_ds = ot.open_grid(inp.GRID_ZARR, with_face=True)
    tile = Tile(grid_ds)
    U0, V0 = _uv(raw, 0)
    U1, V1 = _uv(raw, 1)
    state = _model_fields(raw, 0)
    bp = [fv._pad(tile.fill(np.where(tile.ocean & np.isfinite(a), a, np.nan))) for a in state]
    advect_hour(bp, tile, U0, V0, U1, V1, 'piecewise', clamp=False)
    out = {}
    for k, name in enumerate(TRACERS):
        mine = np.where(tile.ocean, bp[k][fv.PAD:-fv.PAD, fv.PAD:-fv.PAD], np.nan)
        b = raw[name].isel(time=0).load().expand_dims('face').astype('float64')
        ref = np.squeeze(fv.fv_advect(b, 0.5 * (U0 + U1), 0.5 * (V0 + V1), grid_ds, dt=DT).values)
        ok = np.isfinite(mine) & np.isfinite(ref)
        out[name] = dict(bit_identical=bool(np.array_equal(mine, ref, equal_nan=True)),
                         max_abs_diff=float(np.max(np.abs(mine[ok] - ref[ok]))),
                         n_finite=int(ok.sum()), n_finite_ref=int(np.isfinite(ref).sum()))
        say(f'check 1 {name}: bit-identical {out[name]["bit_identical"]}, max |diff| {out[name]["max_abs_diff"]:.2e}')
    out['passed'] = bool(all(out[n]['bit_identical'] for n in TRACERS))
    return out


def _G_of(h, grid_ds, grid):
    b = op.buoyancy(h.expand_dims('face').astype('float64'))
    return np.squeeze(np.asarray(op.gradb2(b, grid_ds, grid).values))


def checks_2_to_4(vel, say=print) -> dict:
    out_path = store_path(vel)
    raw = xr.open_zarr(inp.RAW_ZARR)
    tr = xr.open_zarr(out_path)
    grid_ds = ot.open_grid(inp.GRID_ZARR, with_face=True)
    grid = ot.build_xgcm(grid_ds)
    masks = mk.open_masks(inp.MASKS_NC)
    ana = np.asarray(masks['mask_analysis'].values, bool)
    tile = Tile(grid_ds)
    # check 2: conservation
    cons = {k: dict(mean=[], var=[]) for k in TRACERS + ('b',)}
    cons['margin_inflow_proxy'] = []
    for t in range(tr.sizes['time']):
        h = tr.isel(time=t).load()
        b = np.squeeze(np.asarray(op.buoyancy(h.expand_dims('face').astype('float64')).values))
        for k in TRACERS:
            a = np.asarray(h[k].values, 'float64')[ana]
            cons[k]['mean'].append(float(np.nanmean(a)))
            cons[k]['var'].append(float(np.nanvar(a)))
        cons['b']['mean'].append(float(np.nanmean(b[ana])))
        cons['b']['var'].append(float(np.nanvar(b[ana])))
        # what the clamp lets in: |tracer - model| on the first interior ring is a proxy
        m = np.asarray(raw['Theta'].isel(time=t).values, 'float64')
        ring = tile.interior & ~mk.edge_mask(grid_ds, CLAMP_CELLS + 1)
        cons['margin_inflow_proxy'].append(float(np.nanmean(np.abs(np.asarray(h['Theta'].values, 'float64') - m)[ring])))
    for k in TRACERS + ('b',):
        m0, mN = cons[k]['mean'][0], cons[k]['mean'][-1]
        v0, vN = cons[k]['var'][0], cons[k]['var'][-1]
        cons[k]['mean_drift'] = mN - m0
        cons[k]['mean_drift_over_sd0'] = (mN - m0) / np.sqrt(v0)
        cons[k]['variance_factor_72h'] = vN / v0
    # check 3: the front population
    pop = {}
    for t in CHECK_HOURS:
        if t >= tr.sizes['time']:
            continue
        Gt = _G_of(tr.isel(time=t).load(), grid_ds, grid)
        Gm = _G_of(raw.isel(time=t).load(), grid_ds, grid)
        okt, okm = ana & np.isfinite(Gt), ana & np.isfinite(Gm)
        pt = {p: float(np.percentile(Gt[okt], p)) for p in (50, 90, 99)}
        pm = {p: float(np.percentile(Gm[okm], p)) for p in (50, 90, 99)}
        pop[str(t)] = dict(tracer=pt, model=pm, ratio={p: pt[p] / pm[p] for p in pt},
                           corr=float(np.corrcoef(np.log(Gt[okt & okm] + 1e-30), np.log(Gm[okt & okm] + 1e-30))[0, 1]))
        say(f'check 3 hour {t}: G p90 tracer/model {pop[str(t)]["ratio"][90]:.3f}, p99 {pop[str(t)]["ratio"][99]:.3f}, '
            f'log-G corr {pop[str(t)]["corr"]:.3f}')
    # check 4: the store's invariants
    import series_verify as sv
    ver = sv.verify_series(out_path, timestamps_72()[:tr.sizes['time']], grid_ds=ot.open_grid(inp.GRID_ZARR, with_face=False))
    try:
        ds, _, _, _ = inp.open_inputs(out_path, None, inp.GRID_ZARR, inp.MASKS_NC)
        opens = dict(ok=True, chunk=ds.attrs.get('chunk'), face=ds.attrs.get('face'))
    except Exception as e:                                       # noqa: BLE001 -- recorded
        opens = dict(ok=False, error=f'{type(e).__name__}: {e}')
    say(f'check 4: verify_series ok {ver.get("ok")}; open_inputs(chunk=None) ok {opens["ok"]}')
    return dict(conservation=cons, front_population=pop, verify_series=dict(ok=bool(ver.get('ok')),
                checks={k: (v.get('ok') if isinstance(v, dict) else v) for k, v in ver.items() if k != 'ok'}),
                open_inputs=opens, n_hours=int(tr.sizes['time']))


# ---------------------------------------------------------------------------
# the reading, from the closure summaries
# ---------------------------------------------------------------------------
def reading_L0(r) -> str:
    hi, lo = DECLARED['reading']['thresholds']['hi'], DECLARED['reading']['thresholds']['lo']
    if r >= hi:
        return 'analysis-limited at the grid scale'
    if r <= lo:
        return 'data-limited (the model\'s sub-hourly evolution)'
    return 'both'


def reading_L8(r) -> str:
    return ('not the pipeline' if r <= DECLARED['reading']['thresholds']['lo']
            else 'the pipeline contributes at the filtered scales')


def _closure_numbers(S, L):
    c = S['per_L'][str(L)]['closure']
    return dict(resid_over_meas=c['residual']['over_measured'], explained=c['residual']['explained'],
                resid_on_2F=c['residual']['slope_on_2F'], catchall_over_meas=c['catchall']['over_measured'],
                euler_gap=c['measured_coherence']['rms_euler_minus_semilag'],
                o5_gap=c['measured_coherence']['rms_o5_minus_o3'],
                meas_on_2F=c['measured_on_2F'], n=c['n'],
                subfilter_over_2F=S['per_L'][str(L)]['subfilter']['over_two_F'],
                two_F_over_meas=c['terms']['two_F']['over_measured'],
                euler=S['per_L'][str(L)]['euler'], verdict=c['verdict']['passed'],
                slopes_quotable=S['per_L'][str(L)]['slopes_quotable'],
                slope_primary=(S['per_L'][str(L)].get('slopes') or S['per_L'][str(L)].get('slopes_not_quoted'))['primary'])


def report(say=print) -> dict:
    prod = json.loads((DATA_DIR / 'm3_closure_summary.json').read_text())
    out = dict(declared=DECLARED, created=time.strftime('%Y-%m-%dT%H:%M:%S'), columns={}, product={})
    Ls = [int(L) for L in prod['per_L']]
    for L in Ls:
        out['product'][str(L)] = _closure_numbers(prod, L)
    for vel in VELS:
        p = DATA_DIR / f'm3_closure_{TAG[vel]}_summary.json'
        if not p.exists():
            say(f'{vel}: no closure summary at {p}; skipped')
            continue
        S = json.loads(p.read_text())
        col = dict(tag=TAG[vel], per_L={str(L): _closure_numbers(S, L) for L in Ls if str(L) in S['per_L']},
                   verdict=S['verdict'])
        sp = summary_path(vel)
        if sp.exists():
            col['checks'] = json.loads(sp.read_text())
        r0, r8 = col['per_L']['0']['resid_over_meas'], col['per_L']['8']['resid_over_meas']
        col['reading'] = dict(L0=dict(r_tr=r0, reading=reading_L0(r0)),
                              L8=dict(r_tr=r8, product_catchall=out['product']['8']['catchall_over_meas'],
                                      reading=reading_L8(r8)),
                              slope_label='tracer, not the model: the pipeline\'s calibration',
                              v3b=[0.975, 0.954, 1.003])
        out['columns'][vel] = col
        say(f'{vel}: r_tr {r0:.3f} / {col["per_L"]["2"]["resid_over_meas"]:.3f} / '
            f'{col["per_L"]["4"]["resid_over_meas"]:.3f} / {r8:.3f} -> L0 {col["reading"]["L0"]["reading"]}; '
            f'L8 {col["reading"]["L8"]["reading"]}')
    if 'linear' in out['columns'] and 'piecewise' in out['columns']:
        out['interpolation_effect'] = {str(L): dict(
            linear=out['columns']['linear']['per_L'][str(L)]['resid_over_meas'],
            piecewise=out['columns']['piecewise']['per_L'][str(L)]['resid_over_meas'],
            difference=out['columns']['linear']['per_L'][str(L)]['resid_over_meas']
            - out['columns']['piecewise']['per_L'][str(L)]['resid_over_meas']) for L in Ls}
    out['verdict_note'] = ('the tracer\'s closure is the pipeline\'s floor; the M3 verdict on the model '
                           '(criterion 1 failed at every L) is unchanged')
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    say(f'wrote {OUT_JSON}')
    from m3_tracer_fig import figure
    figure(out, OUT_PNG)
    say(f'wrote {OUT_PNG}')
    return out


def format_report(res) -> str:
    w = '=' * 100
    out = [w, 'M3 DIGGING TASK 4 -- THE OFFLINE TRACER (residual / measured on front & valid; the tracer\'s '
              'residual is the catch-all)', w,
           f'  {"L":>3} {"product 5-term":>15} {"product catch-all":>18} {"tracer linear":>14} '
           f'{"tracer piecewise":>17} {"euler gap: prod / lin / pc":>28} {"expl lin":>9}']
    for L, p in res['product'].items():
        lin = res['columns'].get('linear', {}).get('per_L', {}).get(L)
        pc = res['columns'].get('piecewise', {}).get('per_L', {}).get(L)
        f = (lambda d, k: f'{d[k]:.3f}' if d else 'n/a')
        out.append(f'  {L:>3} {p["resid_over_meas"]:15.3f} {p["catchall_over_meas"]:18.3f} {f(lin, "resid_over_meas"):>14} '
                   f'{f(pc, "resid_over_meas"):>17} {p["euler_gap"]:8.3f} / {f(lin, "euler_gap")} / {f(pc, "euler_gap")}'
                   f' {f(lin, "explained"):>9}')
    for vel, col in res['columns'].items():
        r = col['reading']
        out.append(f'  {vel}: L0 r_tr {r["L0"]["r_tr"]:.3f} -> {r["L0"]["reading"]}; L8 r_tr {r["L8"]["r_tr"]:.3f} '
                   f'(product catch-all {r["L8"]["product_catchall"]:.3f}) -> {r["L8"]["reading"]}; '
                   f'criterion 1 {col["verdict"]["criterion1"]["per_L"]}; slopes quotable (as calibration) '
                   f'{[col["per_L"][L]["slopes_quotable"] for L in col["per_L"]]}')
    out += ['', '  ' + res.get('verdict_note', ''), w]
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--vel', choices=VELS, default='linear')
    ap.add_argument('--hours', default=None, help='a:b slice of hours (a trial run)')
    ap.add_argument('--no-clamp', action='store_true', help='no edge clamp (check 1 and tests only)')
    ap.add_argument('--out', default=None, help='the store (default data/tile330_tracer_{vel}_...)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check1', action='store_true', help='run check 1 only')
    ap.add_argument('--checks', action='store_true', help='run checks 2-4 on the finished store only')
    ap.add_argument('--report', action='store_true', help='apply the declared reading to the closure '
                                                          'summaries and draw the figure')
    args = ap.parse_args(argv)
    if args.report:
        res = report()
        print(format_report(res))
        return 0
    if args.check1:
        c = check_1()
        p = summary_path(args.vel)
        d = json.loads(p.read_text()) if p.exists() else dict(declared=DECLARED, vel=args.vel)
        d['check_1'] = c
        p.write_text(json.dumps(d, indent=1, default=str))
        print(f'check 1 passed: {c["passed"]} -> {p}')
        return 0 if c['passed'] else 1
    hours = None
    if args.hours:
        a, _, b = args.hours.partition(':')
        hours = list(range(int(a or 0), int(b or 72)))
    walls = []
    say = Progress(log_path(args.vel), walls, len(hours) if hours else 72)
    say(f'===== m3_tracer start ({args.vel}, pid {os.getpid()}, host {socket.gethostname()}) =====')
    say('declared: ' + json.dumps(DECLARED['design']))
    p = summary_path(args.vel)
    if not p.exists():
        p.write_text(json.dumps(dict(declared=DECLARED, vel=args.vel,
                                     status='declared; the run has started'), indent=1))
    t0 = time.time()
    status = 'error'
    try:
        if not args.checks:
            done = run_column(args.vel, hours=hours, clamp=not args.no_clamp, out=args.out, say=say,
                              dry=args.dry_run)
            say(f'advection: {done}')
        if not args.dry_run and args.out is None:
            d = json.loads(p.read_text())
            d.update(declared=DECLARED, vel=args.vel, status='run complete; checks follow')
            d['checks_2_to_4'] = checks_2_to_4(args.vel, say=say)
            d['check_1'] = d.get('check_1') or check_1(say=say)
            d['passed'] = bool(d['check_1']['passed'] and d['checks_2_to_4']['verify_series']['ok']
                               and d['checks_2_to_4']['open_inputs']['ok'])
            d['status'] = 'complete'
            d['wall_s'] = round(time.time() - t0, 1)
            p.write_text(json.dumps(d, indent=1, default=str))
            say(f'wrote {p}: checks passed {d["passed"]}')
        status = 'ok'
    except BaseException as e:                                   # noqa: BLE001 -- recorded
        import traceback
        say('EXCEPTION: ' + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        status = 'error'
    done_path(args.vel).write_text(json.dumps(dict(status=status, vel=args.vel, wall_s=round(time.time() - t0, 1),
                                                   finished=datetime.now(timezone.utc).isoformat(timespec='seconds')),
                                              indent=1))
    say(f'===== m3_tracer end: {status}, {time.time() - t0:.0f} s =====')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
