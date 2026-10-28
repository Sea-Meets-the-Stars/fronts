"""Follow one strong front through the window — the M3 interim report's case study.

The field-level gate (``m3_closure.py``) pools 1.87 M front pixels over 71 hourly pairs and
reports one number per filter scale.  That is the right way to judge closure and the wrong way
to *show* it: a reader cannot see a front in it.  This module takes a single, strong,
persistent front, **follows it along the flow** for a day, and asks the budget's question of
that one feature — does its strength change the way the frontogenesis tendency says it should?

**No front finding.**  Prompt 4's "Do not" list reserves ``tile_find``, labelling, thinning and
despurring for M4.  Nothing here labels an object: the case is a fixed box chosen by the
persistence of *front pixels* (the §3.4 percentile, unchanged), and the parcels followed inside
it are the pixels of a narrow corridor across one filament whose ``G`` is at or above the
analysis mask's 90th percentile at the seed time -- §3.4's own rule, applied inside a box drawn
by hand.  What is followed is a set of water parcels, not a front object.

**The trajectory is the budget's own.**  Parcels live on the midpoint-time grid (the pair-``t``
midpoint is ``t + 0.5`` h) and are stepped by the model's velocity at the integer hour between
two midpoints, through the same midpoint iteration and the same cubic interpolation as
``semilag.departure_index`` — forward instead of backward.  Two checks keep that honest and are
written into the summary:

* ``check_trajectory`` — the integrator alone: start a parcel at the departure point
  ``semilag.departure_index`` gives for a grid point, step it forward one hour, and it must
  return to that grid point.  This is the forward form inverting the backward one and it tests
  no physics, only the index conventions;
* ``check_rate`` — the case's own Lagrangian rate ``[G(x_{n+1}) - G(x_n)]/dt`` against the
  budget's stored ``DGDt_semilag`` averaged over the two pairs.  The two are estimates of the
  same rate on time bases offset by half an hour (the store differences the hour's *endpoints*,
  the case differences consecutive *midpoints*), so they should correlate tightly and differ by
  some tens of per cent in r.m.s. — and if they do not, the sampling is wrong.

Output: ``data/m3_case_summary.json`` (every number the report quotes) and
``data/m3_case_fields.npz`` (the small box arrays the figures draw, so that
``report/make_case_figs.py`` opens no store).  Nothing in the derived stores is modified.

Run:  <frontogenesis env python> py/m3_case.py          (from dev/frontogenesis)
"""
import json
import pathlib
import sys

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import inputs as inp                                                # noqa: E402
import semilag as sl                                                # noqa: E402

DERIVED = ROOT / 'data' / 'tile330_derived_L{L}.zarr'
OUT_JSON = ROOT / 'data' / 'm3_case_summary.json'
OUT_NPZ = ROOT / 'data' / 'm3_case_fields.npz'

DT = 3600.0
# The case box.  Chosen from the persistence of front pixels over all 71 pairs (the scan is in
# the log entry): the upwelling filament at ~36.5-37.5 N, ~124-125 W, 140-200 km offshore --
# the strongest sustained front on the tile inside the analysis mask, and the one the day-3
# pairs cut across (M2-Q7's j = 176 transect).
BOX = dict(j0=140, j1=236, i0=20, i1=176)          # what the maps show
# The front itself.  ``i`` runs north -> south on this face, so a near-zonal filament is a
# narrow band in ``i``: at the seed time the median G along ``i = 78`` (37.08 N) is 7.7e-14,
# an order of magnitude above its neighbours two cells away.  Seeding inside this corridor
# follows ONE filament; seeding on the whole display box follows every front in it.
SEED_BOX = dict(j0=150, j1=220, i0=74, i1=85)
T0 = 30              # the seed pair, at the centre of a symmetric window
NB = 24              # hours followed backwards from the seed
NF = 24              # hours followed forwards from the seed
N_CONTROL = 2500     # unselected parcels, for the regression-to-the-mean control
SEED_PCT = None      # seeds are the box's front pixels (G >= p90) -- §3.4's own rule
L_CASE = 0           # the grid scale: where the budget is hardest and the case is sharpest
TERMS = ('two_F', 'subfilter', 'vertical', 'surface_flux', 'residual')


# ---------------------------------------------------------------------------
# sampling at parcel positions
# ---------------------------------------------------------------------------
def sample(field, pos, order=3):
    """``field`` (2-D, positional ``(j, i)``) at fractional positions ``pos`` ``(n, 2)``
    ``(j, i)``, cubic, NaN outside.  NaNs in the support propagate, which is what we want:
    a parcel that reaches land or the tile edge drops out rather than being filled in."""
    j, i = pos[:, 0], pos[:, 1]
    out = map_coordinates(np.nan_to_num(field, nan=0.0), [j, i], order=order,
                          mode='constant', cval=np.nan)
    bad = map_coordinates((~np.isfinite(field)).astype('f8'), [j, i], order=1,
                          mode='constant', cval=1.0)
    out[bad > 1e-6] = np.nan
    inside = ((j >= 0) & (j <= field.shape[0] - 1) & (i >= 0) & (i <= field.shape[1] - 1))
    out[~inside] = np.nan
    return out


def index_velocity(U, V, grid_ds, grid, dt=DT):
    """The model's velocity at one snapshot as a displacement per ``dt`` in cells,
    ``(ui, vj)`` positional ``(j, i)`` — ``semilag.departure_index``'s own first step."""
    u_c, v_c = sl.centre_velocities(U, V, grid_ds, grid)
    dx_c, dy_c = sl._spacing_at_centres(grid_ds)
    ui = np.squeeze(np.asarray(u_c.values)) * dt / dx_c
    vj = np.squeeze(np.asarray(v_c.values)) * dt / dy_c
    return ui, vj


def step_forward(pos, ui, vj, n_iter=3, vel_order=3):
    """One forward hour by the midpoint rule: ``d = u(x + d/2)``, ``x <- x + d``.  The
    backward form of exactly this is ``semilag.departure_index``."""
    d = np.stack([sample(vj, pos, vel_order), sample(ui, pos, vel_order)], axis=1)
    for _ in range(int(n_iter)):
        half = pos + 0.5 * d
        d = np.stack([sample(vj, half, vel_order), sample(ui, half, vel_order)], axis=1)
    return pos + d, d


# ---------------------------------------------------------------------------
# the case
# ---------------------------------------------------------------------------
def seed_parcels(front_box, box):
    """Parcel seeds: every **front pixel** of the box at the seed time — ``G`` at or above its
    90th percentile over the whole analysis mask, which is §3.4's definition unchanged.  Seeding
    on a far higher percentile instead (the box's own top 1 %) picks the instantaneous peak and
    then follows its regression to the mean, which is a property of the selection and not of
    the front."""
    jj, ii = np.where(front_box)
    return np.stack([jj + box['j0'], ii + box['i0']], axis=1).astype('f8')


def _trajectory(ds, grid_ds, grid, seeds, t0, n_back, n_fwd):
    """Positions of ``seeds`` at the midpoint time of every pair from ``t0 - n_back`` to
    ``t0 + n_fwd``.  Stepping from pair ``t``'s midpoint to pair ``t+1``'s uses the model's
    velocity at integer hour ``t+1``, the instant exactly between them; the backward step from
    pair ``t`` to ``t-1`` uses hour ``t``, negated.  ``departure_index`` is the same rule."""
    n = n_back + n_fwd + 1
    pos = np.empty((n, len(seeds), 2))
    pos[n_back] = seeds
    disp = np.zeros((n - 1, len(seeds), 2))
    for m in range(n_fwd):                                   # forwards
        ui, vj = index_velocity(*_uv(ds, t0 + m + 1), grid_ds, grid)
        pos[n_back + m + 1], disp[n_back + m] = step_forward(pos[n_back + m], ui, vj)
    for m in range(n_back):                                  # backwards
        ui, vj = index_velocity(*_uv(ds, t0 - m), grid_ds, grid)
        pos[n_back - m - 1], d = step_forward(pos[n_back - m], -ui, -vj)
        disp[n_back - m - 1] = -d
    return pos, disp


def _uv(ds, t):
    h = ds.isel(time=int(t)).load()
    return h['U'], h['V']


def run(t0=T0, n_back=NB, n_fwd=NF, box=BOX, seed_box=SEED_BOX, L=L_CASE,
        n_control=N_CONTROL, seed=0):
    """Follow the box's front pixels through a window centred on pair ``t0`` and write the
    summary and the figure arrays.

    Time base: everything lives on the **midpoint-time** grid, where pair ``t``'s fields sit at
    ``t + 0.5`` h.  A parcel is carried from one midpoint to the next by the model's velocity at
    the integer hour between them, so the step is centred.  The rate over an interval is
    differenced from the two midpoints it joins; the prediction over the same interval is the
    mean of the two pairs' terms, centred on the same instant.

    **The window is symmetric about the seed time and there is an unselected control**, because
    following the pixels that are strongest *now* makes them weaken afterwards whatever the
    physics does.  Regression to the mean is symmetric in time and acts on any selected set;
    frontolysis is neither.  Both diagnostics are in the summary and in the figure.
    """
    der = xr.open_zarr(str(DERIVED).format(L=L))
    sj, si = slice(box['j0'], box['j1']), slice(box['i0'], box['i1'])
    ds, grid_ds, grid, masks = inp.open_inputs()
    dt, n_t = DT, n_back + n_fwd + 1
    pairs = np.arange(t0 - n_back, t0 + n_fwd + 1)

    G_all = der['G'].values                                   # midpoint G, every pair
    sbj, sbi = slice(seed_box['j0'], seed_box['j1']), slice(seed_box['i0'], seed_box['i1'])
    seeds = seed_parcels(der['front'].isel(time=t0).values[sbj, sbi], seed_box)
    valid_box = der['valid'].isel(time=t0).values[sj, si]
    jj, ii = np.where(valid_box)
    pick = np.random.default_rng(seed).choice(len(jj), size=min(n_control, len(jj)),
                                              replace=False)
    ctrl_seeds = np.stack([jj[pick] + box['j0'], ii[pick] + box['i0']], axis=1).astype('f8')

    pos, disp = _trajectory(ds, grid_ds, grid, seeds, t0, n_back, n_fwd)
    cpos, _ = _trajectory(ds, grid_ds, grid, ctrl_seeds, t0, n_back, n_fwd)

    # --- sample the budget along both trajectories -------------------------
    keys = ('G', 'stored_rate') + TERMS
    series = {k: np.full((n_t, len(seeds)), np.nan) for k in keys}
    ctrl_G = np.full((n_t, len(ctrl_seeds)), np.nan)
    for n, t in enumerate(pairs):
        series['G'][n] = sample(G_all[t], pos[n])
        ctrl_G[n] = sample(G_all[t], cpos[n])
        series['stored_rate'][n] = sample(der['DGDt_semilag'].isel(time=t).values, pos[n])
        for k in TERMS:
            series[k][n] = sample(der[k].isel(time=t).values, pos[n])

    alive = np.isfinite(series['G']).all(axis=0)
    for k in series:
        series[k] = series[k][:, alive]
    pos, disp = pos[:, alive], disp[:, alive]
    c_alive = np.isfinite(ctrl_G).all(axis=0)
    ctrl_G = ctrl_G[:, c_alive]

    # --- rates over each interval, centred on the integer hour --------------
    rate_meas = (series['G'][1:] - series['G'][:-1]) / dt     # the case's own, Lagrangian
    rate_store = 0.5 * (series['stored_rate'][:-1] + series['stored_rate'][1:])
    rate = {k: 0.5 * (series[k][:-1] + series[k][1:]) for k in TERMS}

    G_obs = np.nanmedian(series['G'], axis=1)
    G_ctrl = np.nanmedian(ctrl_G, axis=1)
    med_rate = {k: np.nanmedian(v, axis=1) for k, v in rate.items()}
    med_rate['measured'] = np.nanmedian(rate_store, axis=1)       # the budget's own column
    med_rate['differenced'] = np.nanmedian(rate_meas, axis=1)     # the cross-check only

    # --- the predicted curves, integrated from the seed time outwards -------
    def integrate(r):
        out = np.empty(n_t)
        out[n_back] = G_obs[n_back]
        for m in range(n_back, n_t - 1):                      # forwards from the seed
            out[m + 1] = out[m] + r[m] * dt
        for m in range(n_back, 0, -1):                        # backwards from the seed
            out[m - 1] = out[m] - r[m - 1] * dt
        return out

    G_from_2F = integrate(med_rate['two_F'])
    G_from_budget = integrate(sum(med_rate[k] for k in
                                  ('two_F', 'subfilter', 'vertical', 'surface_flux')))

    # --- checks -------------------------------------------------------------
    U, V = _uv(ds, t0 + 1)
    u_c, v_c = sl.centre_velocities(U, V, grid_ds, grid)
    di, dj = sl.departure_index(u_c, v_c, grid_ds, dt=dt)
    di_a, dj_a = (np.squeeze(np.asarray(x.values)) for x in (di, dj))
    gj, gi = np.mgrid[box['j0']:box['j1']:4, box['i0']:box['i1']:4]
    arr = np.stack([gj.ravel(), gi.ravel()], axis=1).astype('f8')
    dep = arr - np.stack([dj_a[gj.ravel(), gi.ravel()], di_a[gj.ravel(), gi.ravel()]], axis=1)
    ui, vj = index_velocity(U, V, grid_ds, grid)
    back, _ = step_forward(dep, ui, vj)
    err = np.hypot(*(back - arr).T)
    check_traj = dict(
        median_return_error_cells=float(np.nanmedian(err)),
        p99_return_error_cells=float(np.nanpercentile(err, 99)),
        n=int(np.isfinite(err).sum()),
        note=('a parcel started at the departure point semilag.departure_index gives for a grid '
              'point, stepped forward one hour by this module, must come back to that grid '
              'point; this tests the index conventions and no physics'))
    ok = np.isfinite(rate_meas) & np.isfinite(rate_store)
    check_rate = dict(
        corr=float(np.corrcoef(rate_meas[ok], rate_store[ok])[0, 1]),
        rms_ratio=float(np.sqrt(np.nanmean((rate_meas[ok] - rate_store[ok]) ** 2))
                        / np.sqrt(np.nanmean(rate_store[ok] ** 2))),
        median_abs_ratio=float(np.nanmedian(np.abs(rate_meas[ok]))
                               / np.nanmedian(np.abs(rate_store[ok]))),
        n=int(ok.sum()),
        note=('two estimates of the same Lagrangian rate on time bases half an hour apart: the '
              'store differences the hour endpoints, the case differences consecutive midpoint '
              'fields.  The aggregate gap between two such estimates is 0.672 of measured at '
              'L = 0 (the sampling limit), so this is the same number, not a new error'))

    # --- regression to the mean, measured rather than assumed ---------------
    sym = dict(
        G_before=float(G_obs[0]), G_seed=float(G_obs[n_back]), G_after=float(G_obs[-1]),
        fall_before_over_after=float((G_obs[n_back] / G_obs[0])
                                     / (G_obs[n_back] / G_obs[-1])),
        control_G_seed=float(G_ctrl[n_back]), control_G_after=float(G_ctrl[-1]),
        control_change_factor=float(G_ctrl[-1] / G_ctrl[n_back]),
        selected_change_factor=float(G_obs[-1] / G_obs[n_back]),
        n_control=int(c_alive.sum()),
        note=('the selected parcels are at a maximum of G at the seed time by construction and '
              'fall away on BOTH sides of it; the control is an unselected sample of the same '
              'box carried by the same flow.  The ratio of the two change factors is what the '
              'selection does not explain'))

    # --- the budget of this one front ---------------------------------------
    def rms(a):
        return float(np.sqrt(np.nanmean(np.asarray(a, dtype='f8') ** 2)))

    r_meas = rms(rate_store)            # normalise by the budget's measured side, as the
    front_terms = {k: dict(rms_over_measured=rms(rate[k]) / r_meas)   # aggregate gate does
                   for k in TERMS}
    disp_cells = np.hypot(disp[..., 0], disp[..., 1])
    f = slice(n_back, n_t - 1)                                # the forward half, for the rates
    dG = G_obs[-1] - G_obs[n_back]

    out = dict(
        created=str(np.datetime64('now')),
        source=dict(derived=str(DERIVED).format(L=L), raw='tile330_raw_20120702T00_72h.zarr'),
        case=dict(
            L=L, box=box, seed_box=seed_box, t0=int(t0), n_back=int(n_back), n_fwd=int(n_fwd),
            seed_rule=('front pixels (G >= p90 over the analysis mask) inside the seed '
                       'corridor at t0 -- one filament, not every front in the display box'),
            seed_lat=[float(der['YC'].values[sbj, sbi].min()),
                      float(der['YC'].values[sbj, sbi].max())],
            seed_lon=[float(der['XC'].values[sbj, sbi].min()),
                      float(der['XC'].values[sbj, sbi].max())],
            n_seeded=int(len(seeds)), n_followed=int(alive.sum()),
            lat=[float(np.nanmin(der['YC'].values[sj, si])),
                 float(np.nanmax(der['YC'].values[sj, si]))],
            lon=[float(np.nanmin(der['XC'].values[sj, si])),
                 float(np.nanmax(der['XC'].values[sj, si]))],
            coast_km=float(np.nanmedian(der['coast_distance_km'].values[sj, si])),
            t_seed=str(der['time'].values[t0])[:16],
            t_start=str(der['time'].values[pairs[0]])[:16],
            t_end=str(der['time'].values[pairs[-1]])[:16],
            note='front pixels by percentile only; no labelling, thinning or despurring (M4)'),
        trajectory=dict(
            displacement_cells=dict(
                median=float(np.nanmedian(disp_cells)),
                p90=float(np.nanpercentile(disp_cells, 90)),
                max=float(np.nanmax(disp_cells))),
            net_displacement_cells=float(np.nanmedian(np.hypot(*(pos[-1] - pos[0]).T))),
            net_displacement_km=float(np.nanmedian(np.hypot(*(pos[-1] - pos[0]).T)) * 1.9),
            spread_cells_start=float(np.nanmedian(np.std(pos[n_back], axis=0))),
            spread_cells_end=float(np.nanmedian(np.std(pos[-1], axis=0)))),
        strength=dict(
            G_seed=float(G_obs[n_back]), G_end=float(G_obs[-1]),
            G_max=float(np.nanmax(G_obs)), G_min=float(np.nanmin(G_obs)),
            G_max_hour=float((pairs - t0)[int(np.nanargmax(G_obs))]),
            change_factor=float(G_obs[-1] / G_obs[n_back]),
            G_from_2F_end=float(G_from_2F[-1]), G_from_budget_end=float(G_from_budget[-1]),
            predicted_over_observed_change=float((G_from_2F[-1] - G_obs[n_back]) / dG),
            budget_over_observed_change=float((G_from_budget[-1] - G_obs[n_back]) / dG)),
        rates=dict(
            corr_measured_2F=float(np.corrcoef(med_rate['measured'][f],
                                               med_rate['two_F'][f])[0, 1]),
            sign_agreement=float(np.mean(np.sign(med_rate['measured'][f])
                                         == np.sign(med_rate['two_F'][f]))),
            hours_measured_positive=int((med_rate['measured'][f] > 0).sum()),
            hours_2F_positive=int((med_rate['two_F'][f] > 0).sum()),
            n_intervals=int(n_fwd), rms_measured=r_meas,
            note='the measured side is the budget\'s own DGDt_semilag column, averaged over the two pairs that straddle each interval'),
        selection=sym,
        terms=front_terms,
        residual_over_measured=front_terms['residual']['rms_over_measured'],
        checks=dict(trajectory=check_traj, rate=check_rate),
    )
    OUT_JSON.write_text(json.dumps(out, indent=1))

    # --- the small arrays the figures draw ----------------------------------
    map_pairs = [t0, t0 + n_fwd // 3, t0 + 2 * n_fwd // 3, t0 + n_fwd]
    np.savez_compressed(
        OUT_NPZ,
        G_box=np.stack([G_all[t][sj, si] for t in map_pairs]).astype('f4'),
        b_box=np.stack([der['b'].isel(time=t).values[sj, si] for t in map_pairs]).astype('f4'),
        two_F_box=np.stack([der['two_F'].isel(time=t).values[sj, si]
                            for t in map_pairs]).astype('f4'),
        map_pairs=np.array(map_pairs), map_index=np.array([n_back + (t - t0) for t in map_pairs]),
        pos=pos.astype('f4'), box=np.array([box['j0'], box['j1'], box['i0'], box['i1']]),
        hours=(pairs - t0).astype('f8'), rate_hours=(pairs[:-1] - t0).astype('f8') + 0.5,
        G_obs=G_obs, G_ctrl=G_ctrl, G_from_2F=G_from_2F, G_from_budget=G_from_budget,
        G_q25=np.nanpercentile(series['G'], 25, axis=1),
        G_q75=np.nanpercentile(series['G'], 75, axis=1),
        med_measured=med_rate['measured'], med_differenced=med_rate['differenced'],
        med_two_F=med_rate['two_F'],
        med_residual=med_rate['residual'], med_subfilter=med_rate['subfilter'],
        med_vertical=med_rate['vertical'], med_surface_flux=med_rate['surface_flux'],
        term_rms=np.array([front_terms[k]['rms_over_measured'] for k in TERMS]),
        term_names=np.array(TERMS), n_back=np.array(n_back),
        XC=der['XC'].values[sj, si].astype('f4'), YC=der['YC'].values[sj, si].astype('f4'),
    )
    return out


if __name__ == '__main__':
    s = run()
    c, k, sel = s['case'], s['checks'], s['selection']
    print(f"box j{c['box']['j0']}-{c['box']['j1']} i{c['box']['i0']}-{c['box']['i1']}  seed "
          f"{c['t_seed']}  -{c['n_back']}/+{c['n_fwd']} h  {c['n_followed']}/{c['n_seeded']} parcels")
    print(f"check trajectory: median return error "
          f"{k['trajectory']['median_return_error_cells']:.2e} cells "
          f"(p99 {k['trajectory']['p99_return_error_cells']:.2e})")
    print(f"check rate: corr {k['rate']['corr']:.4f}  rms ratio {k['rate']['rms_ratio']:.3f}  "
          f"median |ratio| {k['rate']['median_abs_ratio']:.3f}")
    print(f"G seed {s['strength']['G_seed']:.3e} -> end {s['strength']['G_end']:.3e}  "
          f"x{s['strength']['change_factor']:.2f}   (before seed {sel['G_before']:.3e})")
    print(f"selection: selected x{sel['selected_change_factor']:.2f} vs control "
          f"x{sel['control_change_factor']:.2f}  (n_control {sel['n_control']})")
    print(f"predicted / observed change: 2F {s['strength']['predicted_over_observed_change']:.3f}"
          f"   full budget {s['strength']['budget_over_observed_change']:.3f}")
    print(f"rates: corr(meas, 2F) {s['rates']['corr_measured_2F']:.3f}  sign agreement "
          f"{s['rates']['sign_agreement']:.3f}  2F positive {s['rates']['hours_2F_positive']}"
          f"/{s['rates']['n_intervals']}  measured positive {s['rates']['hours_measured_positive']}")
    print('terms/measured: ' + '  '.join(f"{k2} {v['rms_over_measured']:.3f}"
                                         for k2, v in s['terms'].items()))
