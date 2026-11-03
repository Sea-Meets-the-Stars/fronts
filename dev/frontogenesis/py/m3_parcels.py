""" M3-digging task 6: following the parcels -- the velocity field at the seed and what
the followed parcels do relative to the front (JXP, 2026-11-02).

The §5.1 case study follows 450 parcels seeded on a front; by +16 h most are no longer on
the front features they started on.  This module draws the horizontal velocity at the seed
hour and quantifies the motion so the discussion can be about numbers:

* the velocity field at the seed hour (geographic components: on face 10 ``u_east = V``,
  ``v_north = -U``), over ``log10 G``, with the seed parcels;
* the parcels' speed split into **along-front** and **cross-front** components, the front
  normal being ``grad b / |grad b|`` at the parcel's position and time;
* the fraction of parcels that are on a front pixel (``G >= p90``; also ``p80``) hour by
  hour, and the distance to the nearest front pixel -- for the followed set and for the
  unselected control of the case (recomputed with the case's own seed);
* the box-mean velocity over the window as a hodograph, with least-squares amplitudes at
  the semidiurnal (12.42 h) and inertial (``2 pi / f``) periods, to show how much of the
  hourly velocity is oscillatory;
* the parcel path length against the net displacement (oscillation vs drift).

Reads the raw store (``U``, ``V``), the ``L = 0`` derived store (``G``, ``b``, ``front``,
``valid``, ``front_width``) and the case products; writes
``data/m3_parcels_summary.json``, ``figs/m3_parcels_velocity.png``,
``figs/m3_parcels_motion.png``.  Nothing in the stores or the case products is modified.

Run:  <frontogenesis env python> py/m3_parcels.py      (from dev/frontogenesis; ~1 min)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget as bg                                            # noqa: E402
import inputs as inp                                           # noqa: E402
import m3_case as case                                         # noqa: E402
import operators as op                                         # noqa: E402
import semilag as sl                                           # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
OUT_JSON = DATA_DIR / 'm3_parcels_summary.json'
OMEGA = 7.2921e-5
KM_PER_CELL = 1.873                                            # tile330_flow.zarr km_per_px


def geographic(u_c, v_c, grid_ds):
    """Model-basis centred velocities -> (u_east, v_north) by the grid's CS/SN."""
    cs = float(np.nanmedian(np.asarray(grid_ds['CS'].values)))
    sn = float(np.nanmedian(np.asarray(grid_ds['SN'].values)))
    u, v = (np.squeeze(np.asarray(a.values, 'float64')) for a in (u_c, v_c))
    return cs * u - sn * v, sn * u + cs * v


def load(t0, n_back, n_fwd):
    ds, grid_ds, grid, masks = inp.open_inputs()
    der = xr.open_zarr(bg.derived_path(0))
    pairs = np.arange(t0 - n_back, t0 + n_fwd + 1)
    hours = np.arange(t0 - n_back, t0 + n_fwd + 2)              # the velocities the steps use
    vel = {}
    for h in hours:
        U, V = case._uv(ds, h)
        u_c, v_c = sl.centre_velocities(U, V, grid_ds, grid)
        vel[int(h)] = dict(model=(np.squeeze(u_c.values), np.squeeze(v_c.values)),
                           geo=geographic(u_c, v_c, grid_ds))
    fields = {}
    for p in pairs:
        d = der.isel(time=int(p)).load()
        b = d['b'].expand_dims('face')
        b_x, b_y = op.grad_b(b, grid_ds, grid)
        fields[int(p)] = dict(G=np.asarray(d['G'].values, 'float64'),
                              front=np.asarray(d['front'].values, bool),
                              valid=np.asarray(d['valid'].values, bool),
                              b_x=np.squeeze(np.asarray(b_x.values, 'float64')),
                              b_y=np.squeeze(np.asarray(b_y.values, 'float64')),
                              width=np.asarray(d['front_width'].values, 'float64'),
                              p80=np.asarray(d['valid'].values, bool)
                              & (np.asarray(d['G'].values, 'float64')
                                 >= np.percentile(np.asarray(d['G'].values, 'float64')[np.asarray(d['valid'].values, bool)], 80)),
                              lst=float(inp.local_solar_hour(d['time_mid'].values,
                                                             float(der.attrs.get('utc_offset_h', inp.UTC_OFFSET_H)))))
    XC, YC = np.asarray(der['XC'].values), np.asarray(der['YC'].values)
    return ds, grid_ds, grid, der, vel, fields, XC, YC, pairs


def along_cross(u, v, b_x, b_y):
    """Cross-front (along grad b, signed towards denser water) and along-front
    speeds [m/s] from model-basis velocities and gradients."""
    g = np.hypot(b_x, b_y)
    with np.errstate(invalid='ignore', divide='ignore'):
        vn = (u * b_x + v * b_y) / g
        vt = (-u * b_y + v * b_x) / g
    return vn, vt


def sample_series(pos, pairs, vel, fields, t0):
    """Per time index n (pair pairs[n]): the parcels' speed, cross/along-front
    speeds, on-front flags, distance to the nearest front pixel."""
    n_t, n_p = pos.shape[:2]
    out = {k: np.full((n_t, n_p), np.nan) for k in
           ('speed', 'v_cross', 'v_along', 'G', 'dist_front', 'dist_p80', 'width')}
    out['on_front'] = np.zeros((n_t, n_p), bool)
    out['on_p80'] = np.zeros((n_t, n_p), bool)
    for n, p in enumerate(pairs):
        f = fields[int(p)]
        # the velocity the step from this midpoint uses is at hour p + 1 (m3_case._trajectory)
        u, v = vel[int(p) + 1]['model']
        us, vs = case.sample(u, pos[n], 1), case.sample(v, pos[n], 1)
        bx, by = case.sample(f['b_x'], pos[n], 1), case.sample(f['b_y'], pos[n], 1)
        out['speed'][n] = np.hypot(us, vs)
        vn, vt = along_cross(us, vs, bx, by)
        out['v_cross'][n], out['v_along'][n] = vn, vt
        out['G'][n] = case.sample(f['G'], pos[n], 1)
        out['width'][n] = case.sample(f['width'], pos[n], 1)
        jj, ii = np.clip(np.rint(pos[n, :, 0]).astype(int), 0, 719), np.clip(np.rint(pos[n, :, 1]).astype(int), 0, 719)
        out['on_front'][n] = f['front'][jj, ii]
        out['on_p80'][n] = f['p80'][jj, ii]
        for key, mask in (('dist_front', f['front']), ('dist_p80', f['p80'])):
            d = ndimage.distance_transform_edt(~mask)
            out[key][n] = case.sample(d, pos[n], 1)
    return out


def box_mean_series(vel, hours, box, fields_any):
    """The ocean-mean geographic velocity over the case box per hour, and
    least-squares amplitudes at the semidiurnal and inertial periods."""
    sj, si = slice(box['j0'], box['j1']), slice(box['i0'], box['i1'])
    ue = np.array([np.nanmean(vel[int(h)]['geo'][0][sj, si]) for h in hours])
    vn = np.array([np.nanmean(vel[int(h)]['geo'][1][sj, si]) for h in hours])
    t = (hours - hours[0]).astype('float64')
    return ue, vn, t


def fit_periods(t, y, periods):
    """Mean + trend + sinusoids at ``periods`` [h]: amplitudes and the residual rms."""
    cols = [np.ones_like(t), t - t.mean()]
    for P in periods:
        w = 2 * np.pi / P
        cols += [np.cos(w * t), np.sin(w * t)]
    X = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    amps = {f'{P:.2f}h': float(np.hypot(c[2 + 2 * k], c[3 + 2 * k])) for k, P in enumerate(periods)}
    res = y - X @ c
    return dict(mean=float(c[0]), trend_per_h=float(c[1]), amplitudes=amps,
                residual_rms=float(np.sqrt(np.mean(res ** 2))), rms=float(np.sqrt(np.mean((y - y.mean()) ** 2))))


def run(say=print):
    t_all = time.time()
    cs = json.loads(case.OUT_JSON.read_text())
    z = np.load(case.OUT_NPZ)
    c = cs['case']
    t0, n_back, n_fwd, box, seed_box = c['t0'], c['n_back'], c['n_fwd'], c['box'], c['seed_box']
    ds, grid_ds, grid, der, vel, fields, XC, YC, pairs = load(t0, n_back, n_fwd)
    pos = np.asarray(z['pos'], 'float64')                     # (n_t, 450, 2), (j, i)
    say(f'loaded: {len(vel)} hours of velocity, {len(fields)} pairs of fields ({time.time() - t_all:.0f} s)')

    # --- the control, recomputed as the case does it ------------------------
    sj, si = slice(box['j0'], box['j1']), slice(box['i0'], box['i1'])
    valid_box = der['valid'].isel(time=t0).values[sj, si]
    jj, ii = np.where(valid_box)
    pick = np.random.default_rng(0).choice(len(jj), size=min(c.get('n_control', 2500), len(jj)), replace=False)
    ctrl_seeds = np.stack([jj[pick] + box['j0'], ii[pick] + box['i0']], axis=1).astype('f8')
    cpos, _ = case._trajectory(ds, grid_ds, grid, ctrl_seeds, t0, n_back, n_fwd)
    say(f'control trajectory: {cpos.shape[1]} parcels ({time.time() - t_all:.0f} s)')

    sel = sample_series(pos, pairs, vel, fields, t0)
    ctl = sample_series(cpos, pairs, vel, fields, t0)
    hrs = (pairs - t0).astype('float64')
    n0 = n_back

    # --- the seed hour: the field, the parcels' components -----------------
    f0 = fields[int(t0)]
    u0, v0 = vel[int(t0)]['model']
    vn0, vt0 = along_cross(u0, v0, f0['b_x'], f0['b_y'])
    speed0 = np.hypot(u0, v0)
    seed_ok = f0['front'] & f0['valid']
    corr = seed_box
    corr_mask = np.zeros_like(seed_ok)
    corr_mask[corr['j0']:corr['j1'], corr['i0']:corr['i1']] = True
    on_seed = seed_ok & corr_mask
    def q(a, m):
        a = a[m]
        a = a[np.isfinite(a)]
        return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                    p90=float(np.percentile(a, 90)), mean=float(a.mean()), n=int(a.size))
    seed_stats = dict(
        speed_ms=q(speed0, on_seed), cross_ms=q(np.abs(vn0), on_seed), along_ms=q(np.abs(vt0), on_seed),
        cross_signed_ms=q(vn0, on_seed),
        box_speed_ms=q(speed0, f0['valid'] & np.isfinite(speed0)
                       & (np.arange(720)[:, None] >= box['j0']) & (np.arange(720)[:, None] < box['j1'])
                       & (np.arange(720)[None, :] >= box['i0']) & (np.arange(720)[None, :] < box['i1'])),
        front_width_dx=q(f0['width'], on_seed),
        G_seed=float(np.nanmedian(f0['G'][on_seed])))
    w_km = seed_stats['front_width_dx']['median'] * KM_PER_CELL
    seed_stats['front_width_km'] = w_km
    seed_stats['crossing_time_h'] = w_km * 1e3 / (seed_stats['cross_ms']['median'] * 3600.0)
    seed_stats['cells_per_hour_median'] = seed_stats['speed_ms']['median'] * 3600 / (KM_PER_CELL * 1e3)

    # --- the series ---------------------------------------------------------
    def series(S, P):
        d = np.diff(P, axis=0)
        step = np.hypot(d[..., 0], d[..., 1])                                 # cells per hour
        path = np.concatenate([[0.0], np.cumsum(np.nanmedian(step, axis=1))])
        net = np.hypot(*(P - P[n0]).transpose(2, 0, 1))                       # (n_t, n_p) cells from the seed
        cross_cum = np.concatenate([[0.0], np.cumsum(np.nanmedian(S['v_cross'][:-1], axis=1) * 3600 / 1e3)])  # km, signed
        return dict(
            on_front=(S['on_front'].mean(axis=1)).tolist(),
            on_p80=(S['on_p80'].mean(axis=1)).tolist(),
            dist_front_km=(np.nanmedian(S['dist_front'], axis=1) * KM_PER_CELL).tolist(),
            dist_p80_km=(np.nanmedian(S['dist_p80'], axis=1) * KM_PER_CELL).tolist(),
            speed_ms=np.nanmedian(S['speed'], axis=1).tolist(),
            cross_abs_ms=np.nanmedian(np.abs(S['v_cross']), axis=1).tolist(),
            along_abs_ms=np.nanmedian(np.abs(S['v_along']), axis=1).tolist(),
            cross_signed_ms=np.nanmedian(S['v_cross'], axis=1).tolist(),
            G_median=np.nanmedian(S['G'], axis=1).tolist(),
            step_cells_median=np.nanmedian(step, axis=1).tolist(),
            path_cells=path.tolist(), net_from_seed_cells=np.nanmedian(net, axis=1).tolist(),
            cross_cum_km_from_start=cross_cum.tolist())
    S_sel, S_ctl = series(sel, pos), series(ctl, cpos)
    # path vs net over the forward day
    fwd = slice(n0, None)
    for S, P, name in ((S_sel, pos, 'selected'), (S_ctl, cpos, 'control')):
        d = np.diff(P[fwd], axis=0)
        step = np.hypot(d[..., 0], d[..., 1])
        path_fwd = np.nansum(step, axis=0)                                   # per parcel
        net_fwd = np.hypot(*(P[-1] - P[n0]).T)
        S['forward_day'] = dict(path_cells_median=float(np.nanmedian(path_fwd)),
                                net_cells_median=float(np.nanmedian(net_fwd)),
                                straightness=float(np.nanmedian(net_fwd / path_fwd)),
                                net_km_median=float(np.nanmedian(net_fwd) * KM_PER_CELL))
        # cross-front displacement over the forward day, parcel by parcel (km)
        xc = np.nansum(np.abs(sel['v_cross'][n0:-1] if name == 'selected' else ctl['v_cross'][n0:-1]), axis=0) * 3600 / 1e3
        S['forward_day']['cross_path_km_median'] = float(np.nanmedian(xc))
        xs = np.nansum(sel['v_cross'][n0:-1] if name == 'selected' else ctl['v_cross'][n0:-1], axis=0) * 3600 / 1e3
        S['forward_day']['cross_net_km_median_abs'] = float(np.nanmedian(np.abs(xs)))

    # --- the box-mean velocity: hodograph, periods --------------------------
    hours = np.arange(t0 - n_back, t0 + n_fwd + 2)
    ue, vn, t = box_mean_series(vel, hours, box, fields)
    lat = float(np.nanmean(YC[sj, si]))
    f_cor = 2 * OMEGA * np.sin(np.deg2rad(lat))
    T_in = 2 * np.pi / f_cor / 3600.0
    periods = (12.42, T_in)
    fit_u, fit_v = fit_periods(t, ue, periods), fit_periods(t, vn, periods)
    # rotation rate of the box-mean velocity vector (deg per hour, +: anticlockwise)
    ang = np.unwrap(np.arctan2(vn, ue))
    rot = np.degrees(np.diff(ang))
    box_series = dict(hours_from_seed=(hours - t0).tolist(), u_east=ue.tolist(), v_north=vn.tolist(),
                      lat=lat, inertial_period_h=T_in, fit_u=fit_u, fit_v=fit_v,
                      rotation_deg_per_h=dict(median=float(np.median(rot)), mean=float(rot.mean())),
                      mean_speed_ms=float(np.hypot(ue.mean(), vn.mean())),
                      rms_about_mean_ms=float(np.sqrt(np.mean((ue - ue.mean()) ** 2 + (vn - vn.mean()) ** 2))))

    out = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'), case=c, km_per_cell=KM_PER_CELL,
               seed=seed_stats, selected=S_sel, control=S_ctl, box_mean=box_series,
               hours_from_seed=hrs.tolist(), n_control=int(cpos.shape[1]),
               wall_s=round(time.time() - t_all, 1))
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    say(f'wrote {OUT_JSON} ({out["wall_s"]} s)')

    from m3_parcels_fig import figure_velocity, figure_motion
    figure_velocity(out, f0, vel[int(t0)]['geo'], speed0, vn0, XC, YC, pos[n0], ctrl_seeds, box,
                    FIG_DIR / 'm3_parcels_velocity.png')
    figure_motion(out, f0, XC, YC, pos, cpos, box, FIG_DIR / 'm3_parcels_motion.png')
    say('wrote the two figures')
    return out


if __name__ == '__main__':
    s = run()
    sd, se, ct, bm = s['seed'], s['selected'], s['control'], s['box_mean']
    print(f"seed: speed {sd['speed_ms']['median']:.3f} m/s (p10-p90 {sd['speed_ms']['p10']:.3f}-{sd['speed_ms']['p90']:.3f}) "
          f"= {sd['cells_per_hour_median']:.2f} cells/h; cross {sd['cross_ms']['median']:.3f}, along {sd['along_ms']['median']:.3f} m/s; "
          f"box speed {sd['box_speed_ms']['median']:.3f}; width {sd['front_width_dx']['median']:.2f} dx = {sd['front_width_km']:.1f} km; "
          f"crossing time {sd['crossing_time_h']:.1f} h")
    for n, h in enumerate(s['hours_from_seed']):
        if h in (-24, -16, -8, 0, 4, 8, 12, 16, 20, 24):
            print(f"  t{h:+3.0f} h: on front {se['on_front'][n]:.2f} (p80 {se['on_p80'][n]:.2f}; control {ct['on_front'][n]:.2f}) "
                  f"dist {se['dist_front_km'][n]:.1f} km (ctl {ct['dist_front_km'][n]:.1f}); speed {se['speed_ms'][n]:.3f} "
                  f"cross {se['cross_abs_ms'][n]:.3f} along {se['along_abs_ms'][n]:.3f}; G {se['G_median'][n]:.2e}; "
                  f"net {se['net_from_seed_cells'][n]:.1f} cells")
    print('forward day selected', se['forward_day'], '\nforward day control', ct['forward_day'])
    print(f"box mean: speed {bm['mean_speed_ms']:.3f} m/s, rms about mean {bm['rms_about_mean_ms']:.3f}; "
          f"u amps {bm['fit_u']['amplitudes']}, v amps {bm['fit_v']['amplitudes']}; inertial period {bm['inertial_period_h']:.1f} h; "
          f"rotation {bm['rotation_deg_per_h']}")
