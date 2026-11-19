""" Track analysis, Surface-flux task 1 (JXP, 2026-11-18): is the surface-flux term acting on
the tracked fronts?

``matched_parcels.py`` put the term on one front (label 81).  This runs the same band means
(band 1 + centred matched parcels, M4-Q3 (a)) on **every** flow-informed track of
``m4_tracks.json`` and adds what a judgement on the term needs:

* per track, the front-mean budget: r.m.s. of measured - predicted with ``2F``, ``2F + V``,
  ``2F + V + S(1 m)``, ``2F + V + S(h_bl)``, against both measured rates;
* per track and pooled over tracks, the report's §7.2 multiplier on each form of ``S`` (the
  weight that minimises the residual; +1 for a correct term), with a moving-block bootstrap
  interval over hours (``BLOCK_H``), so that "no effect" is told from "undetectable";
* the sign: ``S(1 m)`` split into non-solar, shortwave and freshwater parts
  (``m3_term_tests.split_components``) on the parcels, and the thermal fraction of the
  buoyancy gradient ``R_T = grad b . grad b_T / |grad b|^2`` (``b_T``: the part of ``b`` due
  to temperature).  ``R_T < 0`` is a front whose warm side is the dense side
  (salinity-controlled), where a heat loss that grows with temperature is frontogenetic;
* the same means on the budget's pooled front pixels (the store's ``front & valid``), for
  the comparison with the report's §8.

No slope here is an efficiency.  Nothing is written but ``data/surface_flux_tracks.json``.

Run:  <frontogenesis env python> surface_flux_tracks.py     (from py/; ~15 min)
"""
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matched_parcels as mp                                    # noqa: E402
import tracking as tr                                           # noqa: E402

OUT_JSON = mp.DATA_DIR / 'surface_flux_tracks.json'
BLOCK_H = 6                     # moving-block length [h] of the bootstrap over hours (declared before the run)
N_BOOT = 2000
PARTS = ('nonsolar', 'shortwave', 'freshwater')
MEASURED = ('r_store', 'r_matched')
FORMS = dict(S_1m='S_1m_mean', S_kpp='S_kpp_mean')


def _rms(x):
    x = np.asarray(x, 'float64')
    return float(np.sqrt(np.nanmean(x ** 2)))


def multiplier(S, I):
    """The weight ``a`` minimising ``rms(I - a S)`` (through the origin, as report §7.2)."""
    return float(np.sum(S * I) / np.sum(S ** 2))


def block_bootstrap(series, n_boot=N_BOOT, block=BLOCK_H, seed=0):
    """Moving-block bootstrap of the multiplier.  ``series`` is a list of ``(S, I)`` pairs,
    one per track, each in time order; blocks are drawn within each track and pooled."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        num = den = 0.0
        for S, I in series:
            n = S.size
            starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
            num += float(np.sum(S[idx] * I[idx]))
            den += float(np.sum(S[idx] ** 2))
        out[b] = num / den
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def projection(dfs, form, measured):
    """Multiplier, interval and correlation of one form of ``S`` on the imbalance
    ``measured - 2F - V``, for a list of per-track tables (pooled when more than one)."""
    series = []
    for d in dfs:
        ok = d[[FORMS[form], measured, 'two_F', 'V_mean']].notna().all(axis=1).to_numpy()
        S = d[FORMS[form]].to_numpy('float64')[ok]
        I = (d[measured] - d['two_F'] - d['V_mean']).to_numpy('float64')[ok]
        series.append((S, I))
    S, I = (np.concatenate([s[k] for s in series]) for k in (0, 1))
    a = multiplier(S, I)
    return dict(multiplier=a, ci95=block_bootstrap(series), corr=float(np.corrcoef(S, I)[0, 1]),
                rms_S=_rms(S), rms_I=_rms(I), rms_I_minus_S=_rms(I - S), rms_I_minus_aS=_rms(I - a * S),
                n=int(S.size))


def track_budget(d):
    """The front-mean budget of one track (band means over the hours with a rate)."""
    d = d[d['r_store'].notna()]
    out = dict(n_steps=int(len(d)),
               mean={k: float(d[c].mean()) for k, c in (('two_F', 'two_F'), ('r_store', 'r_store'),
                                                         ('r_matched', 'r_matched'), ('V', 'V_mean'),
                                                         ('S_1m', 'S_1m_mean'), ('S_kpp', 'S_kpp_mean'))},
               rms={k: _rms(d[c]) for k, c in (('two_F', 'two_F'), ('r_store', 'r_store'),
                                               ('r_matched', 'r_matched'), ('V', 'V_mean'),
                                               ('S_1m', 'S_1m_mean'), ('S_kpp', 'S_kpp_mean'))},
               frac_pos={k: float(np.mean(d[c] > 0)) for k, c in (('S_1m', 'S_1m_mean'), ('S_kpp', 'S_kpp_mean'),
                                                                 ('two_F', 'two_F'), ('r_store', 'r_store'),
                                                                 ('r_matched', 'r_matched'))},
               corr_store_two_F=float(np.corrcoef(d['r_store'], d['two_F'])[0, 1]),
               hbl_median=float(d['hbl_median'].median()), residual_rms={}, projection={})
    for m in MEASURED:
        base = d[m] - d['two_F']
        out['residual_rms'][m] = dict(two_F=_rms(base), two_F_V=_rms(base - d['V_mean']),
                                      full_1m=_rms(base - d['V_mean'] - d['S_1m_mean']),
                                      full_kpp=_rms(base - d['V_mean'] - d['S_kpp_mean']))
        out['projection'][m] = {f: projection([d], f, m) for f in FORMS}
    sc = mp.set_change_summary(d, dict(two_F='two_F', r_store='r_store', r_matched='r_matched',
                                       S_1m='S_1m_mean', S_kpp='S_kpp_mean'))
    out['set_change_corr'] = {k: v['corr'] for k, v in sc['series'].items()}
    out['integral_1e14'] = {k: mp._trapz(d, c) / 1e-14 for k, c in (('two_F', 'two_F'), ('r_store', 'r_store'),
                                                                    ('r_matched', 'r_matched'),
                                                                    ('S_1m', 'S_1m_mean'), ('S_kpp', 'S_kpp_mean'))}
    return out


def sign_maps(ds, grid_ds, grid, drF, step):
    """Full-tile maps at one pair: the three parts of ``S(1 m)`` and of ``S(h_bl)``, the
    recomputed full terms, and the thermal / haline fractions of the buoyancy gradient."""
    import inputs as inp
    import m3_term_tests as tt
    import operators as op
    import vertical as vt
    h0, h1 = inp.hour_pair(ds, int(step))
    sp = tt.split_components(h0, h1, 0, grid_ds, grid, drF)
    m = tt.midpoint_inputs(h0, h1, 0, grid_ds, grid)
    alpha, beta, rho = vt.expansion_coefficients(m['Th'], m['Sa'])
    k = vt.G / vt.RHO0_REFERENCE
    Tx, Ty = op.grad_b(m['Th'], grid_ds, grid)
    Sx, Sy = op.grad_b(m['Sa'], grid_ds, grid)
    sq = lambda a: np.squeeze(np.asarray(getattr(a, 'values', a), 'float64'))       # noqa: E731
    bx, by = sq(m['b_x']), sq(m['b_y'])
    cT, cS = -k * sq(rho) * sq(alpha), k * sq(rho) * sq(beta)      # code b: d b = cT dTheta + cS dS
    g2 = bx ** 2 + by ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        R_T = (bx * cT * sq(Tx) + by * cT * sq(Ty)) / g2
        R_S = (bx * cS * sq(Sx) + by * cS * sq(Sy)) / g2
    out = dict(R_T=R_T, R_S=R_S, full_top=sp['top']['full'], full_kpp=sp['kpp']['full'])
    for dep in ('top', 'kpp'):
        for p in PARTS:
            out[f'{dep}_{p}'] = sp[dep][p]
    return out


def run(log=print):
    import budget
    import inputs as inp
    import xarray as xr
    from m4_track import FLOW_PATH, TRACKS_PATH
    t_all = time.time()
    flow = tr.Flow(store=FLOW_PATH)
    labels = tr.open_fronts_series(tr.fronts_path(0))
    keys = list(labels)
    fields = tr.DerivedFields(budget.derived_path(0))
    tracks = tr.tracks_from_json(json.load(open(TRACKS_PATH))['flow'])

    # --- 1. the band means of every track -------------------------------------------------
    dfs, seeds = [], []
    for k, trk in enumerate(tracks):
        t0 = time.time()
        df = mp.analyse_track(trk, labels, flow, fields, check=False)
        dfs.append(df)
        sd = {}
        for s in trk.steps():
            pl = mp.parcel_link(s, trk.labels[s], labels, flow, fields)
            if pl is not None and pl['alive'].any():
                sd[int(s)] = (pl['j'][pl['alive']], pl['i'][pl['alive']])
        seeds.append(sd)
        log(f'track {k} (label {trk.anchor.label}): {len(df)} steps, {time.time() - t0:.0f} s')

    # --- 2. the sign: parts of S and the thermal fraction, on the parcels and on the pool --
    ds, grid_ds, grid, masks = inp.open_inputs()
    der = xr.open_zarr(budget.derived_path(0))
    drF = inp.drF(ds)
    names = ['R_T', 'R_S', 'full_top', 'full_kpp'] + [f'{d}_{p}' for d in ('top', 'kpp') for p in PARTS]
    per = [{n: {} for n in names + ['frac_RT_neg', 'S_where_RT_neg', 'S_where_RT_pos', 'stored_S']}
           for _ in tracks]
    pool = {n: [] for n in names + ['frac_RT_neg', 'S_where_RT_neg', 'S_where_RT_pos', 'stored_S',
                                    'stored_S_kpp', 'n', 'step']}
    ident = []
    steps_all = sorted(set().union(*[set(sd) for sd in seeds]))
    for s in steps_all:
        t0 = time.time()
        mps = sign_maps(ds, grid_ds, grid, drF, s)
        d = der.isel(time=int(s))
        S_st = np.asarray(d['surface_flux'].values, 'float64')
        S_kp = np.asarray(d['surface_flux_kpp'].values, 'float64')
        fr = np.asarray(d['front'].values, bool) & np.asarray(d['valid'].values, bool) & np.isfinite(S_st)
        ident.append(_rms((mps['full_top'] - S_st)[fr]) / _rms(S_st[fr]))
        for n in names:
            pool[n].append(float(np.nanmean(mps[n][fr])))
        rt = mps['R_T'][fr]
        pool['frac_RT_neg'].append(float(np.mean(rt < 0)))
        pool['S_where_RT_neg'].append(float(S_st[fr][rt < 0].mean()))
        pool['S_where_RT_pos'].append(float(S_st[fr][rt >= 0].mean()))
        pool['stored_S'].append(float(S_st[fr].mean()))
        pool['stored_S_kpp'].append(float(np.nanmean(S_kp[fr])))
        pool['n'].append(int(fr.sum()))
        pool['step'].append(int(s))
        for k, sd in enumerate(seeds):
            if s not in sd:
                continue
            jj, ii = sd[s]
            for n in names:
                per[k][n][s] = float(np.nanmean(mps[n][jj, ii]))
            rt, Sv = mps['R_T'][jj, ii], S_st[jj, ii]
            per[k]['frac_RT_neg'][s] = float(np.mean(rt < 0))
            per[k]['S_where_RT_neg'][s] = float(Sv[rt < 0].mean()) if (rt < 0).any() else np.nan
            per[k]['S_where_RT_pos'][s] = float(Sv[rt >= 0].mean()) if (rt >= 0).any() else np.nan
            per[k]['stored_S'][s] = float(np.nanmean(Sv))
        if s % 10 == 0:
            log(f'  pair {s}: {time.time() - t0:.1f} s; identity {ident[-1]:.2e}')
    der.close()

    # --- 3. assemble -----------------------------------------------------------------------
    out_tracks = []
    for k, (trk, df) in enumerate(zip(tracks, dfs)):
        tb = track_budget(df)
        sg = {n: float(np.nanmean(list(v.values()))) for n, v in per[k].items()}
        sg['S_1m_hours_positive'] = float(np.mean(np.array(list(per[k]['stored_S'].values())) > 0))
        sg['corr_hourly_S_1m_R_T'] = float(np.corrcoef(list(per[k]['stored_S'].values()),
                                                      list(per[k]['R_T'].values()))[0, 1])
        out_tracks.append(dict(index=k, label=int(trk.anchor.label), anchor_step=int(trk.anchor.step),
                               npix_median=float(df['npix'].median()), budget=tb, sign=sg,
                               rows=df[['step', 'npix', 'two_F', 'r_store', 'r_matched', 'V_mean', 'S_1m_mean',
                                        'S_kpp_mean', 'hbl_median', 'lst']].to_dict('records')))
    ok = [d[d['r_store'].notna()] for d in dfs]
    pooled = dict(n_track_hours=int(sum(len(d) for d in ok)),
                  projection={m: {f: projection(ok, f, m) for f in FORMS} for m in MEASURED},
                  residual_rms={m: dict(
                      two_F=_rms(np.concatenate([(d[m] - d['two_F']).to_numpy() for d in ok])),
                      two_F_V=_rms(np.concatenate([(d[m] - d['two_F'] - d['V_mean']).to_numpy() for d in ok])),
                      full_1m=_rms(np.concatenate([(d[m] - d['two_F'] - d['V_mean'] - d['S_1m_mean']).to_numpy() for d in ok])),
                      full_kpp=_rms(np.concatenate([(d[m] - d['two_F'] - d['V_mean'] - d['S_kpp_mean']).to_numpy() for d in ok])),
                      measured=_rms(np.concatenate([d[m].to_numpy() for d in ok]))) for m in MEASURED},
                  rms={c: _rms(np.concatenate([d[c].to_numpy() for d in ok]))
                       for c in ('two_F', 'r_store', 'r_matched', 'V_mean', 'S_1m_mean', 'S_kpp_mean')},
                  mean={c: float(np.mean(np.concatenate([d[c].to_numpy() for d in ok])))
                        for c in ('two_F', 'r_store', 'r_matched', 'V_mean', 'S_1m_mean', 'S_kpp_mean')})
    pool_summary = {n: float(np.mean(v)) for n, v in pool.items() if n not in ('n', 'step')}
    pool_summary.update(hours_S_negative=float(np.mean(np.array(pool['stored_S']) < 0)),
                        n_pixels=int(np.sum(pool['n'])), n_pairs=len(pool['step']))
    out = dict(params=dict(L=0, band=mp.BAND, centred=True, block_h=BLOCK_H, n_boot=N_BOOT,
                           imbalance='measured - 2F - V (band means, hourly)',
                           R_T='grad b . grad b_T / |grad b|^2 at the midpoint; b_T the thermal part of code b'),
               tracks=out_tracks, pooled_tracks=pooled, front_pool=pool_summary,
               front_pool_series={k: pool[k] for k in ('step', 'stored_S', 'stored_S_kpp', 'R_T', 'frac_RT_neg', 'n')},
               identity_rel_rms_max=float(np.max(ident)), seconds=time.time() - t_all,
               created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'))
    tr.dump_json(out, OUT_JSON)
    log(f'wrote {OUT_JSON} ({time.time() - t_all:.0f} s)')
    return out


if __name__ == '__main__':
    run()
