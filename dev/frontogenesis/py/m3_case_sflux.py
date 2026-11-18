""" M3-digging task 9: the surface-flux term on the primary example -- the followed front of
the interim's §5.1 (JXP, 2026-11-05).

Along the 450 parcels of ``m3_case.py`` (pairs 6..54, the seed at pair 30) this samples the
stored budget columns and recomputes the surface-flux term's three parts -- non-solar heat,
shortwave, freshwater (``m3_term_tests.split_components``, ``L = 0``) -- in both depth forms
(the 1 m top cell, primary; the KPP boundary-layer depth, the §6.3 sensitivity), and writes:

* the parcel-median rates along the trajectory (measured, ``2F``, the imbalance
  ``I = measured - 2F - tau - V``, ``S``, its parts, ``S`` in the boundary-layer form);
* the integrals from the seed outwards, as the case does them, with and without ``S``;
* the pointwise relation of ``S`` (and its parts) to the imbalance on the parcels over the
  forward day: correlation, OLS slope, explained fraction;
* the composite of ``S`` and its parts by local solar hour along the parcels;
* the maps at the seed in the case box: ``S`` and its three parts, the imbalance, ``KPPhbl``.

Products: ``data/m3_case_sflux.json`` (every number the report quotes) and
``data/m3_case_sflux.npz`` (the arrays the figure draws), so ``report/make_case_figs.py``
opens no store.  The case products and the derived stores are not modified.

Run:  <frontogenesis env python> py/m3_case_sflux.py      (from dev/frontogenesis; ~4 min)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget as bg                                            # noqa: E402
import inputs as inp                                           # noqa: E402
import m3_case as case                                         # noqa: E402
import m3_closure as mc                                        # noqa: E402
import m3_term_tests as tt                                     # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

OUT_JSON = DATA_DIR / 'm3_case_sflux.json'
OUT_NPZ = DATA_DIR / 'm3_case_sflux.npz'
COLS = ('DGDt_semilag', 'two_F', 'subfilter', 'vertical', 'surface_flux', 'surface_flux_kpp',
        'residual', 'KPPhbl', 'G')
PARTS = tt.COMP                                                # nonsolar, shortwave, freshwater


def integrate(r, G0, n_back, dt=case.DT):
    """The case's own integral: from the seed outwards, both ways."""
    n_t = r.size + 1
    out = np.empty(n_t)
    out[n_back] = G0
    for m in range(n_back, n_t - 1):
        out[m + 1] = out[m] + r[m] * dt
    for m in range(n_back, 0, -1):
        out[m - 1] = out[m] - r[m - 1] * dt
    return out


def run(say=print):
    t_all = time.time()
    cs = json.loads(case.OUT_JSON.read_text())
    z = np.load(case.OUT_NPZ)
    c = cs['case']
    t0, n_back, n_fwd, box = c['t0'], c['n_back'], c['n_fwd'], c['box']
    pos = np.asarray(z['pos'], 'float64')
    pairs = np.arange(t0 - n_back, t0 + n_fwd + 1)
    n_t, n_p = pos.shape[:2]
    ds, grid_ds, grid, masks = inp.open_inputs()
    der = xr.open_zarr(bg.derived_path(0))
    drF = inp.drF(ds)
    offset = float(der.attrs.get('utc_offset_h', inp.UTC_OFFSET_H))
    sj, si = slice(box['j0'], box['j1']), slice(box['i0'], box['i1'])

    series = {k: np.full((n_t, n_p), np.nan) for k in COLS}
    for d in tt.DEPTHS:
        for p in PARTS:
            series[f'{d}_{p}'] = np.full((n_t, n_p), np.nan)
    lst = np.empty(n_t)
    maps = {}
    say(f'sampling {n_t} pairs along {n_p} parcels')
    for n, p in enumerate(pairs):
        tp = time.time()
        d = der.isel(time=int(p)).load()
        lst[n] = float(inp.local_solar_hour(d['time_mid'].values, offset))
        for k in COLS:
            series[k][n] = case.sample(np.asarray(d[k].values, 'float64'), pos[n], 3)   # cubic, as the case
        h0, h1 = inp.hour_pair(ds, int(p))
        sp = tt.split_components(h0, h1, 0, grid_ds, grid, drF)
        for dep in tt.DEPTHS:
            for part in PARTS:
                series[f'{dep}_{part}'][n] = case.sample(sp[dep][part], pos[n], 3)
        if int(p) == t0:
            f = lambda a: np.asarray(a, 'float32')[sj, si]                      # noqa: E731
            maps = dict(S=f(d['surface_flux'].values), S_kpp=f(d['surface_flux_kpp'].values),
                        I=f(np.asarray(d['DGDt_semilag'].values, 'float64') - np.asarray(d['two_F'].values, 'float64')
                            - np.asarray(d['subfilter'].values, 'float64') - np.asarray(d['vertical'].values, 'float64')),
                        two_F=f(d['two_F'].values), KPPhbl=f(d['KPPhbl'].values), G=f(d['G'].values),
                        **{f'S_{part}': f(sp['top'][part]) for part in PARTS},
                        identity_rel=float(sp['top']['identity_rel_rms']))
        if n % 8 == 0:
            say(f'  pair {int(p)} ({n + 1}/{n_t}) {time.time() - tp:.1f} s')
    der.close()

    # --- rates: parcel medians per interval (centred on the integer hour, as the case) ----
    def rate(k):
        a = series[k]
        return np.nanmedian(0.5 * (a[:-1] + a[1:]), axis=1)
    I_series = (series['DGDt_semilag'] - series['two_F'] - series['subfilter'] - series['vertical'])
    series['I'] = I_series
    med = {k: rate(k) for k in ('DGDt_semilag', 'two_F', 'subfilter', 'vertical', 'surface_flux',
                                'surface_flux_kpp', 'residual', 'I')}
    for dep in tt.DEPTHS:
        for part in PARTS:
            med[f'{dep}_{part}'] = rate(f'{dep}_{part}')
    G_obs = np.nanmedian(series['G'], axis=1)
    G0 = float(G_obs[n_back])
    integ = {
        'two_F': integrate(med['two_F'], G0, n_back),
        'two_F_S': integrate(med['two_F'] + med['surface_flux'], G0, n_back),
        'two_F_tau_V': integrate(med['two_F'] + med['subfilter'] + med['vertical'], G0, n_back),
        'all_four': integrate(med['two_F'] + med['subfilter'] + med['vertical'] + med['surface_flux'], G0, n_back),
        'all_four_kpp': integrate(med['two_F'] + med['subfilter'] + med['vertical'] + med['surface_flux_kpp'], G0, n_back),
        'S_alone': integrate(med['surface_flux'], G0, n_back),
    }
    dG = float(G_obs[-1] - G0)
    fwd = slice(n_back, n_t - 1)
    dt = case.DT
    rms = (lambda a: float(np.sqrt(np.nanmean(np.asarray(a, 'float64') ** 2))))
    r_meas = rms(med['DGDt_semilag'])

    # --- the pointwise relation on the parcels over the forward day ---------------------
    def pointwise(x_series, y_series):
        x = 0.5 * (x_series[:-1] + x_series[1:])[fwd].ravel()
        y = 0.5 * (y_series[:-1] + y_series[1:])[fwd].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        f = mc._fit(x[ok], y[ok])
        res = y[ok] - f['ols'] * x[ok]
        return dict(corr=f['corr'], ols=f['ols'], n=int(ok.sum()),
                    explained_at_unit=float(1 - np.var(y[ok] - x[ok]) / np.var(y[ok])),
                    explained_at_ols=float(1 - np.var(res) / np.var(y[ok])),
                    rms_x_over_y=rms(x[ok]) / rms(y[ok]))
    pw = dict(S=pointwise(series['surface_flux'], I_series),
              S_kpp=pointwise(series['surface_flux_kpp'], I_series),
              **{f'S_{part}': pointwise(series[f'top_{part}'], I_series) for part in PARTS},
              two_F_vs_measured=pointwise(series['two_F'], series['DGDt_semilag']))

    # --- the diurnal composite along the parcels -------------------------------------------
    hr = np.floor(0.5 * (lst[:-1] + lst[1:])).astype(int) % 24
    # normalised by the rms of the measured rate over every parcel and interval (as the
    # amplitude ratios below), not by the rms of the median series
    r_norm = rms(0.5 * (series['DGDt_semilag'][:-1] + series['DGDt_semilag'][1:]))
    diurnal = {}
    for k in ('surface_flux', 'surface_flux_kpp', 'I', 'DGDt_semilag') + tuple(f'top_{p}' for p in PARTS):
        diurnal[k] = {int(h): float(np.nanmean(med[k][hr == h]) / r_norm) for h in np.unique(hr)}
    s_d = diurnal['surface_flux']
    hmin, hmax = min(s_d, key=s_d.get), max(s_d, key=s_d.get)

    # --- the numbers --------------------------------------------------------------------------
    rate_fwd = {k: med[k][fwd] for k in med}
    # the amplitude ratios AS THE CASE QUOTES THEM: the rms over every parcel and interval of
    # the forward day (the medians above are for the curves; the median measured rate is near
    # zero because the parcels' rates alternate in sign, so a ratio of medians means nothing)
    def pr(k):
        a = series[k]
        return 0.5 * (a[:-1] + a[1:])[fwd]
    r_meas_all = rms(pr('DGDt_semilag'))
    out = dict(
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        source=dict(derived=str(bg.derived_path(0)), case=str(case.OUT_JSON), raw=str(inp.RAW_ZARR),
                    chunk=str(inp.CHUNK_ZARR)),
        case=dict(t0=t0, n_back=n_back, n_fwd=n_fwd, n_parcels=int(n_p), L=0, box=box,
                  t_seed=c['t_seed'], lst_seed=float(lst[n_back])),
        identity_split_rel_rms=float(maps['identity_rel']),
        rms_over_measured=dict(
            S=rms(pr('surface_flux')) / r_meas_all,
            S_kpp=rms(pr('surface_flux_kpp')) / r_meas_all,
            I=rms(pr('I')) / r_meas_all,
            residual=rms(pr('residual')) / r_meas_all,
            two_F=rms(pr('two_F')) / r_meas_all,
            **{f'S_{part}': rms(pr(f'top_{part}')) / r_meas_all for part in PARTS},
            note='rms over every parcel and interval of the forward day, as the case quotes its terms'),
        parts_over_S=dict(**{part: rms(pr(f'top_{part}')) / rms(pr('surface_flux')) for part in PARTS},
                          sum_check=float(rms(sum(pr(f'top_{p}') for p in PARTS) - pr('surface_flux'))
                                          / rms(pr('surface_flux')))),
        sign=dict(S_mean_over_rms_meas_fwd=float(np.nanmean(pr('surface_flux')) / r_meas_all),
                  S_negative_intervals_fwd=int((rate_fwd['surface_flux'] < 0).sum()),
                  S_negative_parcel_intervals_fraction=float(np.nanmean(pr('surface_flux') < 0)),
                  I_mean_over_rms_meas_fwd=float(np.nanmean(pr('I')) / r_meas_all),
                  n_intervals=int(n_fwd),
                  S_negative_intervals_all=int((med['surface_flux'] < 0).sum()), n_intervals_all=int(n_t - 1)),
        integrals=dict(
            G_seed=G0, G_end=float(G_obs[-1]), dG_observed=dG,
            S_integral_fwd=float(np.nansum(rate_fwd['surface_flux']) * dt),
            S_kpp_integral_fwd=float(np.nansum(rate_fwd['surface_flux_kpp']) * dt),
            S_integral_over_dG=float(np.nansum(rate_fwd['surface_flux']) * dt / dG),
            S_integral_over_G_seed=float(np.nansum(rate_fwd['surface_flux']) * dt / G0),
            **{f'S_{part}_integral_over_dG': float(np.nansum(rate_fwd[f'top_{part}']) * dt / dG) for part in PARTS},
            end=dict(observed=float(G_obs[-1]), two_F=float(integ['two_F'][-1]), two_F_S=float(integ['two_F_S'][-1]),
                     two_F_tau_V=float(integ['two_F_tau_V'][-1]), all_four=float(integ['all_four'][-1]),
                     all_four_kpp=float(integ['all_four_kpp'][-1])),
            over_observed_change={k: float((integ[k][-1] - G0) / dG) for k in integ if k != 'S_alone'}),
        pointwise=pw,
        diurnal=dict(composites=diurnal, S_min_hour=int(hmin), S_max_hour=int(hmax),
                     S_min=float(s_d[hmin]), S_max=float(s_d[hmax]),
                     note='parcel-median rate by local solar hour of the interval, over rms measured; '
                          'negative = frontolytic'),
        KPPhbl=dict(median_along_parcels_m=float(np.nanmedian(series['KPPhbl'])),
                    min_median_hour=float(np.nanmin(np.nanmedian(series['KPPhbl'], axis=1))),
                    max_median_hour=float(np.nanmax(np.nanmedian(series['KPPhbl'], axis=1))),
                    S_kpp_over_S=rms(pr('surface_flux_kpp')) / rms(pr('surface_flux'))),
        wall_s=round(time.time() - t_all, 1))
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(
        OUT_NPZ, hours=(pairs - t0).astype('f8'), rate_hours=(pairs[:-1] - t0).astype('f8') + 0.5,
        lst=lst, n_back=np.array(n_back), G_obs=G_obs,
        **{f'med_{k}': v for k, v in med.items()}, **{f'int_{k}': v for k, v in integ.items()},
        **{f'map_{k}': v for k, v in maps.items() if k != 'identity_rel'},
        pos_seed=pos[n_back].astype('f4'), box=np.array([box['j0'], box['j1'], box['i0'], box['i1']]),
        XC=z['XC'], YC=z['YC'],
        pw_S_x=(0.5 * (series['surface_flux'][:-1] + series['surface_flux'][1:])[fwd].ravel()).astype('f4'),
        pw_S_y=(0.5 * (I_series[:-1] + I_series[1:])[fwd].ravel()).astype('f4'),
        diurnal_hours=np.array(sorted(diurnal['surface_flux'])),
        **{f'diurnal_{k}': np.array([diurnal[k][h] for h in sorted(diurnal[k])]) for k in diurnal})
    say(f'wrote {OUT_JSON} and {OUT_NPZ} ({out["wall_s"]} s)')
    return out


if __name__ == '__main__':
    o = run()
    r, s, i, p = o['rms_over_measured'], o['sign'], o['integrals'], o['pointwise']
    print(f"S/meas {r['S']:.3f} (kpp {r['S_kpp']:.3f}); parts/S ns {o['parts_over_S']['nonsolar']:.3f} sw "
          f"{o['parts_over_S']['shortwave']:.3f} fw {o['parts_over_S']['freshwater']:.3f}; I/meas {r['I']:.3f}")
    print(f"sign: S mean/rms {s['S_mean_over_rms_meas_fwd']:+.3f}, negative {s['S_negative_intervals_fwd']}/{s['n_intervals']} fwd; "
          f"I mean {s['I_mean_over_rms_meas_fwd']:+.3f}")
    print(f"integrals: dG {i['dG_observed']:.2e}; int S {i['S_integral_fwd']:.2e} = {i['S_integral_over_dG']:+.2f} of dG, "
          f"{i['S_integral_over_G_seed']:+.2f} of G_seed; over observed change: " + ', '.join(f'{k} {v:+.2f}' for k, v in i['over_observed_change'].items()))
    print('pointwise on the parcels: ' + ', '.join(f"{k}: corr {v['corr']:+.3f} ols {v['ols']:+.3f} expl@1 {v['explained_at_unit']:+.3f}" for k, v in p.items()))
    print(f"diurnal: S min {o['diurnal']['S_min']:+.3f} at {o['diurnal']['S_min_hour']} LST, max {o['diurnal']['S_max']:+.3f} at {o['diurnal']['S_max_hour']} LST; KPPhbl median {o['KPPhbl']['median_along_parcels_m']:.1f} m")
