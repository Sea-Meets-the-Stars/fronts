""" M2 task 3, extra step (M2-Q3): is Figure 2's baseline 0.981 [0.970, 0.994]
a property of the operators or of one hour?  -> figs/m2_v3_stability.png

Re-runs V3's real-velocity discrete null (``validate.test_discrete_null(
velocities='llc')``) on hour pairs spread over the tidal and diurnal cycle
of the 72-hour store, with **exactly** V3's pre-declared construction:
front pixels ``G_mid >= p90`` on ``mask_analysis`` (``validate.
FRONT_PERCENTILE``), the OLS-with-intercept gate estimator, the 32-cell
block bootstrap (1000 draws), ``form='discrete'`` as the gate with
``form='chain'`` (and ``'discrete_o2'``) fitted alongside for reference,
``order = 3``, ``vel_order = 3``.  Nothing is redefined; the only new
inputs are ``store=`` (the 72-hour store) and ``t0=`` (the pair's first
hour), the backward-compatible keywords added to ``_llc_inputs`` /
``test_discrete_null`` for this step.  Pair 0-1 is the reproduction check
(must give 0.9806); ``--check-default`` also runs the unchanged default
call (M0's two-hour store) and compares the two bit for bit.

The pairs (chosen from ``m2_qa.py``'s Eta and KPPhbl phases; at least one
per day):

    0   07-02 00-01  high tide (Eta +1.30 m), ML deepening (17.5 m), 16 h solar   -- M1's pair
    9   07-02 09-10  low tide (-0.41 m), ML max (25.1 m), 01 h solar
    21  07-02 21-22  rising, ML min of day 1 (13.1 m), 13 h solar; p99 displacement 1.35
    33  07-03 09-10  low tide (-0.47 m), ML max of the window (26.8 m), 01 h solar
    45  07-03 21-22  low tide (+0.03 m), ML min of day 2 (11.8 m), 13 h solar; largest analysis-mask displacement (p99 1.34)
    62  07-04 14-15  high-ish tide (+0.43 m), ML 21.5 m, 06 h solar; smallest displacement of the window (median 0.27, p99 1.05)
    69  07-04 21-22  (extra, 7th) ML min of the window (7.9 m), 13 h solar

Per-pair results are cached in ``data/m2_v3_stability.json`` (resumable,
~40 s per pair); run in batches under ``timeout 300``:

    timeout 300 ~/miniforge3/envs/frontogenesis/bin/python m2_baseline_stability.py --pairs 0,9,21
    timeout 300 ~/miniforge3/envs/frontogenesis/bin/python m2_baseline_stability.py --pairs 33,45,62,69
    timeout 300 ~/miniforge3/envs/frontogenesis/bin/python m2_baseline_stability.py --check-default
    timeout 300 ~/miniforge3/envs/frontogenesis/bin/python m2_baseline_stability.py            # figure + table
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osn_tiles as ot                                           # noqa: E402
import validate as va                                            # noqa: E402
from validate_figs import COL, _save                             # noqa: E402

RAW72 = ot.DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'
OUT_JSON = ot.DATA_DIR / 'm2_v3_stability.json'
PNG = 'm2_v3_stability.png'
PAIRS = (0, 9, 21, 33, 45, 62, 69)
LABELS = {0: 'high tide, ML deepening\n16 h solar (M1 pair)', 9: 'low tide, ML max\n01 h solar',
          21: 'rising, ML min d1\n13 h solar', 33: 'low tide, ML max (window)\n01 h solar',
          45: 'low tide, ML min d2\n13 h solar, max displ.', 62: 'high tide, min displ.\n06 h solar',
          69: 'ML min (window)\n13 h solar (extra)'}
BASELINE = dict(slope=0.981, ci=(0.970, 0.994))          # M1 task 6, Figure 2's baseline (M1-Q4)
V3B_BAND = (0.954, 1.003)                                # V3b's model-advection systematic (M1-Q2)
CHAIN_REF = dict(slope=0.7914, ci=(0.753, 0.815))        # V3 llc, form='chain'
GATE = (0.95, 1.05)
N_BOOT = va.N_BOOT


def _load():
    return json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}


def _dump(obj):
    tmp = OUT_JSON.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(OUT_JSON)


def _strip(res):
    """The JSON-able part of a ``test_discrete_null`` result."""
    keep = {}
    for k, v in res.items():
        if k in ('png',):
            continue
        try:
            json.dumps(v)
            keep[k] = v
        except TypeError:
            pass
    return keep


def run_pair(t0, grid_ds, say, store=RAW72, n_boot=N_BOOT):
    tic = time.time()
    res = va.test_discrete_null(velocities='llc', png=False, grid_ds=grid_ds, n_boot=n_boot, changes=False,
                                store=None if store is None else str(store), t0=t0)
    res = _strip(res)
    res['wall_s'] = time.time() - tic
    f = res['fits']
    say(f'pair {t0:2d}-{t0 + 1:2d} {res.get("hours", ["M0 hour 0", ""])[0]}: discrete {res["slope"]:.4f} '
        f'[{res["ci"][0]:.4f}, {res["ci"][1]:.4f}] n {res["n_front"]} gate {"PASS" if res["gate"]["passed"] else "FAIL"} | '
        f'chain {f["chain"]["ols"]:.4f} [{f["chain"]["bootstrap"]["ci"][0]:.4f}, {f["chain"]["bootstrap"]["ci"][1]:.4f}] | '
        f'o2 {f["discrete_o2"]["ols"]:.4f} | corr {f["discrete"]["corr"]:.3f} | displ. front median {res["displacement_cells"]["median"]:.3f} '
        f'p99 {res["displacement_cells"]["p99"]:.2f} max {res["displacement_cells"]["max"]:.2f} | {res["wall_s"]:.0f} s')
    return res


def roughness(t0, grid_ds, raw):
    """Grid-scale content of the pair's midpoint centred velocity on
    ``mask_analysis``: ``rms(u_c - lowpass(u_c, 2)) / rms(u_c)`` (both
    components), the quantity M1 task 6 identified as the source of the
    -2% (it vanishes when the velocity is low-passed)."""
    import xarray as xr
    import operators as op
    import masking as mk
    ana = mk.open_masks(ot.DATA_DIR / 'tile330_masks.nc')['mask_analysis'].values
    ds = [xr.merge([raw.isel(time=k).load().expand_dims('face'), grid_ds], compat='override',
                   combine_attrs='override').astype('float64') for k in (t0, t0 + 1)]
    grid = ot.build_xgcm(grid_ds)
    U = 0.5 * (ds[0].U + ds[1].U); V = 0.5 * (ds[0].V + ds[1].V)
    uc, vc = op.centred_model_velocity(U, V, grid)
    u, v = uc.values[0], vc.values[0]
    hf = np.sqrt(np.nanmean((u - op.lowpass(u, 2))[ana] ** 2 + (v - op.lowpass(v, 2))[ana] ** 2))
    return float(hf / np.sqrt(np.nanmean(u[ana] ** 2 + v[ana] ** 2)))


def robust(t0, grid_ds):
    """Leverage diagnostics of the pre-declared pool for one pair (the
    construction is unchanged; these are *additional* estimators on the
    same front pixels): OLS with the top 1% / 5% of |2F| pixels dropped,
    the kurtosis of 2F, the largest leave-one-block-out shift of the OLS
    slope, that block's share of the 2F variance and its position."""
    inp = va._llc_inputs(grid_ds, store=str(RAW72), t0=t0)
    st, fits = va._fit_llc(inp, inp['b_t'], inp['U'], inp['V'], ('discrete',), 3, boot=20)
    fr = st['front']
    x, y, blk = st['two_F_discrete'][fr], st['measured'][fr], inp['block'][fr]

    def ols(x, y):
        xm, ym = x.mean(), y.mean()
        return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    full = ols(x, y)
    ids = np.unique(blk)
    loo = np.array([ols(x[blk != b], y[blk != b]) for b in ids])
    sxx = np.array([np.sum((x[blk == b] - x.mean()) ** 2) for b in ids])
    top = ids[int(np.argmax(sxx))]
    jj, ii = np.where(inp['block'] == top)
    g2 = grid_ds.squeeze('face')
    return dict(ols=full, ols_trim1=ols(*(a[np.abs(x) < np.percentile(np.abs(x), 99)] for a in (x, y))),
                ols_trim5=ols(*(a[np.abs(x) < np.percentile(np.abs(x), 95)] for a in (x, y))),
                kurt_2F=float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2),
                loo_max_shift=float(loo[np.argmax(np.abs(loo - full))] - full),
                top_block_share_sxx=float(sxx.max() / sxx.sum()),
                top_block_lon=float(g2.XC.values[jj, ii].mean()), top_block_lat=float(g2.YC.values[jj, ii].mean()),
                top_block_min_edge_cells=int(min(jj.min(), ii.min(), inp['ana'].shape[0] - 1 - jj.max(),
                                                 inp['ana'].shape[1] - 1 - ii.max())))


def table(results):
    rows = []
    for t0 in sorted(results, key=int):
        r = results[t0]
        f = r['fits']
        rows.append(dict(t0=int(t0), hours=r.get('hours', ['2012-07-02T00:00:00', '2012-07-02T01:00:00']),
                         slope=r['slope'], ci=r['ci'], se=f['discrete']['bootstrap']['se'], n=r['n_front'],
                         n_valid=r['n_valid'], gate=r['gate']['passed'], corr=f['discrete']['corr'],
                         orthogonal=f['discrete']['orthogonal'], ols_inverse=f['discrete']['ols_inverse'],
                         geometric_mean=f['discrete']['geometric_mean'], ratio=f['discrete']['ratio'],
                         chain=f['chain']['ols'], chain_ci=f['chain']['bootstrap']['ci'],
                         discrete_o2=f['discrete_o2']['ols'], threshold=r['threshold'],
                         displacement=r['displacement_cells'], signal=r['signal'], wall_s=r.get('wall_s')))
    return rows


def spread(rows):
    s = np.array([r['slope'] for r in rows])
    se = np.array([r['se'] for r in rows])
    c = np.array([r['chain'] for r in rows])
    w = 1 / se ** 2
    mean_w = float(np.sum(w * s) / np.sum(w))
    chi2 = float(np.sum(((s - mean_w) / se) ** 2))
    dof = len(s) - 1
    inside_baseline = [bool(BASELINE['ci'][0] <= v <= BASELINE['ci'][1]) for v in s]
    ci_overlaps_baseline = [bool(r['ci'][0] <= BASELINE['ci'][1] and r['ci'][1] >= BASELINE['ci'][0]) for r in rows]
    inside_v3b = [bool(V3B_BAND[0] <= v <= V3B_BAND[1]) for v in s]
    return dict(n_pairs=len(s), mean=float(s.mean()), std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                min=float(s.min()), max=float(s.max()), range=float(s.max() - s.min()),
                weighted_mean=mean_w, mean_se=float(se.mean()), chi2=chi2, dof=dof,
                chi2_per_dof=chi2 / dof if dof else float('nan'),
                excess_scatter=float(np.sqrt(max(s.var(ddof=1) - np.mean(se ** 2), 0.0))) if len(s) > 1 else 0.0,
                all_pass_gate=bool(all(r['gate'] for r in rows)),
                all_ci_exclude_1=bool(all(r['ci'][1] < 1.0 for r in rows)),
                n_inside_baseline_band=int(sum(inside_baseline)), inside_baseline_band=inside_baseline,
                n_ci_overlap_baseline=int(sum(ci_overlaps_baseline)), ci_overlaps_baseline=ci_overlaps_baseline,
                n_inside_v3b_band=int(sum(inside_v3b)),
                chain=dict(mean=float(c.mean()), std=float(c.std(ddof=1)) if len(c) > 1 else 0.0,
                           min=float(c.min()), max=float(c.max())),
                reproduction=dict(hour0_slope=float(rows[0]['slope']) if rows and rows[0]['t0'] == 0 else None,
                                  target=0.9806,
                                  ok=bool(rows and rows[0]['t0'] == 0 and abs(rows[0]['slope'] - 0.9806) < 5e-4)))


def fig(rows, sp, default_check=None, chosen=None, sp_chosen=None, rough=None, rb=None):
    full = len(rows) > len(PAIRS)                       # the 71-pair series is on disk
    t_all = np.array([r['t0'] + 0.5 for r in rows])
    s_all = np.array([r['slope'] for r in rows])
    lo_all = np.array([r['ci'][0] for r in rows]); hi_all = np.array([r['ci'][1] for r in rows])
    c_all = np.array([r['chain'] for r in rows])
    rows_m = chosen if (full and chosen) else rows      # the marked pairs
    t = np.array([r['t0'] + 0.5 for r in rows_m])
    s = np.array([r['slope'] for r in rows_m])
    lo = np.array([r['ci'][0] for r in rows_m]); hi = np.array([r['ci'][1] for r in rows_m])
    c = np.array([r['chain'] for r in rows_m])
    clo = np.array([r['chain_ci'][0] for r in rows_m]); chi = np.array([r['chain_ci'][1] for r in rows_m])
    f = plt.figure(figsize=(17, 11.5))
    gs = f.add_gridspec(2, 2, height_ratios=(1.25, 1), hspace=0.62, wspace=0.18, left=0.06, right=0.98, top=0.90, bottom=0.07)
    pa = f.add_subplot(gs[0, :]); pb = f.add_subplot(gs[1, 0]); pc = f.add_subplot(gs[1, 1])
    # (a) the gate form across the window
    pa.axhspan(*GATE, color='#eeeeee', zorder=0, label='gate 1 +/- 0.05')
    pa.axhspan(*V3B_BAND, color=COL['order1'], alpha=0.18, zorder=1,
               label=f'V3b band {V3B_BAND[0]:.3f}-{V3B_BAND[1]:.3f} (model-advection systematic)')
    pa.axhspan(*BASELINE['ci'], color=COL['order3'], alpha=0.22, zorder=2,
               label=f'M1 baseline {BASELINE["slope"]:.3f} [{BASELINE["ci"][0]:.3f}, {BASELINE["ci"][1]:.3f}] (hour 0-1)')
    pa.axhline(BASELINE['slope'], color=COL['order3'], lw=1.2, zorder=3)
    pa.axhline(1.0, color='black', lw=0.8, ls='--', zorder=3)
    if full:
        pa.fill_between(t_all, lo_all, hi_all, color=COL['grey'], alpha=0.25, lw=0, zorder=3,
                        label=f'all {len(rows)} pairs: 95% CI band')
        pa.plot(t_all, s_all, '-', color='black', lw=1.0, zorder=4,
                label=f'all {len(rows)} pairs: slope {sp["min"]:.3f}-{sp["max"]:.3f}, mean {sp["mean"]:.4f} (std {sp["std"]:.4f}), '
                      f'{sum(r["gate"] for r in rows)}/{len(rows)} pass the gate, chi2/dof {sp["chi2_per_dof"]:.1f}')
    if rb and all(str(r['t0']) in rb for r in rows):
        tr1 = np.array([rb[str(r['t0'])]['ols_trim1'] for r in rows])
        pa.plot(t_all, tr1, '--', color=COL['order5'], lw=1.2, zorder=4,
                label=f'all pairs, OLS with the top 1% |2F| pixels dropped (not the gate): {tr1.min():.3f}-{tr1.max():.3f}, '
                      f'mean {tr1.mean():.4f} (std {tr1.std(ddof=1):.4f})')
    pa.errorbar(t, s, yerr=[s - lo, hi - s], fmt='o', color=COL['red'], ecolor=COL['red'], capsize=4, ms=7, lw=1.5,
                zorder=5, label=f"marked pairs: form='discrete' (the gate), OLS +/- 32-cell block-bootstrap 95% CI, {N_BOOT} draws")
    spm = sp_chosen if (full and sp_chosen) else sp
    pa.axhline(spm['weighted_mean'], color=COL['red'], lw=0.9, ls=':', zorder=4,
               label=f'marked pairs: weighted mean {spm["weighted_mean"]:.4f}, spread {spm["min"]:.4f}-{spm["max"]:.4f} '
                     f'(std {spm["std"]:.4f}), chi2/dof {spm["chi2_per_dof"]:.1f}')
    ylo = min(GATE[0], float(lo_all.min())) - 0.01
    for k, (r, x, y) in enumerate(zip(rows_m, t, s)):
        pa.text(x, hi[k] + 0.003, f'{y:.4f}', fontsize=7.5, ha='center', va='bottom', color=COL['red'])
        pa.text(x, ylo + 0.003 + (0.016 if k % 2 else 0.0), LABELS.get(r['t0'], ''), fontsize=6.5, ha='center',
                va='bottom', color='#444444')
    for d in (24, 48):
        pa.axvline(d, color='#888888', lw=0.6, ls=':')
    pa.set_xlim(-2, 73); pa.set_ylim(ylo, GATE[1] + 0.005)
    pa.set_xlabel('first hour of the pair (hours since 2012-07-02 00:00 UTC; dotted: day boundaries)')
    pa.set_ylabel('OLS slope of measured DG/Dt on 2F, front pixels')
    rep = sp['reproduction']
    pa.set_title(f'(a) V3 real-velocity null across the window, {len(rows)} pairs: {sum(r["gate"] for r in rows)}/{len(rows)} pass the gate, '
                 f'{sum(r["ci"][1] < 1 for r in rows)}/{len(rows)} CIs below 1, {sp["n_ci_overlap_baseline"]}/{len(rows)} CIs overlap the '
                 f'baseline band, {sp["n_inside_v3b_band"]}/{len(rows)} slopes inside the V3b band\n'
                 f'hour 0-1 reproduction {rep["hour0_slope"]:.4f} vs 0.9806: {"OK" if rep["ok"] else "MISMATCH"}'
                 + (f'; the default call (M0 two-hour store) is bit-identical: {default_check["identical"]}' if default_check else ''),
                 fontsize=10)
    pa.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    pa.grid(alpha=0.3)
    # (b) the chain form for reference
    pb.axhspan(*CHAIN_REF['ci'], color=COL['grey'], alpha=0.25, zorder=1,
               label=f"M1 form='chain' {CHAIN_REF['slope']:.3f} [{CHAIN_REF['ci'][0]:.3f}, {CHAIN_REF['ci'][1]:.3f}] (hour 0-1)")
    pb.axhline(CHAIN_REF['slope'], color=COL['grey'], lw=1.2)
    if full:
        pb.fill_between(t_all, [r['chain_ci'][0] for r in rows], [r['chain_ci'][1] for r in rows], color=COL['grey'],
                        alpha=0.25, lw=0)
        pb.plot(t_all, c_all, '-', color='black', lw=1.0, label=f"all {len(rows)} pairs, form='chain'")
    pb.errorbar(t, c, yerr=[c - clo, chi - c], fmt='s', color=COL['purple'], ecolor=COL['purple'], capsize=4, ms=6, lw=1.4,
                label="marked pairs, form='chain' (reference, not the gate)")
    pb.plot(t, [r['discrete_o2'] for r in rows_m], 'x', color=COL['order5'], ms=7, label="marked pairs, form='discrete_o2' (rejected variant)")
    pb.axhline(1.0, color='black', lw=0.8, ls='--')
    for d in (24, 48):
        pb.axvline(d, color='#888888', lw=0.6, ls=':')
    pb.set_xlim(-2, 73)
    pb.set_xlabel('first hour of the pair')
    pb.set_ylabel('OLS slope')
    pb.set_title(f"(b) reference forms: chain {sp['chain']['min']:.3f}-{sp['chain']['max']:.3f} (std {sp['chain']['std']:.3f}), "
                 f'moving with the gate form;\nfront pixels n {min(r["n"] for r in rows):,} of {rows[0]["n_valid"]:,} in every pair, '
                 f'corr {min(r["corr"] for r in rows):.3f}-{max(r["corr"] for r in rows):.3f}', fontsize=10)
    pb.legend(fontsize=7.5, loc='lower left'); pb.grid(alpha=0.3)
    # (c) the slope against the grid-scale velocity content of the pair
    if rough:
        x = np.array([rough.get(str(r['t0']), np.nan) for r in rows])
        ok = np.isfinite(x)
        cc = float(np.corrcoef(x[ok], s_all[ok])[0, 1])
        fit = np.polyfit(x[ok], s_all[ok], 1)
        day = np.array([r['t0'] // 24 for r in rows])
        for d, col, lab in ((0, COL['order3'], 'day 1 (07-02)'), (1, COL['order5'], 'day 2 (07-03)'), (2, COL['red'], 'day 3 (07-04)')):
            m = ok & (day == d)
            pc.errorbar(x[m], s_all[m], yerr=[s_all[m] - lo_all[m], hi_all[m] - s_all[m]], fmt='o', color=col, ecolor=col,
                        alpha=0.8, ms=5, lw=0.8, capsize=2, label=lab)
        xx = np.linspace(x[ok].min(), x[ok].max(), 10)
        pc.plot(xx, np.polyval(fit, xx), '--', color='black', lw=1.0,
                label=f'linear fit: slope {fit[0]:.2f} per unit, corr {cc:.2f}')
        pc.axhline(BASELINE['slope'], color=COL['order3'], lw=1.0)
        pc.axhspan(*BASELINE['ci'], color=COL['order3'], alpha=0.15, lw=0)
        pc.axhline(1.0, color='black', lw=0.8, ls='--')
        for r, xi, yi in zip(rows, x, s_all):
            if r['t0'] in PAIRS or yi < 0.955:
                pc.text(xi, yi, f' {r["t0"]}', fontsize=7, va='center', color='#444444')
        pc.set_xlabel('grid-scale content of the midpoint velocity on mask_analysis: rms(u_c - lowpass(u_c, L = 2)) / rms(u_c)')
        pc.set_ylabel("OLS slope, form='discrete'")
        pc.set_title(f'(c) slope against the grid-scale velocity content of the pair (corr {cc:.2f}): a weak dependence. The dips are '
                     f'leverage events --\na sharp front strip 7-12 cells inside the northern tile edge on day 3 (OLS 0.71-0.83 there, '
                     f'0.98 beyond 13 cells); numbers mark the first hour', fontsize=10)
        pc.legend(fontsize=7.5, loc='lower left'); pc.grid(alpha=0.3)
    else:
        pc.axis('off')
    f.suptitle('M2 task 3 (M2-Q3): stability of the V3 baseline over the 72-hour window -- same construction as M1 task 6 '
               '(G_mid >= p90 on mask_analysis, OLS gate, 32-cell blocks, order 3, vel_order 3)', fontsize=11.5)
    return _save(f, PNG)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pairs', default='', help='comma-separated first hours to (re)run this batch')
    ap.add_argument('--all', default='', help='a:b -- run every pair a..b-1 not yet cached (all 71 at ~2.5 s each)')
    ap.add_argument('--changes', default='',
                    help='comma-separated first hours on which to also record M1\'s diagnostic variants '
                         '(changes_tried: bilinear velocity, velocity / b low-passed L = 8, strain_seen)')
    ap.add_argument('--check-default', action='store_true',
                    help='run the unchanged default call (M0 two-hour store, hour 0) and compare with pair 0')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--roughness', action='store_true', help='grid-scale velocity content per cached pair (panel c)')
    ap.add_argument('--robust', default='', help='a:b -- leverage diagnostics (trimmed OLS, leave-one-block-out) for cached pairs a..b-1')
    args = ap.parse_args(argv)

    def say(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)

    results = _load()
    todo = [int(x) for x in args.pairs.split(',') if x]
    if args.all:
        a, b = (int(x) for x in args.all.split(':'))
        todo += [k for k in range(a, b) if str(k) not in results and k not in todo]
    changes = [int(x) for x in args.changes.split(',') if x]
    if todo or changes or args.check_default:
        g = ot.open_grid(with_face=True)
        for t0 in todo:
            results[str(t0)] = run_pair(t0, g, say, n_boot=args.n_boot)
            _dump(results)
        for t0 in changes:
            say(f'pair {t0}: M1\'s diagnostic variants (changes=True) ...')
            tic = time.time()
            r = _strip(va.test_discrete_null(velocities='llc', png=False, grid_ds=g, n_boot=200, changes=True,
                                             store=str(RAW72), t0=t0))
            if str(t0) not in results:
                results[str(t0)] = run_pair(t0, g, say, n_boot=args.n_boot)
            results[str(t0)]['changes_tried'] = r['changes_tried']
            results[str(t0)]['strain_seen'] = r['strain_seen']
            results[str(t0)]['first_attempt'] = r['first_attempt']['ols']
            say(f'pair {t0}: ' + '; '.join(f'{k}: {v:.4f}' for k, v in r['changes_tried'].items())
                + f'; strain seen {r["strain_seen"]["departure_vs_jacobian"]:.3f} / {r["strain_seen"]["jacobian_vs_fluxform"]:.3f} / '
                  f'{r["strain_seen"]["departure_vs_fluxform"]:.3f} [{time.time() - tic:.0f} s]')
            _dump(results)
        if args.check_default:
            say('default call: test_discrete_null("llc", changes=False) on M0\'s two-hour store ...')
            d = run_pair(0, g, say, store=None, n_boot=args.n_boot)
            ref = results.get('0')
            same = ref is not None and all(d['fits'][f][k] == ref['fits'][f][k]
                                           for f in ('chain', 'discrete_o2', 'discrete')
                                           for k in ('ols', 'intercept', 'corr', 'n', 'orthogonal', 'ratio')) \
                and d['ci'] == ref['ci'] and d['threshold'] == ref['threshold']
            results['_default_check'] = dict(identical=bool(same), default_slope=d['slope'], default_ci=d['ci'],
                                             pair0_slope=ref['slope'] if ref else None, default_n=d['n_front'],
                                             default_threshold=d['threshold'], wall_s=d['wall_s'])
            say(f'default call: slope {d["slope"]:.6f} ci {d["ci"]}; identical to the 72-h store pair 0: {same}')
            _dump(results)
    pairs = {k: v for k, v in results.items() if not k.startswith('_')}
    if args.roughness and pairs:
        import xarray as xr
        g = ot.open_grid(with_face=True)
        raw = xr.open_zarr(RAW72)
        rough = results.get('_roughness', {})
        for k in sorted(pairs, key=int):
            if k not in rough:
                rough[k] = roughness(int(k), g, raw)
        results['_roughness'] = rough
        _dump(results)
        say(f'roughness: {len(rough)} pairs, {min(rough.values()):.4f}-{max(rough.values()):.4f}')
    if args.robust and pairs:
        a, b = (int(x) for x in args.robust.split(':'))
        g = ot.open_grid(with_face=True)
        rb = results.get('_robust', {})
        tic = time.time()
        for k in sorted(pairs, key=int):
            if a <= int(k) < b and k not in rb:
                rb[k] = robust(int(k), g)
                r = rb[k]
                say(f'robust pair {k}: ols {r["ols"]:.4f} trim1 {r["ols_trim1"]:.4f} trim5 {r["ols_trim5"]:.4f} kurt {r["kurt_2F"]:.0f} '
                    f'LOO max shift {r["loo_max_shift"]:+.4f} top block share {r["top_block_share_sxx"]:.2f} at '
                    f'({r["top_block_lon"]:.1f}, {r["top_block_lat"]:.1f}), {r["top_block_min_edge_cells"]} cells from the edge [{time.time() - tic:.0f} s]')
                results['_robust'] = rb
                _dump(results)
    if not pairs:
        say('no pairs cached; run with --pairs')
        return 1
    rows = table(pairs)
    sp = spread(rows)
    chosen = [r for r in rows if r['t0'] in PAIRS]
    sp_chosen = spread(chosen) if chosen else None
    results['_table'] = rows
    results['_spread'] = sp
    results['_spread_chosen'] = sp_chosen
    results['_png'] = fig(rows, sp, results.get('_default_check'), chosen=chosen, sp_chosen=sp_chosen,
                          rough=results.get('_roughness'), rb=results.get('_robust'))
    _dump(results)
    say(f'wrote {results["_png"]}')
    print(f'{"pair":>7} {"hours (UTC)":>22} {"discrete":>8} {"CI":>18} {"n":>6} gate {"chain":>7} {"o2":>7} {"corr":>6} {"disp med/p99/max":>18}')
    for r in rows:
        print(f'{r["t0"]:2d}-{r["t0"] + 1:2d}   {r["hours"][0][5:16]:>22} {r["slope"]:8.4f} [{r["ci"][0]:.4f}, {r["ci"][1]:.4f}] {r["n"]:6d} '
              f'{"PASS" if r["gate"] else "FAIL"} {r["chain"]:7.4f} {r["discrete_o2"]:7.4f} {r["corr"]:6.3f} '
              f'{r["displacement"]["median"]:.3f}/{r["displacement"]["p99"]:.2f}/{r["displacement"]["max"]:.2f}')
    print('spread (all cached pairs):', json.dumps({k: v for k, v in sp.items() if not isinstance(v, list)}, indent=1))
    if sp_chosen and len(rows) > len(chosen):
        print('spread (the chosen pairs):', json.dumps({k: v for k, v in sp_chosen.items() if not isinstance(v, list)}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
