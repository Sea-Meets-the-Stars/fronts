""" M2-Q7 (JXP option (a)): is the day-3 V3 failure at the northern tile
edge tile-edge contamination (``edge_cells = 7`` too narrow) or a real
front?  -> figs/m2_q7_edge_margin.png

1. ``--pairs a:b``: the V3 real-velocity discrete null of M2 task 3
   (``m2_baseline_stability.py``) on every hour pair, refitted on
   ``masking.analysis_mask(edge_cells=E)`` for E in ``EDGES``, built in
   memory.  The null step (``validate.null_step``) does not depend on the
   mask, so it is computed once per pair and the pre-declared fit of
   ``validate._fit_llc`` is applied per mask (same lines: valid =
   mask & finite on every side, ``front_pixels`` = ``G_mid >= p90``
   **recomputed on each mask**, OLS + 32-cell block bootstrap, 1000 draws,
   forms ``chain``, ``discrete_o2``, ``discrete`` = the gate).  E = 7 must
   reproduce ``data/m2_v3_stability.json`` exactly.
2. ``--crop``: the crop test on the worst pairs: the null step on the tile
   cropped by N cells at the northern edge (low ``i`` on face 10) vs the
   full tile, at the same cells.
3. ``--motion`` / ``--support``: does the low-slope strip move with the
   flow over hours 55-70, and how far does the departure support reach
   towards the northern edge.
4. no flag: the figure and summary.

Caches go to ``--cache`` (default the session scratchpad given below, or
``data/``); every python run under ``timeout 300``.
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
import masking as mk                                             # noqa: E402
import semilag as sl                                             # noqa: E402
from validate_figs import COL, _save                             # noqa: E402

RAW72 = ot.DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'
TASK3 = ot.DATA_DIR / 'm2_v3_stability.json'
EDGES = (7, 10, 13, 16)
FORMS = ('chain', 'discrete_o2', 'discrete')
NPAIRS = 71
NORTH_J = (160, 192)                  # the 32-cell block of task 3 (lon -124.3, lat 38.0), i = 0-31
BANDS = ((7, 10), (10, 13), (13, 16), (16, 20), (20, 32))   # distance from the northern edge (i)
CROPS = (4, 8, 16)
CACHE = Path('/private/tmp/claude-501/-Users-xavier-Oceanography-python-llc4320-native-grid-'
             'preprocessing/3436e395-e6fd-4557-ad52-7e1e69c69dee/scratchpad/m2q7')


def ols(x, y):
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))


def masks_by_edge(g):
    out = {E: mk.analysis_mask(g, edge_cells=E) for E in EDGES}
    ref = mk.open_masks(ot.DATA_DIR / 'tile330_masks.nc')['mask_analysis'].values
    assert np.array_equal(out[7], ref), 'analysis_mask(edge_cells=7) != tile330_masks.nc'
    return out


def fit_mask(st, ana, block, n_boot=va.N_BOOT):
    """``validate._fit_llc``'s fit, line for line, on a given mask."""
    valid = ana & np.isfinite(st['measured']) & np.isfinite(st['G_mid'])
    for f in FORMS:
        valid &= np.isfinite(st[f'two_F_{f}'])
    front, thr = va.front_pixels(st['G_mid'], valid)
    fits = {f: va._fit(st[f'two_F_{f}'][front], st['measured'][front], block[front], n_boot) for f in FORMS}
    return front, thr, valid, fits


def run_pairs(pairs, g, cache):
    cache.mkdir(parents=True, exist_ok=True)
    masks = masks_by_edge(g)
    for t0 in pairs:
        fn = cache / f'pair_{t0:02d}.json'
        if fn.exists():
            continue
        tic = time.time()
        inp = va._llc_inputs(g, store=str(RAW72), t0=t0)
        st = va.null_step(inp['b_t'], inp['U'], inp['V'], inp['g'], inp['grid'], order=3, forms=FORMS,
                          vel_order=3)
        rec = dict(t0=t0, hours=inp['hours'], edges={})
        x_all, y_all = st['two_F_discrete'], st['measured']
        ii = np.arange(x_all.shape[1])[None, :] * np.ones((x_all.shape[0], 1), int)
        jj = np.arange(x_all.shape[0])[:, None] * np.ones((1, x_all.shape[1]), int)
        for E in EDGES:
            front, thr, valid, fits = fit_mask(st, masks[E], inp['block'])
            x, y = x_all[front], y_all[front]
            keep = np.abs(x) < np.percentile(np.abs(x), 99)
            nb = front & (jj >= NORTH_J[0]) & (jj < NORTH_J[1])
            bands = {}
            for a, b in BANDS:
                m = nb & (ii >= a) & (ii < b)
                bands[f'{a}-{b - 1}'] = dict(n=int(m.sum()), ols=ols(x_all[m], y_all[m]) if m.sum() > 5 else None)
            fd = fits['discrete']
            rec['edges'][str(E)] = dict(
                slope=fd['ols'], ci=fd['bootstrap']['ci'], se=fd['bootstrap']['se'], n_front=fd['n'],
                n_valid=int(valid.sum()), n_mask=int(masks[E].sum()), threshold=thr,
                passed=fd['gate']['passed'], corr=fd['corr'], orthogonal=fd['orthogonal'],
                chain=fits['chain']['ols'], discrete_o2=fits['discrete_o2']['ols'],
                ols_trim1=ols(x[keep], y[keep]), north_block_bands=bands,
                north_block_n_front=int(nb.sum()))
        # the northern strip (i < 48) for the motion test
        sl_ = np.s_[:, :48]
        np.savez_compressed(cache / f'strip_{t0:02d}.npz', G_mid=st['G_mid'][sl_], meas=st['measured'][sl_],
                            twoF=st['two_F_discrete'][sl_], di=st['di'][sl_], dj=st['dj'][sl_],
                            front7=fit_mask(st, masks[7], inp['block'], n_boot=10)[0][sl_])
        rec['wall_s'] = time.time() - tic
        tmp = fn.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(fn)
        e = rec['edges']
        print(f'pair {t0:2d}: ' + ' | '.join(f'E{E} {e[str(E)]["slope"]:.4f} {"P" if e[str(E)]["passed"] else "F"}'
                                            for E in EDGES) + f' | {rec["wall_s"]:.1f} s', flush=True)


def reproduction(cache):
    t3 = json.loads(TASK3.read_text())
    worst = 0.0
    for t0 in range(NPAIRS):
        r = json.loads((cache / f'pair_{t0:02d}.json').read_text())['edges']['7']
        a = t3[str(t0)]
        for k_new, k_old in (('slope', 'slope'), ('n_front', 'n_front'), ('threshold', 'threshold')):
            worst = max(worst, abs(r[k_new] - a[k_old]))
        worst = max(worst, abs(r['ci'][0] - a['ci'][0]), abs(r['ci'][1] - a['ci'][1]),
                    abs(r['chain'] - a['fits']['chain']['ols']))
        rb = t3['_robust'].get(str(t0)) if isinstance(t3.get('_robust'), dict) else None
        if rb is not None:
            worst = max(worst, abs(r['ols_trim1'] - rb['ols_trim1']))
    return worst


def summary(cache):
    rows = [json.loads((cache / f'pair_{t0:02d}.json').read_text()) for t0 in range(NPAIRS)]
    out = {}
    for E in EDGES:
        s = np.array([r['edges'][str(E)]['slope'] for r in rows])
        tr = np.array([r['edges'][str(E)]['ols_trim1'] for r in rows])
        se = np.array([r['edges'][str(E)]['se'] for r in rows])
        p = [r['t0'] for r in rows if not r['edges'][str(E)]['passed']]
        out[E] = dict(n_pass=int(NPAIRS - len(p)), failing=p, mean=float(s.mean()), std=float(s.std(ddof=1)),
                      min=float(s.min()), max=float(s.max()), argmin=int(s.argmin()),
                      weighted_mean=float(np.sum(s / se ** 2) / np.sum(1 / se ** 2)),
                      day3={int(t): float(s[t]) for t in (36, 62, 63, 64, 65, 66, 67)},
                      trim_mean=float(tr.mean()), trim_std=float(tr.std(ddof=1)), trim_min=float(tr.min()),
                      trim_max=float(tr.max()), n_front=rows[0]['edges'][str(E)]['n_front'],
                      n_mask=rows[0]['edges'][str(E)]['n_mask'], hour0=float(s[0]),
                      hour0_ci=rows[0]['edges'][str(E)]['ci'])
    return rows, out


# ---------------------------------------------------------------------------
# crop test
# ---------------------------------------------------------------------------
def crop_test(t0, g, N):
    """Null step on the full tile and on the tile with the first N columns
    (the northern edge, low i) removed; differences at the same cells as a
    function of distance from the *full-tile* edge."""
    inp = va._llc_inputs(g, store=str(RAW72), t0=t0)
    st = va.null_step(inp['b_t'], inp['U'], inp['V'], inp['g'], inp['grid'], order=3, forms=('discrete',))
    gc = inp['g'].isel(i=slice(N, None), i_g=slice(N, None))
    gridc = ot.build_xgcm(gc)
    stc = va.null_step(inp['b_t'].isel(i=slice(N, None)), inp['U'].isel(i_g=slice(N, None)),
                       inp['V'].isel(i=slice(N, None)), gc, gridc, order=3, forms=('discrete',))
    out = dict(t0=t0, N=N, cols={})
    j0, j1 = 16, 704                                    # away from the j edges
    for name in ('G_mid', 'two_F_discrete', 'measured'):
        full, part = st[name][j0:j1, N:], stc[name][j0:j1]
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = np.abs(part - full) / np.nanmax(np.abs(full), axis=0, keepdims=True)
            bad = ~np.isclose(full, part, rtol=1e-9, atol=0, equal_nan=True)
        prof = []
        for c in range(0, 40 - N if 40 - N > 0 else 1):
            fin = np.isfinite(full[:, c])
            prof.append(dict(dist_full=N + c, dist_new=c, frac_changed=float(bad[:, c].sum() / max(fin.sum(), 1)),
                             max_rel=float(np.nanmax(rel[:, c])) if np.isfinite(rel[:, c]).any() else None,
                             n_nan_new=int((np.isfinite(full[:, c]) & ~np.isfinite(part[:, c])).sum())))
        out['cols'][name] = prof
        # the deepest column (from the new edge) still holding a changed cell
        changed = [p['dist_new'] for p in prof if p['frac_changed'] > 0]
        out[f'{name}_max_changed_dist_new'] = max(changed) if changed else None
    # slopes in bands on the full-tile front pixels of the northern block (E = 7)
    masks = masks_by_edge(g)
    front, *_ = fit_mask(st | {'two_F_chain': st['two_F_discrete'], 'two_F_discrete_o2': st['two_F_discrete']},
                         masks[7], inp['block'], n_boot=10)
    bands = {}
    for a, b in BANDS:
        if a < N:
            continue
        m = np.zeros_like(front)
        m[NORTH_J[0]:NORTH_J[1], a:b] = True
        m &= front
        mc = m[:, N:]
        xf, yf = st['two_F_discrete'][m], st['measured'][m]
        xc, yc = stc['two_F_discrete'][mc], stc['measured'][mc]
        ok = np.isfinite(xc) & np.isfinite(yc)
        bands[f'{a}-{b - 1}'] = dict(n=int(m.sum()), n_ok_cropped=int(ok.sum()), full=ols(xf, yf),
                                     cropped=ols(xc[ok], yc[ok]) if ok.sum() > 5 else None,
                                     full_same=ols(xf[ok], yf[ok]) if ok.sum() > 5 else None)
    out['north_bands'] = bands
    return out


# ---------------------------------------------------------------------------
# support test
# ---------------------------------------------------------------------------
def support(cache, masks):
    """Departure support of measured_DGDt towards the northern edge on the
    analysis mask and the northern block: the lowest i index touched is
    floor(i - di) - 1 (order-3 nodes start at floor - 1 for a 4-node
    kernel: ks = -1..2) - 1 (the W stencil neighbour), and departure_index's
    own midpoint interpolation reaches floor(i - di/2) - 1."""
    out = {}
    for E in EDGES:
        ana = masks[E][:, :48]
        rows = []
        for t0 in range(NPAIRS):
            z = np.load(cache / f'strip_{t0:02d}.npz')
            di = z['di']
            ii = np.arange(48)[None, :] + np.zeros((720, 1))
            m = ana & np.isfinite(di)
            low = np.floor(ii - di) - 1 - 1
            mb = m.copy(); mb[:NORTH_J[0]] = False; mb[NORTH_J[1]:] = False
            rows.append(dict(t0=t0, min_node_mask=int(low[m].min()), min_node_block=int(low[mb].min()) if mb.any() else None,
                             max_toward_edge_block=float(di[mb].max()) if mb.any() else None,
                             max_toward_edge_mask_northrow=float(di[m & (ii < E + 3)].max())))
        out[E] = rows
    return out


# ---------------------------------------------------------------------------
# motion test
# ---------------------------------------------------------------------------
def motion(cache, pairs=range(55, NPAIRS)):
    """Per pair, in the northern block region (j 140-230, i 0-47): the
    position of the strongest G_mid (front axis) and of the largest
    |measured - 2F| residual on E = 7 front pixels, plus the band slopes."""
    out = []
    for t0 in pairs:
        z = np.load(cache / f'strip_{t0:02d}.npz')
        G, res, fr = z['G_mid'], z['meas'] - z['twoF'], z['front7']
        win = np.s_[140:230, :]
        Gw = np.where(np.isfinite(G[win]), G[win], 0.0)
        # front axis: G-weighted mean i per j (columns >= 7 only, the analysis side)
        Gw7 = Gw.copy(); Gw7[:, :7] = 0
        top = Gw7 >= np.percentile(Gw7[:, 7:], 99)
        jt, it = np.where(top)
        rw = np.where(fr[win] & np.isfinite(res[win]), np.abs(res[win]), 0.0)
        topr = rw >= np.percentile(rw[rw > 0], 99) if (rw > 0).any() else rw > 0
        jr, ir = np.where(topr & (rw > 0))
        out.append(dict(t0=t0, G_top_i_median=float(np.median(it)), G_top_j_median=float(140 + np.median(jt)),
                        G_top_i_p10_p90=[float(np.percentile(it, 10)), float(np.percentile(it, 90))],
                        res_top_i_median=float(np.median(ir)) if ir.size else None,
                        res_top_j_median=float(140 + np.median(jr)) if jr.size else None,
                        res_top_i_p10_p90=[float(np.percentile(ir, 10)), float(np.percentile(ir, 90))] if ir.size else None,
                        mean_di_block=float(np.nanmean(z['di'][160:192, 7:32])),
                        mean_dj_block=float(np.nanmean(z['dj'][160:192, 7:32]))))
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(rows, summ, crops):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[2.2, 1.2, 1.2]))
    ax = axs[0]
    t = np.arange(NPAIRS)
    cols = {7: COL.get('ink', 'k') if isinstance(COL, dict) else 'k'}
    palette = ['#222222', '#1f77b4', '#d62728', '#2ca02c']
    for k, E in enumerate(EDGES):
        s = np.array([r['edges'][str(E)]['slope'] for r in rows])
        ax.plot(t, s, '-o', ms=3, lw=1.2, color=palette[k], label=f'edge_cells = {E}: '
                f'{summ[E]["n_pass"]}/71 pass, mean {summ[E]["mean"]:.3f}')
    ax.axhspan(0.95, 1.05, color='0.9', zorder=0, label='gate 1 ± 0.05')
    ax.axhline(0.981, color='0.5', ls=':', lw=1, label='M1 baseline 0.981')
    ax.set_xlabel('hour pair t0 (2012-07-02 00 UTC + t0 h)')
    ax.set_ylabel('V3 llc gate slope (OLS, discrete form)')
    ax.set_title('(a) V3 real-velocity null vs tile-edge margin')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.88, 1.02)
    # (b) crop: the largest change at the same cells vs distance from the new edge
    ax = axs[1]
    mk_ = {4: 'o', 8: 's', 16: '^'}
    pc = {63: '#1f77b4', 64: '#d62728', 0: '#888888'}
    for c in crops:
        for name, ls in (('measured', '-'), ('two_F_discrete', '--')):
            prof = [p for p in c['cols'][name] if p['dist_new'] <= 24]
            d = [p['dist_new'] for p in prof]
            # NaN-in-cropped cells shown as 1 (the edge removes them); else max |change| / column max
            v = [1.0 if p['n_nan_new'] > 0 else max(p['max_rel'] or 0.0, 1e-16) for p in prof]
            ax.plot(d, v, ls, marker=mk_[c['N']], ms=3, lw=0.8, color=pc.get(c['t0'], 'k'),
                    label=f'pair {c["t0"]}, N = {c["N"]}, {"DG/Dt" if name == "measured" else "2F"}')
    ax.set_yscale('log')
    ax.set_ylim(1e-17, 3)
    ax.axvline(7, color='0.5', ls=':')
    ax.text(7.3, 1e-2, 'edge_cells = 7', fontsize=7, color='0.4')
    ax.set_xlabel('distance from the cropped (new) northern edge, cells')
    ax.set_ylabel('max |cropped - full| / max |full| per column\n(1 = cell lost to NaN)')
    ax.set_title('(b) crop test: the edge reaches <= 5 cells, as NaN')
    ax.legend(fontsize=5, ncol=2)
    # (c) the raw Theta across the northern edge: the front is in the model data
    ax = axs[2]
    import xarray as xr
    th = xr.open_zarr(RAW72).Theta
    for t, col in ((0, '#888888'), (40, '#2ca02c'), (52, '#1f77b4'), (64, '#d62728'), (70, '#ff7f0e')):
        a = np.asarray(th.isel(time=t).values).squeeze()
        ax.plot(np.arange(40), a[176, :40], '-o', ms=2.5, lw=1, color=col, label=f'hour {t}')
    for E, ls in ((7, '-'), (10, '--'), (13, ':')):
        ax.axvline(E, color='k', ls=ls, lw=0.8)
    ax.set_xlabel('i, cells from the northern tile edge (j = 176, lon -124.3)')
    ax.set_ylabel('Theta, raw model, deg C')
    ax.set_title('(c) the day-3 strip is a 2 C front in the raw data')
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, 'm2_q7_edge_margin.png')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', default=None)
    p.add_argument('--crop', default=None, help='t0 list, e.g. 63,64')
    p.add_argument('--cache', default=str(CACHE))
    p.add_argument('--analyse', action='store_true')
    a = p.parse_args(argv)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    g = ot.open_grid(with_face=True)
    g = g if 'face' in g.dims else g.expand_dims('face')
    if a.pairs:
        lo, hi = (int(x) for x in a.pairs.split(':'))
        run_pairs(range(lo, hi), g, cache)
        return
    if a.crop:
        for t0 in (int(x) for x in a.crop.split(',')):
            for N in CROPS:
                fn = cache / f'crop_{t0:02d}_{N:02d}.json'
                if fn.exists():
                    continue
                r = crop_test(t0, g, N)
                fn.write_text(json.dumps(r, indent=1))
                print(f'crop pair {t0} N {N}: deepest changed col from new edge: '
                      f'G {r["G_mid_max_changed_dist_new"]}, 2F {r["two_F_discrete_max_changed_dist_new"]}, '
                      f'meas {r["measured_max_changed_dist_new"]} | bands ' +
                      ', '.join(f'{k}: {v["full"]:.3f}->{v["cropped"] if v["cropped"] is None else round(v["cropped"], 3)}'
                                for k, v in r['north_bands'].items()), flush=True)
        return
    rows, summ = summary(cache)
    print('reproduction max |diff| vs task 3:', reproduction(cache))
    masks = masks_by_edge(g)
    res = dict(summary={str(k): v for k, v in summ.items()}, support={str(k): v for k, v in support(cache, masks).items()},
               motion=motion(cache))
    crops = [json.loads(f.read_text()) for f in sorted(cache.glob('crop_*.json'))]
    res['crops'] = crops
    (cache / 'analysis.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['summary'], indent=1))
    if crops:
        print('png:', figure(rows, summ, crops))


if __name__ == '__main__':
    main()
