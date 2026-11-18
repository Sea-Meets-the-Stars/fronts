""" M3-digging task 5: a second tile, surface-only (vet report §6 item 5c; M3-Q30 (a),
M3D-Q2 (a): tile 329; M3D-Q5 (a): the same 72 h).

Tile 329 is tile 330's western neighbour on the rect grid: face 7, face-local ``j``
3600:4320, ``i`` 2880:3600, longitude 143 W to 128 W at the same latitudes (26.7-38.3 N) --
the open NE Pacific with **no land in the tile**.  The budget on it is the surface-only,
three-term budget (``measured``, ``2F``, ``tau``; the chunk terms absent, ``--chunk none``):
criterion 1 is read against the product's **two-term catch-all**, not its five-term ratio.

Steps (each logged with its numbers; ``--step`` runs one):
1. ``resolve``: ``osn_tiles.tile_spec(i_rect, j_rect)`` -> the ``TileInfo``;
   ``grid``: the static grid from OSN (``load_grid(tile=)``, ``write_grid``) with the
   orientation (``CS``, ``SN``), spacing, extent and land count;
2. ``masks``: ``masking.build_masks`` with the product's parameters (halo 7, 100 km, edge 7);
   with no land ``coast_distance_km`` is +inf and ``mask_offshore`` all-true (recorded);
3. ``pull``: ``osn_tiles.pull_series`` of the 72 h, resumable, with this tile's grid for
   ``XC``/``YC``; ``verify_series`` after;
   ``qa``: the orientation QA -- the V3 discrete null (``validate.test_discrete_null``,
   ``'llc'``, the ``discrete`` form) on this tile's hour 0 -> 1 with this tile's masks;
   **slope within [0.95, 1.05] is the acceptance**, and the sweep does not run without it;
4. the sweep and closure are the generic tools: ``m3_run.py --raw ... --grid ... --masks ...
   --chunk none --tag tile329`` and ``m3_closure.py --tag tile329 --grid ... --masks ...``;
5. ``compare``: one table per ``L`` from the two closure summaries, the front population
   from the two stores, ``figs/m3_tile2.png``, ``data/m3_tile2_summary.json``.

Run (from dev/frontogenesis):
    <python> py/m3_tile2.py --step prep            # grid, masks, hours 0-1, the QA  (~2 min)
    nohup <python> py/m3_tile2.py --step pull > data/m3_tile2_pull.nohup 2>&1 &
    nohup <python> py/m3_run.py --raw data/tile329_raw_20120702T00_72h.zarr --grid data/tile329_grid.zarr \\
          --masks data/tile329_masks.nc --chunk none --tag tile329 > data/m3_run_tile329.nohup 2>&1 &
    <python> py/m3_closure.py --tag tile329 --grid data/tile329_grid.zarr --masks data/tile329_masks.nc
    <python> py/m3_tile2.py --step compare
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget as bg                                            # noqa: E402
import inputs as inp                                           # noqa: E402
import masking as mk                                           # noqa: E402
import osn_tiles as ot                                         # noqa: E402
import series_verify as sv                                     # noqa: E402
from m2_pull import timestamps_72                              # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

FIG_DIR = HERE.parent / 'figs'
NAME = 'tile329'
RECT = (12600, 9720)                           # the rect pixel that resolves tile 329 (M3D-Q2 (a))
TAG = NAME
QA_BAND = (0.95, 1.05)                         # the orientation acceptance on the V3 llc slope
FACE10 = dict(CS=0.0, SN=-1.0)
OUT_JSON = DATA_DIR / 'm3_tile2_summary.json'
OUT_PNG = FIG_DIR / 'm3_tile2.png'
LOG_PATH = DATA_DIR / 'm3_tile2.log'
DECLARED = dict(
    source='frontogenesis_prompt_M3_digging.md task 5; M3-Q30 (a), M3D-Q2 (a) tile 329, M3D-Q5 (a) '
           'the same 72 h; fixed 2026-11-02 before m3_tile2.py was run',
    tile=NAME, rect_pixel=list(RECT), window='2012-07-02 00 .. 2012-07-04 23 UTC (timestamps_72)',
    budget='surface-only: measured - 2F - tau; the chunk terms ABSENT (--chunk none)',
    criterion1='the tile\'s catch-all against the PRODUCT\'s two-term catch-all (0.739 / 0.728 / '
               '0.713 / 0.685), and the gate\'s 0.50 as for the product',
    orientation_qa=dict(test='V3 discrete null (validate.test_discrete_null, llc, discrete form) on '
                             'hour 0 -> 1 of this tile with this tile\'s masks',
                        acceptance=f'OLS slope of measured on 2F within {list(QA_BAND)}',
                        gate='the sweep does not run without it'),
    masks=dict(halo_cells=7, min_km=100.0, edge_cells=7, no_land='coast_distance_km = +inf, '
               'mask_offshore all-true, the Gulf-of-California check not run (tile 330\'s)'),
    no_chunk_pull='M3-Q30 (b) declined: no full-depth chunk store for this tile')


def paths(name=NAME) -> dict:
    return dict(grid=DATA_DIR / f'{name}_grid.zarr', masks=DATA_DIR / f'{name}_masks.nc',
                raw=DATA_DIR / f'{name}_raw_20120702T00_72h.zarr',
                closure=DATA_DIR / f'm3_closure_{name}_summary.json',
                done=DATA_DIR / f'm3_tile2_{name}_done.json')


class Say:
    def __init__(self, path=LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, 'a')

    def __call__(self, msg):
        line = f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}  {msg}'
        self.fh.write(line + '\n')
        self.fh.flush()
        print(line, flush=True)

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# 1. resolve and the grid
# ---------------------------------------------------------------------------
def resolve(rect=RECT) -> dict:
    t = ot.tile_spec(*rect)
    return dict(tile_idx=int(t.tile_idx), face=int(t.face_idx),
                j_face=[t.j_face_slice.start, t.j_face_slice.stop],
                i_face=[t.i_face_slice.start, t.i_face_slice.stop],
                rect_j=[t.rect_j_slice.start, t.rect_j_slice.stop],
                rect_i=[t.rect_i_slice.start, t.rect_i_slice.stop], rect_pixel=list(rect),
                tile=t)


def grid_step(tile, path, say=print, clobber=False) -> tuple:
    if path.exists() and not clobber:
        g = ot.open_grid(path, with_face=True)
        say(f'grid: {path.name} on disk')
    else:
        t = time.time()
        g = ot.load_grid(tile=tile)
        ot.write_grid(g, out=path, clobber=clobber)
        say(f'grid: pulled and written {path.name} in {time.time() - t:.0f} s')
        g = ot.open_grid(path, with_face=True)
    cs, sn = float(np.nanmedian(g['CS'].values)), float(np.nanmedian(g['SN'].values))
    sp = mk.spacing_km(g)
    oc = mk.ocean_mask(g)
    info = dict(path=str(path), face=int(g.attrs.get('face_index', -1)), CS=cs, SN=sn,
                same_orientation_as_face10=bool(abs(cs - FACE10['CS']) < 1e-6 and abs(sn - FACE10['SN']) < 1e-6),
                orientation=str(g.attrs.get('orientation', '')),
                lon=[float(g['XC'].min()), float(g['XC'].max())], lat=[float(g['YC'].min()), float(g['YC'].max())],
                spacing_km={k: round(v, 4) for k, v in sp.items()},
                n_cells=int(oc.size), n_ocean=int(oc.sum()), n_land=int((~oc).sum()),
                depth_m=[float(np.nanmin(g['Depth'].values)), float(np.nanmax(g['Depth'].values))])
    say(f'grid: face {info["face"]}, CS {cs:.3g}, SN {sn:.3g} (face 10\'s: {info["same_orientation_as_face10"]}); '
        f'lon {info["lon"][0]:.2f}..{info["lon"][1]:.2f}, lat {info["lat"][0]:.2f}..{info["lat"][1]:.2f}; '
        f'dxC median {sp["dxC_median_km"]:.3f} km, dyC {sp["dyC_median_km"]:.3f} km; land cells {info["n_land"]}')
    return g, info


# ---------------------------------------------------------------------------
# 2. the masks
# ---------------------------------------------------------------------------
def masks_step(g, path, say=print, clobber=False) -> tuple:
    if path.exists() and not clobber:
        m = mk.open_masks(path)
        say(f'masks: {path.name} on disk')
    else:
        m = mk.build_masks(g, halo_cells=7, min_km=100.0, edge_cells=7)
        mk.write_masks(m, out=path, clobber=clobber)
        m = mk.open_masks(path)
        say(f'masks: built and written {path.name}')
    d = m['coast_distance_km'].values
    info = dict(path=str(path), **{k: int(m[k].values.sum()) for k in
                                   ('mask_ocean', 'mask_halo', 'mask_offshore', 'mask_edge', 'mask_analysis')},
                coast_distance=('+inf everywhere (no land)' if np.isposinf(d[np.isfinite(d) | np.isposinf(d)]).all()
                                else f'finite, median {float(np.nanmedian(d)):.1f} km'),
                face_index=int(m.attrs.get('face_index', -1)), n_analysis_tile330=262925)
    say(f'masks: analysis cells {info["mask_analysis"]:,} (tile 330: 262,925); ocean {info["mask_ocean"]:,}, '
        f'halo {info["mask_halo"]:,}, offshore {info["mask_offshore"]:,}, edge {info["mask_edge"]:,}; '
        f'coast distance {info["coast_distance"]}')
    return m, info


# ---------------------------------------------------------------------------
# 3. the pull, the verification, the orientation QA
# ---------------------------------------------------------------------------
def pull_step(tile, g, path, hours=None, say=print) -> dict:
    ts = timestamps_72()
    if hours is not None:
        ts = [ts[h] for h in hours]
    rep = {}
    t = time.time()
    ot.pull_series(ts, path, tile=tile, grid_ds=g, log=say, report=rep)
    rep['wall_total_s'] = round(time.time() - t, 1)
    walls = list(rep.get('wall_s', {}).values())
    rep['wall_per_hour'] = (dict(median=float(np.median(walls)), min=float(np.min(walls)),
                                 max=float(np.max(walls)), n=len(walls)) if walls else None)
    rep['bytes'] = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    say(f'pull: {len(rep.get("pulled", []))} pulled, {len(rep.get("skipped", []))} skipped, '
        f'{len(rep.get("failed", []))} failed in {rep["wall_total_s"]:.0f} s; '
        f'{rep["bytes"] / 1e6:.0f} MB on disk')
    return rep


def verify_step(path, g, n_hours=None, say=print) -> dict:
    ts = timestamps_72()
    if n_hours is not None:
        ts = ts[:n_hours]
    res = sv.verify_series(path, ts, grid_ds=g.squeeze('face') if 'face' in g.dims else g)
    say('verify: ' + sv.summarize(res).splitlines()[0])
    return dict(ok=bool(res.get('ok')), checks={k: (v.get('ok') if isinstance(v, dict) else v)
                                                 for k, v in res.items() if k not in ('ok', 'path')})


def qa_step(g, raw, masks, say=print, n_boot=1000) -> dict:
    import validate as va
    t = time.time()
    res = va.test_discrete_null(velocities='llc', png=False, order=3, forms=('discrete',),
                                grid_ds=g, store=raw, t0=0, masks=masks, changes=False, n_boot=n_boot)
    slope, ci = float(res['slope']), [float(x) for x in res['ci']]
    out = dict(slope=slope, ci=ci, n_front=int(res['n_front']), band=list(QA_BAND),
               passed=bool(QA_BAND[0] <= slope <= QA_BAND[1]),
               hours=res.get('hours'), tile330_reference=dict(slope=0.9806, ci=[0.9698, 0.9942]),
               displacement_cells=res.get('displacement_cells'), wall_s=round(time.time() - t, 1))
    say(f'orientation QA: V3 llc discrete slope {slope:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] on {out["n_front"]:,} '
        f'front pixels (tile 330: 0.9806 [0.9698, 0.9942]) -> {"PASS" if out["passed"] else "FAIL"} '
        f'(band {QA_BAND}); {out["wall_s"]} s')
    return out


# ---------------------------------------------------------------------------
# 5. the comparison
# ---------------------------------------------------------------------------
def _nums(S, L):
    c = S['per_L'][str(L)]['closure']
    return dict(resid_over_meas=c['residual']['over_measured'], catchall_over_meas=c['catchall']['over_measured'],
                explained=c['residual']['explained'], catchall_explained=c['catchall']['explained'],
                resid_on_2F=c['residual']['slope_on_2F'],
                euler_gap=c['measured_coherence']['rms_euler_minus_semilag'],
                o5_gap=c['measured_coherence']['rms_o5_minus_o3'],
                subfilter_over_2F=S['per_L'][str(L)]['subfilter']['over_two_F'],
                two_F_over_meas=c['terms']['two_F']['over_measured'],
                meas_on_2F=c['measured_on_2F'], n=c['n'], rms_measured=c['rms_measured'],
                rms_two_F=c['rms_two_F'], euler=S['per_L'][str(L)]['euler'],
                verdict=c['verdict']['passed'], n_pairs=c['n_pairs'])


def front_population(tag, Ls=(0, 2, 4, 8)) -> dict:
    """The p90 G threshold (median over pairs) and the front-pixel count per
    pair, from the derived stores."""
    out = {}
    for L in Ls:
        ds = xr.open_zarr(bg.derived_path(L, tag=tag))
        thr, nf = [], []
        for p in range(ds.sizes['time']):
            h = ds.isel(time=p)
            v = np.asarray(h['valid'].values, bool)
            G = np.asarray(h['G'].values, 'float64')
            thr.append(float(np.percentile(G[v], 90.0)))
            nf.append(int(np.asarray(h['front'].values, bool).sum()))
        ds.close()
        out[str(L)] = dict(G_p90_threshold_median=float(np.median(thr)), n_front_median=float(np.median(nf)),
                           n_front_total=int(np.sum(nf)))
    return out


def compare(name=NAME, say=print) -> dict:
    P = paths(name)
    prod = json.loads((DATA_DIR / 'm3_closure_summary.json').read_text())
    tile = json.loads(P['closure'].read_text())
    Ls = [int(L) for L in prod['per_L']]
    out = dict(per_L={})
    say('front population: the product')
    pop_prod = front_population(None, Ls)
    say(f'front population: {name}')
    pop_tile = front_population(TAG, Ls)
    for L in Ls:
        a, b = _nums(prod, L), _nums(tile, L)
        row = dict(product=a, tile=b, product_front=pop_prod[str(L)], tile_front=pop_tile[str(L)],
                   tile_minus_product_catchall=b['catchall_over_meas'] - a['catchall_over_meas'],
                   tile_catchall_passes_gate=bool(b['catchall_over_meas'] <= 0.5),
                   G_p90_ratio_tile_over_product=pop_tile[str(L)]['G_p90_threshold_median'] / pop_prod[str(L)]['G_p90_threshold_median'],
                   n_front_ratio=pop_tile[str(L)]['n_front_total'] / pop_prod[str(L)]['n_front_total'])
        out['per_L'][str(L)] = row
        say(f'L={L}: resid/meas product 5-term {a["resid_over_meas"]:.3f} catch-all {a["catchall_over_meas"]:.3f} | '
            f'{name} catch-all {b["catchall_over_meas"]:.3f} (expl {b["explained"]:+.3f}; euler gap {b["euler_gap"]:.3f} '
            f'vs {a["euler_gap"]:.3f}; tau/2F {b["subfilter_over_2F"]:.3f} vs {a["subfilter_over_2F"]:.3f}; '
            f'm~2F {b["meas_on_2F"]["ols"]:.3f} vs {a["meas_on_2F"]["ols"]:.3f}) | G p90 thr ratio '
            f'{row["G_p90_ratio_tile_over_product"]:.2f}, n_front x{row["n_front_ratio"]:.2f}')
    out['verdict'] = dict(tile=tile['verdict'], note='criterion 1 on the tile is read against the product\'s '
                                                       'two-term catch-all, and against the 0.50 gate')
    return out


def figure(res, path, dpi=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Ls = [int(L) for L in res['compare']['per_L']]
    rows = res['compare']['per_L']
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for a, (k, lab) in zip(ax[0], (('catchall_over_meas', 'residual / measured'), ('explained', 'explained fraction'),
                                   ('euler_gap', 'Eulerian - semi-Lagrangian gap / measured'))):
        a.plot(Ls, [rows[str(L)]['product']['resid_over_meas' if k == 'catchall_over_meas' else k] for L in Ls], 'o-',
               color='k', label='tile 330, 5 terms')
        if k != 'euler_gap':
            a.plot(Ls, [rows[str(L)]['product']['catchall_over_meas' if k == 'catchall_over_meas' else 'catchall_explained'] for L in Ls],
                   's--', color='0.5', label='tile 330, 2-term catch-all')
        a.plot(Ls, [rows[str(L)]['tile'][k] for L in Ls], 'D-', color='#e76f51',
               label=f'{NAME} (no land), 2-term catch-all')
        if k == 'catchall_over_meas':
            a.axhline(0.5, color='#b22222', ls='--', lw=1, label='gate 0.50')
        if k == 'explained':
            a.axhline(0, color='0.7', lw=0.8)
        a.set_xlabel('L (cells)')
        a.set_ylabel(lab)
        a.set_xticks(Ls)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    a = ax[1, 0]
    w = 0.35
    x = np.arange(len(Ls))
    a.bar(x - w / 2, [rows[str(L)]['product']['subfilter_over_2F'] for L in Ls], w, color='0.4', label='tile 330')
    a.bar(x + w / 2, [rows[str(L)]['tile']['subfilter_over_2F'] for L in Ls], w, color='#e76f51', label=NAME)
    a.set_xticks(x)
    a.set_xticklabels([f'L={L}' for L in Ls])
    a.set_ylabel('tau / 2F (rms)')
    a.set_title('the sub-filter term', fontsize=10)
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.bar(x - w / 2, [rows[str(L)]['product']['meas_on_2F']['ols'] for L in Ls], w, color='0.4', label='tile 330')
    a.bar(x + w / 2, [rows[str(L)]['tile']['meas_on_2F']['ols'] for L in Ls], w, color='#e76f51', label=NAME)
    a.axhline(1, color='0.7', lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels([f'L={L}' for L in Ls])
    a.set_ylabel('OLS slope of measured on 2F')
    a.set_title('measured on 2F (not an efficiency: criterion 1 fails)', fontsize=10)
    a.legend(fontsize=8)
    a = ax[1, 2]
    a.bar(x - w / 2, [rows[str(L)]['G_p90_ratio_tile_over_product'] for L in Ls], w, color='#e76f51', label='G p90 threshold')
    a.bar(x + w / 2, [rows[str(L)]['n_front_ratio'] for L in Ls], w, color='#f4a259', label='n_front')
    a.axhline(1, color='0.7', lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels([f'L={L}' for L in Ls])
    a.set_ylabel(f'{NAME} / tile 330')
    a.set_title('the front population', fontsize=10)
    a.legend(fontsize=8)
    qa = res.get('orientation_qa', {})
    fig.suptitle(f'M3 digging task 5 -- a second tile: {NAME} (face {res.get("grid", {}).get("face", "?")}, '
                 f'no land, {res.get("grid", {}).get("lon", ["?", "?"])[0]:.0f} to {res.get("grid", {}).get("lon", ["?", "?"])[1]:.0f} E) '
                 f'beside tile 330   |   orientation QA: V3 slope {qa.get("slope", float("nan")):.3f} '
                 f'{"PASS" if qa.get("passed") else "FAIL"}', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.text(0.5, 0.008, 'surface-only budget on the second tile (measured - 2F - tau; the chunk terms absent); '
                         'the same 72 h window; front & valid of each tile; local solar hour from each tile\'s median longitude.',
             ha='center', fontsize=7.5, color='0.3')
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def _load_summary():
    return json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else dict(declared=DECLARED)


def _save_summary(d):
    d['declared'] = DECLARED
    d['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    OUT_JSON.write_text(json.dumps(d, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--step', choices=('prep', 'grid', 'masks', 'pull', 'verify', 'qa', 'compare', 'dry-run'),
                    default='prep')
    ap.add_argument('--hours', default=None, help='a:b slice of hours for --step pull')
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args(argv)
    P = paths()
    say = Say()
    if args.step == 'dry-run':
        ts = timestamps_72()
        r = resolve()
        print(f'tile {r["tile_idx"]}: face {r["face"]}, j {r["j_face"]}, i {r["i_face"]}')
        print(f'{len(ts)} timestamps: {ts[0]} .. {ts[-1]}')
        for k, p in P.items():
            print(f'  {k:8s} {p} ({"exists" if p.exists() else "absent"})')
        return 0
    d = _load_summary()
    if 'declared' not in d or not OUT_JSON.exists():
        _save_summary(d)
        say(f'declared -> {OUT_JSON}')
    say(f'===== m3_tile2 --step {args.step} (pid {os.getpid()}, host {socket.gethostname()}) =====')
    t0 = time.time()
    try:
        r = resolve()
        tile = r.pop('tile')
        d['resolve'] = r
        say(f'resolve: tile {r["tile_idx"]} = face {r["face"]}, j {r["j_face"]}, i {r["i_face"]} '
            f'(rect pixel {r["rect_pixel"]})')
        if args.step in ('prep', 'grid', 'masks', 'pull', 'verify', 'qa'):
            g, d['grid'] = grid_step(tile, P['grid'], say=say)
        if args.step in ('prep', 'masks', 'qa'):
            m, d['masks'] = masks_step(g, P['masks'], say=say)
        if args.step == 'prep':
            d['pull_hours_0_1'] = pull_step(tile, g, P['raw'], hours=[0, 1], say=say)
            d['verify_2h'] = verify_step(P['raw'], g, n_hours=2, say=say)
            d['orientation_qa'] = qa_step(g, P['raw'], m, say=say, n_boot=args.n_boot)
            if not d['orientation_qa']['passed']:
                say('ORIENTATION QA FAILED: the sweep must not run (prompt task 5, step 1)')
        if args.step == 'qa':
            d['orientation_qa'] = qa_step(g, P['raw'], m, say=say, n_boot=args.n_boot)
        if args.step == 'pull':
            if not d.get('orientation_qa', {}).get('passed'):
                raise RuntimeError('the orientation QA has not passed; run --step prep first')
            hours = None
            if args.hours:
                a, _, b = args.hours.partition(':')
                hours = list(range(int(a or 0), int(b or 72)))
            d['pull'] = pull_step(tile, g, P['raw'], hours=hours, say=say)
            d['verify'] = verify_step(P['raw'], g, say=say)
        if args.step == 'verify':
            d['verify'] = verify_step(P['raw'], g, say=say)
        if args.step == 'compare':
            d['compare'] = compare(say=say)
            figure(d, OUT_PNG)
            say(f'wrote {OUT_PNG}')
        d['wall_s_last_step'] = round(time.time() - t0, 1)
        _save_summary(d)
        say(f'wrote {OUT_JSON}')
        status = 'ok'
    except BaseException as e:                                   # noqa: BLE001 -- recorded
        import traceback
        say('EXCEPTION: ' + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        status = 'error'
    P['done'].write_text(json.dumps(dict(status=status, step=args.step, wall_s=round(time.time() - t0, 1),
                                         finished=datetime.now(timezone.utc).isoformat(timespec='seconds')), indent=1))
    say(f'===== m3_tile2 --step {args.step} end: {status}, {time.time() - t0:.0f} s =====')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
