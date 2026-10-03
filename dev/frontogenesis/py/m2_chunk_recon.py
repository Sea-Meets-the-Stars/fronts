"""M2 task 4 -- read-only reconnaissance of the hourly full-depth `monterey_bay` chunk store.

Source B (prompt 3): ``s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/`` on NRP Nautilus
(``https://s3-west.nrp-nautilus.io``), one zarr-v3 store per hour ``{YYYYMMDDTHH}.zarr`` plus a
static ``grid.zarr``.  Credentialed bucket: s3fs picks up the default AWS profile of this
machine; nothing here reads or prints a secret.

What it does (no bulk load, nothing written except the JSON summary):

1. inventory  -- list the stores; which of the 72 window hours exist; extent beyond the window;
                 per store: variables, dims, shapes, chunk shapes, codecs, object bytes, the
                 provenance attrs (face, j/i start, selected date / MIT iteration) -- from
                 metadata and listings only.
2. grid       -- drF, Z, Zl, Zu, Zp1 for k = 0..2 (+ bottoms); grid.zarr XC/YC/hFacC/... vs
                 our ``tile330_grid.zarr`` (tile coverage and orientation).
3. Eta, all 72 hours -- the cheap 2-D field (1.2 MB/hour), chunk vs OSN, to check the
                 timestamp/iteration mapping over the whole window.
4. consistency at one hour (default 2012-07-03 00:00) -- chunk Theta/Salt/U/V at k=0 and
                 W(k_p1=0) vs OSN; W(k_p1=1,2,51) statistics; land convention; plausibility of
                 oceQnet/oceQsw/oceFWflx.  Each 3-D read fetches the whole 51-level object
                 (one chunk per variable per hour), kept in memory only.
5. first/last hour Theta k=0 vs OSN (edge-of-window time mapping).

Usage (each `timeout 300` run fits one or two sections; results merge into the JSON)::

    PY=~/miniforge3/envs/frontogenesis/bin/python
    timeout 300 $PY dev/frontogenesis/py/m2_chunk_recon.py --sections inventory,grid
    timeout 300 $PY dev/frontogenesis/py/m2_chunk_recon.py --sections hour [--hour '2012-07-03 00:00:00']
    timeout 300 $PY dev/frontogenesis/py/m2_chunk_recon.py --sections eta,edges   # then: --sections fluxes

Writes ``data/m2_chunk_recon.json`` (git-ignored) after every section.
"""
from __future__ import annotations

import argparse
import json
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from numcodecs import Zstd

ENDPOINT = 'https://s3-west.nrp-nautilus.io'
PREFIX = 'dbof/LLC4320_RAW/CHUNKS/monterey_bay'
WINDOW_START = datetime(2012, 7, 2, 0)
N_HOURS = 72
TILE = dict(face=10, j0=0, j1=720, i0=2880, i1=3600)
OSN_OFFSET = 10368          # osn_date_to_iteration = mit + 10368 (dbof date_iterations)
TS_PER_HOUR = 144

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / 'data'
OSN_STORE = DATA / 'tile330_raw_20120702T00_72h.zarr'
GRID_STORE = DATA / 'tile330_grid.zarr'
OUT_JSON = DATA / 'm2_chunk_recon.json'
# compressed objects fetched for the checks are cached outside data/ (session scratchpad, or
# M2_RECON_CACHE); delete freely
import os
CACHE = Path(os.environ.get('M2_RECON_CACHE', '/private/tmp/claude-501/-Users-xavier-Oceanography-python-llc4320-native-grid-preprocessing/3436e395-e6fd-4557-ad52-7e1e69c69dee/scratchpad/chunk_cache'))

SUMMARY: dict = {}


def say(msg):
    print(msg, flush=True)


def save():
    OUT_JSON.write_text(json.dumps(SUMMARY, indent=1, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def make_fs():
    return s3fs.S3FileSystem(client_kwargs={'endpoint_url': ENDPOINT},
                             config_kwargs={'s3': {'addressing_style': 'path'},
                                            'retries': {'max_attempts': 5, 'mode': 'adaptive'}},
                             default_cache_type='none')


def window_stamps():
    return [WINDOW_START + timedelta(hours=h) for h in range(N_HOURS)]


def stamp(dt):
    return dt.strftime('%Y%m%dT%H')


def mit_iter(dt):
    """MIT iteration (dbof mit_date_to_iteration), from the OSN anchor 2012-07-02 00 = 1022976."""
    osn = 1022976 + int(round((dt - WINDOW_START).total_seconds() / 3600)) * TS_PER_HOUR
    return osn - OSN_OFFSET


# --------------------------------------------------------------------------- 1. inventory
def store_meta(fs, name):
    """Listing + metadata for one hourly store (no array data except the 1-element time)."""
    base = f'{PREFIX}/{name}'
    files = fs.find(base, detail=True)
    meta_paths = [p for p in files if p.endswith('zarr.json')]
    blobs = fs.cat(meta_paths)                       # one concurrent batch of small GETs
    group = json.loads(blobs[f'{base}/zarr.json'])
    arrays = {}
    nbytes = {}
    for path, info in files.items():
        rel = path[len(base) + 1:]
        var = rel.split('/')[0]
        if rel.endswith('zarr.json') and rel.count('/') == 1:
            m = json.loads(blobs[path])
            arrays[var] = dict(shape=m['shape'], dims=m.get('dimension_names'),
                               dtype=m['data_type'] if isinstance(m['data_type'], str)
                               else m['data_type'].get('name'),
                               chunks=m['chunk_grid']['configuration']['chunk_shape'],
                               codecs=[c['name'] for c in m['codecs']],
                               fill=m.get('fill_value'),
                               units=m.get('attributes', {}).get('units'))
        elif '/c/' in rel:
            nbytes.setdefault(var, [0, 0])
            nbytes[var][0] += info['size']
            nbytes[var][1] += 1
    # the 1-element time array: bytes(little-endian int64 ns) + zstd, decoded directly
    raw = Zstd().decode(fs.cat(f'{base}/time/c/0'))
    t = np.frombuffer(raw, dtype='<i8').astype('datetime64[ns]')
    return dict(name=name, attrs=group.get('attributes', {}), arrays=arrays,
                object_bytes={k: v[0] for k, v in nbytes.items()},
                n_objects={k: v[1] for k, v in nbytes.items()},
                time=str(t[0]) if t.size else None)


def inventory(fs):
    t0 = _time.time()
    names = sorted(p.split('/')[-1] for p in fs.ls(PREFIX))
    stores = [n for n in names if n.endswith('.zarr') and n != 'grid.zarr']
    want = {stamp(d) + '.zarr': d for d in window_stamps()}
    present = [n for n in want if n in stores]
    missing = [n for n in want if n not in stores]
    outside = [n for n in stores if n not in want]
    say(f'[inventory] {len(stores)} hourly stores + grid.zarr={"grid.zarr" in names}; '
        f'window: {len(present)}/72 present, missing {missing}; outside window: {outside}')

    # stores outside the window: variable list only (one listing each)
    outside_vars = {n: sorted(x.split('/')[-1] for x in fs.ls(f'{PREFIX}/{n}')
                              if not x.endswith('zarr.json')) for n in outside}
    metas = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for m in ex.map(lambda n: store_meta(make_fs(), n), present):
            metas[m['name']] = m
    # uniformity across the window
    ref = metas[present[0]]
    var_sets = {n: sorted(m['arrays']) for n, m in metas.items()}
    uniform_vars = all(v == var_sets[present[0]] for v in var_sets.values())
    layout_diffs = [n for n, m in metas.items()
                    if any(m['arrays'][v] != ref['arrays'].get(v) for v in m['arrays'])]
    time_ok, iter_ok, attr_ok = [], [], []
    for n in present:
        m, dt = metas[n], want[n]
        time_ok.append(m['time'] is not None and np.datetime64(m['time']) == np.datetime64(dt))
        iter_ok.append(int(m['attrs'].get('selected_iteration', -1)) == mit_iter(dt))
        a = m['attrs']
        attr_ok.append(a.get('resolved_face') == TILE['face'] and a.get('j_start') == TILE['j0']
                       and a.get('i_start') == TILE['i0'] and a.get('tile_size') == 720)
    per_hour_bytes = {n: sum(m['object_bytes'].values()) for n, m in metas.items()}
    hb = np.array(list(per_hour_bytes.values()))
    k03_vars = ('Theta', 'Salt', 'W', 'oceQnet', 'oceQsw', 'oceFWflx')
    k03 = np.array([sum(metas[n]['object_bytes'][v] for v in k03_vars) for n in present])
    SUMMARY['inventory'] = dict(
        endpoint=ENDPOINT, prefix=PREFIX, format='zarr v3, one store per hour, no consolidated md',
        n_stores=len(stores), all_stores=stores, window_present=present, window_missing=missing,
        outside_window=outside, outside_window_variables=outside_vars, variables=var_sets[present[0]], uniform_variables=uniform_vars,
        layout_differs_from_first=layout_diffs, arrays=ref['arrays'],
        object_bytes_example={present[0]: ref['object_bytes']},
        group_attrs_example=ref['attrs'],
        time_matches_name=int(sum(time_ok)), mit_iteration_matches=int(sum(iter_ok)),
        provenance_matches_tile330=int(sum(attr_ok)),
        bytes_per_hour_all=dict(min=int(hb.min()), median=float(np.median(hb)), max=int(hb.max()),
                                total_window=int(hb.sum())),
        bytes_per_hour_needed_vars=dict(vars=list(k03_vars), min=int(k03.min()),
                                        median=float(np.median(k03)), max=int(k03.max()),
                                        total_window=int(k03.sum())),
        seconds=round(_time.time() - t0, 1))
    for n, v in outside_vars.items():
        say(f'[inventory] outside window {n}: has oceQsw={"oceQsw" in v}, oceFWflx={"oceFWflx" in v}')
    say(f'[inventory] vars {var_sets[present[0]]} uniform={uniform_vars}, '
        f'layout diffs {layout_diffs}; time==name {sum(time_ok)}/{len(present)}, '
        f'MIT iter ok {sum(iter_ok)}/{len(present)}, tile attrs ok {sum(attr_ok)}/{len(present)}')
    for v, a in ref['arrays'].items():
        say(f'    {v:9s} {str(a["dims"]):34s} {str(a["shape"]):18s} chunks {a["chunks"]} '
            f'{a["dtype"]} {a["codecs"]} {ref["object_bytes"].get(v, 0)/1e6:7.2f} MB')
    say(f'[inventory] bytes/hour all vars: median {np.median(hb)/1e6:.1f} MB '
        f'({hb.min()/1e6:.1f}-{hb.max()/1e6:.1f}); window total {hb.sum()/1e9:.2f} GB')
    say(f'[inventory] bytes/hour fetched by a k=0..2 read of {k03_vars}: median '
        f'{np.median(k03)/1e6:.1f} MB; window total {k03.sum()/1e9:.2f} GB  ({_time.time()-t0:.0f} s)')
    save()
    return present


# --------------------------------------------------------------------------- 2. grid
def open_chunk(fs, name):
    return xr.open_zarr(fs.get_mapper(f'{PREFIX}/{name}'), consolidated=False, chunks=None)


def grid_check(fs):
    t0 = _time.time()
    g = open_chunk(fs, 'grid.zarr')
    out = {}
    for v in ('drF', 'Z', 'Zl', 'Zu', 'Zp1', 'drC'):
        if v in g:
            a = g[v].values
            out[v] = dict(dim=g[v].dims[0], n=a.size, k0_2=a[:3].tolist(), last=a[-2:].tolist())
    out['drF_sum'] = float(g['drF'].values.sum())
    ok_drF = float(g['drF'].values[0]) == 1.0
    ok_Z = float(g['Z'].values[0]) == -0.5
    out['confirm'] = dict(drF0_eq_1=ok_drF, Z0_eq_minus_half=ok_Z,
                          Zl0=float(g['Zl'].values[0]), Zl1=float(g['Zl'].values[1]))
    # tile coverage and orientation vs the OSN grid store
    ours = xr.open_zarr(GRID_STORE)
    cmp = {}
    cmp['index_ranges'] = dict(face=g['face'].values.tolist(),
                               j=[int(g.j[0]), int(g.j[-1])], i=[int(g.i[0]), int(g.i[-1])],
                               j_g=[int(g.j_g[0]), int(g.j_g[-1])], i_g=[int(g.i_g[0]), int(g.i_g[-1])])
    cmp['covers_tile330'] = (int(g.face[0]) == TILE['face'] and g.sizes['j'] == 720
                             and int(g.j[0]) == TILE['j0'] and int(g.i[0]) == TILE['i0']
                             and g.sizes['i'] == 720)
    for v in ('XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rA', 'rAz', 'CS', 'SN', 'Depth'):
        a = g[v].isel(face=0).values
        b = ours[v].values
        d = np.abs(a.astype(np.float64) - b)
        cmp[v] = dict(identical=bool(np.array_equal(a, b, equal_nan=True)),
                      max_abs=float(np.nanmax(d)))
    for v, oursv in (('hFacC', 'hFacC'), ('hFacW', 'hFacW'), ('hFacS', 'hFacS')):
        a = g[v].isel(face=0, k=0).values
        b = ours[oursv].values
        cmp[v + '_k0'] = dict(identical=bool(np.array_equal(a, b, equal_nan=True)),
                              max_abs=float(np.nanmax(np.abs(a - b))))
    out['vs_osn_grid'] = cmp
    out['grid_attrs'] = {k: v for k, v in g.attrs.items()}
    out['grid_vars'] = sorted(g.data_vars)
    out['seconds'] = round(_time.time() - t0, 1)
    SUMMARY['grid'] = out
    say(f'[grid] drF[0:3]={out["drF"]["k0_2"]}  Z[0:3]={out["Z"]["k0_2"]}  '
        f'Zl[0:3]={out["Zl"]["k0_2"]}  Zp1[0:3]={out.get("Zp1", {}).get("k0_2")}  '
        f'Zu[0:3]={out.get("Zu", {}).get("k0_2")}')
    say(f'[grid] n levels: drF {out["drF"]["n"]}, Zp1 {out["Zp1"]["n"]}; sum(drF)={out["drF_sum"]:.2f} m; '
        f'confirm drF[0]==1.0 {ok_drF}, Z[0]==-0.5 {ok_Z}')
    say(f'[grid] covers tile 330: {cmp["covers_tile330"]} {cmp["index_ranges"]}')
    say('[grid] vs tile330_grid.zarr: ' + ', '.join(
        f'{k} {"==" if v["identical"] else "max|d|=%.3g" % v["max_abs"]}'
        for k, v in cmp.items() if isinstance(v, dict) and 'identical' in v))
    save()
    return g


# --------------------------------------------------------------------------- helpers
def compare(a, b):
    """Stats of chunk field a vs OSN field b (same shape)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = a[both].astype(np.float64) - b[both]
    scale = np.nanmax(np.abs(b[both])) if both.any() else np.nan
    return dict(bit_identical=bool(np.array_equal(a, b, equal_nan=True)),
                nan_pattern_equal=bool(np.array_equal(fa, fb)),
                n_finite_chunk=int(fa.sum()), n_finite_osn=int(fb.sum()),
                n_zero_chunk=int((a == 0).sum()), n_zero_osn=int((b == 0).sum()),
                max_abs_diff=float(np.abs(d).max()) if d.size else None,
                rms_diff=float(np.sqrt((d ** 2).mean())) if d.size else None,
                frac_differing=float((d != 0).mean()) if d.size else None,
                max_abs_osn=float(scale))


def fmt(c):
    return (f'bit-identical={c["bit_identical"]} NaN-pattern-equal={c["nan_pattern_equal"]} '
            f'max|d|={c["max_abs_diff"]:.3g} rms={c["rms_diff"]:.3g} '
            f'frac!=0={c["frac_differing"]:.3g} (finite {c["n_finite_chunk"]}/{c["n_finite_osn"]}, '
            f'zeros {c["n_zero_chunk"]}/{c["n_zero_osn"]})')


# --------------------------------------------------------------------------- 3. Eta, 72 h
def eta_window(fs, present, osn):
    t0 = _time.time()

    def one(name):
        f = make_fs()
        ds = open_chunk(f, name)
        return name, ds['Eta'].isel(face=0).values, str(ds['time'].values[0])

    rows = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, eta, t in ex.map(one, present):
            dt = datetime.strptime(name[:-5], '%Y%m%dT%H')
            o = osn['Eta'].sel(time=np.datetime64(dt)).values
            c = compare(eta, o)
            # best-matching OSN hour (time-offset test)
            best = None
            if not c['bit_identical']:
                errs = {}
                for off in (-1, 1):
                    tt = np.datetime64(dt + timedelta(hours=off))
                    if tt in osn.time.values:
                        e = eta.astype(np.float64) - osn['Eta'].sel(time=tt).values
                        errs[off] = float(np.nanmax(np.abs(e)))
                best = errs
            rows[name] = dict(time=t, identical=c['bit_identical'], max_abs=c['max_abs_diff'],
                              nan_equal=c['nan_pattern_equal'], neighbours=best)
    n_ident = sum(r['identical'] for r in rows.values())
    SUMMARY['eta_72h'] = dict(n=len(rows), n_bit_identical=n_ident,
                              max_abs_over_window=max(r['max_abs'] for r in rows.values()),
                              rows=rows, seconds=round(_time.time() - t0, 1))
    say(f'[Eta 72 h] chunk vs OSN bit-identical in {n_ident}/{len(rows)} hours; max|d| over window '
        f'{SUMMARY["eta_72h"]["max_abs_over_window"]:.3g}  ({_time.time()-t0:.0f} s)')
    save()


# --------------------------------------------------------------------------- 4. one hour
def read_array(fs, name, var):
    """Decode one variable of one hourly store (a single zarr-v3 object: bytes LE + zstd).

    Each object holds the whole variable (all 51/52 levels), so there is no partial read.  The
    compressed object is cached in CACHE (scratchpad, outside data/) so that a run split across
    several ``timeout 300`` calls fetches each object once.
    """
    base = f'{PREFIX}/{name}/{var}'
    m = json.loads(fs.cat(f'{base}/zarr.json'))
    assert m['chunk_grid']['configuration']['chunk_shape'] == m['shape'], 'expected one object'
    assert [c['name'] for c in m['codecs']] == ['bytes', 'zstd']
    key = '/'.join(['c'] + ['0'] * len(m['shape']))
    cache = CACHE / name / var
    if cache.exists():
        blob = cache.read_bytes()
    else:
        t0 = _time.time()
        blob = fs.cat(f'{base}/{key}')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
        FETCH_LOG.append(dict(store=name, var=var, bytes=len(blob), seconds=round(_time.time() - t0, 1)))
        say(f'    fetched {name}/{var}: {len(blob)/1e6:.1f} MB in {_time.time()-t0:.0f} s '
            f'({len(blob)/1e6/max(_time.time()-t0, 1e-3):.2f} MB/s)')
    a = np.frombuffer(Zstd().decode(blob), dtype='<f4').reshape(m['shape'])
    return a, m.get('dimension_names')


FETCH_LOG: list = []


def one_hour(fs, ts, osn, grid_chunk, vars_=('Theta', 'Salt', 'U', 'V', 'W', '2d')):
    """Consistency at one hour; per-variable so a slow link can split it across runs."""
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    name = stamp(dt) + '.zarr'
    o = osn.sel(time=np.datetime64(dt))
    res = SUMMARY.setdefault('one_hour', {})
    if res.get('hour') != ts:
        res.clear()
    attrs = json.loads(fs.cat(f'{PREFIX}/{name}/zarr.json'))['attributes']
    res.update(hour=ts, store=name, osn_niter=int(o['niter'].values),
               chunk_mit_iter=attrs.get('selected_iteration'),
               chunk_selected_date=attrs.get('selected_date_utc'))
    res['iter_check'] = (int(o['niter'].values) - OSN_OFFSET == res['chunk_mit_iter'])
    say(f'[hour {ts}] chunk selected_date {res["chunk_selected_date"]}, MIT iter '
        f'{res["chunk_mit_iter"]} + {OSN_OFFSET} = OSN niter {res["osn_niter"]}: {res["iter_check"]}')
    hfac = grid_chunk['hFacC'].isel(face=0, k=slice(0, 3)).values
    hfacw = grid_chunk['hFacW'].isel(face=0, k=slice(0, 3)).values
    hfacs = grid_chunk['hFacS'].isel(face=0, k=slice(0, 3)).values
    landmask = {'Theta': hfac, 'Salt': hfac, 'U': hfacw, 'V': hfacs}
    for v in vars_:
        t0 = _time.time()
        if v in landmask:
            a, dims = read_array(fs, name, v)
            a3 = a[:3, 0]
            c = compare(a3[0], o[v].values)
            res[v] = dict(dims=dims, k0_vs_osn=c,
                          k0_2_range=[[float(np.nanmin(x)), float(np.nanmax(x))] for x in a3],
                          nan_equals_hFac0_k0_2=[bool(np.array_equal(~np.isfinite(a3[k]),
                                                                     landmask[v][k] == 0))
                                                 for k in range(3)],
                          ocean_frac_k0_2=[float((landmask[v][k] > 0).mean()) for k in range(3)])
            say(f'  {v:5s} k=0 vs OSN: {fmt(c)}')
            say(f'        NaN == (hFac==0) at k=0,1,2: {res[v]["nan_equals_hFac0_k0_2"]}; '
                f'ranges k=0..2 {[[round(x, 4) for x in r] for r in res[v]["k0_2_range"]]}')
        elif v == 'W':
            a, dims = read_array(fs, name, 'W')
            W = a[:, 0]
            r = dict(dims=dims, n=int(W.shape[0]), kp1_0_vs_osn=compare(W[0], o['W'].values))
            say(f'  W dims {dims}; W(k_p1=0) vs OSN W: {fmt(r["kp1_0_vs_osn"])}')
            stats = {}
            for k in (0, 1, 2, 3, 49, 50, 51):
                w = W[k]
                f = np.isfinite(w)
                stats[k] = dict(n_finite=int(f.sum()), n_zero=int((w[f] == 0).sum()),
                                median_abs=float(np.median(np.abs(w[f]))) if f.any() else None,
                                p99_abs=float(np.percentile(np.abs(w[f]), 99)) if f.any() else None)
                say(f'    W[k_p1={k:2d}] finite {stats[k]["n_finite"]:6d} zeros {stats[k]["n_zero"]:6d} '
                    f'median|w| {stats[k]["median_abs"]} p99|w| {stats[k]["p99_abs"]}')
            r['stats_by_kp1'] = stats
            r['nan_equals_hFacC0_kp1_0_2'] = [bool(np.array_equal(~np.isfinite(W[k]), hfac[k] == 0))
                                             for k in range(3)]
            say(f'    W NaN == (hFacC[k]==0) for k_p1=k=0,1,2: {r["nan_equals_hFacC0_kp1_0_2"]}')
            # W(k_p1=0) as dEta/dt: centred OSN difference
            tt = [np.datetime64(dt + timedelta(hours=h)) for h in (-1, 1)]
            if all(t in osn.time.values for t in tt):
                detadt = (osn['Eta'].sel(time=tt[1]).values.astype(np.float64)
                          - osn['Eta'].sel(time=tt[0]).values) / 7200.
                for k in (0, 1, 2):
                    f = np.isfinite(detadt) & np.isfinite(W[k])
                    r[f'corr_kp1_{k}_with_dEta_dt'] = float(np.corrcoef(W[k][f], detadt[f])[0, 1])
                say('    corr(W[k_p1=k], centred dEta/dt), k=0,1,2: ' + ', '.join(
                    f'{r[f"corr_kp1_{k}_with_dEta_dt"]:.3f}' for k in (0, 1, 2)))
            # Continuity per cell k (MITgcm, w up, w[k_p1=k] on the TOP face of cell k):
            #   rA (w[k] - w[k+1]) + sum_out(U dyG drF hFacW, V dxG drF hFacS) = 0
            # Tests that k_p1 = k is the top interface of cell k (i.e. k_p1[0:51] == k_l) and
            # W is the model's own; uses U, V from the cache (same hour), interior cells only.
            Uc, _ = read_array(fs, name, 'U')
            Vc, _ = read_array(fs, name, 'V')
            g = grid_chunk
            dyG = g['dyG'].isel(face=0).values.astype(np.float64)
            dxG = g['dxG'].isel(face=0).values.astype(np.float64)
            rA = g['rA'].isel(face=0).values.astype(np.float64)
            drF = g['drF'].values.astype(np.float64)
            cont = {}
            for k in (0, 1, 2, 49, 50):
                hk = g['hFacC'].isel(face=0, k=k).values
                tu = np.nan_to_num(Uc[k, 0].astype(np.float64)) * dyG * drF[k] * \
                    g['hFacW'].isel(face=0, k=k).values
                tv = np.nan_to_num(Vc[k, 0].astype(np.float64)) * dxG * drF[k] * \
                    g['hFacS'].isel(face=0, k=k).values
                F = (tu[:-1, 1:] - tu[:-1, :-1]) + (tv[1:, :-1] - tv[:-1, :-1])   # cells j<719, i<719
                wt = W[k, :-1, :-1].astype(np.float64)
                wb = np.nan_to_num(W[k + 1, :-1, :-1].astype(np.float64))
                pred = wb - F / rA[:-1, :-1]
                ocean = (hk[:-1, :-1] > 0) & np.isfinite(wt)
                resid = (wt - pred)[ocean]
                cont[k] = dict(rms_resid=float(np.sqrt((resid ** 2).mean())),
                               max_abs_resid=float(np.abs(resid).max()),
                               rms_w=float(np.sqrt((wt[ocean] ** 2).mean())))
                say(f'    continuity cell k={k}: rms(w[k]-(w[k+1]-F/rA)) {cont[k]["rms_resid"]:.3g} '
                    f'(max {cont[k]["max_abs_resid"]:.3g}) vs rms w[k] {cont[k]["rms_w"]:.3g}')
            r['continuity'] = cont
            del Uc, Vc
            res['W'] = r
            del W, a
        elif v == '2d':
            for v2 in ('oceQnet', 'oceQsw', 'oceFWflx', 'Eta', 'oceTAUX', 'oceTAUY', 'SIarea'):
                a, dims = read_array(fs, name, v2)
                a = a[0]
                f = np.isfinite(a)
                r = dict(dims=dims, range=[float(np.nanmin(a)), float(np.nanmax(a))] if f.any() else None,
                         n_finite=int(f.sum()),
                         nan_equals_hFacC0=bool(np.array_equal(~f, hfac[0] == 0)))
                if v2 in o:
                    r['vs_osn'] = compare(a, o[v2].values)
                res[v2] = r
                rng = '..'.join(f'{x:.4g}' for x in r['range']) if r['range'] else 'all-NaN'
                say(f'  {v2:8s} {dims} range {rng}, finite {r["n_finite"]}, NaN==land {r["nan_equals_hFacC0"]}'
                    + (f'; vs OSN: {fmt(r["vs_osn"])}' if 'vs_osn' in r else ''))
        res.setdefault('seconds', {})[v] = round(_time.time() - t0, 1)
        SUMMARY['fetch_log'] = SUMMARY.get('fetch_log', []) + FETCH_LOG
        FETCH_LOG.clear()
        save()


# --------------------------------------------------------------------------- 5. edges
def edge_hours(fs, osn):
    t0 = _time.time()
    out = {}
    for dt in (window_stamps()[0], window_stamps()[-1]):
        name = stamp(dt) + '.zarr'
        th = read_array(fs, name, 'Theta')[0][0, 0]
        out[name] = compare(th, osn['Theta'].sel(time=np.datetime64(dt)).values)
        say(f'[edge {name}] Theta k=0 vs OSN: {fmt(out[name])}')
    SUMMARY['edge_hours_theta'] = dict(rows=out, seconds=round(_time.time() - t0, 1))
    save()


# --------------------------------------------------------------------------- 6. flux diurnal cycle
def flux_diurnal(fs, day=datetime(2012, 7, 3)):
    """Tile-mean oceQsw / oceQnet / oceFWflx for the 24 hours of one day (2-D, 1.2 MB each).

    Sign-convention check: the store's attrs say oceQsw/oceQnet are '+=down, >0 increases
    theta'.  Local solar time on the tile is ~UTC-8 h (lon -128..-113), so a downward-positive
    shortwave must be >= 0 everywhere, ~0 at night (UTC 04-13) and peak near 20 UTC.
    """
    t0 = _time.time()
    rows = {}

    def one(h):
        f = make_fs()
        name = stamp(day + timedelta(hours=h)) + '.zarr'
        return h, {v: read_array(f, name, v)[0][0] for v in ('oceQsw', 'oceQnet', 'oceFWflx')}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for h, d in ex.map(one, range(24)):
            rows[h] = {v: dict(mean=float(np.nanmean(a)), min=float(np.nanmin(a)),
                               max=float(np.nanmax(a)), frac_pos=float(np.nanmean(a > 0)))
                       for v, a in d.items()}
    sw = np.array([rows[h]['oceQsw']['mean'] for h in range(24)])
    qn = np.array([rows[h]['oceQnet']['mean'] for h in range(24)])
    out = dict(day=str(day.date()), rows=rows,
               oceQsw_mean_range=[float(sw.min()), float(sw.max())],
               oceQsw_hour_of_extreme=int(np.argmax(np.abs(sw))),
               oceQsw_any_positive=bool(any(rows[h]['oceQsw']['max'] > 0 for h in range(24))),
               oceQnet_mean_range=[float(qn.min()), float(qn.max())],
               seconds=round(_time.time() - t0, 1))
    SUMMARY['flux_diurnal'] = out
    say(f'[fluxes {day.date()}] tile-mean by UTC hour:')
    for h in range(24):
        r = rows[h]
        say(f'    {h:02d} UTC  oceQsw {r["oceQsw"]["mean"]:8.1f} [{r["oceQsw"]["min"]:7.1f},'
            f'{r["oceQsw"]["max"]:7.1f}]  oceQnet {r["oceQnet"]["mean"]:8.1f}  '
            f'oceFWflx {r["oceFWflx"]["mean"]: .3e}')
    say(f'    oceQsw any positive value: {out["oceQsw_any_positive"]}; |extreme| at {out["oceQsw_hour_of_extreme"]} UTC')
    save()


SECTIONS = ('inventory', 'grid', 'eta', 'hour', 'edges', 'fluxes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hour', default='2012-07-03 00:00:00')
    p.add_argument('--hour-vars', default='Theta,Salt,U,V,W,2d',
                   help='subset of Theta,Salt,U,V,W,2d for the hour section (~60-120 s per 3-D '
                        'variable at the ~0.55 MB/s measured from this machine)')
    p.add_argument('--sections', default=','.join(SECTIONS),
                   help='comma list of %s; sections not run keep their previous JSON entry '
                        '(each run merges into data/m2_chunk_recon.json), so the slow parts can '
                        'be split across several `timeout 300` runs' % (SECTIONS,))
    args = p.parse_args()
    run = set(args.sections.split(','))
    t0 = _time.time()
    if OUT_JSON.exists():
        SUMMARY.update(json.loads(OUT_JSON.read_text()))
    SUMMARY['created'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    fs = make_fs()
    present = inventory(fs) if 'inventory' in run else SUMMARY['inventory']['window_present']
    if run & {'grid', 'hour'}:
        g = grid_check(fs) if 'grid' in run else open_chunk(fs, 'grid.zarr')
    osn = xr.open_zarr(OSN_STORE)
    if 'eta' in run:
        eta_window(fs, present, osn)
    if 'hour' in run:
        one_hour(fs, args.hour, osn, g, args.hour_vars.split(','))
    if 'edges' in run:
        edge_hours(fs, osn)
    if 'fluxes' in run:
        flux_diurnal(fs)
    SUMMARY.setdefault('runs', []).append(dict(sections=sorted(run), seconds=round(_time.time() - t0, 1)))
    save()
    say(f'[done] {_time.time() - t0:.0f} s -> {OUT_JSON}')


if __name__ == '__main__':
    main()
