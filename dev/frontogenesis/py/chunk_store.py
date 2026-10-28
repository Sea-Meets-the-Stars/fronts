""" The chunk-store reader (M2 task 5; coding §3.3): :func:`load_chunk_levels`
pulls ``k = 0..2`` of ``Theta, Salt, W`` and the three surface fluxes from
the hourly full-depth ``monterey_bay`` chunk store into the §3.3 zarr.

Moved here from ``vertical.py`` in M3 task 2 (decided 2026-10-07, M3-Q7 (a))
so that ``vertical.py`` holds only the §4.6 physics (``b_z``,
``vertical_term``, ``surface_flux_term``); ``vertical.load_chunk_levels``
remains as a re-export, so ``m2_chunk_pull.py`` and the store's recorded
provenance strings ("vertical.load_chunk_levels") stay valid.  The strings
written into the store's attrs are **unchanged** on purpose: a resume or
no-op re-run re-syncs the root attrs (``osn_tiles._sync_attrs``), and a
changed string would rewrite a store that is otherwise byte-identical.

The source (M2 task 4)
----------------------
``s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/{YYYYMMDDTHH}.zarr`` on NRP
Nautilus (``https://s3-west.nrp-nautilus.io``, path-style addressing,
credentialed: s3fs uses the default AWS profile), one zarr-v3 store per
hour plus a static ``grid.zarr``; exactly tile 330 (face 10, j 0:720,
i 2880:3600).  Every variable is **one object per hour** -- 3-D chunks
``(51, 1, 720, 720)``, ``W`` ``(52, 1, 720, 720)``, 2-D ``(1, 720, 720)``,
codecs bytes (little-endian) + zstd -- so a ``k = 0..2`` read must fetch the
whole 51-level object (~174 MB per hour for the six variables) and keep
three levels.  The objects are read and decoded directly here (zarr.json,
then the single chunk, zstd, reshape) rather than through ``xr.open_zarr``,
so every object can be validated, and re-fetched if Nautilus serves corrupt
bytes (dbof's reader warns that it intermittently does).

What is checked before an hour is written
-----------------------------------------
* every object: the layout (one chunk, bytes+zstd, little-endian), a
  successful zstd decode of exactly ``prod(shape) * itemsize`` bytes, the
  expected shape, and plausible finite values;
* land: NaN exactly where the chunk grid's ``hFacC[k] == 0`` for
  ``Theta``/``Salt`` at ``k = 0..k_max``, ``W`` at ``k_p1 = 0..k_max`` and
  the 2-D fluxes (``hFacC[0]``) -- task 4 found these equal;
* time: the store's own ``time`` equals the requested timestamp, its
  ``selected_date_utc`` attr too, and ``selected_iteration`` (the MIT
  iteration) equals the OSN iteration minus 10368; optionally (``osn_store``)
  the chunk ``Eta`` is bit-identical to the OSN store's ``Eta`` for that hour
  (task 4: true in all 72 hours), which pins the hour to the OSN series;
* sign: the raw ``oceQsw`` is nowhere above +1 W m^-2 (it is upward-positive,
  see below); a store whose shortwave were already downward-positive fails
  here instead of being silently negated twice.

A failed check is retried like a failed read (``zarr_series.with_retries``,
per object, so a corrupt ``Theta`` re-fetches 57 MB, not the hour), and an
hour that still fails stops the run at a gap, as ``pull_series`` does.

Sign convention (M2-Q6, JXP: option (a))
----------------------------------------
The source attrs call ``oceQnet``/``oceQsw``/``oceFWflx`` "+=down", but the
data are MITgcm's upward-positive forcing (task 4: ``oceQsw <= 0`` at every
pixel and hour, about -589 W m^-2 tile mean at local noon; ``oceQnet`` +115
at night, -454 at noon; ``oceFWflx`` about +2.5e-5 kg m^-2 s^-1, net
evaporation).  The three are **negated at write time**, so the store holds
the documented downward-positive convention that coding §3.3/§4.6 assume;
each variable carries ``sign_convention`` (correct), ``source_sign_convention``
(the original long_name) and ``sign_conversion``, and the root attrs record
the negation in ``provenance``.

Resumability is ``zarr_series``'s (task 1): stage the hour in memory, one
atomic per-hour append of the time-dimensioned variables, repair on resume,
"present" from the store's own ``time`` coord, stop at the first hour that
still fails.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Zstd

import zarr_series as zs
import osn_tiles as ot
from osn_tiles import DATA_DIR
from dbof.llc4320_ingestion.date_iterations import osn_date_to_iteration, DATE_FMT
from dbof.llc4320_ingestion.grid import ensure_comodo_attrs

CHUNK_ENDPOINT = 'https://s3-west.nrp-nautilus.io'
CHUNK_PREFIX = 'dbof/LLC4320_RAW/CHUNKS/monterey_bay'
CHUNK_ZARR = DATA_DIR / 'tile330_chunk_20120702T00_72h.zarr'
OSN_RAW_ZARR = DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'

LEVEL_VARS = ('Theta', 'Salt')                  # on k
FLUX_VARS = ('oceQnet', 'oceQsw', 'oceFWflx')   # 2-D, negated (M2-Q6)
OSN_MINUS_MIT = 10368                           # osn_date_to_iteration = MIT + 10368
TILE_FACE, J_START, I_START = 10, 0, 2880       # tile 330 (task 4: the store is exactly it)

# finite values outside these are a corrupt read, not ocean (raw source units/sign).  Generous
# on purpose: they catch garbage that decodes, not physics.  Salt reaches 48.6 psu in the
# northern Gulf of California (OSN 72 h, -115.7 E 32.5 N; a 45 bound stopped the first launch)
PLAUSIBLE = {'Theta': (-3.0, 45.0), 'Salt': (0.0, 70.0), 'W': (-1.0, 1.0),
             'oceQnet': (-3000.0, 3000.0), 'oceQsw': (-2000.0, 1.0),   # upward-positive: <= 0
             'oceFWflx': (-0.1, 0.1), 'Eta': (-20.0, 20.0)}

SIGN_CONVENTION = {
    'oceQnet': 'positive downward (into the ocean); >0 increases theta',
    'oceQsw': 'positive downward (into the ocean); >0 increases theta',
    'oceFWflx': 'positive downward (into the ocean); >0 decreases salinity',
}
LONG_NAME = {
    'oceQnet': 'net surface heat flux into the ocean, positive downward',
    'oceQsw': 'net shortwave radiation into the ocean, positive downward',
    'oceFWflx': 'net surface fresh-water flux into the ocean, positive downward',
}
SIGN_CONVERSION = ('negated at write time by vertical.load_chunk_levels (M2-Q6): the source '
                   'attrs say +=down but the source data are upward-positive (MITgcm forcing '
                   'convention; M2 task 4: oceQsw <= 0 everywhere, oceFWflx > 0 = evaporation)')
W_MAPPING = ('W(k_l=n) is the source W(k_p1=n), n=0..k_max: the source puts W on k_p1 (52 '
             'interfaces), and k_p1=n is the TOP face of cell n, i.e. k_l=n -- verified by '
             'continuity in M2 task 4 (rA*(W[k]-W[k+1]) + horizontal divergence closes to rms '
             '7e-12 m/s for cells 0..2). W(k_l=0) = dEta/dt (linear free surface); W(k_l=1) is '
             'the top-cell base velocity vertical_term takes (coding §4.6).')
FORCING_NOTE = ('the surface fluxes are 6-hourly forcing linearly interpolated by the model '
                '(kinks at 03/09/15/21 UTC, M2 task 4): the diurnal shortwave shape is a '
                'triangle, not resolved insolation')


class CorruptRead(ValueError):
    """An object that did not decode to the expected bytes or values."""


# ---------------------------------------------------------------------------
# reading one object
# ---------------------------------------------------------------------------
def make_fs(endpoint: str = CHUNK_ENDPOINT):
    """s3fs on Nautilus, path-style, default AWS profile (no secret handled
    here).  No block cache: each object is read once."""
    import s3fs
    return s3fs.S3FileSystem(client_kwargs={'endpoint_url': endpoint},
                             config_kwargs={'s3': {'addressing_style': 'path'},
                                            'connect_timeout': 60, 'read_timeout': 300,
                                            'retries': {'max_attempts': 5, 'mode': 'adaptive'}},
                             default_cache_type='none')


def _cat(fs, path: str) -> bytes:
    """The one network primitive (monkeypatched by the tests)."""
    return fs.cat(path)


def _np_dtype(data_type) -> np.dtype:
    if isinstance(data_type, dict):                  # 'numpy.datetime64' extension dtype
        if data_type.get('name') == 'numpy.datetime64':
            return np.dtype('<i8')                   # ns since epoch, viewed by the caller
        raise CorruptRead(f'unexpected data_type {data_type}')
    return np.dtype(data_type).newbyteorder('<')


def _read_object(fs, prefix: str, store: str, var: str) -> tuple:
    """Fetch and decode one single-chunk zarr-v3 array.

    Returns ``(array, meta, n_bytes_fetched)``.  Raises :class:`CorruptRead`
    if the layout is not the one task 4 found or the bytes do not decode to
    exactly the array's size.
    """
    base = f'{prefix}/{store}/{var}'
    try:
        meta = json.loads(_cat(fs, f'{base}/zarr.json'))
    except json.JSONDecodeError as e:
        raise CorruptRead(f'{store}/{var}: zarr.json does not parse ({e})') from e
    shape = tuple(meta['shape'])
    chunks = tuple(meta['chunk_grid']['configuration']['chunk_shape'])
    codecs = meta.get('codecs', [])
    if chunks != shape:
        raise CorruptRead(f'{store}/{var}: chunks {chunks} != shape {shape} (expected one object)')
    if [c['name'] for c in codecs] != ['bytes', 'zstd'] or \
            codecs[0].get('configuration', {}).get('endian', 'little') != 'little':
        raise CorruptRead(f'{store}/{var}: codecs {codecs} != bytes(little) + zstd')
    if meta.get('chunk_key_encoding', {}).get('name', 'default') != 'default':
        raise CorruptRead(f'{store}/{var}: chunk_key_encoding {meta["chunk_key_encoding"]}')
    dtype = _np_dtype(meta['data_type'])
    blob = _cat(fs, f'{base}/' + '/'.join(['c'] + ['0'] * len(shape)))
    try:
        raw = Zstd().decode(blob)
    except Exception as e:                           # noqa: BLE001 -- any decoder failure
        raise CorruptRead(f'{store}/{var}: zstd decode failed on {len(blob)} bytes '
                          f'({type(e).__name__}: {e})') from e
    want = int(np.prod(shape)) * dtype.itemsize
    if len(raw) != want:
        raise CorruptRead(f'{store}/{var}: decoded {len(raw)} bytes, expected {want}')
    return np.frombuffer(raw, dtype=dtype).reshape(shape), meta, len(blob)


def _check_values(name: str, a: np.ndarray, land: np.ndarray, rng: tuple):
    """NaN exactly on land, finite values within ``rng``."""
    nan = np.isnan(a)
    bad = int((nan != land).sum())
    if bad:
        raise CorruptRead(f'{name}: NaN pattern differs from hFacC == 0 in {bad} cells')
    if (~nan).any():
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if lo < rng[0] or hi > rng[1]:
            hint = (' -- positive source shortwave: is the source already downward-positive? '
                    'then the M2-Q6 negation would be wrong' if name.endswith('oceQsw')
                    and hi > rng[1] else '')
            raise CorruptRead(f'{name}: values [{lo:.4g}, {hi:.4g}] outside plausible {rng}{hint}')


# ---------------------------------------------------------------------------
# the static levels (grid.zarr), once per run
# ---------------------------------------------------------------------------
def _load_levels(fs, prefix: str, k_max: int, local_grid=None) -> dict:
    """``drF``/``Z``/``Zl`` for ``k = 0..k_max``, ``hFacC`` there (the land
    reference), ``XC``/``YC`` and the index coords, from the chunk store's
    ``grid.zarr`` (task 4: bit-identical to ``tile330_grid.zarr``)."""
    g = {}
    for v in ('drF', 'Z', 'Zl', 'hFacC', 'XC', 'YC', 'j', 'i'):
        g[v] = _read_object(fs, prefix, 'grid.zarr', v)[0]
    n_lev = g['drF'].shape[0]
    if g['hFacC'].ndim != 4 or g['hFacC'].shape[:2] != (n_lev, 1):
        raise CorruptRead(f'grid.zarr hFacC shape {g["hFacC"].shape}')
    if not k_max < n_lev:
        raise ValueError(f'k_max={k_max} but the store has {n_lev} levels')
    lev = dict(n_lev=n_lev, nj=g['hFacC'].shape[2], ni=g['hFacC'].shape[3],
               drF=np.array(g['drF'][:k_max + 1], dtype='f4'),
               Z=np.array(g['Z'][:k_max + 1], dtype='f4'),
               Zl=np.array(g['Zl'][:k_max + 1], dtype='f4'),
               land=np.array(g['hFacC'][:k_max + 1, 0]) == 0,
               XC=np.array(g['XC'][0], dtype='f4'), YC=np.array(g['YC'][0], dtype='f4'),
               j=g['j'].astype('int64'), i=g['i'].astype('int64'))
    # §3.3/§4.6 and task 4: the OSN top-cell values
    if lev['drF'][0] != 1.0 or lev['Z'][0] != -0.5:
        raise ValueError(f'drF[0]={lev["drF"][0]}, Z[0]={lev["Z"][0]}; expected 1.0, -0.5')
    if lev['j'][0] != J_START or lev['i'][0] != I_START:
        raise ValueError(f'grid.zarr starts at j={lev["j"][0]}, i={lev["i"][0]}; expected '
                         f'{J_START}, {I_START} (tile 330)')
    if local_grid is not None:
        # same tile, same land, same coordinates as M0's grid (task 4: bit-identical)
        loc = xr.open_zarr(local_grid)
        for v, a in (('hFacC', ~lev['land'][0]), ('XC', lev['XC']), ('YC', lev['YC'])):
            b = loc[v].values
            same = np.array_equal(a, b != 0) if v == 'hFacC' else np.array_equal(a, b)
            if not same:
                raise ValueError(f'chunk grid.zarr {v} differs from {local_grid}')
    return lev


# ---------------------------------------------------------------------------
# one hour
# ---------------------------------------------------------------------------
def _store_name(ts: str) -> str:
    return datetime.strptime(ts, DATE_FMT).strftime('%Y%m%dT%H') + '.zarr'


def _as_dt64(ts: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(ts, DATE_FMT), 's')


def _osn_eta_reader(osn_store):
    """``ts -> Eta(j, i)`` from the OSN series store, or None if absent."""
    if osn_store is None or not Path(osn_store).exists():
        return None
    ds = xr.open_zarr(osn_store)

    def read(ts):
        t = _as_dt64(ts)
        if t not in ds['time'].values.astype('datetime64[s]'):
            return None
        return ds['Eta'].sel(time=t.astype('datetime64[ns]')).values
    return read


def _load_hour(fs, prefix: str, ts: str, lev: dict, k_max: int, osn_eta=None, *,
               attempts: int, backoff, sleep, log) -> tuple:
    """Fetch, validate and subset one hour; nothing is written here.

    Returns ``(ds_hour, info)``; ``info`` has the bytes fetched and the
    MIT iteration.  Every fetch is retried on its own.
    """
    store = _store_name(ts)
    nk, nj, ni, n_lev = k_max + 1, lev['nj'], lev['ni'], lev['n_lev']
    osn_it = int(osn_date_to_iteration(ts))
    mit_it = osn_it - OSN_MINUS_MIT
    fetched = {'bytes': 0}

    def get(var, check):
        def once():
            t0 = time.time()
            a, meta, nb = _read_object(fs, prefix, store, var)
            out = check(a)
            fetched['bytes'] += nb
            dt = time.time() - t0
            if nb > 5e6:                             # one line per large object
                zs._say(f'    {store}/{var}: {nb / 1e6:.1f} MB in {dt:.0f} s '
                        f'({nb / 1e6 / max(dt, 1e-3):.2f} MB/s)', log)
            return out, meta
        return zs.with_retries(once, attempts=attempts, backoff=backoff, sleep=sleep, log=log,
                               what=f'{store}/{var}')

    # time and provenance of the hour (small)
    def group_attrs():
        try:
            a = json.loads(_cat(fs, f'{prefix}/{store}/zarr.json')).get('attributes', {})
        except json.JSONDecodeError as e:
            raise CorruptRead(f'{store}: zarr.json does not parse ({e})') from e
        want = dict(selected_date_utc=ts, selected_iteration=mit_it, resolved_face=TILE_FACE,
                    j_start=J_START, i_start=I_START, tile_size=nj)
        bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
        if bad:
            raise ValueError(f'{store}: attrs (found, expected) {bad}')
        return a
    src_attrs = zs.with_retries(group_attrs, attempts=attempts, backoff=backoff, sleep=sleep,
                                log=log, what=f'{store}/zarr.json')

    def check_time(a):
        got = a.view('datetime64[ns]').astype('datetime64[s]')
        if got.shape != (1,) or got[0] != _as_dt64(ts):
            raise ValueError(f'{store}: store time {got} != requested {ts}')
    get('time', check_time)

    def levels(var, n):
        def check(a):
            if a.shape != (n, 1, nj, ni):
                raise CorruptRead(f'{store}/{var}: shape {a.shape} != {(n, 1, nj, ni)}')
            sub = np.array(a[:nk, 0], dtype='f4')    # copy: frees the 51-level buffer
            for k in range(nk):
                _check_values(f'{store}/{var}[{k}]', sub[k], lev['land'][k], PLAUSIBLE[var])
            return sub
        return check

    def surface(var):
        def check(a):
            if a.shape != (1, nj, ni):
                raise CorruptRead(f'{store}/{var}: shape {a.shape} != {(1, nj, ni)}')
            sub = np.array(a[0], dtype='f4')
            _check_values(f'{store}/{var}', sub, lev['land'][0], PLAUSIBLE[var])
            return sub
        return check

    data, attrs = {}, {}
    for v in LEVEL_VARS:
        data[v], meta = get(v, levels(v, n_lev))
        attrs[v] = meta.get('attributes', {})
    # W lives on k_p1 (n_lev + 1 interfaces); see W_MAPPING for k_p1 -> k_l
    data['W'], meta = get('W', levels('W', n_lev + 1))
    attrs['W'] = meta.get('attributes', {})
    for v in FLUX_VARS:
        data[v], meta = get(v, surface(v))
        attrs[v] = meta.get('attributes', {})
    if osn_eta is not None:
        ref = osn_eta(ts)
        if ref is not None:
            def check_eta(a):
                e = surface('Eta')(a)
                if not np.array_equal(e, ref, equal_nan=True):
                    raise ValueError(f'{store}: Eta differs from the OSN store at {ts} '
                                     f'(max |d| {np.nanmax(np.abs(e - ref)):.3g}) -- time offset?')
            get('Eta', check_eta)

    t = np.array([_as_dt64(ts)]).astype('datetime64[ns]')
    var_attrs = {v: dict(attrs[v]) for v in data}
    var_attrs['W'].update(source_dim='k_p1', interface_mapping=W_MAPPING)
    for v in FLUX_VARS:
        a = attrs[v]
        var_attrs[v] = dict(units=a.get('units'), standard_name=a.get('standard_name', v),
                            long_name=LONG_NAME[v], sign_convention=SIGN_CONVENTION[v],
                            source_sign_convention=a.get('long_name', ''),
                            sign_conversion=SIGN_CONVERSION, forcing_note=FORCING_NOTE)
    dvars = {v: (('time', 'k', 'j', 'i'), data[v][None], var_attrs[v]) for v in LEVEL_VARS}
    dvars['W'] = (('time', 'k_l', 'j', 'i'), data['W'][None], var_attrs['W'])
    for v in FLUX_VARS:
        # M2-Q6 (a): upward-positive source -> downward-positive store
        dvars[v] = (('time', 'j', 'i'), (-data[v])[None], var_attrs[v])
    dvars['drF'] = (('k',), lev['drF'], dict(long_name='cell z size', units='m',
                                             source='CHUNKS/monterey_bay/grid.zarr'))
    ds = xr.Dataset(dvars, coords=dict(
        time=('time', t), niter=('time', [osn_it]), mit_iteration=('time', [mit_it]),
        face=TILE_FACE, k=('k', np.arange(nk, dtype='int64')),
        k_l=('k_l', np.arange(nk, dtype='int64')), j=('j', lev['j']), i=('i', lev['i']),
        XC=(('j', 'i'), lev['XC']), YC=(('j', 'i'), lev['YC']),
        Z=('k', lev['Z'], dict(long_name='vertical coordinate of cell center', units='m')),
        Zl=('k_l', lev['Zl'], dict(long_name='vertical coordinate of upper cell interface '
                                   '(k_l=n is the top of cell n)', units='m'))))
    ds['niter'].attrs.update(long_name='OSN iteration (MIT iteration + 10368), as in §3.2')
    ds['mit_iteration'].attrs.update(long_name='MITgcm iteration (source selected_iteration)')
    ds = ensure_comodo_attrs(ds)
    return ds, dict(bytes=fetched['bytes'], mit_iteration=mit_it, source_attrs=src_attrs)


def _encoding(ds: xr.Dataset) -> dict:
    """One chunk per hour per variable; ``time`` as in §3.2."""
    enc = {v: {'chunks': tuple(1 if d == 'time' else n for d, n in zip(ds[v].dims, ds[v].shape))}
           for v in ds.data_vars}
    enc['time'] = {'units': 'seconds since 2011-09-10', 'dtype': 'int64'}
    return enc


def _series_attrs(stored: list, k_max: int, endpoint: str, prefix: str) -> dict:
    """The §3.3 root attrs for the hours ``stored`` (full list on every append)."""
    osn = [int(osn_date_to_iteration(ts)) for ts in stored]
    return dict(
        source='CHUNKS/monterey_bay', levels=f'k=0..{k_max}, k_l=0..{k_max}',
        source_path=f's3://{prefix}/{{YYYYMMDDTHH}}.zarr (+ grid.zarr)', endpoint=endpoint,
        addressing='path-style', iterations=osn, mit_iterations=[i - OSN_MINUS_MIT for i in osn],
        timestamps=list(stored), face_index=TILE_FACE, j_face_start=J_START,
        i_face_start=I_START, land_fill='NaN', flux_sign_convention='positive downward',
        flux_sign_conversion=SIGN_CONVERSION, w_interfaces=W_MAPPING, forcing_note=FORCING_NOTE,
        provenance=[
            'vertical.load_chunk_levels (M2 task 5): k=0..k_max of Theta/Salt, W(k_p1=0..k_max) '
            'renamed k_l, oceQnet/oceQsw/oceFWflx, drF/Z/Zl from grid.zarr; float32',
            'oceQnet, oceQsw, oceFWflx NEGATED (source upward-positive -> stored downward-'
            'positive; M2-Q6 (a), JXP 2026-10-03)',
            'each object validated before writing: zstd decode and size, shape, NaN == '
            '(hFacC == 0), plausible range; time == request; MIT iteration == OSN - 10368'],
        **ot._provenance())


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def load_chunk_levels(window, k_max: int = 2, out_zarr=None, *, clobber: bool = False,
                      endpoint: str = CHUNK_ENDPOINT, prefix: str = CHUNK_PREFIX, fs=None,
                      osn_store=None, local_grid=None, attempts: int = 3,
                      backoff=zs.RETRY_BACKOFF_S, sleep=time.sleep, log=None,
                      report: dict = None):
    """Pull ``k = 0..k_max`` of the chunk store into the §3.3 schema.

    Parameters
    ----------
    window : sequence of str
        Hourly timestamps, ``dbof`` format, strictly increasing.
    k_max : int
        Deepest cell (and interface) kept; 2 per §3.3.
    out_zarr : path-like, optional
        The §3.3 store (``CHUNK_ZARR`` for the 72-hour window), written
        hour by hour, resumably; returns the path.  ``None`` returns the
        hours as one in-memory Dataset (and raises if any hour fails).
    clobber : bool
        Remove ``out_zarr`` first.
    endpoint, prefix, fs
        The source; ``fs`` defaults to :func:`make_fs` (any fsspec
        filesystem works -- the tests use a local one).
    osn_store : path-like, optional
        The OSN series store (``OSN_RAW_ZARR``); if given, each hour's chunk
        ``Eta`` must be bit-identical to it (a time-alignment check; 1.2 MB).
    local_grid : path-like, optional
        ``tile330_grid.zarr``; if given, the chunk grid's ``hFacC[0]``,
        ``XC``, ``YC`` must equal it.
    attempts, backoff, sleep, log, report
        As in ``osn_tiles.pull_series``; ``report`` also gets ``bytes``
        (fetched per hour).

    Returns
    -------
    str or xarray.Dataset
    """
    window = list(window)
    want = np.array([_as_dt64(ts) for ts in window], dtype='datetime64[s]')
    if len(set(window)) != len(window):
        raise ValueError('duplicated timestamps')
    if len(want) > 1 and not np.all(np.diff(want) > np.timedelta64(0, 's')):
        raise ValueError('timestamps must be strictly increasing')
    rep = {} if report is None else report
    rep.update(pulled=[], skipped=[], failed=[], not_attempted=[], repaired=0, wall_s={},
               bytes={})
    say = lambda msg: zs._say(msg, log)
    fs = make_fs(endpoint) if fs is None else fs
    retry = dict(attempts=attempts, backoff=backoff, sleep=sleep, log=log)
    eta = _osn_eta_reader(osn_store)
    lev = None

    def levels():
        nonlocal lev
        if lev is None:
            lev = zs.with_retries(lambda: _load_levels(fs, prefix, k_max, local_grid),
                                  what='grid.zarr', **retry)
            say(f'grid.zarr: {lev["n_lev"]} levels, tile {lev["nj"]}x{lev["ni"]}, '
                f'drF[0..{k_max}]={lev["drF"].tolist()}, Z={lev["Z"].tolist()}')
        return lev

    if out_zarr is None:                             # in memory: no resume, failures raise
        hours = [_load_hour(fs, prefix, ts, levels(), k_max, eta, **retry)[0] for ts in window]
        ds = xr.concat(hours, dim='time', data_vars='minimal', coords='minimal',
                       compat='override', combine_attrs='override')
        ds.attrs = _series_attrs(window, k_max, endpoint, prefix)
        return ds

    out = Path(out_zarr)
    if clobber and out.exists():
        say(f'clobber: removing {out}')
        shutil.rmtree(out)
    rep['repaired'] = zs.repair_trailing(out, log=log)
    present = zs.present_times(out)
    stored = [str(t).replace('T', ' ') for t in present]
    if len(present):
        new = want[~np.isin(want, present)]
        if len(new) and new.min() <= present.max():
            raise ValueError(f'{out} ends at {present.max()}; cannot append earlier hours '
                             f'{new[new <= present.max()].astype(str).tolist()} -- clobber '
                             'or pull them into a new store')
        ot._sync_attrs(out, present, _series_attrs(stored, k_max, endpoint, prefix), say)
    say(f'{out.name}: {len(present)} hours present, {len(window)} requested')

    for n, ts in enumerate(window):
        if want[n] in present:
            rep['skipped'].append(ts)
            continue
        t0 = time.time()
        try:
            hour, info = _load_hour(fs, prefix, ts, levels(), k_max, eta, **retry)
        except Exception as e:                       # noqa: BLE001 -- stop at the gap
            rep['failed'].append(ts)
            rep['not_attempted'] = window[n + 1:]
            say(f'{ts}: FAILED after retries ({type(e).__name__}: {e}); stopping here so the '
                f'store stays gap-free -- {len(rep["not_attempted"])} hours not attempted, '
                're-run to resume')
            break
        stored.append(ts)
        zs.append_hour(out, hour, encoding=_encoding(hour),
                       attrs=_series_attrs(stored, k_max, endpoint, prefix))
        # the store's own clock, read back, must be the requested hour
        got = zs.present_times(out)
        if len(got) != len(stored) or got[-1] != want[n]:
            raise RuntimeError(f'{out}: after appending {ts} the store time ends at '
                               f'{got[-1] if len(got) else None} ({len(got)} hours)')
        rep['pulled'].append(ts)
        rep['wall_s'][ts] = round(time.time() - t0, 1)
        rep['bytes'][ts] = info['bytes']
        say(f'{ts}: pulled and appended in {rep["wall_s"][ts]:.1f} s, '
            f'{info["bytes"] / 1e6:.0f} MB fetched ({len(stored)}/{len(window)} on disk)')
    say(f'{out.name}: done -- {len(rep["pulled"])} pulled, {len(rep["skipped"])} skipped, '
        f'{len(rep["failed"])} failed, {len(rep["not_attempted"])} not attempted')
    return str(out)
