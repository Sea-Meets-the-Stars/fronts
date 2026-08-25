"""Real-data provider, built on the preprocessing repo's readers.

``GlobalZarrDatasetReader`` already resolves
``{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}``, which is both
the surface and depth layouts; ``GlobalGridZarrReader`` gives XC/YC and the
land mask.  Nothing here re-implements either.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

import numpy as np
import pandas as pd

from fronts.viz.apps import config
from fronts.viz.apps.common import cache, tilestore
from fronts.viz.apps.common.sources import DataProvider, NotWiredUp


def _readers():
    config.ensure_dbof()
    from dbof.global_dataset_creation.zarr_dataset_global import (
        GlobalZarrDatasetReader)
    from dbof.global_dataset_creation.zarr_grid_global import (
        GlobalGridZarrReader)
    from dbof.io.filesystems import create_s3_filesystems
    return GlobalZarrDatasetReader, GlobalGridZarrReader, create_s3_filesystems


@lru_cache(maxsize=1)
def _filesystems():
    *_, create = _readers()
    return create(config.S3_ENDPOINT)


def _subset_names(pipeline: str) -> list[str]:
    config.ensure_dbof()
    from dbof.global_dataset_creation import subset_definitions as sd
    table = (sd.DEPTH_SUBSETS if pipeline == "DEPTH" else sd.SURFACE_SUBSETS)
    return [d["dataset_name"] for d in table.values()]


class S3Provider(DataProvider):
    """Reads the real stores from S3."""

    mode = "s3"
    synthetic = False

    def __init__(self, pipeline: str = "SURF"):
        self.pipeline = pipeline
        self.folder = (config.DEPTH_FOLDER if pipeline == "DEPTH"
                       else config.SURFACE_FOLDER)
        self.run_id = (config.DEPTH_RUN_ID if pipeline == "DEPTH"
                       else config.SURFACE_RUN_ID)
        # Front products are published to their own location, which for
        # DEPTH is not the folder the depth fields live in.
        self.fronts_folder = (config.DEPTH_FRONTS_FOLDER if pipeline == "DEPTH"
                              else config.SURFACE_FRONTS_FOLDER)
        self.fronts_run_id = (config.DEPTH_FRONTS_RUN_ID if pipeline == "DEPTH"
                              else config.SURFACE_FRONTS_RUN_ID)

    # -- stores ----------------------------------------------------------

    def _reader(self, date: str, dataset_name: str):
        return _cached_reader(self.folder, self.run_id,
                              config.date_to_prefix(date), dataset_name)

    def _index(self, date: str) -> dict[str, str]:
        """channel name -> dataset_name, across every subset for this date."""
        return _cached_index(self.folder, self.run_id,
                             config.date_to_prefix(date), self.pipeline)

    # -- interface -------------------------------------------------------

    def dates(self) -> list[str]:
        return _cached_dates(self.folder, self.run_id)

    def coords(self, date: str):
        return _grid_plane("XC"), _grid_plane("YC")

    def field_names(self, date: str) -> list[str]:
        return sorted(self._index(date))

    def field(self, date: str, name: str) -> np.ndarray:
        index = self._index(date)
        if name not in index:
            raise KeyError(f"no channel {name!r} for {date}")

        key = cache.make_key("field", self.folder, self.run_id, date, name)
        return cache.array(
            key,
            lambda: self._reader(date, index[name]).get_channel_snapshot(name),
        )

    def ice_mask(self, date: str):
        """Cached to disk.  Rebuilding it per layer per redraw cost a
        grid-sized comparison plus allocation every time."""
        if config.ICE_CHANNEL not in self.field_names(date):
            return None
        key = cache.make_key("ice", self.folder, self.run_id, date,
                             config.ICE_THRESHOLD)

        def build():
            area = np.asarray(self.field(date, config.ICE_CHANNEL))
            return np.isfinite(area) & (area > config.ICE_THRESHOLD)

        return cache.array(key, build)

    def land_mask(self, date: str, reference: str | None = None) -> np.ndarray:
        """Land from the reference field's NaNs.

        The grid store's hFacC would also answer this, but it is another
        grid-sized plane to fetch (~0.9 GB) for a coastline the field
        already carries: LLC masks land with NaN, so the field being drawn
        gives the same answer for free and cannot disagree with itself.
        """
        return super().land_mask(date, reference)

    def front_binary(self, date: str) -> np.ndarray:
        return _product_array(self.fronts_folder, self.fronts_run_id,
                              date, "binary")

    def labels(self, date: str) -> np.ndarray:
        return _product_array(self.fronts_folder, self.fronts_run_id,
                              date, "labels")

    def geometry(self, date: str) -> pd.DataFrame:
        return _product_table(self.fronts_folder, self.fronts_run_id,
                              date, "geometry")

    def colocation(self, date: str) -> pd.DataFrame:
        return _product_table(self.fronts_folder, self.fronts_run_id,
                              date, "colocation")

    def has_fronts(self, date: str) -> bool:
        """Cheap existence check: list the prefix, do not read the map.

        The base implementation loads the labels, which here would mean
        downloading a grid-sized array to answer a yes/no question.
        """
        try:
            _product_path(self.fronts_folder, self.fronts_run_id,
                          date, "labels")
        except Exception:                                   # noqa: BLE001
            return False
        return True

    def tile(self, date: str, tile_idx: int, prop: str, region: str | None = None):
        """A 3-D tile: from the store when it is there, generated when not.

        *region* names the store slot.  It is only a label -- the data is
        decided by *tile_idx* -- so a caller that has no region falls back
        to the tile number, which is stable and unambiguous.
        """
        slot = region or f"tile_{int(tile_idx):03d}"
        try:
            return tilestore.read(date, slot, prop)
        except FileNotFoundError:
            pass
        except Exception as exc:                            # noqa: BLE001
            print(f"[tilestore] ignoring unreadable store for "
                  f"{slot}/{prop}: {exc}")

        ds = _generate_tile(date, tile_idx, prop, chunk=None)

        if config.TILE_STORE_WRITE_BACK:
            try:
                tilestore.write(ds, date, slot, prop)
            except Exception as exc:                        # noqa: BLE001
                print(f"[tilestore] could not store {slot}/{prop}: {exc}")
        return ds

    # -- chunks ----------------------------------------------------------

    def chunks(self) -> list[str]:
        _, fs_sync = _filesystems()
        base = f"{config.S3_BUCKET}/{config.CHUNK_FOLDER}"
        return sorted(p.rsplit("/", 1)[-1] for p in fs_sync.ls(base))

    def chunk_timesteps(self, chunk: str) -> list[str]:
        """Chunk snapshots that also have fronts, in time order.

        The chunk folder holds every transferred snapshot, but fronts were
        not found for all of them -- Monterey has 18 stores and 17 sets of
        fronts.  Offering a step whose fronts are missing puts the failure
        in the middle of a movie build instead of before it, so the
        listing is intersected here.  ``has_fronts`` only lists a prefix,
        so this costs one listing per snapshot and no grid-sized reads.

        The cadence is **not** uniform: each chunk is a week of daily
        snapshots wrapped around one intensive day (3-hourly for Monterey,
        hourly for Scotia).  Anything that treats a step index as a
        constant time increment is wrong -- use the timestamps.
        """
        return list(_chunk_times(chunk, self.fronts_folder,
                                 self.fronts_run_id))

    def chunk_tile(self, chunk: str, step: int, prop: str):
        times = self.chunk_timesteps(chunk)
        return _generate_tile(times[int(step)], 0, prop, chunk=chunk)

    def chunk_location(self, chunk: str) -> tuple[float, float]:
        return _chunk_centre(chunk)

    def chunk_labels(self, chunk: str, step: int):
        """Labelled fronts for one chunk step.

        There is no chunk-specific front product: a chunk is floored onto
        the 720-cell tile lattice, so it *is* a rect tile, and the fronts
        for its timestamp are the global ones sliced to that window --
        exactly what ``pipeline.tile_labels`` does for a tile.

        Labels are assigned per date, so the same physical front carries a
        different label at every step.  Nothing here tries to hide that;
        following a front across steps is ``evolution.tracking``'s job.
        """
        times = self.chunk_timesteps(chunk)
        if not times:
            raise NotWiredUp(f"front-bearing timesteps for chunk {chunk!r}",
                             "No snapshot in this chunk has fronts.")
        date = times[int(step)]

        # Cache the 720x720 *window*, not the global plane it came from.
        # The label product is one 0.9 GB .npy with no partial reads, so
        # the first visit to a step costs a full download -- but a chunk
        # has ~17 steps, which is ~15 GB of global planes: more than the
        # cache cap, so they would evict each other and every revisit
        # would pay again.  The slices are ~2 MB each and all fit.
        key = cache.make_key("chunk-labels-v1", self.fronts_folder,
                             self.fronts_run_id, chunk, date)

        def build():
            js, iss = _chunk_window(chunk)
            path = _product_path(self.fronts_folder, self.fronts_run_id,
                                 date, "labels")
            try:
                return _product_window(path, js, iss)
            except Exception as exc:                        # noqa: BLE001
                # Fall back to the whole plane rather than failing: the
                # band read depends on the .npy layout, and being slow is
                # better than being broken if that ever changes.
                print(f"[chunk_labels] band read failed ({exc}); "
                      "falling back to the full plane", flush=True)
                return np.asarray(self.labels(date)[js, iss])

        return cache.array(key, build)


# --------------------------------------------------------------------------
# Cached readers
# --------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cached_reader(folder, run_id, date_prefix, dataset_name):
    Reader, _, _ = _readers()
    fs, _ = _filesystems()
    return Reader(bucket=config.S3_BUCKET, folder=folder, run_id=run_id,
                  dataset_name=dataset_name, fs=fs, date_prefix=date_prefix)


@lru_cache(maxsize=1)
def _cached_grid():
    _, GridReader, _ = _readers()
    fs, _ = _filesystems()
    return GridReader(bucket=config.S3_BUCKET, folder=config.GRID_FOLDER,
                      dataset_name=config.GRID_STORE, fs=fs)


# --------------------------------------------------------------------------
# Front products
# --------------------------------------------------------------------------
#: What ``build_v5`` step 5 pushes into ``{date_prefix}/Fronts/``, keyed by
#: the name the pages ask for.  The run tag varies with the finding config,
#: so these are matched as globs rather than rebuilt from parts -- see
#: ``fronts.llc.publish.PRODUCT_PATTERNS``.
PRODUCT_GLOBS = {
    # Anchored on LLC4320_ deliberately: the label map is also *_bfronts.npy,
    # and it sorts after the binary map, so a looser pattern silently returns
    # labels where a binary mask was asked for.
    "binary": "LLC4320_*_bfronts.npy",
    "labels": "labeled_fronts_global_*.npy",
    "geometry": "global_front_geometry_*.parquet",
    "colocation": "front_properties_*.parquet",
    "index": "front_index_*.parquet",
}

_STEP = {"binary": "step 2 (find)", "labels": "step 3 (group)",
         "geometry": "step 3 (group)", "colocation": "step 4 (colocate)",
         "index": "step 3 (group)"}


@lru_cache(maxsize=64)
def _product_path(folder, run_id, date: str, kind: str) -> str:
    """Locate one pushed front product on S3, or say which step is missing."""
    import fnmatch

    _, fs_sync = _filesystems()
    prefix = (f"{config.S3_BUCKET}/{folder}/{run_id}/"
              f"{config.date_to_prefix(date)}/{config.FRONTS_SUBFOLDER}")
    try:
        names = [p.rsplit("/", 1)[-1] for p in fs_sync.ls(prefix)]
    except FileNotFoundError:
        names = []

    hits = sorted(n for n in names
                  if fnmatch.fnmatch(n, PRODUCT_GLOBS[kind]))
    if not hits:
        raise NotWiredUp(
            f"{kind} for {date}",
            f"build_v5 {_STEP[kind]} has not been pushed for this date "
            f"(nothing matching {PRODUCT_GLOBS[kind]} under {prefix}).")
    return f"{prefix}/{hits[-1]}"


def _product_array(folder, run_id, date: str, kind: str) -> np.ndarray:
    """A grid-sized ``.npy`` product, cached and memory-mapped like a field."""
    path = _product_path(folder, run_id, date, kind)

    def build():
        _, fs_sync = _filesystems()
        with fs_sync.open(path, "rb") as fh:
            return np.load(fh, allow_pickle=False)

    return cache.array(cache.make_key("product", path), build)


def _product_window(path: str, j_slice: slice, i_slice: slice) -> np.ndarray:
    """Read only the rows a window needs out of a grid-sized ``.npy``.

    The label map is 12960 x 17280 of **int64** -- ``measure.label``
    returns ``np.intp`` and nothing downcasts it -- so the file is 1.67 GB
    even though it is ~99% zeros, because ``.npy`` is uncompressed.  A
    720-cell window is 4 MB of that, and pulling 1.67 GB to keep 4 MB is
    what made an Evolution build spend twenty minutes downloading.

    A C-order ``.npy`` stores rows contiguously, so the window's rows are
    one contiguous byte range: seek past the header, read 720 rows
    (95 MB, 5.6% of the file), then take the columns in memory.  An 18x
    saving for a range request and no change to what is published.

    Columns cannot be narrowed the same way -- they are strided -- so 95 MB
    is the floor for this file layout.  Publishing the labels as a chunked,
    compressed store instead would make it ~4 MB, but that is a change to
    the products rather than to the app.
    """
    _, fs_sync = _filesystems()
    with fs_sync.open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"unsupported .npy version {version} for {path}")

        if fortran or len(shape) != 2:
            raise ValueError(
                f"{path} is not a C-order 2-D array (shape={shape}, "
                f"fortran={fortran}); a row-range read would be wrong")

        start = int(j_slice.start or 0)
        stop = int(j_slice.stop if j_slice.stop is not None else shape[0])
        stop = min(stop, shape[0])
        if start >= stop:
            raise ValueError(f"empty row range {j_slice} for shape {shape}")

        row_bytes = int(shape[1]) * dtype.itemsize
        fh.seek(fh.tell() + start * row_bytes)
        raw = fh.read((stop - start) * row_bytes)

    rows = np.frombuffer(raw, dtype=dtype).reshape(stop - start, shape[1])
    return np.ascontiguousarray(rows[:, i_slice])


@lru_cache(maxsize=8)
def _product_table(folder, run_id, date: str, kind: str) -> pd.DataFrame:
    """A parquet product.  Small enough to keep in memory."""
    _, fs_sync = _filesystems()
    with fs_sync.open(_product_path(folder, run_id, date, kind), "rb") as fh:
        return pd.read_parquet(fh)


def _grid_plane(name: str) -> np.ndarray:
    """A grid plane (XC, YC, land_mask), memory-mapped from the disk cache.

    ``GlobalGridZarrReader`` exposes these as properties that re-read the
    whole 0.9 GB chunk on every access, and the pages call ``coords`` once
    per pyramid level and once per box selection.  Read once, map after.
    """
    key = f"grid_{config.GRID_FOLDER}_{config.GRID_STORE}_{name}".replace(
        "/", "_")

    def build():
        plane = getattr(_cached_grid(), name)
        if plane is None:                 # land_mask, if hFacC was not stored
            raise KeyError(f"grid store has no {name}")
        return plane

    return cache.array(key, build)


@lru_cache(maxsize=16)
def _cached_dates(folder, run_id):
    _, fs_sync = _filesystems()
    out = []
    for path in fs_sync.ls(f"{config.S3_BUCKET}/{folder}/{run_id}"):
        name = path.rsplit("/", 1)[-1]
        if len(name) == 15 and name[8] == "_":         # YYYYMMDD_HHMMSS
            out.append(f"{name[:4]}-{name[4:6]}-{name[6:8]}"
                       f"T{name[9:11]}_{name[11:13]}_{name[13:15]}")
    return sorted(out)


@lru_cache(maxsize=32)
def _cached_index(folder, run_id, date_prefix, pipeline):
    index = {}
    for dataset_name in _subset_names(pipeline):
        try:
            reader = _cached_reader(folder, run_id, date_prefix, dataset_name)
        except Exception:
            continue                                   # subset not built yet
        for channel in reader.channel_names:
            index.setdefault(channel, dataset_name)
    return index


#: What ``_compose_tile`` needs from ``dbof.tiles.tile_utils``.
#:
#: The app reproduces steps 1-7 of ``tile_utils.run`` rather than calling
#: it, because ``run`` writes a NetCDF and returns a path -- there is no
#: way to ask it for a Dataset.  The price is that this reaches into the
#: module's internals, and those differ between branches of the
#: preprocessing repo.  Checking up front turns "no attribute
#: 'resolve_property'" three frames down into a sentence that says which
#: branch is checked out and what it is missing.
_TILE_API = ("resolve_property", "_build_tile_context",
             "_load_grid_for_tile", "_load_tracers_for_tile",
             "compute_tile_property", "_build_output_dataset",
             "mit_date_to_iteration", "rect_ij_to_tile")


def _check_tile_api(T):
    """Fail with the diagnosis rather than the symptom."""
    missing = [name for name in _TILE_API if not hasattr(T, name)]
    if not missing:
        return

    branch = ""
    try:
        import pathlib as _pl
        import subprocess
        root = _pl.Path(T.__file__).resolve().parents[3]
        branch = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:                                       # noqa: BLE001
        pass

    raise RuntimeError(
        "the checked-out llc4320-native-grid-preprocessing branch"
        + (f" ({branch})" if branch else "")
        + " has a different tile_utils API: missing "
        + ", ".join(missing)
        + ".  Tile *generation* needs the API the app was built against "
          "(branch 'tiles-viz'); stored tiles under s3://dbof/tiles/ are "
          "unaffected, which is why only fields with no stored tile fail.")


def _generate_tile(date: str, tile_idx: int, prop: str, chunk: str | None):
    """Build a tile in memory, straight from zarr -- no NetCDF, no profx.

    ``tile_utils.run`` does this and then writes a NetCDF, and its
    in-memory variant lives on a branch of the preprocessing repo rather
    than on all of them.  Rather than depend on which branch is checked
    out, the same steps are composed here from helpers that exist on every
    branch -- steps 1-7 of ``run``, stopping before it saves.
    """
    config.ensure_dbof()
    from dbof.tiles import tile_utils as T
    _check_tile_api(T)

    stamp = date.replace("T", " ").replace("_", ":")

    if chunk:
        return _compose_chunk_tile(T, chunk, date, prop)

    i_rect, j_rect = _tile_origin(tile_idx)
    return _compose_tile(T, stamp, i_rect, j_rect, prop)


def _compose_tile(T, stamp: str, i_rect: int, j_rect: int, prop_name: str):
    """Steps 1-7 of ``tile_utils.run``, returning the Dataset it would save."""
    import xarray as xr

    prop = T.resolve_property(prop_name)
    s3_cfg = T._resolve_s3_source(None)
    tile = T.rect_ij_to_tile(i_rect, j_rect)

    ds_grid = T._load_grid_for_tile(s3_cfg, tile)
    ds_tracers = (
        T._load_tracers_for_tile(s3_cfg, stamp, tile,
                                 vars_needed=list(prop.vars_needed))
        if prop.vars_needed else xr.Dataset()
    )

    ds_merge, xgrid = T._build_tile_context(ds_tracers, ds_grid)
    field = T.compute_tile_property(ds_merge, xgrid, prop, mask_land=True)

    return T._build_output_dataset(
        field=field, ds_grid_tile=ds_grid, tile=tile, prop=prop,
        date_str=stamp, iteration=T.mit_date_to_iteration(stamp),
        rect_i_user=i_rect, rect_j_user=j_rect,
    )


#: The comodo annotations xgcm uses to find its horizontal axes.
#:
#: On the global grid these are stamped at load time by
#: ``get_llc_depth_gridfile``.  A chunk's own ``grid.zarr`` comes back
#: without them -- coordinate attrs do not survive the transfer -- so
#: ``Grid()`` silently finds zero axes and ``_build_tile_context`` raises
#: "missing axes ['X', 'Y']" for every step of every chunk.  Same values
#: as the preprocessing repo uses, so the two cannot drift apart.
_COMODO_ATTRS = {
    "j": {"axis": "Y"},
    "j_g": {"axis": "Y", "c_grid_axis_shift": 0.5},
    "i": {"axis": "X"},
    "i_g": {"axis": "X", "c_grid_axis_shift": 0.5},
}


def _stamp_comodo(grid):
    """Put the xgcm axis annotations back on a chunk grid."""
    import xarray as xr

    updates = {}
    for dim, attrs in _COMODO_ATTRS.items():
        if dim not in grid.dims:
            continue
        existing = (grid.coords[dim] if dim in grid.coords
                    else xr.DataArray(np.arange(grid.sizes[dim]), dims=dim))
        updates[dim] = existing.assign_attrs(attrs)
    return grid.assign_coords(updates) if updates else grid


def _chunk_stores(chunk: str, date: str):
    """The chunk's timestep store and its own grid, as Datasets.

    A chunk is a 720 x 720 box saved whole, so there is no face slicing
    to do -- which is the only thing ``tile_utils`` would have added.
    Reading the two stores here keeps the chunk path off the branch that
    carries ``run(chunk=True)``.

    The grid needs repair on the way through: see :data:`_COMODO_ATTRS`.
    """
    import xarray as xr

    fs, _ = _filesystems()
    base = f"{config.S3_BUCKET}/{config.CHUNK_FOLDER}/{chunk}"
    stamp = config.date_to_tile_stamp(date)               # 20121103T07

    ds = xr.open_zarr(fs.get_mapper(f"{base}/{stamp}.zarr"))
    if ds.sizes.get("time") == 1:
        ds = ds.isel(time=0, drop=True)
    grid = xr.open_zarr(
        fs.get_mapper(f"{base}/{config.CHUNK_GRID_STORE}")).compute()

    # Only the comodo repair.  Do NOT drop the `face` dimension:
    # compute_tile_property selects on it, so removing it fails with
    # "Dimensions {'face'} do not exist" for every step.
    return ds.compute(), _stamp_comodo(grid)


def _compose_chunk_tile(T, chunk: str, date: str, prop_name: str):
    """One chunk timestep as a tile-shaped Dataset.

    Same steps as :func:`_compose_tile`, but the tracers and the grid come
    from the chunk's own stores rather than from a slice of the global
    raw store.
    """
    prop = T.resolve_property(prop_name)
    ds_tracers, ds_grid = _chunk_stores(chunk, date)

    # Provenance: the transfer floors a chunk onto the 720-cell tile
    # lattice, so it does correspond to a real rect tile -- which is what
    # lets the label alignment work exactly as it does for a tile.
    from fronts.viz.apps.common import regions as regions_mod
    XC, YC = _grid_plane("XC"), _grid_plane("YC")
    lat, lon = _chunk_centre(chunk)
    i_rect, j_rect = regions_mod.nearest_ij(XC, YC, lat, lon)
    tile = T.rect_ij_to_tile(i_rect, j_rect)

    ds_merge, xgrid = T._build_tile_context(ds_tracers, ds_grid)
    field = T.compute_tile_property(ds_merge, xgrid, prop, mask_land=True)

    stamp = date.replace("T", " ").replace("_", ":")
    return T._build_output_dataset(
        field=field, ds_grid_tile=ds_grid, tile=tile, prop=prop,
        date_str=stamp, iteration=T.mit_date_to_iteration(stamp),
        rect_i_user=i_rect, rect_j_user=j_rect,
    )


@lru_cache(maxsize=8)
def _chunk_centre(chunk: str) -> tuple[float, float]:
    """Centre of a chunk, from its own grid.zarr -- no hard-coded table."""
    import xarray as xr

    fs, _ = _filesystems()
    store = (f"{config.S3_BUCKET}/{config.CHUNK_FOLDER}/{chunk}/"
             f"{config.CHUNK_GRID_STORE}")
    grid = xr.open_zarr(fs.get_mapper(store))
    return float(grid.YC.mean()), float(grid.XC.mean())


@lru_cache(maxsize=8)
def _chunk_times(chunk: str, fronts_folder: str, fronts_run_id: str):
    """Chunk snapshots that have fronts, as a tuple so it can be cached."""
    _, fs_sync = _filesystems()
    base = f"{config.S3_BUCKET}/{config.CHUNK_FOLDER}/{chunk}"

    stamps = []
    for path in fs_sync.ls(base):
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".zarr") and name != config.CHUNK_GRID_STORE:
            stamp = name[:-5]                              # 20120629T12
            stamps.append(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
                          f"T{stamp[9:11]}_00_00")

    out = []
    for date in sorted(stamps):
        try:
            _product_path(fronts_folder, fronts_run_id, date, "labels")
        except NotWiredUp:
            continue
        out.append(date)
    return tuple(out)


@lru_cache(maxsize=8)
def _chunk_window(chunk: str):
    """The chunk's window on the global rect grid, as ``(j_slice, i_slice)``.

    Derived the same way :func:`_compose_chunk_tile` derives the tile
    identity -- centre from the chunk's own grid, then the enclosing tile
    -- so the labels and the data cannot disagree about where the chunk is.

    **This is expensive and the answer is four integers**, so it is cached
    on disk rather than only per process.  Resolving a lat/lon on the rect
    grid means a search -- the grid is stitched from rotated faces, so
    there is no formula -- and that search reads both 0.9 GB coordinate
    planes and makes several full-size temporaries.  Paying ~2 GB once per
    chunk ever is fine; paying it on the first click after every server
    restart is what made *Load chunk* look hung.
    """
    key = cache.make_key("chunk-window-v1", chunk)

    def build():
        config.ensure_dbof()
        from dbof.tiles import tile_utils as T
        from fronts.viz.apps.common import regions as regions_mod

        lat, lon = _chunk_centre(chunk)
        i_rect, j_rect = regions_mod.nearest_ij(
            _grid_plane("XC"), _grid_plane("YC"), lat, lon)
        info = T.rect_ij_to_tile(i_rect, j_rect)
        return np.array([info.rect_j_slice.start, info.rect_j_slice.stop,
                         info.rect_i_slice.start, info.rect_i_slice.stop])

    j0, j1, i0, i1 = (int(v) for v in cache.array(key, build))
    return slice(j0, j1), slice(i0, i1)


def _tile_origin(tile_idx: int) -> tuple[int, int]:
    tj, ti = divmod(int(tile_idx), config.N_TILE_I)
    return ti * config.TILE_SIZE, tj * config.TILE_SIZE
