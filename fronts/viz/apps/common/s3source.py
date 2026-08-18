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

    def land_mask(self, date: str, reference: str | None = None) -> np.ndarray:
        try:
            return _grid_plane("land_mask")
        except Exception:
            return super().land_mask(date, reference)

    def front_binary(self, date: str) -> np.ndarray:
        return _product_array(self.folder, self.run_id, date, "binary")

    def labels(self, date: str) -> np.ndarray:
        return _product_array(self.folder, self.run_id, date, "labels")

    def geometry(self, date: str) -> pd.DataFrame:
        return _product_table(self.folder, self.run_id, date, "geometry")

    def colocation(self, date: str) -> pd.DataFrame:
        return _product_table(self.folder, self.run_id, date, "colocation")

    def has_fronts(self, date: str) -> bool:
        """Cheap existence check: list the prefix, do not read the map.

        The base implementation loads the labels, which here would mean
        downloading a grid-sized array to answer a yes/no question.
        """
        try:
            _product_path(self.folder, self.run_id, date, "labels")
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
        _, fs_sync = _filesystems()
        base = f"{config.S3_BUCKET}/{config.CHUNK_FOLDER}/{chunk}"
        out = []
        for path in fs_sync.ls(base):
            name = path.rsplit("/", 1)[-1]
            if name.endswith(".zarr"):
                stamp = name[:-5]                      # 20120629T12
                out.append(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
                           f"T{stamp[9:11]}_00_00")
        return sorted(out)

    def chunk_tile(self, chunk: str, step: int, prop: str):
        times = self.chunk_timesteps(chunk)
        return _generate_tile(times[int(step)], 0, prop, chunk=chunk)

    def chunk_location(self, chunk: str) -> tuple[float, float]:
        return _chunk_centre(chunk)

    def chunk_labels(self, chunk: str, step: int):
        raise NotWiredUp("labelled fronts for chunks",
                         "Fronts have not been found for the chunk window.")


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


def _generate_tile(date: str, tile_idx: int, prop: str, chunk: str | None):
    """Build a tile in memory, straight from zarr -- no NetCDF, no profx.

    ``tile_utils.run`` does this and then writes a NetCDF, and its
    in-memory variant lives on a branch of the preprocessing repo rather
    than on all of them.  Rather than depend on which branch is checked
    out, the same steps are composed here from helpers that exist on every
    branch -- steps 1-7 of ``run``, stopping before it saves.
    """
    from dbof.tiles import tile_utils as T

    stamp = date.replace("T", " ").replace("_", ":")

    if chunk:
        lat, lon = _chunk_centre(chunk)
        return T.run(lat=lat, lon=lon, timestamp=stamp, property=prop,
                     config_path=_chunk_config(chunk), chunk=True, write=False)

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
def _chunk_config(chunk: str):
    """A minimal s3_source YAML pointing tile_utils at a chunk folder.

    grid_folder is the chunk folder too: the transfer writes a per-chunk
    grid.zarr at the same extent as the tracers.
    """
    import tempfile
    import yaml

    path = pathlib.Path(tempfile.gettempdir()) / f"chunk_source_{chunk}.yaml"
    path.write_text(yaml.safe_dump({"s3_source": {
        "s3_endpoint": config.S3_ENDPOINT,
        "bucket": config.S3_BUCKET,
        "folder": f"{config.CHUNK_FOLDER}/{chunk}",
        "grid_folder": f"{config.CHUNK_FOLDER}/{chunk}",
    }}))
    return path


def _tile_origin(tile_idx: int) -> tuple[int, int]:
    tj, ti = divmod(int(tile_idx), config.N_TILE_I)
    return ti * config.TILE_SIZE, tj * config.TILE_SIZE
