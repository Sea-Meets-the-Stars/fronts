"""Where the pages get their data.

One small interface, two implementations:

``SyntheticProvider``
    Fabricates everything (:mod:`~fronts.viz.apps.common.synthetic`).  The
    default, so the app runs before any data is wired up.

``S3Provider``
    Reads the real stores.  Every method that needs a store layout we have
    not confirmed raises :class:`NotWiredUp` with the exact question it
    needs answered, rather than guessing a path and failing obscurely.

Pages never construct a provider directly; they call :func:`get_provider`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd

from fronts.viz.apps import config


class NotWiredUp(NotImplementedError):
    """Raised when the real data layout has not been supplied yet.

    Carries the specific thing that needs confirming so the message in the
    UI is actionable rather than a bare traceback.
    """

    def __init__(self, what: str, how: str = ""):
        msg = f"Not wired up yet: {what}"
        if how:
            msg += f"\n  -> {how}"
        super().__init__(msg)
        self.what = what
        self.how = how


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

class DataProvider(ABC):
    """Everything the three pages need from a timestamp."""

    #: Human-readable name shown in the status bar.
    mode: str = "?"

    #: True when the numbers are fabricated, so the UI can say so loudly.
    synthetic: bool = False

    @abstractmethod
    def dates(self) -> list[str]:
        """Timestamps available."""

    @abstractmethod
    def coords(self, date: str) -> tuple[np.ndarray, np.ndarray]:
        """``(XC, YC)`` -- 2-D longitude and latitude on the native grid."""

    @abstractmethod
    def field_names(self, date: str) -> list[str]:
        """Channels available for this date."""

    @abstractmethod
    def field(self, date: str, name: str) -> np.ndarray:
        """One 2-D field.  Land is NaN."""

    @abstractmethod
    def front_binary(self, date: str) -> np.ndarray:
        """Binary front mask, 1 = front."""

    @abstractmethod
    def labels(self, date: str) -> np.ndarray:
        """Integer labelled fronts, 0 = background."""

    @abstractmethod
    def geometry(self, date: str) -> pd.DataFrame:
        """Per-front geometry table."""

    @abstractmethod
    def colocation(self, date: str) -> pd.DataFrame:
        """Per-front colocated property table."""

    @abstractmethod
    def tile(self, date: str, tile_idx: int, prop: str):
        """A 3-D tile as an :class:`xarray.Dataset`."""

    # -- depth ------------------------------------------------------------

    def dates_3d(self) -> list[str]:
        """Timestamps with full 3-D raw data.

        Everything depth-resolved is limited to these: the Depth page,
        Tiles, Evolution, and the depth mode of Bivariate.
        """
        available = set(self.dates())
        return [d for d in config.DATES_3D if d in available]

    def depth_levels(self, date: str) -> list[str]:
        """Depth-level labels available for a date.

        Only the 3-D dates carry depth-resolved channels; everywhere else
        the only level is the surface.
        """
        if date not in self.dates_3d():
            return ["Surface"]
        return list(config.DEPTH_LEVELS)

    def channel(self, field: str, depth: str | None = None) -> str:
        """Resolve a field (+ depth level) to the channel name in the store.

        The SURF pipeline emits bare names; the DEPTH pipeline suffixes
        them -- ``relative_vorticity`` becomes ``relative_vorticity_mld``.
        A depth of ``None`` or ``'Surface'`` on a surface store gives the
        bare name back.
        """
        if depth is None:
            return field
        suffix = config.DEPTH_LEVELS.get(depth)
        if suffix is None:
            raise KeyError(f"unknown depth level {depth!r}")
        return f"{field}_{suffix}"

    def front_stats(self, date: str) -> list[str]:
        """Per-front statistic suffixes actually present in the colocation table.

        Built from the columns rather than assumed, so a re-run of step 4
        with extra percentiles shows up with no code change.
        """
        try:
            df = self.colocation(date)
        except Exception:                                   # noqa: BLE001
            return list(config.FRONT_STATS)

        found = set()
        for col in df.columns:
            if "_" not in col:
                continue
            suffix = col.rsplit("_", 1)[1]
            if suffix in config.FRONT_STATS or (
                suffix.startswith("p") and suffix[1:].isdigit()
            ):
                found.add(suffix)

        ordered = [s for s in config.FRONT_STATS if s in found]
        extra = sorted(found - set(config.FRONT_STATS))
        return ordered + extra

    # -- derived, shared by both implementations -------------------------

    def land_mask(self, date: str, reference: str | None = None) -> np.ndarray:
        """Land as a boolean array, taken from a reference field's NaNs.

        The LLC output masks land with NaN, so the model's own coastline is
        available for free and matches the grid exactly -- no external
        coastline dataset, and no mismatch between the two.
        """
        if reference is None:
            names = self.field_names(date)
            reference = "gradb2" if "gradb2" in names else names[0]
        return ~np.isfinite(self.field(date, reference))

    def resolve_channels(self, date: str) -> dict[str, str | None]:
        """Map the kinematic roles onto whatever this store calls them.

        The SURFACE pipeline emits ``relative_vorticity`` / ``strain_mag`` /
        ``coriolis_f``; the DEPTH pipeline suffixes the first two
        (``relative_vorticity_sfc``, ...) and moves Coriolis to the extra
        channels.  Rather than assume, look.
        """
        have = set(self.field_names(date))
        out: dict[str, str | None] = {}
        for role, candidates in config.KINEMATIC_ROLES.items():
            out[role] = next((c for c in candidates if c in have), None)
        return out


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------

class SyntheticProvider(DataProvider):
    """Fabricated data, so the pages run with nothing installed."""

    mode = "synthetic"
    synthetic = True

    def _world(self, date):
        from fronts.viz.apps.common import synthetic
        return synthetic.get_world(date)

    def dates(self):
        return list(config.DATES)

    def coords(self, date):
        w = self._world(date)
        return w.XC, w.YC

    def field_names(self, date):
        return sorted(self._world(date).fields)

    def field(self, date, name):
        """One field, resolving a depth suffix if the name carries one.

        The synthetic world holds surface fields only.  A depth-suffixed
        name is served by modulating the surface field, which is enough to
        exercise the Depth page's plumbing without pretending to model
        anything.
        """
        w = self._world(date)
        if name in w.fields:
            return w.fields[name]

        base, suffix = _split_depth_suffix(name)
        if base in w.fields:
            return w.depth_variant(base, suffix)

        raise KeyError(f"no synthetic field {name!r}")

    def field_names_at(self, date, depth=None):
        """Channel names as they would appear at a depth level."""
        base = self.field_names(date)
        if depth in (None, "Surface") and date not in self.dates_3d():
            return base
        suffix = config.DEPTH_LEVELS.get(depth or "Surface")
        return [f"{n}_{suffix}" for n in base]

    def front_binary(self, date):
        return self._world(date).fronts

    def labels(self, date):
        return self._world(date).labels

    def geometry(self, date):
        return _cached_geometry(date)

    def colocation(self, date):
        return _cached_colocation(date)

    def tile(self, date, tile_idx, prop):
        return self._world(date).tile_dataset(tile_idx, prop)


_DEPTH_SUFFIXES = set(config.DEPTH_LEVELS.values())


def _split_depth_suffix(name: str) -> tuple[str, str | None]:
    """``'relative_vorticity_mld'`` -> ``('relative_vorticity', 'mld')``.

    Longest suffix wins, so ``mld_mean`` is not mistaken for ``mld``.
    """
    for suffix in sorted(_DEPTH_SUFFIXES, key=len, reverse=True):
        if name.endswith("_" + suffix):
            return name[: -len(suffix) - 1], suffix
    return name, None


@lru_cache(maxsize=8)
def _cached_geometry(date):
    from fronts.viz.apps.common import synthetic
    return synthetic.get_world(date).geometry_table()


@lru_cache(maxsize=8)
def _cached_colocation(date):
    from fronts.viz.apps.common import synthetic
    return synthetic.get_world(date).colocation_table()


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------

class S3Provider(DataProvider):
    """Reads the real global products and pre-generated tiles.

    Deliberately incomplete.  The store layout under
    ``globals_for_cutouts/v2_2_01/`` has not been confirmed, so the methods
    that need it raise :class:`NotWiredUp` naming exactly what to check.
    Filling these in is a small, mechanical job once the listing exists --
    see ``docs/viz/apps/WIRING.md``.
    """

    mode = "s3"
    synthetic = False

    def __init__(self, root: str | None = None, tile_dir=None):
        self.root = (root or config.S3_ROOT).rstrip("/")
        self.tile_dir = tile_dir or config.TILE_DIR

    # -- helpers ---------------------------------------------------------

    def _prefix(self, date: str) -> str:
        return f"{self.root}/{config.date_to_prefix(date)}"

    def _fs(self):
        import s3fs
        return s3fs.S3FileSystem(anon=False)

    def listing(self, date: str) -> list[str]:
        """Everything in the date's directory.  Used by the wiring script."""
        fs = self._fs()
        return sorted(fs.ls(self._prefix(date).replace("s3://", "")))

    # -- interface -------------------------------------------------------

    def dates(self):
        return list(config.DATES)

    def coords(self, date):
        raise NotWiredUp(
            "the coordinate source for the global grid",
            "Confirm whether XC/YC ship inside the per-date store or come "
            "from a separate LLC_coords_lat_lon.nc, then fill in "
            "S3Provider.coords.",
        )

    def field_names(self, date):
        raise NotWiredUp(
            "the channel list for a date",
            "Run the listing command in docs/viz/apps/WIRING.md; the names "
            "decide whether kinematic channels carry depth suffixes.",
        )

    def field(self, date, name):
        raise NotWiredUp(
            f"the file layout for channel {name!r}",
            "Need one example filename to pin the naming convention.",
        )

    def front_binary(self, date):
        raise NotWiredUp("the binary-fronts filename")

    def labels(self, date):
        raise NotWiredUp("the labelled-fronts filename")

    def geometry(self, date):
        raise NotWiredUp("the geometry parquet filename")

    def colocation(self, date):
        raise NotWiredUp("the colocation parquet filename")

    def tile(self, date, tile_idx, prop):
        """Read a pre-generated tile NetCDF.

        This one *is* implemented -- the tile layout is known, because
        ``dbof.cli.generate_tile`` writes it and ``density_utils.load_tile``
        validates it.  It just needs the files to exist.
        """
        from pathlib import Path
        import xarray as xr

        stamp = config.date_to_tile_stamp(date)
        prefix = "density" if prop in ("density", "sigma0") else prop.lower()
        path = Path(self.tile_dir) / f"{prefix}_tile{tile_idx}_{stamp}.nc"
        if not path.exists():
            raise NotWiredUp(
                f"tile file {path}",
                "Generate it with `python -m dbof.cli.generate_tile "
                f"--i <I> --j <J> --timestamp '<TS>' --property {prop} "
                "--output $TILES` (see docs/viz/apps/WIRING.md).",
            )
        return xr.open_dataset(path)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_OVERRIDE: DataProvider | None = None


def set_provider(provider: DataProvider | None) -> None:
    """Force a provider.  Used by tests and by ``serve.py`` at start-up."""
    global _OVERRIDE
    _OVERRIDE = provider
    get_provider.cache_clear()


@lru_cache(maxsize=1)
def get_provider() -> DataProvider:
    """The provider chosen by ``FRONTS_APP_DATA``."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    mode = os.environ.get("FRONTS_APP_DATA", config.DATA_MODE).lower()
    if mode == "s3":
        return S3Provider()
    return SyntheticProvider()
