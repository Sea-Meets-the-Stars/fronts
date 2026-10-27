"""Where the pages get their data.

One small interface, two implementations:

``SyntheticProvider``
    Fabricates everything (:mod:`~fronts.viz.apps.common.synthetic`).  The
    default, so the app runs before any data is wired up.

``S3Provider`` (:mod:`~fronts.viz.apps.common.s3source`)
    Reads the real stores through the preprocessing repo's readers.
    Anything the pipeline has not produced yet raises :class:`NotWiredUp`
    naming which build_v5 step is outstanding.

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
    def tile(self, date: str, tile_idx: int, prop: str,
             region: str | None = None):
        """A 3-D tile as an :class:`xarray.Dataset`."""

    # -- depth ------------------------------------------------------------

    def dates_3d(self) -> list[str]:
        """Timestamps with full 3-D raw data.

        Everything depth-resolved is limited to these: the Depth page,
        Tiles, Evolution, and the depth mode of Bivariate.
        """
        available = set(self.dates())
        return [d for d in config.DATES_3D if d in available]

    def has_fronts(self, date: str) -> bool:
        """Whether the labelled fronts exist for a date."""
        try:
            self.labels(date)
        except Exception:                                   # noqa: BLE001
            return False
        return True

    def dates_with_fronts(self, candidates=None) -> list[str]:
        """The subset of *candidates* that has a label map.

        build_v5 runs date by date, so for a long while only some dates
        have fronts.  A page that needs them offers only those, rather
        than a dropdown where most entries fail.
        """
        if candidates is None:
            candidates = self.dates()
        return [d for d in candidates if self.has_fronts(d)]

    def depth_levels(self, date: str) -> list[str]:
        """Depth-level labels actually present for a date.

        Read from the channel names rather than assumed from config: a
        depth build can be partial -- run with ``depth_suffixes: [sfc]``,
        or still in progress -- and offering a level whose channels do not
        exist puts the failure after the click instead of before it.

        Falls back to the full list when nothing can be read, so the
        control is never empty.
        """
        if date not in self.dates_3d():
            return ["Surface"]

        try:
            names = set(self.field_names(date))
        except Exception:                                   # noqa: BLE001
            return list(config.DEPTH_LEVELS)

        found = [label for label, suffix in config.DEPTH_LEVELS.items()
                 if any(n.endswith(f"_{suffix}") for n in names)]
        return found or list(config.DEPTH_LEVELS)

    # -- depth channel naming ---------------------------------------------
    #
    # The DEPTH pipeline emits three kinds of channel, and the page has to
    # tell them apart:
    #
    #   suffixed  N2_sfc, N2_z25m, N2_mld, N2_mld_mean   (compute channels)
    #   bare      mixed_layer_depth, ml_heat_content     (extra_channels)
    #   bare      oceTAUX, oceQnet, SIarea, coriolis_f   (surface-only subsets)
    #
    # The selector must offer *roots* -- "N2", not four N2s -- and the depth
    # control must be free to move without resetting the field.  So the root
    # list is derived by stripping the suffixes that are actually present,
    # and the channel name is resolved back by looking in the store rather
    # than by assuming the suffix exists.

    def refresh(self) -> None:
        """Forget cached listings, so a new store is noticed.

        A no-op for a provider that reads nothing remote.
        """

    def field_roots(self, date: str) -> list[str]:
        """Selectable field names, with the depth suffix stripped.

        A bare channel is its own root, so mixed-layer depth and the wind
        appear once each alongside the depth-resolved fields.
        """
        suffixes = tuple(f"_{s}" for s in config.DEPTH_LEVELS.values())
        roots = set()
        for name in self.field_names(date):
            for suffix in suffixes:
                if name.endswith(suffix):
                    roots.add(name[: -len(suffix)])
                    break
            else:
                roots.add(name)
        return sorted(roots)

    def channel_in(self, date: str, field: str,
                   depth: str | None = None) -> str:
        """Like :meth:`channel`, but checked against what the store holds.

        Two things go wrong without this.  A bare channel -- the wind, the
        mixed-layer quantities -- has no suffix to add, so asking for
        ``mixed_layer_depth_mld`` fails on a field that is present.  And a
        depth-resolved field may not have been built at every suffix, in
        which case saying so beats a KeyError from three frames down.
        """
        names = set(self.field_names(date))
        if depth is not None:
            suffixed = self.channel(field, depth)
            if suffixed in names:
                return suffixed
        if field in names:
            return field                      # bare: nothing to suffix
        if depth is not None:
            raise KeyError(
                f"{field!r} is not in this store at {depth!r} (looked for "
                f"{self.channel(field, depth)!r} and {field!r})")
        raise KeyError(f"{field!r} is not in this store")

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

    # -- evolution chunks -------------------------------------------------

    def chunks(self) -> list[str]:
        """Named chunks available for the Evolution page."""
        return list(config.EVOLUTION_CHUNKS)

    def chunk_timesteps(self, chunk: str) -> list[str]:
        """The consecutive timestamps saved for a chunk."""
        raise NotWiredUp(f"the timestep list for chunk {chunk!r}")

    def chunk_tile(self, chunk: str, step: int, prop: str):
        """One timestep of a chunk, as a tile-shaped Dataset."""
        raise NotWiredUp(f"the store layout for chunk {chunk!r}")

    def chunk_labels(self, chunk: str, step: int):
        """Labelled fronts for one timestep of a chunk."""
        raise NotWiredUp(f"the labelled fronts for chunk {chunk!r}")

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

    def ice_mask(self, date: str) -> np.ndarray | None:
        """Cells under sea ice, or ``None`` when the store has no ice channel.

        Values under the ice pack are not comparable with the open ocean
        and are extreme enough to set the colour limits for a whole
        hemisphere, so both the map and the statistics drop them.
        """
        if config.ICE_CHANNEL not in self.field_names(date):
            return None
        area = np.asarray(self.field(date, config.ICE_CHANNEL))
        return np.isfinite(area) & (area > config.ICE_THRESHOLD)

    def drop_ice(self, date: str, name: str, values: np.ndarray) -> np.ndarray:
        """NaN the ice-covered cells of *values*, unless *name* is the ice."""
        if name in config.ICE_EXEMPT:
            return values
        ice = self.ice_mask(date)
        if ice is None:
            return values
        return np.where(ice, np.nan, values)

    def ice_exclusion(self, date: str, name: str) -> np.ndarray | None:
        """The ice mask to skip for *name*, or ``None`` if there is none.

        The mask, not a masked copy.  Binning already walks the cells one
        by one, so it can drop the ice ones for free -- whereas NaN-ing a
        copy of the field costs a grid-sized allocation (~0.9 GB) on every
        redraw of every layer.
        """
        if name in config.ICE_EXEMPT:
            return None
        return self.ice_mask(date)

    def resolve_channels(self, date: str) -> dict[str, str | None]:
        """Map the kinematic roles onto whatever this store calls them.

        Returns **root** names.  The caller applies the depth level, so a
        role follows the field being examined rather than being pinned to
        whichever level happened to be listed.

        A role counts as present if the store holds it at *any* level: the
        page then resolves it to the selected one, or to the bare channel
        where there is no depth variant (Coriolis is a function of
        latitude alone).  Matching on the exact channel name instead meant
        a store built with only ``_mld`` reported every role missing, and
        the joint PDFs came out blank with nothing to say why.
        """
        from fronts.viz.field_styles import strip_depth_suffix

        have = {strip_depth_suffix(n) for n in self.field_names(date)}
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

    def tile(self, date, tile_idx, prop, region=None):
        return self._world(date).tile_dataset(tile_idx, prop)

    # -- evolution chunks -------------------------------------------------

    def chunk_timesteps(self, chunk):
        from fronts.viz.apps.common import chunks as chunk_mod
        return chunk_mod.get_chunk(chunk).times

    def chunk_tile(self, chunk, step, prop):
        from fronts.viz.apps.common import chunks as chunk_mod
        return chunk_mod.get_chunk(chunk).dataset(int(step), prop)

    def chunk_labels(self, chunk, step):
        from fronts.viz.apps.common import chunks as chunk_mod
        return chunk_mod.get_chunk(chunk).labels(int(step))


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

# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_OVERRIDE: DataProvider | None = None


def set_provider(provider: DataProvider | None) -> None:
    """Force a provider.  Used by tests and by ``serve.py`` at start-up."""
    global _OVERRIDE
    _OVERRIDE = provider
    get_provider.cache_clear()


@lru_cache(maxsize=4)
def get_provider(pipeline: str = "SURF") -> DataProvider:
    """The provider chosen by ``FRONTS_APP_DATA``.

    *pipeline* selects which store the real provider reads.  ``SURF`` is
    the surface globals; ``DEPTH`` is the separate prefix built by
    ``run_v5_depth.yaml``, where the channels carry a depth suffix.

    This matters more than it looks.  Without it every page shared one
    SURF provider, so the Depth page would look for ``N2_mld`` in the
    *surface* store, find only bare names, and quietly show surface fields
    under a depth selector that changed nothing -- wrong with no error
    anywhere.

    The synthetic provider ignores the pipeline: the fake world has one
    store and answers for both.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    mode = os.environ.get("FRONTS_APP_DATA", config.DATA_MODE).lower()
    if mode in ("s3", "profx"):
        from fronts.viz.apps.common.s3source import S3Provider
        return S3Provider(pipeline=pipeline)
    return SyntheticProvider()
