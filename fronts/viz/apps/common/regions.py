"""The six named regions for page 2.

Each region is a point on the globe.  That point resolves to one 720x720
tile on the rect grid via the preprocessing repo's ``rect_ij_to_tile``.
The resolved index is what the tile filenames carry, so it is recorded
here once the resolution has been run -- see ``resolve_all``.

The centres below are **placeholders chosen from the region names**, not
from the science.  Replace them before generating tiles; the numbers here
only need to land inside the intended current system, since a tile spans
roughly 15 degrees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fronts.viz.apps import config


@dataclass(frozen=True)
class Region:
    """One named region."""

    key: str
    name: str
    lat: float
    lon: float
    #: Resolved rect tile index.  ``None`` until ``resolve_all`` has run
    #: against the real grid and the value has been written back here.
    tile_idx: int | None = None

    def label(self) -> str:
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{self.name} ({abs(self.lat):.1f}{ns}, {abs(self.lon):.1f}{ew})"


REGIONS: tuple[Region, ...] = (
    Region("southern_ocean", "Southern Ocean", -55.0, 40.0),
    Region("gulf_stream", "Gulf Stream", 38.0, -68.0),
    Region("california", "California Current System", 36.4, -124.2),
    Region("eq_pacific", "Equatorial Tropical Pacific", 0.5, -140.0),
    Region("agulhas", "Agulhas Current", -37.0, 22.0),
    # tile_idx supplied directly, so the centre is only used to place the
    # box on the overview map -- the tile is 407 whatever the search says.
    Region("se_greenland", "SE of Greenland", 62.0, -40.0, tile_idx=407),
)

BY_KEY = {r.key: r for r in REGIONS}


def names() -> list[str]:
    """Display names, for the dropdown."""
    return [r.name for r in REGIONS]


def by_name(name: str) -> Region:
    for r in REGIONS:
        if r.name == name:
            return r
    raise KeyError(f"no region named {name!r}")


def nearest(lat: float, lon: float, *, max_deg: float = 25.0) -> Region | None:
    """The region nearest a clicked point, or ``None`` if the click missed.

    Longitude difference is wrapped, and scaled by cos(lat) so a degree of
    longitude counts for less at high latitude.
    """
    import math

    best, best_d = None, float("inf")
    for r in REGIONS:
        dlon = (lon - r.lon + 180.0) % 360.0 - 180.0
        dlon *= math.cos(math.radians(0.5 * (lat + r.lat)))
        d = math.hypot(lat - r.lat, dlon)
        if d < best_d:
            best, best_d = r, d
    return best if best_d <= max_deg else None


# --------------------------------------------------------------------------
# Tile resolution
# --------------------------------------------------------------------------

def _import_tile_mapping():
    """Import the preprocessing repo's ``tile_mapping``, robustly.

    Resolution order matches ``dev/mld/density_utils.py``: the installed
    ``dbof`` package first, then ``LLC4320_PREPROC_SRC``.
    """
    try:
        from dbof.tiles import tile_mapping
        return tile_mapping
    except ImportError:
        pass

    src = os.environ.get("LLC4320_PREPROC_SRC")
    if src:
        import sys
        for cand in (src, os.path.join(src, "src")):
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            from dbof.tiles import tile_mapping
            return tile_mapping
        except ImportError:
            pass

    raise ImportError(
        "Could not import dbof.tiles.tile_mapping.  Either `pip install -e` "
        "the llc4320-native-grid-preprocessing repo, or set "
        "LLC4320_PREPROC_SRC to its src/ directory."
    )


def resolve_tile(region: Region, latlon_to_ij) -> int:
    """Resolve a region's centre to a rect tile index.

    Parameters
    ----------
    region : Region
    latlon_to_ij : callable
        ``(lat, lon) -> (i_rect, j_rect)``.  Supplied by the caller so this
        module does not depend on how the coordinate lookup is done -- the
        real one is ``fronts.llc.coords``, and tests pass a stub.

    Returns
    -------
    int
        The flat rect tile index, 0..431.
    """
    tile_mapping = _import_tile_mapping()
    i_rect, j_rect = latlon_to_ij(region.lat, region.lon)
    return tile_mapping.rect_ij_to_tile(int(i_rect), int(j_rect)).tile_idx


def resolve_all(latlon_to_ij) -> dict[str, int]:
    """Resolve every region.  Print the result into ``REGIONS`` afterwards."""
    return {r.key: resolve_tile(r, latlon_to_ij) for r in REGIONS}


def nearest_ij(XC, YC, lat: float, lon: float) -> tuple[int, int]:
    """The rect pixel closest to a lat/lon, by search over the real grid.

    The rect grid is stitched from rotated faces, so there is no formula
    from lat/lon to (i, j) -- the only correct answer is a search.  One
    pass over 224 million cells in float32 is about a second, and XC/YC
    are memory-mapped by then, so this is cheap enough to do on demand
    rather than baking a table of indices into the source.
    """
    import numpy as np

    dlon = (np.asarray(XC, dtype=np.float32) - np.float32(lon) + 180.0)
    dlon %= 360.0
    dlon -= 180.0
    dlon *= np.float32(np.cos(np.radians(lat)))     # degrees -> comparable
    dlat = np.asarray(YC, dtype=np.float32) - np.float32(lat)

    d2 = dlon * dlon
    d2 += dlat * dlat
    j, i = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(i), int(j)


def tile_extent(provider, date: str, tile_idx: int):
    """The lon/lat box a tile actually covers, from its own coordinates.

    The overview used to draw each region at its *configured* centre,
    which is not where the tile is: the centre is resolved to a grid cell
    and then floored onto the 720-cell tile lattice, so the box and the
    tile could be a long way apart.  Reading the corner coordinates of
    the tile window is exact and cannot drift from the tile it labels.

    Returns ``(lon0, lat0, lon1, lat1)`` on the 0..360 axis the maps use.
    A tile straddling the seam comes back with ``lon1 > 360``.
    """
    import numpy as np

    from fronts.viz.apps import config

    XC, YC = provider.coords(date)
    nj, ni = XC.shape

    # The tile lattice is a property of the grid, not a constant: the
    # synthetic world is smaller and uses a smaller tile.
    n = (config.SYNTH_TILE_SIZE if getattr(provider, "synthetic", False)
         else config.TILE_SIZE)
    per_row = max(ni // n, 1)
    tj, ti = divmod(int(tile_idx), per_row)

    js = slice(min(tj * n, nj), min(tj * n + n, nj))
    iss = slice(min(ti * n, ni), min(ti * n + n, ni))
    if js.stop <= js.start or iss.stop <= iss.start:
        raise IndexError(f"tile {tile_idx} is outside a {nj}x{ni} grid")

    lon = np.asarray(XC[js, iss], dtype=float) % 360.0
    lat = np.asarray(YC[js, iss], dtype=float)

    lon0, lon1 = float(np.nanmin(lon)), float(np.nanmax(lon))
    if lon1 - lon0 > 180.0:
        # Straddles longitude 0: unwrap the low side so the box is one
        # interval rather than spanning the whole map.
        shifted = np.where(lon < 180.0, lon + 360.0, lon)
        lon0, lon1 = float(np.nanmin(shifted)), float(np.nanmax(shifted))

    return lon0, float(np.nanmin(lat)), lon1, float(np.nanmax(lat))


def tile_index_for(provider, date: str, region: Region) -> int:
    """Resolve a region to its rect tile index against the provider's grid."""
    XC, YC = provider.coords(date)
    tile_mapping = _import_tile_mapping()
    i_rect, j_rect = nearest_ij(XC, YC, region.lat, region.lon)
    return tile_mapping.rect_ij_to_tile(i_rect, j_rect).tile_idx


def synthetic_tile_idx(region: Region) -> int:
    """A stable pseudo-tile index for synthetic mode.

    The synthetic world has no LLC face geometry, so tiles are just blocks
    of the fake grid.  This keeps each region pointing at its own block.
    """
    nj, ni = config.SYNTH_SHAPE
    n = config.SYNTH_TILE_SIZE
    n_tiles = max((nj // n) * (ni // n), 1)
    return REGIONS.index(region) % n_tiles
