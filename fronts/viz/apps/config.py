"""Configuration for the front visualisation web pages.

Everything environment- or deployment-specific lives here so the page
modules stay free of paths and magic numbers.

Environment variables
---------------------
FRONTS_APP_DATA : {'synthetic', 's3'}
    Which data provider the pages use.  Defaults to ``'synthetic'`` so the
    app runs with no data at all.  Set to ``'s3'`` once the real stores are
    wired up (see :mod:`fronts.viz.apps.common.sources`).
FRONTS_APP_S3_ROOT : str
    Root of the global products, e.g. ``s3://dbof/globals_for_cutouts/v2_2_01``.
FRONTS_APP_TILE_DIR : str
    Directory holding the pre-generated 3-D tile NetCDFs.
FRONTS_APP_CACHE : str
    Directory for the on-disk statistics cache.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
#: Timestamps offered on the surface page.  The real build has 100; the
#: synthetic provider ships a handful so the selector is exercised.
DATES: list[str] = [
    "2012-02-29T18_00_00",
    "2012-05-16T06_00_00",
    "2012-09-18T11_00_00",
    "2012-11-09T12_00_00",
]

#: The timestamps with full 3-D raw data.  Everything depth-resolved -- the
#: Depth page, Tiles, Evolution, and the depth mode of Bivariate -- is
#: limited to these.  Same four: the V5 build covers exactly these dates.
DATES_3D: list[str] = list(DATES)

DEFAULT_DATE: str = DATES[0]


# --------------------------------------------------------------------------
# Which pages are served
# --------------------------------------------------------------------------
#: Routes ``serve.py`` publishes.  Dropping a name hides the page without
#: touching its module -- the Depth page and the Bivariate depth mode are
#: complete and tested, just not offered for now.  Put "depth" back here
#: (and "Depth" in BIVARIATE_MODES) to bring them back.
ENABLED_PAGES: tuple[str, ...] = (
    "surface",
    "bivariate",
    "tiles",
    "evolution",
)

#: Modes the Bivariate page offers.  ``("Surface", "Depth")`` for both.
BIVARIATE_MODES: tuple[str, ...] = ("Surface",)


# --------------------------------------------------------------------------
# Depth levels
# --------------------------------------------------------------------------
#: Maps the page's label onto the suffix the channel already carries.  The
#: suffixes are not invented here -- they are
#: ``dbof.global_dataset_creation.subset_definitions.DEFAULT_DEPTH_SUFFIXES``.
DEPTH_LEVELS: dict[str, str] = {
    "Surface": "sfc",
    "25 m": "z25m",
    "Mixed layer depth": "mld",
    "Mean over mixed layer": "mld_mean",
}

DEFAULT_DEPTH_LEVEL = "Surface"


# --------------------------------------------------------------------------
# Per-front statistics
# --------------------------------------------------------------------------
#: Colocation column suffixes the front-properties panels can plot.
#:
#: ``run_v5_100_timesteps.yaml`` sets ``percentiles: [25, 75, 90]``, so p95
#: does NOT exist -- p90 does, and median is p50.  The selector is built
#: from the columns actually present, so adding 95 to the config and
#: re-running step 4 makes it appear with no code change.
FRONT_STATS: tuple[str, ...] = ("mean", "median", "p25", "p75", "p90")

DEFAULT_FRONT_STAT = "median"


def date_to_prefix(date: str) -> str:
    """``'2012-05-16T06_00_00'`` -> ``'20120516_060000'`` (store directory)."""
    d, t = date.split("T")
    return f"{d.replace('-', '')}_{t.replace('_', '')}"


def date_to_tile_stamp(date: str) -> str:
    """``'2012-05-16T06_00_00'`` -> ``'20120516T06'`` (tile filename stamp)."""
    d, t = date.split("T")
    return f"{d.replace('-', '')}T{t.split('_')[0]}"


# --------------------------------------------------------------------------
# The LLC4320 rectangular grid
# --------------------------------------------------------------------------
RECT_H = 12960          # 3 * 4320
RECT_W = 17280          # 4 * 4320
TILE_SIZE = 720
N_TILE_I = RECT_W // TILE_SIZE      # 24
N_TILE_J = RECT_H // TILE_SIZE      # 18


# --------------------------------------------------------------------------
# Display pyramid
# --------------------------------------------------------------------------
#: Regular lat/lon raster widths, coarsest first.  The map picks the finest
#: level whose cells are still smaller than a screen pixel at current zoom.
PYRAMID_WIDTHS: tuple[int, ...] = (1440, 2880, 5760, 11520)

#: Latitude range covered by the display pyramid.  LLC4320 stops short of
#: the poles; anything outside this is empty in every level.
PYRAMID_LAT_RANGE: tuple[float, float] = (-80.0, 80.0)


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------
DATA_MODE = os.environ.get("FRONTS_APP_DATA", "synthetic").lower()

S3_ROOT = os.environ.get(
    "FRONTS_APP_S3_ROOT", "s3://dbof/globals_for_cutouts/v2_2_01"
)

#: S3 layout, matching GlobalZarrDatasetReader's
#: {bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}.
def _default_endpoint() -> str:
    """The endpoint the preprocessing repo already uses."""
    try:
        from dbof.global_dataset_creation.data_sources import LLC_DEPTH_SOURCE
        return LLC_DEPTH_SOURCE["s3_endpoint"]
    except Exception:
        return "https://s3-west.nrp-nautilus.io"


S3_ENDPOINT = os.environ.get("FRONTS_APP_S3_ENDPOINT") or _default_endpoint()
S3_BUCKET = os.environ.get("FRONTS_APP_S3_BUCKET", "dbof")

#: The V5 build: subsets and Fronts/ products for the four dates above,
#: all under one prefix --
#: ``s3://dbof/globals_for_chunks/V5/{date}/{subset.zarr | Fronts/}``.
#: Surface and depth read the same folder; they differ only in the channel
#: names they ask for (depth channels carry a suffix).
SURFACE_FOLDER = os.environ.get("FRONTS_APP_SURFACE_FOLDER",
                                "globals_for_chunks")
SURFACE_RUN_ID = os.environ.get("FRONTS_APP_SURFACE_RUN_ID", "V5")

# The depth fields are built by run_v5_depth.yaml into their own prefix, so
# they cannot collide with the surface globals -- different channel names
# (depth channels carry a suffix), different store, different folder.
DEPTH_FOLDER = os.environ.get("FRONTS_APP_DEPTH_FOLDER",
                              "globals_for_chunks_depth")
DEPTH_RUN_ID = os.environ.get("FRONTS_APP_DEPTH_RUN_ID", "V5")

#: The 2-D (rect) grid, read by GlobalGridZarrReader -- XC/YC and hFacC for
#: the whole 12960 x 17280 grid.  Distinct from the per-face grid.zarr in
#: LLC4320_RAW/{SURFACE,DEPTH}, which is what tile_utils reads.
GRID_FOLDER = os.environ.get("FRONTS_APP_GRID_FOLDER", "LLC4320_GRID_2D")
GRID_STORE = os.environ.get("FRONTS_APP_GRID_STORE", "llc4320_grid.zarr")

#: Pre-generated 3-D tiles, one zarr per (date, region, field):
#: ``s3://{bucket}/{TILE_STORE_FOLDER}/{date_prefix}/{region}/{field}.zarr``.
#: Generating a tile is ~15 s, so the page reads these when they exist.
TILE_STORE_FOLDER = os.environ.get("FRONTS_APP_TILE_STORE", "tiles")

#: Whether a tile generated on demand is written back to the store.  With
#: this on, the store fills in as the page is used and the batch builder is
#: a warm-up rather than a prerequisite.
TILE_STORE_WRITE_BACK = (
    os.environ.get("FRONTS_APP_TILE_WRITE_BACK", "1") != "0")

#: Fields the batch builder does by default -- the ones the demo needs,
#: not all of TILE_FIELDS_3D.  density is always included: the 3-D geometry
#: is built from it whatever field is being coloured.
TILE_STORE_DEFAULT_FIELDS: tuple[str, ...] = (
    "density", "Ri", "N2", "relative_vorticity", "gradb2", "turner_angle",
)

#: Where build_v5 step 5 puts the front products, inside each date's
#: directory alongside the zarr stores they came from.  See
#: ``fronts.llc.publish``.
FRONTS_SUBFOLDER = os.environ.get("FRONTS_APP_FRONTS_SUBFOLDER", "Fronts")

#: The front products do not always live beside the fields they came from.
#: Surface: ``globals_for_cutouts/v2_2_01/{date}/Fronts/`` -- the same
#: folder as the surface stores.  Depth: ``globals_for_chunks/V5/{date}/
#: Fronts/``, which is *not* where the depth fields are, so it is
#: configured separately rather than derived.
SURFACE_FRONTS_FOLDER = os.environ.get(
    "FRONTS_APP_SURFACE_FRONTS_FOLDER", "globals_for_chunks")
SURFACE_FRONTS_RUN_ID = os.environ.get(
    "FRONTS_APP_SURFACE_FRONTS_RUN_ID", "V5")

DEPTH_FRONTS_FOLDER = os.environ.get(
    "FRONTS_APP_DEPTH_FRONTS_FOLDER", "globals_for_chunks")
DEPTH_FRONTS_RUN_ID = os.environ.get(
    "FRONTS_APP_DEPTH_FRONTS_RUN_ID", "V5")

#: Raw 3-D and chunk stores, for tile generation.
RAW_DEPTH_FOLDER = os.environ.get("FRONTS_APP_RAW_DEPTH_FOLDER",
                                  "LLC4320_RAW/DEPTH")
CHUNK_FOLDER = os.environ.get("FRONTS_APP_CHUNK_FOLDER",
                              "LLC4320_RAW/CHUNKS")

#: Per-chunk grid store, written once by the transfer alongside the
#: timestep stores.  Carries the chunk's own XC/YC, so its location is
#: read rather than configured.
CHUNK_GRID_STORE = "grid.zarr"

TILE_DIR = Path(os.environ.get("FRONTS_APP_TILE_DIR", "./tiles")).expanduser()

CACHE_DIR = Path(
    os.environ.get("FRONTS_APP_CACHE", "~/.cache/fronts-viz")
).expanduser()

#: Disk budget for everything under CACHE_DIR -- cached coordinate planes,
#: pyramid levels, and statistics.  Least-recently-used files are evicted
#: once the total goes over.  XC and YC alone are 0.9 GB each.
CACHE_CAP_BYTES = int(float(os.environ.get("FRONTS_APP_CACHE_GB", "10")) * 2**30)


# --------------------------------------------------------------------------
# Synthetic-world size
# --------------------------------------------------------------------------
#: Grid used in synthetic mode.  Deliberately far smaller than the real
#: rect grid so the prototype is fast, and deliberately *irregular* in
#: latitude so the pyramid and selection code paths are genuinely exercised.
SYNTH_SHAPE: tuple[int, int] = (540, 720)

#: Tile size in synthetic mode.  The real one is 720; a 3-D synthetic tile
#: at that width would be ~170 MB per field.
SYNTH_TILE_SIZE: int = 180

#: Depth levels in a synthetic 3-D tile.
SYNTH_NZ: int = 40


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------
#: Fields the pages know how to normalise into the (zeta/f, sigma/|f|) plane.
#: Resolved against whatever the store actually calls them at start-up --
#: see ``sources.resolve_channels``.
KINEMATIC_ROLES = {
    "vorticity": ("relative_vorticity", "relative_vorticity_sfc"),
    "strain": ("strain_mag", "strain_mag_sfc"),
    "coriolis": ("coriolis_f",),
}

#: 3-D tile fields offered on page 2.
#:
#: ``TileProperty`` carries no dimensionality flag, so this cannot be
#: derived from the registry -- the 2-D channels (``mixed_layer_depth``,
#: ``Eta``, ``oceTAUX``, ...) are only rejected at load time, by which
#: point the user has already picked one.  Hence an explicit list.
#:
#: ``test_tile_field_list_matches_the_registry`` asserts every name here
#: still exists in ``TILE_PROPERTIES``.  It cannot assert they are all
#: 3-D -- confirm that against a real tile at Milestone 0 and prune
#: anything that turns out to be surface-only.
TILE_FIELDS_3D: tuple[str, ...] = (
    # stratification and shear
    "Ri", "N2", "vertical_shear", "Fr", "Ro", "Bu", "R_ib",
    # kinematics
    "relative_vorticity", "rossby_number",
    "strain_mag", "strain_n", "strain_s", "divergence", "okubo_weiss",
    # frontogenesis -- the total only.  The geostrophic and ageostrophic
    # parts are surface-only quantities (they need the SSH-derived
    # geostrophic velocity), so they have no meaning on a depth-resolved
    # tile and are not offered here.
    "frontogenesis_tendency",
    # potential vorticity
    "ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt",
    # tracers and derived scalars.  KE is left out: it is built from the
    # mixed-layer depth, so it exists at one level only and has nothing to
    # section through.
    "density", "buoyancy", "turner_angle", "Theta", "Salt",
    "gradb2", "gradrho2", "gradtheta2", "gradsalt2",
    # velocity and buoyancy fluxes.  ug / vg are surface-only for the same
    # reason as the frontogenesis split, so they are left out too.
    "U", "V", "W", "uB", "vB", "wB",
)

#: Surface-only properties a chunk store can produce.
#:
#: These were always in ``dbof.tiles.field_registry`` -- the app simply
#: never offered them, because the only field list it had was the
#: depth-resolved one.  They have no ``Z`` axis, so they can colour a plan
#: view and nothing else: no curtain, no profile, no isopycnal.  Kept in
#: their own list so that limit is structural rather than remembered.
CHUNK_SURFACE_FIELDS: tuple[str, ...] = (
    # surface_wind
    "oceTAUX", "oceTAUY", "wind_stress_curl",
    "ekman_pumping", "u_ekman", "v_ekman", "oceQnet",
    # depth-integrated or mixed-layer quantities: a single number per
    # column, so they are surface-only for the same reason the wind is --
    # there is no profile to section, however much depth went into them.
    "mixed_layer_depth", "ml_heat_content", "KE",
)


def is_surface_only(field: str) -> bool:
    """True for fields that exist at the surface alone."""
    return field in CHUNK_SURFACE_FIELDS


#: Field used for the 3-D geometry.  Always density.
TILE_GEOMETRY_FIELD = "density"


# --------------------------------------------------------------------------
# The preprocessing repo
# --------------------------------------------------------------------------
def ensure_dbof() -> None:
    """Make ``dbof`` importable, or say exactly how to fix it.

    Resolution order is the one ``dev/mld/density_utils.py`` already uses:
    the installed package first, then ``LLC4320_PREPROC_SRC``.  Idempotent,
    so it is cheap to call from every entry point.

    This exists because the failure it replaces was a bare
    ``ModuleNotFoundError: No module named 'dbof'`` raised four frames deep
    inside a page build, which says nothing about the actual cause -- an
    interpreter without the repo, usually the wrong conda env.  Naming the
    interpreter is what makes that obvious.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("dbof") is not None:
        return

    src = os.environ.get("LLC4320_PREPROC_SRC")
    if src:
        for cand in (src, os.path.join(src, "src")):
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        importlib.invalidate_caches()
        if importlib.util.find_spec("dbof") is not None:
            return

    raise ModuleNotFoundError(
        "the llc4320-native-grid-preprocessing package (`dbof`) is not "
        f"importable from this interpreter:\n    {sys.executable}\n"
        "Either activate the environment it is installed in, `pip install "
        "-e` the repo, or set LLC4320_PREPROC_SRC to its src/ directory."
    )


# --------------------------------------------------------------------------
# Sea ice
# --------------------------------------------------------------------------
#: Cells under sea ice carry values that are not comparable with the open
#: ocean -- gradients under the ice pack are large enough to set the colour
#: limits for the whole map and swamp everything else.  They are dropped
#: from both the display and the statistics.
ICE_CHANNEL = "SIarea"

#: Ice concentration above which a cell counts as ice-covered.  0.15 is the
#: usual sea-ice-extent convention.
ICE_THRESHOLD = float(os.environ.get("FRONTS_APP_ICE_THRESHOLD", "0.15"))

#: Channels that describe the ice itself, and so are never ice-masked.
ICE_EXEMPT: frozenset[str] = frozenset({ICE_CHANNEL, "__land__"})


# --------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------
#: Named chunks: one spatial box, saved at many consecutive timesteps.
#: A chunk is the same size as a tile (720 x 720) but comes from
#: ``s3://dbof/LLC4320_RAW/CHUNKS/{chunk}/YYYYMMDD_HHMMSS.zarr`` rather
#: than from the full 3-D store.
#:
#: This is an **allow-list**, not a listing.  The chunks folder also holds
#: partial transfers -- amundsen, bellingshausen, ross, weddell each have
#: a couple of timesteps rather than a window -- and offering those on the
#: page just produces a movie that cannot be built.  Add a name here once
#: its transfer is complete.  An empty tuple means "offer whatever is on
#: S3", which is what the checks use.
EVOLUTION_CHUNKS: tuple[str, ...] = (
    "monterey_bay",
    "southern_ocean_scotia_sea",
)

#: Steps per chunk in synthetic mode.  With real data the window is
#: whatever the chunk folder holds -- run_chunks_monterey_bay.yaml has 17
#: timesteps, a daily sequence plus a 3-hourly day, not 24 hourly ones.
EVOLUTION_N_STEPS: int = 24

#: First timestamp of the synthetic evolution window.
EVOLUTION_START: str = "2012-05-16T00_00_00"

#: Per-front statistics drawn as separate, toggleable lines on the field
#: time series.
EVOLUTION_STAT_LINES: tuple[str, ...] = ("mean", "median", "p25", "p75", "p90")

DEFAULT_EVOLUTION_STAT_LINES: tuple[str, ...] = ("median", "p90")

#: Fixed camera for the 3-D frame.  The scene must not be rotatable during
#: playback -- a moving camera and moving data are impossible to read
#: together -- so every frame is rendered from this azimuth/elevation.
EVOLUTION_CAMERA = {"azimuth": 45.0, "elevation": 28.0, "zoom": 1.05}
