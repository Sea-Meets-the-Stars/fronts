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
    "2012-05-16T06_00_00",
    "2012-02-29T18_00_00",
    "2012-11-09T12_00_00",
    "2012-07-21T00_00_00",
    "2012-09-02T12_00_00",
]

#: The timestamps with full 3-D raw data.  Everything depth-resolved -- the
#: Depth page, Tiles, Evolution, and the depth mode of Bivariate -- is
#: limited to these.  A fourth may be added; it is only a list entry.
DATES_3D: list[str] = [
    "2012-05-16T06_00_00",
    "2012-02-29T18_00_00",
    "2012-11-09T12_00_00",
]

DEFAULT_DATE: str = DATES[0]


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

TILE_DIR = Path(os.environ.get("FRONTS_APP_TILE_DIR", "./tiles")).expanduser()

CACHE_DIR = Path(
    os.environ.get("FRONTS_APP_CACHE", "~/.cache/fronts-viz")
).expanduser()


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
    # frontogenesis
    "frontogenesis_tendency", "frontogenesis_geo", "frontogenesis_ageo",
    # potential vorticity
    "ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt",
    # tracers and derived scalars
    "density", "buoyancy", "turner_angle", "Theta", "Salt", "KE",
    "gradb2", "gradrho2", "gradtheta2", "gradsalt2",
    # velocity and buoyancy fluxes
    "U", "V", "W", "ug", "vg", "uB", "vB", "wB",
)

#: Field used for the 3-D geometry.  Always density.
TILE_GEOMETRY_FIELD = "density"
