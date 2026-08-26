"""Headless tests for the visualisation pages.

No browser, no server, no display.  The pages keep their whole selection
in ``param.Parameterized`` classes, so the state machine and every
computation behind it can be driven directly.

The Panel layout itself is not tested, which is the correct thing to
leave untested.
"""

import inspect

import numpy as np
import pytest

from fronts.viz.apps import config
from fronts.viz.apps.common import pyramid, regions, selection, sources
from fronts.viz.apps.common.selection import BBox


@pytest.fixture(scope="module")
def provider():
    return sources.SyntheticProvider()


@pytest.fixture(scope="module")
def date(provider):
    return provider.dates()[0]


# --------------------------------------------------------------------------
# Selection -- the part that has to be right on an irregular grid
# --------------------------------------------------------------------------

def test_bbox_from_bounds_sorts_corners():
    box = BBox.from_bounds((40.0, 50.0, 10.0, 20.0))
    assert box.lon0 == 10.0 and box.lon1 == 40.0
    assert box.lat0 == 20.0 and box.lat1 == 50.0


def test_global_bbox_is_recognised():
    assert BBox.globe().is_global
    assert not BBox(-10, -10, 10, 10).is_global


def test_wrap180_round_trips():
    assert selection.wrap180(190.0) == pytest.approx(-170.0)
    assert selection.wrap180(-190.0) == pytest.approx(170.0)
    assert selection.wrap180(0.0) == pytest.approx(0.0)


def test_bbox_mask_selects_exactly_the_intended_cells():
    """On a deliberately warped grid, selection must follow coordinates.

    Latitude rows are unevenly spaced and each row is shifted in
    longitude, so any code that assumes a regular axis picks the wrong
    cells.  The mask is built from XC/YC, so it cannot.
    """
    lat1d = np.array([-60.0, -20.0, -5.0, 5.0, 25.0, 70.0])       # uneven
    lon1d = np.linspace(-180, 175, 12)
    YC = np.repeat(lat1d[:, None], lon1d.size, axis=1)
    XC = (lon1d[None, :] + np.array([0, 5, -5, 3, -3, 8])[:, None] + 180) % 360 - 180

    box = BBox(-50.0, -10.0, 50.0, 30.0)
    mask = selection.bbox_mask(XC, YC, box)

    expected = (YC >= -10) & (YC <= 30) & (XC >= -50) & (XC <= 50)
    assert np.array_equal(mask, expected)
    # rows outside the latitude band contribute nothing
    assert not mask[0].any() and not mask[-1].any()
    # the selected rows are exactly those whose latitude is in [-10, 30]:
    # lat1d[2] = -5, lat1d[3] = 5, lat1d[4] = 25
    assert set(np.nonzero(mask.any(axis=1))[0]) == {2, 3, 4}


def test_bbox_mask_handles_antimeridian():
    YC = np.zeros((1, 8))
    XC = np.array([[-180.0, -170, -100, -10, 10, 100, 170, 179]])
    box = BBox(150.0, -5.0, -150.0, 5.0)          # wraps
    assert box.wraps()
    mask = selection.bbox_mask(XC, YC, box)
    assert mask.tolist() == [[True, True, False, False,
                              False, False, True, True]]


def test_global_bbox_short_circuits(provider, date):
    XC, YC = provider.coords(date)
    mask = selection.bbox_mask(XC, YC, BBox.globe())
    assert mask.all() and mask.shape == XC.shape


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

def test_synthetic_grid_is_irregular(provider, date):
    """The fixture grid must not be a regular lat/lon grid.

    If it ever becomes one, the tests above stop proving anything.
    """
    XC, YC = provider.coords(date)
    dlat = np.diff(YC[:, 0])
    assert dlat.std() > 1e-3, "latitude spacing should be non-uniform"
    assert XC.ndim == 2 and np.ptp(XC[:, 0]) > 0, "longitude should vary by row"


def test_land_comes_from_the_field_nans(provider, date):
    land = provider.land_mask(date)
    field = provider.field(date, "gradb2")
    assert np.array_equal(land, ~np.isfinite(field))
    assert 0.0 < land.mean() < 0.9


def test_roles_resolve_to_real_channels(provider, date):
    roles = provider.resolve_channels(date)
    names = provider.field_names(date)
    assert set(roles) == set(config.KINEMATIC_ROLES)
    for role, channel in roles.items():
        assert channel in names, role


def test_tables_line_up_with_the_label_mask(provider, date):
    labels = provider.labels(date)
    geom = provider.geometry(date)
    coloc = provider.colocation(date)
    present = set(np.unique(labels)) - {0}
    assert set(geom["label"]) == present
    assert set(coloc["flabel"]) == present
    assert coloc["npix"].sum() == int((labels > 0).sum())


def test_front_products_are_found_by_pattern(monkeypatch):
    """The pushed products are matched as globs; the run tag is not known."""
    from fronts.viz.apps.common import s3source

    pushed = [
        "x/LLC4320_20120516T06_00_00_v2_2_01_bfronts.npy",
        "x/labeled_fronts_global_20120516T06_00_00_v2_2_01_bfronts.npy",
        "x/global_front_geometry_20120516T06_00_00_v2_2_01_bfronts.parquet",
        "x/metadata_20120516T06_00_00_v2_2_01_bfronts.json",
    ]

    class FS:
        def ls(self, prefix):
            return pushed

    monkeypatch.setattr(s3source, "_filesystems", lambda: (None, FS()))

    for kind, expect in (("binary", "LLC4320_"),
                         ("labels", "labeled_fronts_global_"),
                         ("geometry", "global_front_geometry_")):
        path = s3source._product_path("f", "r", "2012-05-16T06_00_00", kind)
        assert expect in path

    # Step 4 has not run, so colocation must name the step that is missing.
    with pytest.raises(sources.NotWiredUp) as exc:
        s3source._product_path("f", "r", "2012-05-16T06_00_00", "colocation")
    assert "step 4" in str(exc.value)


def test_binary_glob_does_not_swallow_the_label_map():
    """Both end in _bfronts.npy; the label map must not match 'binary'."""
    import fnmatch
    from fronts.viz.apps.common.s3source import PRODUCT_GLOBS

    label_file = "labeled_fronts_global_20120516T06_00_00_v1_bfronts.npy"
    assert not fnmatch.fnmatch(label_file, PRODUCT_GLOBS["binary"]), (
        "the binary pattern must exclude the label map")


# --------------------------------------------------------------------------
# Pyramid -- display only, but it must place data correctly
# --------------------------------------------------------------------------

def test_regrid_places_values_at_their_coordinates():
    # Points sit on cell centres, not cell edges: a value exactly on an
    # edge is legitimately ambiguous between two cells, and asserting one
    # of them would be testing a tie-break rather than the placement.
    XC = np.array([[-175.0, 5.0, 175.0]])
    YC = np.array([[-75.0, -5.0, 65.0]])
    vals = np.array([[1.0, 2.0, 3.0]])

    lon, lat, out = pyramid.regrid(vals, XC, YC, width=36)

    for x, y, v in zip(XC[0], YC[0], vals[0]):
        i = int(np.argmin(np.abs(lon - x)))
        j = int(np.argmin(np.abs(lat - y)))
        assert out[j, i] == pytest.approx(v)


def test_regrid_means_collisions():
    XC = np.array([[10.0, 10.4]])
    YC = np.array([[0.0, 0.0]])
    vals = np.array([[2.0, 4.0]])
    _, _, out = pyramid.regrid(vals, XC, YC, width=36)
    assert np.nanmax(out) == pytest.approx(3.0)


def test_regrid_any_and_max_reductions():
    XC = np.array([[10.0, 10.4]])
    YC = np.array([[0.0, 0.0]])
    _, _, any_out = pyramid.regrid(np.array([[0.0, 1.0]]), XC, YC,
                                   width=36, reduce="any")
    assert any_out.max() == 1
    _, _, max_out = pyramid.regrid(np.array([[3.0, 9.0]]), XC, YC,
                                   width=36, reduce="max")
    assert max_out.max() == 9


def test_to_pacific_reorders_into_0_360():
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    arr = np.array([[1.0, 2.0, 3.0, 4.0]])
    out_lon, out_arr = pyramid.to_pacific(lon, arr)
    assert np.all(np.diff(out_lon) > 0)
    assert out_lon.min() >= 0 and out_lon.max() < 360
    assert sorted(out_arr[0].tolist()) == [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------

def test_regions_have_unique_keys_and_names():
    from fronts.viz.apps.common import regions

    keys = [r.key for r in regions.REGIONS]
    names = [r.name for r in regions.REGIONS]
    assert len(set(keys)) == len(keys)
    assert len(set(names)) == len(names)
    assert len(regions.REGIONS) >= 6


def test_pinned_regions_use_their_tile_number(provider, date):
    """A tile number beats a lat/lon search, and must beat it everywhere.

    A centre is resolved to a grid cell and then floored onto the
    720-cell lattice, so one near a tile boundary can land either side.
    The page already honoured the pin; build_tiles and check_align call
    tile_index_for directly, so the check belongs there.
    """
    from fronts.viz.apps.common import regions

    pinned = {r.key: r.tile_idx for r in regions.REGIONS
              if r.tile_idx is not None}
    assert pinned == {"agulhas": 171, "se_greenland": 408,
                      "gulf_of_alaska": 400}

    for region in regions.REGIONS:
        if region.tile_idx is None:
            continue
        # No provider access at all: the answer is the pin.
        assert regions.tile_index_for(None, date, region) == region.tile_idx
        assert f"tile {region.tile_idx}" in region.label()


def test_nearest_region_hits_and_misses():
    ccs = regions.by_name("California Current System")
    assert regions.nearest(ccs.lat + 1, ccs.lon + 1) is ccs
    assert regions.nearest(0.0, 90.0) is None      # nothing within 25 deg


def test_resolve_tile_uses_the_supplied_lookup(monkeypatch):
    """Tile resolution must go through rect_ij_to_tile, not arithmetic here."""
    class FakeInfo:
        tile_idx = 330

    class FakeMapping:
        @staticmethod
        def rect_ij_to_tile(i, j):
            assert (i, j) == (13142, 9956)
            return FakeInfo()

    monkeypatch.setattr(regions, "_import_tile_mapping", lambda: FakeMapping)
    got = regions.resolve_tile(regions.by_name("California Current System"),
                               lambda lat, lon: (13142, 9956))
    assert got == 330


def test_tile_field_list_matches_the_registry():
    """Page 2's 3-D field list is hand-maintained; keep it honest.

    ``TileProperty`` carries no dimensionality flag, so the list cannot be
    derived from the registry -- but every name in it must still exist
    there.  Skipped when the preprocessing repo is not installed.
    """
    registry = pytest.importorskip("dbof.tiles.field_registry")
    known = set(registry.TILE_PROPERTIES) | set(
        getattr(registry, "ALIASES", {})
    )
    unknown = sorted(set(config.TILE_FIELDS_3D) - known)
    assert not unknown, f"not in TILE_PROPERTIES: {unknown}"


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def test_characteristics_state_defaults_and_region(provider):
    from fronts.viz.apps.common.state import CharacteristicsState

    st = CharacteristicsState(provider=provider)
    assert st.field in provider.field_names(st.date)
    assert st.box.is_global
    assert st.region_label() == "global"

    st.set_bounds((200.0, -10.0, 240.0, 20.0))       # 0..360 map coords
    assert not st.box.is_global
    assert st.box.lon0 == pytest.approx(-160.0)
    assert st.box.lon1 == pytest.approx(-120.0)

    st.reset_region()
    assert st.box.is_global


def test_tiles_state_clears_the_front_when_the_region_changes(provider):
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=provider)
    assert st.select_front(7) is True
    assert st.front_label == 7
    assert st.select_front(7) is False

    st.region = regions.names()[1]
    assert st.front_label == 0


def test_tiles_state_resolves_a_tile_index(provider):
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=provider)
    assert isinstance(st.tile_index(), int)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def test_statistics_use_every_cell_in_the_box(provider, date):
    from fronts.viz.apps.characteristics import stats

    box = BBox(-140.0, 10.0, -100.0, 45.0)
    s = stats.extract(provider, date, "gradb2", box, fronts_only=False)
    assert s.n_cells == stats.cost_estimate(provider, date, box)
    assert s.n <= s.n_cells
    assert not s.missing


def test_fronts_only_is_a_subset(provider, date):
    from fronts.viz.apps.characteristics import stats

    box = BBox.globe()
    both = stats.extract_both(provider, date, "gradb2", box)
    assert both["fronts"].n < both["all"].n
    assert both["fronts"].n > 0


def test_equatorial_band_is_excluded_from_normalised_samples(provider, date):
    """f-normalised quantities blow up at the equator and must be dropped."""
    from fronts.viz.apps.characteristics import stats

    box = BBox(-20.0, -1.5, 20.0, 1.5)              # inside the cutoff
    s = stats.extract(provider, date, "gradb2", box)
    assert s.n_cells > 0
    assert s.zeta_f.size == 0


def test_missing_kinematics_are_reported_not_guessed(provider, date):
    from fronts.viz.apps.characteristics import stats

    class Blinkered:
        """A provider whose store lacks the strain channel."""
        mode, synthetic = "test", True
        def __getattr__(self, name):
            return getattr(provider, name)
        def resolve_channels(self, date):
            r = provider.resolve_channels(date)
            r["strain"] = None
            return r

    s = stats.extract(Blinkered(), date, "gradb2", BBox.globe())
    assert s.missing == ("strain",)
    assert not s.has_kinematics
    assert s.n > 0                    # the PDF still has something to draw


# --------------------------------------------------------------------------
# Panels and the tile pipeline
# --------------------------------------------------------------------------

def test_panels_return_figures_without_writing_files(provider, date):
    from matplotlib.figure import Figure

    from fronts.viz.apps.characteristics import panels, stats

    pytest.importorskip("dbof.plotting.jpdfs")

    cols = stats.extract_both(provider, date, "gradb2", BBox.globe())
    bins = panels.pdf_bins(cols, "gradb2")
    for s in cols.values():
        assert isinstance(panels.figure_pdf(s, "gradb2", bins), Figure)
        assert isinstance(panels.figure_jpdf(s), Figure)
        assert isinstance(panels.figure_jpdf_conditional(s, "gradb2"), Figure)


def test_adaptive_bins_are_bounded():
    from fronts.viz.apps.characteristics import panels

    assert panels.adaptive_bins(0) == 20
    assert panels.adaptive_bins(10) == 20
    assert panels.adaptive_bins(10 ** 8) == 175
    assert 20 < panels.adaptive_bins(10_000) < 175


def test_tile_pipeline_builds_a_scene(provider, date):
    from fronts.viz.apps.tiles import pipeline

    tile_idx = regions.synthetic_tile_idx(
        regions.by_name("California Current System")
    )
    labels = pipeline.tile_labels(provider, date, tile_idx,
                                  (config.SYNTH_TILE_SIZE,) * 2)
    available = pipeline.available_fronts(labels)
    assert available, "the fixture tile should contain at least one front"

    scene = pipeline.build_scene(provider, date, tile_idx, "Ri", available[0])
    assert scene.sigma0.shape == scene.color.shape
    assert scene.Z.shape[0] == scene.sigma0.shape[0]
    assert scene.axis_path.shape[1] == 2
    assert len(scene.axis_path) >= 2
    assert scene.front_mask.shape == scene.sigma0.shape[1:]
    assert scene.clim[0] < scene.clim[1]


def test_two_d_figures_do_not_need_pyvista():
    """The five 2-D figures must build on a machine with no 3-D stack.

    This is checked in a subprocess with ``pyvista``/``vtk`` blocked at
    import, because the failure mode it guards against is a module-level
    ``import pyvista`` reached through a chain of otherwise-innocent
    imports -- exactly what made ``fronts.viz.curtains`` unusable without
    PyVista before ``fronts/viz/geometry.py`` existed.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import importlib.abc, sys
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"pyvista", "vtk", "vtkmodules"}:
                    raise ImportError("blocked: " + name)
                return None
        sys.meta_path.insert(0, Blocker())

        from fronts.viz import curtains, geometry          # noqa: F401
        from fronts.viz.apps.tiles import panels, pipeline

        from fronts.viz.apps.common import sources
        p = sources.SyntheticProvider(); d = p.dates()[0]
        labels = pipeline.tile_labels(p, d, 2, (180, 180))
        scene = pipeline.build_scene(
            p, d, 2, "Ri", pipeline.available_fronts(labels)[0]
        )
        idx = panels.pick_perp_index(scene)
        panels.figure_mainaxis(scene, perp_index=idx)
        panels.figure_offsets(scene, n_offsets=1)
        panels.figure_perpendicular(scene, index=idx)
        panels.figure_isopycnal(scene, perp_index=idx)
        panels.figure_inset(scene, perp_index=idx)

        try:
            panels.build_3d(scene)
        except panels.Missing3DStack:
            pass
        else:
            raise AssertionError("build_3d should refuse without PyVista")
        print("OK")
    """)

    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2500:]
    assert "OK" in out.stdout


def test_geometry_module_stays_free_of_pyvista():
    """``fronts/viz/geometry.py`` exists precisely to import nothing heavy."""
    import inspect

    from fronts.viz import geometry

    src = inspect.getsource(geometry)
    assert "pyvista" not in src and "import pv" not in src


def test_fronts_3d_still_exports_the_moved_names():
    """Moving them must not break any existing import."""
    pytest.importorskip("pyvista")
    from fronts.viz import fronts_3d, geometry

    for name in ("front_bbox_and_crop", "truncate_depth",
                 "decompose_front_branches"):
        assert getattr(fronts_3d, name) is getattr(geometry, name)


def test_unknown_front_label_is_rejected(provider, date):
    from fronts.viz.apps.tiles import pipeline

    with pytest.raises(pipeline.NoSuchFront):
        pipeline.build_scene(provider, date, 2, "Ri", 999_999)


def test_provenance_mismatch_is_caught(provider, date):
    from fronts.viz.apps.tiles import pipeline

    a = provider.tile(date, 2, "density")
    b = provider.tile(date, 3, "Ri")            # different window
    with pytest.raises(ValueError, match="provenance"):
        pipeline._check_provenance(a, b)


def test_auto_zscale_keeps_the_scene_proportionate(provider, date):
    from fronts.viz.apps.tiles import panels, pipeline

    tile_idx = regions.synthetic_tile_idx(
        regions.by_name("California Current System")
    )
    labels = pipeline.tile_labels(provider, date, tile_idx,
                                  (config.SYNTH_TILE_SIZE,) * 2)
    scene = pipeline.build_scene(provider, date, tile_idx, "Ri",
                                 pipeline.available_fronts(labels)[0])

    z = panels.auto_zscale(scene)
    depth = float(np.nanmax(scene.Z) - np.nanmin(scene.Z))
    horizontal = max(scene.j_slice.stop - scene.j_slice.start,
                     scene.i_slice.stop - scene.i_slice.start)
    assert 0.2 * horizontal < z * depth < 1.2 * horizontal


# ==========================================================================
# Phase 2: depth, front properties, bivariate, multi-field tiles
# ==========================================================================

def test_depth_levels_only_on_3d_dates(provider):
    """Depth-resolved channels exist only where the 3-D data does."""
    d3 = provider.dates_3d()[0]
    assert provider.depth_levels(d3) == list(config.DEPTH_LEVELS)

    surface_only = [d for d in provider.dates() if d not in provider.dates_3d()]
    if surface_only:
        assert provider.depth_levels(surface_only[0]) == ["Surface"]


def test_channel_resolution_uses_the_registry_suffixes(provider):
    """The suffixes are the preprocessing repo's, not invented here."""
    assert provider.channel("relative_vorticity") == "relative_vorticity"
    assert provider.channel("relative_vorticity", "Surface") == \
        "relative_vorticity_sfc"
    assert provider.channel("relative_vorticity", "Mixed layer depth") == \
        "relative_vorticity_mld"
    assert provider.channel("relative_vorticity", "Mean over mixed layer") == \
        "relative_vorticity_mld_mean"
    with pytest.raises(KeyError):
        provider.channel("relative_vorticity", "no such level")


def test_mld_mean_suffix_is_not_confused_with_mld(provider, date):
    """Longest suffix wins, or 'mld_mean' would resolve as 'mld'."""
    from fronts.viz.apps.common.sources import _split_depth_suffix

    assert _split_depth_suffix("strain_mag_mld_mean") == ("strain_mag", "mld_mean")
    assert _split_depth_suffix("strain_mag_mld") == ("strain_mag", "mld")
    assert _split_depth_suffix("strain_mag") == ("strain_mag", None)


def test_depth_variants_differ_from_the_surface(provider):
    """Each depth level must give a visibly different field.

    Compared as a ratio, not with ``np.allclose``: gradb2 is of order
    1e-14, far below allclose's default ``atol=1e-8``, so an absolute
    comparison calls every pair of values equal.
    """
    d3 = provider.dates_3d()[0]
    base = provider.field(d3, "gradb2")

    seen = [np.nanstd(base)]
    for level in ("25 m", "Mixed layer depth", "Mean over mixed layer"):
        variant = provider.field(d3, provider.channel("gradb2", level))
        assert variant.shape == base.shape
        seen.append(np.nanstd(variant))

    # Every level's spread differs from the surface by at least 10%.
    for other in seen[1:]:
        assert abs(other - seen[0]) / seen[0] > 0.1


def test_front_stats_come_from_the_columns(provider, date):
    """p95 is absent from the config, so it must not be offered."""
    stats = provider.front_stats(date)
    assert "median" in stats and "p25" in stats and "p90" in stats
    assert "p95" not in stats


# -- front properties ------------------------------------------------------

def test_front_property_table_joins_on_label(provider, date):
    from fronts.viz.apps.characteristics import front_props as FP

    table = FP.merged_table(provider, date)
    assert not table.empty
    assert {"length_km", "orientation", "centroid_lat"} <= set(table.columns)
    assert (table["label"] == table["flabel"]).all()


def test_front_region_filter_is_by_centroid(provider, date):
    from fronts.viz.apps.characteristics import front_props as FP

    table = FP.merged_table(provider, date)
    box = BBox(-60.0, -20.0, 40.0, 30.0)
    sub = FP.in_region(table, box)

    assert len(sub) <= len(table)
    if len(sub):
        assert sub["centroid_lat"].between(box.lat0, box.lat1).all()
    assert len(FP.in_region(table, BBox.globe())) == len(table)


def test_front_property_panels_build(provider, date):
    from matplotlib.figure import Figure

    from fronts.viz.apps.characteristics import front_props as FP

    table = FP.merged_table(provider, date)
    assert isinstance(FP.figure_length_pdf(table), Figure)
    assert isinstance(FP.figure_orientation_pdf(table), Figure)
    assert isinstance(FP.figure_lat_vs(table, "length"), Figure)
    assert isinstance(FP.figure_lat_vs(table, "orientation"), Figure)
    assert isinstance(
        FP.figure_field_vs(table, "gradb2", "median", "length"), Figure)
    assert isinstance(
        FP.figure_field_vs(table, "gradb2", "median", "orientation"), Figure)


def test_missing_statistic_falls_back_rather_than_failing(provider, date):
    from fronts.viz.apps.characteristics import front_props as FP

    table = FP.merged_table(provider, date)
    # p99 was never computed; the panel should still resolve something.
    assert FP.stat_column(table, "gradb2", "p99") == "gradb2_median"
    assert FP.stat_column(table, "not_a_field", "median") is None


# -- bivariate -------------------------------------------------------------

def test_bivariate_colormap_shape_and_ordering():
    from fronts.viz import bivariate as BV

    for n in (2, 3, 5):
        grid = BV.bivariate_colormap(n)
        assert grid.shape == (n, n, 3)
        assert grid.min() >= 0 and grid.max() <= 1
        # Lightness must decrease with the A index, or the two axes are
        # not separable by eye.
        for b in range(n):
            lum = grid[:, b].mean(axis=1)
            assert np.all(np.diff(lum) < 0)

    with pytest.raises(ValueError):
        BV.bivariate_colormap(1)


def test_bivariate_uses_the_natural_split_when_there_is_one():
    from fronts.viz import bivariate as BV

    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, 5000)      # entirely positive-ish, mean 5

    quantile = BV.bin_edges(values, 2, field_name="some_field")
    assert abs(quantile[1] - np.median(values)) < 0.1

    # Turner angle divides at 0, not at the median -- even when the data
    # is nowhere near symmetric about it.
    mixed = np.concatenate([values, -values * 0.3])
    natural = BV.bin_edges(mixed, 2, field_name="turner_angle")
    assert natural[1] == 0.0


def test_bivariate_edges_are_monotonic_even_for_constant_fields():
    from fronts.viz import bivariate as BV

    edges = BV.bin_edges(np.full(500, 3.0), 4, field_name="flat")
    assert np.all(np.diff(edges) > 0)


def test_bivariate_bin_assignment_covers_every_bin():
    from fronts.viz import bivariate as BV

    rng = np.random.default_rng(1)
    v = rng.normal(size=4000)
    edges = BV.bin_edges(v, 4)
    idx = BV.assign_bins(v, edges)
    assert set(np.unique(idx)) == {0, 1, 2, 3}

    with_nan = np.append(v, np.nan)
    assert BV.assign_bins(with_nan, edges)[-1] == -1


def test_bivariate_figure_builds(provider, date):
    from matplotlib.figure import Figure

    from fronts.viz import bivariate as BV
    from fronts.viz.apps.characteristics import front_props as FP

    table = FP.merged_table(provider, date)
    a = table["gradb2_median"].to_numpy(dtype=float)
    b = table["relative_vorticity_median"].to_numpy(dtype=float)

    for n in (2, 3):
        fig, scheme = BV.figure_bivariate(table, a, b, n=n,
                                          name_a="gradb2",
                                          name_b="relative_vorticity")
        assert isinstance(fig, Figure)
        assert scheme.n == n
        assert len(scheme.edges_a) == n + 1


# -- page state ------------------------------------------------------------

def test_depth_page_restricts_dates_and_offers_levels(provider):
    from fronts.viz.apps.common.state import CharacteristicsState

    surface = CharacteristicsState(provider=provider, depth_mode=False)
    depth = CharacteristicsState(provider=provider, depth_mode=True)

    assert len(depth.param.date.objects) == len(provider.dates_3d())
    assert len(surface.param.date.objects) == len(provider.dates())
    assert depth.param.depth_level.objects == list(config.DEPTH_LEVELS)


def test_zoom_limits_follow_the_selection(provider):
    from fronts.viz.apps.characteristics import page as PG

    p = PG.CharacteristicsPage.__new__(PG.CharacteristicsPage)
    from fronts.viz.apps.common.state import CharacteristicsState
    p.state = CharacteristicsState(provider=provider)
    p.mode = PG.SURFACE

    xlim, ylim = p._zoom_limits()
    assert xlim == (0, 360)

    p.state.set_bounds((200.0, -10.0, 240.0, 20.0))
    xlim, ylim = p._zoom_limits()
    assert 190 < xlim[0] < 200 and 240 < xlim[1] < 250
    assert ylim[0] < -10 and ylim[1] > 20


def test_tiles_state_caps_fields_and_tracks_dirty(provider):
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=provider)
    assert set(st.param.date.objects) == set(provider.dates_3d())

    st.dirty = False
    st.fields = ["Ri", "N2"]
    assert st.dirty is True                       # a change stales the figures

    st.fields = ["Ri", "N2", "wB", "Theta", "Salt"]
    assert len(st.fields) == TilesState.MAX_FIELDS

    # The map field is independent of the figure columns: the map is for
    # orientation and reading the density range, the columns are the
    # comparison.  Forcing the map field into `fields` meant choosing
    # density on the map silently replaced a column.
    assert st.field == config.TILE_GEOMETRY_FIELD
    assert st.field not in st.fields


# ==========================================================================
# Evolution
# ==========================================================================

def test_chunk_window_is_consecutive_hours(provider):
    chunk = provider.chunks()[0]
    times = provider.chunk_timesteps(chunk)
    assert len(times) == config.EVOLUTION_N_STEPS

    hours = [int(t.split("T")[1].split("_")[0]) for t in times]
    # Consecutive modulo the day boundary.
    for a, b in zip(hours, hours[1:]):
        assert (b - a) % 24 == 1


def test_chunk_step_is_shaped_like_a_tile(provider):
    """A chunk step must be interchangeable with a tile, or the whole
    Tiles pipeline would need a second code path."""
    chunk = provider.chunks()[0]
    ds = provider.chunk_tile(chunk, 0, "density")

    for key in ("tile_index", "face_index", "rect_i_start", "rect_j_start",
                "timestamp"):
        assert key in ds.attrs, key
    var = list(ds.data_vars)[0]
    assert ds[var].ndim == 3
    assert {"XC", "YC", "Z"} <= set(ds.coords)


def test_chunk_front_persists_across_the_window(provider):
    """Label 1 must exist at every step, or nothing can be followed."""
    chunk = provider.chunks()[0]
    for step in range(0, config.EVOLUTION_N_STEPS, 6):
        labels = provider.chunk_labels(chunk, step)
        assert (labels == 1).sum() > 0, step


def test_chunk_actually_evolves(provider):
    """A movie of a static field is not a movie.

    The front must move: its centroid should travel a visible distance
    between the first and last step.
    """
    chunk = provider.chunks()[0]
    first = provider.chunk_labels(chunk, 0) == 1
    last = provider.chunk_labels(chunk, config.EVOLUTION_N_STEPS - 1) == 1

    def centroid(mask):
        jj, ii = np.nonzero(mask)
        return np.array([jj.mean(), ii.mean()])

    assert np.linalg.norm(centroid(last) - centroid(first)) > 3.0


def test_front_list_comes_from_one_step_not_the_whole_window(provider):
    """Enumerating fronts must cost one step's labels, not every step's.

    The old selector walked the window counting label reuse.  With real
    data that is one 0.9 GB label plane per step -- ~15 GB for a chunk,
    over the cache cap, which is what made *Load chunk* hang for a quarter
    of an hour.
    """
    from fronts.viz.apps.evolution import tracking as TR

    chunk = provider.chunks()[0]
    reads = []
    original = provider.chunk_labels

    def counted(chunk_name, step):
        reads.append(int(step))
        return original(chunk_name, step)

    provider.chunk_labels = counted
    try:
        labels = TR.fronts_present(provider.chunk_labels(chunk, 0))
    finally:
        provider.chunk_labels = original

    assert labels                                   # found something
    assert reads == [0]                              # exactly one step


def test_fronts_present_is_numerically_ordered_and_drops_specks(provider):
    """Numerical order: the dropdown is something you search by number."""
    from fronts.viz.apps.evolution import tracking as TR

    labels = np.zeros((40, 40), dtype=int)
    labels[10, 5:35] = 7                             # 30 px
    labels[20, 10:25] = 3                            # 15 px
    labels[30, 0:2] = 9                              # 2 px -- a speck

    assert TR.fronts_present(labels, min_pixels=10) == [3, 7]


def test_available_fronts_is_numerically_ordered(provider):
    """Same rule on the Tiles page."""
    from fronts.viz.apps.tiles import pipeline

    labels = np.zeros((60, 60), dtype=int)
    labels[10, 0:40] = 900
    labels[20, 0:40] = 12
    labels[30, 0:40] = 5000
    labels[40, 0:3] = 7                              # speck

    assert pipeline.available_fronts(labels, min_pixels=25) == [12, 900, 5000]


def test_map_annotates_the_biggest_fronts_not_the_lowest_numbered():
    """Ordering the list numerically must not change which are labelled."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    labels = np.zeros((60, 60), dtype=int)
    labels[10, 0:50] = 9999                          # biggest
    labels[20, 0:5] = 11                             # smallest number
    labels[30, 0:30] = 500

    markers = TilesPage._label_markers(None, labels, [11, 500, 9999], top=2)
    texts = {row[2] for row in markers.data.itertuples(index=False)} \
        if hasattr(markers.data, "itertuples") else \
        {r[2] for r in markers.data}
    assert "9999" in texts and "500" in texts
    assert "11" not in texts


def test_common_labels_refuses_rather_than_being_slow_and_wrong(provider):
    """The retired API must fail loudly, not quietly do the wrong thing."""
    from fronts.viz.apps.evolution import timeseries as TS

    with pytest.raises(NotImplementedError, match="per date"):
        TS.common_labels(provider, provider.chunks()[0])


def test_orientation_matches_the_geometry_convention():
    """0 = north-south, 90 = east-west, always in 0-90."""
    from fronts.viz.apps.evolution import timeseries as TS

    vertical = np.zeros((20, 20), dtype=bool)
    vertical[2:18, 10] = True                      # runs along j -> N-S
    assert TS.orientation_deg(vertical) == pytest.approx(0.0, abs=1.0)

    horizontal = np.zeros((20, 20), dtype=bool)
    horizontal[10, 2:18] = True                    # runs along i -> E-W
    assert TS.orientation_deg(horizontal) == pytest.approx(90.0, abs=1.0)

    assert np.isnan(TS.orientation_deg(np.zeros((5, 5), dtype=bool)))


def test_evolution_series_has_a_value_per_step(provider):
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.evolution import timeseries as TS

    chunk = provider.chunks()[0]
    series = TS.build(provider, chunk,
                      EP.build_track(provider, chunk, 0, 1), "Ri")

    assert series.n == config.EVOLUTION_N_STEPS
    assert len(series.length_km) == series.n
    assert len(series.orientation) == series.n
    for name in config.DEFAULT_EVOLUTION_STAT_LINES:
        assert name in series.stats
        assert len(series.stats[name]) == series.n

    assert series.present().sum() > series.n // 2
    assert np.nanmax(series.length_km) > np.nanmin(series.length_km)


def test_evolution_series_orientation_is_in_range(provider):
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.evolution import timeseries as TS

    chunk0 = provider.chunks()[0]
    series = TS.build(provider, chunk0,
                      EP.build_track(provider, chunk0, 0, 1), "Ri")
    finite = series.orientation[np.isfinite(series.orientation)]
    assert finite.size and finite.min() >= 0.0 and finite.max() <= 90.0


def test_shared_settings_fix_the_transect_and_colour_scale(provider):
    """Every frame must share these, or the movie appears to pulse."""
    from fronts.viz.apps.evolution import pipeline as EP

    chunk0 = provider.chunks()[0]
    shared = EP.shared_settings(
        provider, chunk0, "Ri", EP.build_track(provider, chunk0, 0, 1))
    assert isinstance(shared["perp_index"], int)
    assert shared["clim"] is not None
    assert shared["clim"][0] < shared["clim"][1]


def test_build_step_rejects_an_absent_front(provider):
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.tiles import pipeline as TP

    with pytest.raises(TP.NoSuchFront):
        EP.build_step(provider, provider.chunks()[0], 0, "Ri", 999_999)


def test_build_step_restores_the_patched_lookup(provider):
    """The label patch must not leak into the Tiles page."""
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.tiles import pipeline as TP

    before = TP.tile_labels
    EP.build_step(provider, provider.chunks()[0], 0, "Ri", 1)
    assert TP.tile_labels is before

    with pytest.raises(Exception):
        EP.build_step(provider, provider.chunks()[0], 0, "Ri", 999_999)
    assert TP.tile_labels is before


def test_evolution_state_invalidates_on_change(provider):
    from fronts.viz.apps.evolution.app import EvolutionState

    st = EvolutionState(provider=provider)
    st.built = True
    st.field = "N2"
    assert st.built is False

    # The *point* invalidates, not front_label -- that is a derived
    # readout of which label the point hit at this step, so changing it
    # cannot mean "rebuild".
    st.built = True
    st.front_label = 3
    assert st.built is True, "front_label is an output, not a selection"

    st.anchor_lat = 36.5
    assert st.built is False

    st.built = True
    st.anchor_lon = 237.2
    assert st.built is False


def test_tile_utils_supports_chunk_and_in_memory_output():
    """The two changes the app depends on, on the chunk-transfer branch."""
    import inspect

    tile_utils = pytest.importorskip("dbof.tiles.tile_utils")

    run_args = inspect.signature(tile_utils.run).parameters
    assert run_args["chunk"].default is False
    assert run_args["write"].default is True

    load_args = inspect.signature(
        tile_utils._load_tracers_for_tile).parameters
    assert load_args["chunk"].default is False


def test_tile_origin_round_trips():
    from fronts.viz.apps.common.s3source import _tile_origin

    for tile_idx in (0, 1, 330, 431):
        i, j = _tile_origin(tile_idx)
        assert i % config.TILE_SIZE == 0 and j % config.TILE_SIZE == 0
        assert (j // config.TILE_SIZE) * config.N_TILE_I \
            + (i // config.TILE_SIZE) == tile_idx


# --------------------------------------------------------------------------
# Disk cache for grid-sized arrays
# --------------------------------------------------------------------------

def test_cached_array_is_built_once_and_memory_mapped(tmp_path, monkeypatch):
    from fronts.viz.apps.common import cache

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    calls = []

    def build():
        calls.append(1)
        return np.arange(12, dtype=np.float32).reshape(3, 4)

    first = cache.array("plane", build)
    second = cache.array("plane", build)

    assert len(calls) == 1                      # the second call was a hit
    assert np.array_equal(first, second)
    assert isinstance(second, np.memmap)        # not resident


def test_trim_evicts_oldest_until_under_the_cap(tmp_path, monkeypatch):
    from fronts.viz.apps.common import cache

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    block = np.zeros(4096, dtype=np.float32)    # 16 KB each
    for i in range(4):
        cache.array(f"p{i}", lambda b=block: b)

    removed = cache.trim(cap_bytes=40_000)
    assert removed >= 2
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total <= 40_000


# --------------------------------------------------------------------------
# Zoom, ice, and degrading without the front products
# --------------------------------------------------------------------------

def test_zooming_in_asks_for_a_finer_pyramid_level():
    from fronts.viz.apps.common import basemap

    globe = basemap.width_for_extent(None)
    region = basemap.width_for_extent(((290, 310), (30, 45)))
    assert region > globe, "a zoomed view must buy resolution, not just crop"
    assert region in config.PYRAMID_WIDTHS


def test_cropping_keeps_the_window_and_shrinks_the_raster():
    from fronts.viz.apps.common import basemap

    lon = np.linspace(0.125, 359.875, 1440)
    lat = np.linspace(-79.9, 79.9, 640)
    arr = np.zeros((640, 1440), dtype=np.float32)

    lon2, lat2, arr2 = basemap._crop(lon, lat, arr, ((300, 310), (35, 45)))
    assert arr2.shape[0] < arr.shape[0] and arr2.shape[1] < arr.shape[1]
    assert lon2[0] <= 300 and lon2[-1] >= 310
    assert lat2[0] <= 35 and lat2[-1] >= 45


def test_seam_crossing_window_is_not_cropped():
    from fronts.viz.apps.common import basemap

    lon = np.linspace(0.125, 359.875, 1440)
    lat = np.linspace(-79.9, 79.9, 640)
    arr = np.zeros((640, 1440), dtype=np.float32)

    _, _, arr2 = basemap._crop(lon, lat, arr, ((350, 370), (0, 10)))
    assert arr2.shape == arr.shape


def test_ice_covered_cells_are_dropped_from_a_field():
    provider = sources.get_provider()
    date = provider.dates()[0]
    shape = provider.coords(date)[0].shape

    class Icy:
        """A provider whose northern third is under ice."""
        mode, synthetic = "test", True
        drop_ice = sources.DataProvider.drop_ice
        ice_mask = sources.DataProvider.ice_mask

        def field_names(self, date):
            return [config.ICE_CHANNEL, "gradb2"]

        def field(self, date, name):
            if name == config.ICE_CHANNEL:
                area = np.zeros(shape, dtype=np.float32)
                area[: shape[0] // 3] = 0.9
                return area
            return np.ones(shape, dtype=np.float32)

    icy = Icy()
    out = icy.drop_ice(date, "gradb2", icy.field(date, "gradb2"))
    assert np.isnan(out[: shape[0] // 3]).all(), "ice should be masked"
    assert np.isfinite(out[shape[0] // 3:]).all(), "open ocean should survive"

    # The ice channel itself is never masked by its own mask.
    area = icy.field(date, config.ICE_CHANNEL)
    assert np.array_equal(icy.drop_ice(date, config.ICE_CHANNEL, area), area)


def test_all_points_column_survives_missing_front_products():
    from fronts.viz.apps.characteristics import stats
    from fronts.viz.apps.common.sources import NotWiredUp

    provider = sources.get_provider()
    date = provider.dates()[0]

    class NoFronts:
        """Everything works except the front detection."""
        def __getattr__(self, name):
            return getattr(provider, name)

        def labels(self, date):
            raise NotWiredUp("the labelled-fronts filename")

    columns = stats.extract_both(NoFronts(), date, provider.field_names(date)[0],
                                 BBox.globe(), tag="nofronts")
    assert columns["all"].n > 0, "grid-cell statistics do not need fronts"
    assert columns["fronts"].unavailable
    assert columns["fronts"].n == 0


def test_evolution_offers_only_the_allow_listed_chunks(monkeypatch):
    from fronts.viz.apps.evolution.app import EvolutionState

    provider = sources.get_provider()

    class ManyChunks:
        def __getattr__(self, name):
            return getattr(provider, name)

        def chunks(self):
            return ["amundsen", "monterey_bay", "ross", "weddell"]

        def chunk_timesteps(self, chunk):
            return provider.chunk_timesteps("monterey_bay")

    monkeypatch.setattr(config, "EVOLUTION_CHUNKS", ("monterey_bay",))
    state = EvolutionState(provider=ManyChunks())
    assert list(state.param.chunk.objects) == ["monterey_bay"]
    assert state.chunk == "monterey_bay"

    # With no allow-list, everything found is offered.
    monkeypatch.setattr(config, "EVOLUTION_CHUNKS", ())
    state = EvolutionState(provider=ManyChunks())
    assert len(state.param.chunk.objects) == 4


def test_tiles_offers_only_dates_that_have_fronts():
    """build_v5 runs date by date; the page should not offer empty ones."""
    from fronts.viz.apps.common.state import TilesState

    base = sources.get_provider()
    all_3d = base.dates_3d()
    assert len(all_3d) > 1, "fixture needs more than one 3-D date"
    built = all_3d[:1]

    class PartlyBuilt(type(base)):
        def has_fronts(self, date):
            return date in built

    state = TilesState(provider=PartlyBuilt())
    assert list(state.param.date.objects) == built
    assert state.date == built[0]


def test_tiles_falls_back_when_no_date_has_fronts():
    """With nothing built yet the page still loads, and says why."""
    from fronts.viz.apps.common.state import TilesState

    base = sources.get_provider()

    class NothingBuilt(type(base)):
        def has_fronts(self, date):
            return False

    state = TilesState(provider=NothingBuilt())
    assert list(state.param.date.objects) == base.dates_3d()


# --------------------------------------------------------------------------
# The S3 tile store
# --------------------------------------------------------------------------

def test_tile_store_path_is_the_agreed_layout():
    from fronts.viz.apps.common import tilestore

    p = tilestore.path("2012-05-16T06_00_00", "California Current System", "Ri")
    assert p == (f"{config.S3_BUCKET}/{config.TILE_STORE_FOLDER}/"
                 "20120516_060000/california_current_system/Ri.zarr")


def test_tile_store_round_trips_a_dataset_with_its_provenance(tmp_path,
                                                              monkeypatch):
    """The rect/face attrs decide label alignment, so they must survive."""
    import xarray as xr
    from fronts.viz.apps.common import tilestore

    ds = xr.Dataset(
        {"sigma0": (("k", "j", "i"), np.arange(8.0).reshape(2, 2, 2))},
        attrs={"rect_i_start": np.int64(720), "rect_j_start": np.int32(1440),
               "face_index": np.array(1), "tile_var_name": "sigma0"},
    )

    store = tmp_path / "Ri.zarr"
    monkeypatch.setattr(tilestore, "path", lambda *a, **k: str(store))
    monkeypatch.setattr(tilestore, "_filesystems",
                        lambda: (_Local(), _Local()))

    assert tilestore.write(ds, "2012-05-16T06_00_00", "r", "Ri")
    back = tilestore.read("2012-05-16T06_00_00", "r", "Ri")

    assert back.attrs["rect_i_start"] == 720
    assert back.attrs["rect_j_start"] == 1440
    assert back.attrs["face_index"] == 1
    assert np.array_equal(back["sigma0"].values, ds["sigma0"].values)

    # A second write without clobber must not overwrite.
    assert tilestore.write(ds, "2012-05-16T06_00_00", "r", "Ri") is None


class _Local:
    """Stand-in for the S3 filesystem, backed by ordinary paths."""

    def get_mapper(self, path):
        return path

    def exists(self, path):
        import os
        return os.path.exists(path)


def test_tile_is_composed_without_the_branch_only_run_arguments():
    """The in-memory path must not depend on run(write=...) existing.

    ``write=False`` lives on one branch of the preprocessing repo; the
    helpers composed here are on all of them.
    """
    import xarray as xr
    from fronts.viz.apps.common import s3source

    calls = []

    class FakeTileUtils:
        """Only the helpers that exist on every branch."""

        class _Prop:
            vars_needed = ("Theta", "Salt")
            out_name = "sigma0"

        @staticmethod
        def resolve_property(name):
            calls.append(("resolve_property", name))
            return FakeTileUtils._Prop()

        @staticmethod
        def _resolve_s3_source(path):
            return {"bucket": "dbof"}

        @staticmethod
        def rect_ij_to_tile(i, j):
            calls.append(("rect_ij_to_tile", i, j))
            return f"tile({i},{j})"

        @staticmethod
        def _load_grid_for_tile(cfg, tile):
            return xr.Dataset()

        @staticmethod
        def _load_tracers_for_tile(cfg, stamp, tile, vars_needed):
            calls.append(("tracers", stamp, tuple(vars_needed)))
            return xr.Dataset()

        @staticmethod
        def _build_tile_context(tracers, grid):
            return xr.Dataset(), object()

        @staticmethod
        def compute_tile_property(merge, xgrid, prop, mask_land):
            assert mask_land is True
            return "field"

        @staticmethod
        def mit_date_to_iteration(stamp):
            return 12345

        @staticmethod
        def _build_output_dataset(**kw):
            calls.append(("output", kw["rect_i_user"], kw["rect_j_user"]))
            return xr.Dataset(attrs={"iteration": kw["iteration"]})

        def run(self, *a, **k):                    # pragma: no cover
            raise AssertionError("run() must not be called for a plain tile")

    ds = s3source._compose_tile(FakeTileUtils(), "2012-05-16 06:00:00",
                                720, 1440, "density")

    assert ds.attrs["iteration"] == 12345
    assert ("rect_ij_to_tile", 720, 1440) in calls
    assert ("output", 720, 1440) in calls
    assert ("tracers", "2012-05-16 06:00:00", ("Theta", "Salt")) in calls


# --------------------------------------------------------------------------
# Tile provenance -- the attrs that decide label alignment
# --------------------------------------------------------------------------

def _tile_with(attrs, n=720):
    import xarray as xr
    return xr.Dataset(
        {"sigma0": (("k", "j", "i"), np.zeros((2, n, n), dtype=np.float32))},
        attrs=attrs,
    )


def test_rect_origin_accepts_either_provenance_convention():
    """Different tile_utils versions record the origin differently."""
    from fronts.viz.apps.tiles import pipeline as TP

    explicit = _tile_with({"rect_i_start": 1440, "rect_j_start": 2160,
                           "face_index": 1}, n=8)
    by_tile = _tile_with({"tile_i_rect": 2, "tile_j_rect": 3,
                          "face_index": 1}, n=8)

    assert TP.rect_origin(explicit) == (1440, 2160)
    assert TP.rect_origin(by_tile) == (2 * config.TILE_SIZE,
                                       3 * config.TILE_SIZE)
    # Both conventions describe the same tile, so they must agree.
    assert TP.rect_origin(explicit) == TP.rect_origin(by_tile)


def test_rect_origin_refuses_a_tile_with_no_origin():
    from fronts.viz.apps.tiles import pipeline as TP

    with pytest.raises(KeyError):
        TP.rect_origin(_tile_with({"face_index": 1}, n=8))


def test_real_tile_without_an_origin_does_not_silently_skip_the_remap():
    """Returning None there would misalign labels on every rotated face."""
    from fronts.viz.apps.tiles import pipeline as TP

    ds = _tile_with({"face_index": 1}, n=8)
    assert TP.tile_lookup(ds, synthetic=True) is None       # synthetic is fine
    with pytest.raises(KeyError):
        TP.tile_lookup(ds, synthetic=False)


def test_tile_window_uses_the_tile_size_of_the_dataset():
    from fronts.viz.apps.tiles import pipeline as TP

    ds = _tile_with({"tile_i_rect": 1, "tile_j_rect": 0, "face_index": 0}, n=8)
    js, iss = TP.tile_window(ds)
    assert (js.start, js.stop) == (0, 8)
    assert (iss.start, iss.stop) == (config.TILE_SIZE, config.TILE_SIZE + 8)


# --------------------------------------------------------------------------
# Tile map: same colours as the curtains, real labels in the overlay
# --------------------------------------------------------------------------

def test_bokeh_cmap_resolves_the_cmocean_names_field_styles_uses():
    """field_styles names cmocean maps Bokeh has never heard of."""
    from fronts.viz.apps.common import basemap

    for name in ("dense", "thermal", "haline"):
        colours = basemap.bokeh_cmap(name, n=8)
        assert len(colours) == 8
        assert all(c.startswith("#") for c in colours)

    # Distinct colormaps must not collapse onto the same fallback.
    assert basemap.bokeh_cmap("RdBu_r", 8) != basemap.bokeh_cmap("viridis", 8)
    # An unknown name falls back rather than raising.
    assert basemap.bokeh_cmap("not_a_colormap", 8) == \
        basemap.bokeh_cmap("viridis", 8)


def test_every_demo_field_has_a_resolvable_colormap():
    from fronts.viz import field_styles
    from fronts.viz.apps.common import basemap

    for field in config.TILE_STORE_DEFAULT_FIELDS + ("sigma0",):
        style = field_styles.get_style(field)
        assert len(basemap.bokeh_cmap(style.cmap, 4)) == 4, field


def test_front_overlay_carries_the_true_label_not_the_palette_index():
    """The dropdown shows global labels; hover must show the same number."""
    import holoviews as hv
    hv.extension("bokeh")

    labels = np.zeros((6, 6), dtype=np.int64)
    labels[1:3, 1:3] = 41953
    labels[4:6, 4:6] = 44615

    xs, ys = np.arange(6), np.arange(6)
    palette_idx = np.where(labels > 0, (labels - 1) % 8, np.nan).astype(float)
    true_label = np.where(labels > 0, labels, np.nan).astype(float)

    img = hv.Image((xs, ys, palette_idx, true_label),
                   kdims=["i", "j"], vdims=["front", "label"])

    shown = img.dimension_values("label", flat=False)
    assert np.nanmax(shown) == 44615
    assert 41953 in np.unique(shown[np.isfinite(shown)])
    # The palette index is deliberately not the label.
    assert np.nanmax(img.dimension_values("front", flat=False)) < 8


def test_label_markers_sit_on_their_fronts():
    from fronts.viz.apps.tiles.app import TilesPage

    labels = np.zeros((20, 20), dtype=np.int64)
    labels[2:6, 2:6] = 41953          # centroid ~ (3.5, 3.5)
    labels[14:18, 14:18] = 44615      # centroid ~ (15.5, 15.5)

    # No coords -> pixel positions, which is what this asserts.
    markers = TilesPage._label_markers(None, labels, [41953, 44615],
                                       coords=None)
    rows = {row[2]: (row[0], row[1]) for row in markers.data.itertuples(index=False)} \
        if hasattr(markers.data, "itertuples") else None

    text = list(markers.dimension_values("text"))
    assert set(text) == {"41953", "44615"}
    i_vals = markers.dimension_values("lon")
    j_vals = markers.dimension_values("lat")
    assert np.allclose(sorted(i_vals), [3.5, 15.5])
    assert np.allclose(sorted(j_vals), [3.5, 15.5])


def test_squared_gradient_fields_are_displayed_in_log10():
    """gradb2 spans orders of magnitude; linear limits hide every front."""
    from fronts.viz import field_styles

    for name in ("gradb2", "gradrho2", "gradtheta2", "gradsalt2"):
        style = field_styles.get_style(name)
        assert style.transform == "log10", name
        assert "log10" in style.title, name


def test_gradb2_log10_drops_non_positive_values_rather_than_flooring_them():
    from fronts.viz import field_styles

    style = field_styles.get_style("gradb2")
    vals = np.array([[1e-14, 1e-12], [0.0, -1e-14]])
    disp = field_styles.apply_transform(vals, style)

    assert disp[0, 0] == pytest.approx(-14.0)
    assert disp[0, 1] == pytest.approx(-12.0)
    assert np.isnan(disp[1, 0]), "zero must be NaN, not a floor value"
    assert np.isnan(disp[1, 1]), "negative must be NaN"


def test_the_two_d_map_and_the_tile_map_agree_that_gradb2_is_logged():
    """The global map has its own display table; it must not disagree."""
    from fronts.viz import field_styles
    from fronts.viz.apps.common import basemap

    assert "gradb2" in basemap._LOG_FIELDS
    assert field_styles.get_style("gradb2").transform == "log10"

    _, _, label = basemap.field_display(np.array([[1e-14, 1e-12]]), "gradb2")
    assert label == "log10(gradb2)"


# --------------------------------------------------------------------------
# Tiles page: zoom follows the selection, columns match the request
# --------------------------------------------------------------------------

def _tiles_page():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage
    return TilesPage(provider=sources.get_provider())


def test_tile_map_zooms_to_the_selected_front():
    page = _tiles_page()
    page.draw_tile(force=True)

    def limits():
        return page._tilemap.object.opts.get().kwargs

    whole = limits()
    available = [int(v) for v in page.w_avail.options]
    assert available, "fixture tile needs at least one front"

    page.state.select_front(available[0])
    zoomed = limits()

    # The axes are in degrees, so the window is compared as a fraction of
    # the tile rather than in pixels.
    xlim, ylim = zoomed["xlim"], zoomed["ylim"]
    wx = abs(whole["xlim"][1] - whole["xlim"][0])
    wy = abs(whole["ylim"][1] - whole["ylim"][0])
    assert abs(xlim[1] - xlim[0]) < wx or abs(ylim[1] - ylim[0]) < wy, \
        "selecting a front must tighten the view, not keep the whole tile"
    assert (whole["xlim"], whole["ylim"]) != (xlim, ylim)

    # And the window must lie inside the tile.
    assert min(xlim) >= min(whole["xlim"]) - 1e-6
    assert max(xlim) <= max(whole["xlim"]) + 1e-6


def test_deselecting_returns_the_whole_tile():
    page = _tiles_page()
    page.draw_tile(force=True)
    available = [int(v) for v in page.w_avail.options]
    page.state.select_front(available[0])
    page.state.front_label = 0

    # Degrees, not pixel counts: the whole-tile window is the tile's own
    # coordinate span.
    opts = page._tilemap.object.opts.get().kwargs
    lon, lat = page._tile_coords
    assert opts["xlim"] == (pytest.approx(float(lon[0])),
                            pytest.approx(float(lon[-1])))
    assert opts["ylim"] == (pytest.approx(float(lat[0])),
                            pytest.approx(float(lat[-1])))


def test_regenerate_does_not_leave_a_stale_column_behind():
    """Columns must equal the requested fields, not accumulate."""
    page = _tiles_page()
    assert len(page._columns.objects) == 1        # default single field

    page.state.fields = ["Ri", "N2", "gradb2"]
    page.schedule_figures()

    assert len(page._columns.objects) == 3
    assert page._column_fields == ["Ri", "N2", "gradb2"]
    assert {f for f, _ in page._panes} == {"Ri", "N2", "gradb2"}


def test_over_the_limit_keeps_the_newest_fields_not_the_default():
    from fronts.viz.apps.common.state import TilesState

    state = TilesState(provider=sources.get_provider())
    assert state.fields == ["Ri"]

    state.fields = ["Ri", "N2", "gradb2", "turner_angle"]
    assert len(state.fields) == state.MAX_FIELDS
    assert "turner_angle" in state.fields, \
        "the most recent pick must survive the cap"
    assert "Ri" not in state.fields, "the stale default should be the one to go"


# --------------------------------------------------------------------------
# Re-applied after the perf revert: layout, rebuild, products, ice
# --------------------------------------------------------------------------

def test_native_resolution_window_path_is_not_reintroduced():
    """The revert removed it for being slow; keep it removed."""
    from fronts.viz.apps.common import basemap
    assert not hasattr(pyramid, "regrid_window")
    assert not hasattr(pyramid, "window")
    assert not hasattr(basemap, "_layer_raster")
    assert not hasattr(config, "MAP_WINDOW_RASTER")


def test_global_map_fills_its_container_by_default():
    from fronts.viz.apps.common import basemap

    provider = sources.get_provider()
    date = provider.dates()[0]
    overlay = basemap.global_map(provider, date, "gradb2")
    opts = overlay.opts.get().kwargs
    assert opts.get("responsive") is True
    assert "width" not in opts

    fixed = basemap.global_map(provider, date, "gradb2", width=700)
    assert fixed.opts.get().kwargs.get("width") == 700


def test_changing_a_setting_stales_rather_than_rebuilds():
    from fronts.viz.apps.common.state import CharacteristicsState

    state = CharacteristicsState(provider=sources.get_provider())
    state.dirty = False
    state.field = [f for f in state.param.field.objects if f != state.field][0]
    assert state.dirty

    state.dirty = False
    state.set_bounds((100.0, 10.0, 120.0, 30.0))
    assert state.dirty


def test_characteristics_page_waits_for_rebuild():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    assert page.state.dirty
    assert page._map.object is not None, "the map is cheap and drawn up front"
    assert all(p.object is None for p in page._panes.values())

    page.rebuild()
    assert not page.state.dirty
    assert any(p.object is not None for p in page._panes.values())


def test_front_properties_work_without_colocation():
    from fronts.viz.apps.characteristics import front_props as FP
    from fronts.viz.apps.common.sources import NotWiredUp

    provider = sources.get_provider()

    class NoColocation:
        def __getattr__(self, name):
            return getattr(provider, name)

        def colocation(self, date):
            raise NotWiredUp("the colocation parquet")

    table = FP.merged_table(NoColocation(), provider.dates()[0])
    assert not table.empty
    for column in ("label", "centroid_lat", "centroid_lon", "length_km"):
        assert column in table.columns


def test_bivariate_fields_come_from_the_store_not_from_colocation():
    from fronts.viz.apps.bivariate.app import BivariateState
    from fronts.viz.apps.common.sources import NotWiredUp

    provider = sources.get_provider()

    class NoColocation:
        def __getattr__(self, name):
            return getattr(provider, name)

        def colocation(self, date):
            raise NotWiredUp("the colocation parquet")

    state = BivariateState(provider=NoColocation())
    assert state.field_a and state.field_b, "neither field may be None"
    assert state.field_a in provider.field_names(state.date)
    assert state.resolve(state.field_a) == state.field_a


def test_bivariate_grid_figure_colours_a_raster():
    from fronts.viz import bivariate as BV
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.linspace(0, 20, 40)
    lat = np.linspace(-5, 5, 20)
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((20, 40)), rng.standard_normal((20, 40))
    a[0, 0] = np.nan

    fig, scheme = BV.figure_bivariate_grid(lon, lat, a, b, n=2,
                                           name_a="A", name_b="B")
    assert scheme.n == 2
    plt.close(fig)

    with pytest.raises(ValueError):
        BV.figure_bivariate_grid(lon, lat, np.zeros((3, 4)), np.zeros((3, 5)))


def test_ice_is_excluded_at_bin_time_not_by_copying_the_field():
    XC = np.array([[10.2, 12.3, 14.1]])
    YC = np.array([[1.2, 3.3, 5.1]])
    vals = np.array([[1.0, 2.0, 3.0]])
    ice = np.array([[False, True, False]])

    # Width 720 so the three points land in three cells: at a coarse
    # width they share one and the mean of 1 and 3 is 2.0, which would
    # make the assertion meaningless.
    _, _, out = pyramid.regrid(vals, XC, YC, 720, fill_gaps=False,
                               exclude=ice)
    assert set(np.round(out[np.isfinite(out)], 3)) == {1.0, 3.0}


def test_ice_exclusion_leaves_the_ice_channel_alone():
    provider = sources.get_provider()
    date = provider.dates()[0]
    assert provider.ice_exclusion(date, config.ICE_CHANNEL) is None
    assert provider.ice_exclusion(date, "__land__") is None


def test_surface_and_depth_fields_live_in_separate_prefixes():
    """Depth fields are built by their own run, into their own prefix.

    A sibling rather than a subdirectory, so neither build can overwrite
    the other's stores or mistake one for the other and skip it.  The
    front products stay shared: a front is a front, and the labels do not
    depend on which pipeline computed the fields they are co-located
    against.
    """
    assert config.SURFACE_FOLDER == "globals_for_chunks"
    assert config.DEPTH_FOLDER == "globals_for_chunks_depth"
    assert config.DEPTH_FOLDER != config.SURFACE_FOLDER

    for folder in (config.SURFACE_FRONTS_FOLDER, config.DEPTH_FRONTS_FOLDER):
        assert folder == "globals_for_chunks"
    for run_id in (config.SURFACE_RUN_ID, config.DEPTH_RUN_ID,
                   config.SURFACE_FRONTS_RUN_ID, config.DEPTH_FRONTS_RUN_ID):
        assert run_id == "V5"


def test_only_the_v5_dates_are_offered():
    assert [config.date_to_prefix(d) for d in config.DATES] == [
        "20120229_180000", "20120516_060000",
        "20120918_110000", "20121109_120000",
    ]
    # Depth, Tiles and Evolution are limited to the same four.
    assert config.DATES_3D == config.DATES


def test_scotia_sea_chunk_is_offered():
    assert "southern_ocean_scotia_sea" in config.EVOLUTION_CHUNKS
    assert "monterey_bay" in config.EVOLUTION_CHUNKS



def test_selecting_a_region_moves_the_map_without_a_rebuild():
    """Navigation is immediate; only the figures below wait."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    page.rebuild()

    # The map is a DynamicMap now, so the frame is what carries the zoom.
    before = page._map_for_bounds().opts.get().kwargs
    page._on_bounds((100.0, 10.0, 140.0, 40.0))
    after = page._map_for_bounds().opts.get().kwargs

    assert after["xlim"] != before["xlim"], "the map must zoom to the box"
    assert not page.state.box.is_global, "the box must be recorded"
    assert page.state.dirty, "but the figures below must still be stale"

    page._reset_region()
    assert page._map_for_bounds().opts.get().kwargs["xlim"] == (0, 360)
    assert page.state.dirty


def test_a_box_drawn_on_the_map_survives_a_rebuild():
    """Rebuild used to zoom out and compute globally -- the selection was
    being lost when the stream was recreated on every redraw."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())

    # A box arriving through the map's own stream, as it does in a browser.
    page._bounds.event(bounds=(100.0, 10.0, 140.0, 40.0))
    assert not page.state.box.is_global
    selected = page.state.box.label()

    page.rebuild()
    assert page.state.box.label() == selected, "Rebuild must keep the region"
    assert not page.state.box.is_global, "and must not fall back to global"
    zoomed = page._map_for_bounds().opts.get().kwargs["xlim"]
    assert zoomed != (0, 360), "and must stay zoomed"


def test_the_same_box_is_not_reapplied_on_every_frame():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    page._bounds.event(bounds=(100.0, 10.0, 140.0, 40.0))
    first = page.state.box.label()
    page._bounds.event(bounds=(100.0, 10.0, 140.0, 40.0))
    assert page.state.box.label() == first


def test_cmocean_names_resolve_for_matplotlib():
    """'dense' is not a matplotlib name, and density is the default field."""
    from fronts.viz import field_styles
    import matplotlib.colors as mcolors

    for name in ("dense", "thermal", "haline", "viridis", "RdBu_r", "nope"):
        cmap = field_styles.resolve_cmap(name)
        assert isinstance(cmap, mcolors.Colormap), name

    assert (field_styles.resolve_cmap("RdBu_r")(0.1)
            != field_styles.resolve_cmap("viridis")(0.1))


def test_density_panels_build_without_cmocean_installed(monkeypatch):
    """The style names cmocean maps; matplotlib must still get a Colormap."""
    import builtins
    from fronts.viz import field_styles

    real_import = builtins.__import__

    def no_cmocean(name, *args, **kwargs):
        if name == "cmocean":
            raise ImportError("cmocean blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cmocean)
    cmap = field_styles.resolve_cmap("dense")
    import matplotlib.colors as mcolors
    assert isinstance(cmap, mcolors.Colormap)


def test_bivariate_land_shares_the_fields_longitude_axis():
    """Land on -180..180 under fields on 0..360 is a half-globe shift."""
    provider = sources.get_provider()
    date = provider.dates()[0]

    lon_field, _, field = pyramid.level(provider, date, "gradb2", 720)
    lon_land, _, land = pyramid.level(provider, date, "__land__", 720,
                                      reduce="any")

    assert lon_field[0] == pytest.approx(lon_land[0])
    assert lon_field[-1] == pytest.approx(lon_land[-1])

    agree = (~np.isfinite(field) == (land > 0)).mean()
    assert agree > 0.9, f"land should sit on the field's NaNs, got {agree:.2f}"


def test_only_the_touched_figures_are_marked_stale():
    """A profile click must not rebuild the offsets figure."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage, SECTION_DEPENDS

    page = TilesPage(provider=sources.get_provider())
    page._stale_keys.clear()

    page.state.add_profile_point(4, 4)
    assert page._stale_keys == {"profiles"}

    page._stale_keys.clear()
    page.state.perp_index = 5
    assert page._stale_keys == set(SECTION_DEPENDS["perp_index"])
    assert "offsets" not in page._stale_keys

    page._stale_keys.clear()
    page.state.n_offsets = 2
    assert page._stale_keys == {"offsets"}


def test_offset_side_colours_are_shared_across_figures():
    from fronts.viz.apps.tiles import panels as F
    assert F.OFFSET_PLUS != F.OFFSET_MINUS
    assert F.OFFSET_PLUS.startswith("#") and F.OFFSET_MINUS.startswith("#")


def test_evolution_offers_the_chunk_timesteps():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionPage

    page = EvolutionPage(provider=sources.get_provider())
    assert page.w_when.options, "the timestep list must be populated"
    assert page.w_when.value == page.w_when.options[0]

    page.w_when.value = page.w_when.options[2]
    assert page.state.step == 2, "picking a timestep must move the step"


def test_clicking_the_plan_view_moves_the_axis_vertex_and_the_slider():
    """The slider not moving was the symptom that clicks never arrived."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    scene = page._first_scene()
    path = np.asarray(scene.axis_path)
    assert len(path) > 4

    # The plan view is in degrees, so a click is a lon/lat.
    lon, lat, deg = page._plan_coords(scene)
    target = len(path) - 2
    page._plan_markers(tick=1, x=float(deg[target][1]),
                       y=float(deg[target][0]))

    assert page.state.perp_index == target, "the click must set the vertex"
    assert page.w_axis.value == target, "and the slider must follow it"
    assert page._stale_keys, "and the dependent sections must be stale"


def test_clicking_in_profile_mode_adds_a_location():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    page.state.pick_mode = "profiles"
    before = page.state.perp_index

    # Click at a known crop pixel, expressed in degrees.
    lon, lat, _ = page._plan_coords(page._first_scene())
    page._plan_markers(tick=1, x=float(lon[12]), y=float(lat[8]))

    assert page.state.profile_points == [(8, 12)], (
        "a degree click must map back to the crop pixel")
    assert page.state.perp_index == before, "profile mode must not move it"


def test_the_same_click_is_not_applied_twice():
    """A re-render must not re-run the last pick."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()
    page.state.pick_mode = "profiles"

    lon, lat, _ = page._plan_coords(page._first_scene())
    page._plan_markers(tick=1, x=float(lon[10]), y=float(lat[10]))
    page._plan_markers(tick=2, x=float(lon[10]), y=float(lat[10]))
    assert len(page.state.profile_points) == 1


def test_profiles_pane_is_actually_filled_by_the_builder():
    """The builder had no 'profiles' entry, so the pane never filled."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    page.state.pick_mode = "profiles"
    lon, lat, _ = page._plan_coords(page._first_scene())
    page._plan_markers(tick=1, x=float(lon[8]), y=float(lat[8]))
    assert page.state.profile_points, "the click must land a location"

    page.schedule_figures()
    for field in page.state.fields:
        pane = page._panes[(field, "profiles")]
        assert pane.object, f"profiles pane empty for {field}"


def test_rebuilding_after_a_pick_reuses_the_scenes():
    """Changing a pick is figures only -- no tile fetch."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()
    page.schedule_figures()
    assert page._scenes, "the first build must leave scenes behind"

    provider = page.state.provider
    original = type(provider).tile
    calls = []
    type(provider).tile = lambda self, *a, **k: (
        calls.append(a) or original(self, *a, **k))
    try:
        page.state.perp_index = 3
        page.schedule_figures()
    finally:
        type(provider).tile = original

    assert calls == [], "a second build must not re-fetch a tile"


def test_a_new_front_drops_the_cached_scenes():
    """They are geometry for the old front, so keeping them is wrong."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    available = [int(v) for v in page.w_avail.options]
    page.state.front_label = available[0]
    page.build_plan()
    page.schedule_figures()
    assert page._scenes

    page.state.front_label = available[1] if len(available) > 1 else 999
    assert page._scenes == {}, "a new front must invalidate the scenes"
    assert page._density_scene is None


def test_first_build_fills_every_panel_even_after_a_pick():
    """Filtering by the stale set alone left inset and offsets empty."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    # Pick something *before* the first build -- the realistic order, and
    # what used to leave the unmarked panels permanently blank.
    page.state.pick_mode = "profiles"
    lon, lat, _ = page._plan_coords(page._first_scene())
    page._plan_markers(tick=1, x=float(lon[8]), y=float(lat[8]))
    assert page._stale_keys, "the pick must mark something stale"

    page.schedule_figures()
    field = page.state.fields[0]
    for key in ("inset", "offsets", "isopycnal", "mainaxis",
                "perpendicular", "profiles"):
        assert page._panes[(field, key)].object, f"{key} pane is empty"


def test_the_transect_pick_also_refreshes_the_inset():
    """The inset draws the transect marker, so it depends on the pick."""
    from fronts.viz.apps.tiles.app import SECTION_DEPENDS
    assert "inset" in SECTION_DEPENDS["perp_index"]


def test_a_second_build_filters_to_what_changed():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()
    page.schedule_figures()
    assert page._built_fields, "the first build must record the field"

    page._stale_keys.clear()
    page.state.n_offsets = 2
    assert page._stale_keys == {"offsets"}, (
        "only the offsets figure depends on the offset count")


def test_isopycnal_depth_interpolates_between_levels():
    from fronts.viz.geometry import isopycnal_depth

    Z = np.linspace(0, -200, 11)                     # 0, -20, ... -200
    profile = np.linspace(25.0, 27.0, 11)            # 25.0, 25.2, ... 27.0
    rho = np.repeat(profile[:, None, None], 2, axis=1).repeat(2, axis=2)

    # 26.0 is exactly level 5 -> -100 m.
    assert isopycnal_depth(rho, Z, 26.0)[0, 0] == pytest.approx(-100.0)
    # 26.1 is half a level deeper -> -110 m, not snapped to a level.
    assert isopycnal_depth(rho, Z, 26.1)[0, 0] == pytest.approx(-110.0)


def test_outcropped_and_too_light_columns_are_undefined():
    """Both are physical statements, and both must read as NaN."""
    from fronts.viz.geometry import isopycnal_depth

    Z = np.linspace(0, -200, 11)
    rho = np.zeros((11, 1, 3))
    rho[:, 0, 0] = np.linspace(25.0, 27.0, 11)       # crosses 26
    rho[:, 0, 1] = np.linspace(26.5, 28.0, 11)       # surface already denser
    rho[:, 0, 2] = np.linspace(20.0, 22.0, 11)       # never reaches 26

    out = isopycnal_depth(rho, Z, 26.0)
    assert np.isfinite(out[0, 0])
    assert np.isnan(out[0, 1]), "outcropped column must be undefined"
    assert np.isnan(out[0, 2]), "column lighter than sigma must be undefined"


def test_isopycnal_depth_rejects_mismatched_inputs():
    from fronts.viz.geometry import isopycnal_depth

    with pytest.raises(ValueError):
        isopycnal_depth(np.zeros((4, 4)), np.zeros(4), 26.0)      # not 3-D
    with pytest.raises(ValueError):
        isopycnal_depth(np.zeros((5, 2, 2)), np.zeros(4), 26.0)   # Z mismatch


def test_isopycnal_depth_ignores_nan_columns():
    from fronts.viz.geometry import isopycnal_depth

    Z = np.linspace(0, -100, 6)
    rho = np.full((6, 1, 1), np.nan)
    assert np.isnan(isopycnal_depth(rho, Z, 26.0)[0, 0])


def test_default_sigma_exists_in_the_volume():
    """A fixed default would often name a surface nowhere in the tile."""
    from fronts.viz.apps.tiles import panels as F

    class Scene:
        sigma0 = np.linspace(24.0, 28.0, 8).reshape(8, 1, 1) * np.ones((8, 3, 3))

    sigma = F.default_sigma(Scene())
    finite = Scene.sigma0[np.isfinite(Scene.sigma0)]
    assert finite.min() <= sigma <= finite.max()


def test_sigma_change_stales_the_tiles_figures():
    from fronts.viz.apps.common.state import TilesState

    state = TilesState(provider=sources.get_provider())
    state.dirty = False
    state.sigma = 26.5
    assert state.dirty


def test_tiles_page_does_not_touch_a_tile_until_asked():
    """A page view must not fetch or generate a 3-D tile."""
    page = _tiles_page()
    assert page._tilemap.object is None
    assert page._tile_cache is None
    assert page._overview.object is not None, "the 2-D overview is cheap"

    page.draw_tile(force=True)
    assert page._tile_cache is not None
    assert page._tilemap.object is not None


def test_changing_region_drops_the_tile_without_fetching_another():
    page = _tiles_page()
    page.draw_tile(force=True)
    assert page._tile_cache is not None

    other = [r for r in page.state.param.region.objects
             if r != page.state.region][0]
    page.state.region = other
    assert page._tile_cache is None, "the old tile is wrong for the new region"
    assert page._tilemap.object is not None, "the stale map stays until reload"


def test_front_toggle_does_not_fetch_when_no_tile_is_loaded():
    page = _tiles_page()
    page.state.show_fronts = not page.state.show_fronts
    assert page._tile_cache is None


def test_evolution_page_does_not_read_a_chunk_until_asked():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionPage

    page = EvolutionPage(provider=sources.get_provider())
    assert page._chunkmap.object is None
    assert page._overview.object is not None

    page.load_chunk()
    assert page._chunkmap.object is not None


def test_only_the_configured_dates_are_offered_even_if_the_store_has_more():
    """config.DATES is the allow-list; the store listing is not."""
    from fronts.viz.apps.common.state import PageState

    base = sources.get_provider()
    extra = "2011-01-01T00_00_00"

    class ExtraDates(type(base)):
        def dates(self):
            return [extra] + list(base.dates())

    state = PageState(provider=ExtraDates())
    assert extra not in state.param.date.objects
    assert set(state.param.date.objects) <= set(config.DATES)


def test_the_depth_page_is_served_and_bivariate_depth_mode_is_not():
    """Depth is back; the Bivariate depth mode is still parked.

    They were hidden together, but for different reasons: the Depth page
    only needed its store built, while the Bivariate depth mode needs
    colocation at depth, which is a separate question.
    """
    from fronts.viz.apps import config

    assert "depth" in config.ENABLED_PAGES
    assert "Depth" not in config.BIVARIATE_MODES
def test_bivariate_offers_only_the_configured_dates():
    from fronts.viz.apps.bivariate.app import BivariateState

    base = sources.get_provider()
    extra = "2011-01-01T00_00_00"

    class ExtraDates(type(base)):
        def dates(self):
            return [extra] + list(base.dates())

    state = BivariateState(provider=ExtraDates())
    assert extra not in state.param.date.objects
    assert set(state.param.date.objects) <= set(config.DATES)


def test_chunk_tiles_do_not_go_through_run():
    """The chunk path must not need the branch that has run(chunk=...)."""
    import xarray as xr
    from fronts.viz.apps.common import s3source

    calls = []

    class FakeTileUtils:
        class _Prop:
            vars_needed = ("Theta",)
            out_name = "sigma0"

        @staticmethod
        def resolve_property(name):
            return FakeTileUtils._Prop()

        @staticmethod
        def rect_ij_to_tile(i, j):
            calls.append(("tile", i, j))
            return "tileinfo"

        @staticmethod
        def _build_tile_context(tracers, grid):
            calls.append(("context",))
            return xr.Dataset(), object()

        @staticmethod
        def compute_tile_property(merge, xgrid, prop, mask_land):
            return "field"

        @staticmethod
        def mit_date_to_iteration(stamp):
            return 7

        @staticmethod
        def _build_output_dataset(**kw):
            return xr.Dataset(attrs={"iteration": kw["iteration"]})

        def run(self, *a, **k):                    # pragma: no cover
            raise AssertionError("run() must not be called for a chunk")

    s3source._chunk_stores = lambda chunk, date: (xr.Dataset(), xr.Dataset())
    s3source._chunk_centre = lambda chunk: (36.8, -121.9)
    s3source._grid_plane = lambda name: np.zeros((4, 4), dtype=np.float32)

    ds = s3source._compose_chunk_tile(FakeTileUtils(), "monterey_bay",
                                      "2012-11-03T07_00_00", "density")
    assert ds.attrs["iteration"] == 7
    assert ("context",) in calls
    assert any(c[0] == "tile" for c in calls)



# --------------------------------------------------------------------------
# Perpendicular selection and vertical profiles (phase 3, A and C)
# --------------------------------------------------------------------------


def test_density_is_the_default_map_field():
    """The isopycnal control is the next step, and it needs the range."""
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=sources.get_provider())
    assert st.field == "density" == config.TILE_GEOMETRY_FIELD


def test_profile_points_are_capped_and_cleared_with_the_front():
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=sources.get_provider())
    for n in range(st.MAX_PROFILES):
        assert st.add_profile_point(n, n) is True
    assert st.add_profile_point(99, 99) is False, "must stop at the limit"
    assert len(st.profile_points) == st.MAX_PROFILES

    st.front_label = 1234
    assert st.profile_points == [], "a new front invalidates the locations"
    assert st.perp_index == -1


def test_picking_an_axis_point_stales_only_the_sections():
    """Stage 1 stales stage 2; stage 2 never stales stage 1."""
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=sources.get_provider())
    st.dirty = False
    st.sections_dirty = False

    st.perp_index = 4
    assert st.sections_dirty is True
    assert st.dirty is False, "choosing a point must not require a refetch"

    st.sections_dirty = False
    st.add_profile_point(3, 3)
    assert st.sections_dirty is True
    assert st.dirty is False


def test_axis_ticks_run_from_start_to_end_in_km():
    from fronts.viz.apps.tiles import panels as F

    class Scene:
        axis_path = np.stack([np.arange(20), np.arange(20)], axis=1)
        metrics = {"dist_km": np.linspace(0.0, 95.0, 20)}

    ticks = F.axis_ticks(Scene(), n=5)
    assert len(ticks) == 5
    assert ticks[0] == (0, 0.0)
    assert ticks[-1][0] == 19
    assert ticks[-1][1] == pytest.approx(95.0)
    # Monotonic, so the labels read left to right along the front.
    assert [km for _, km in ticks] == sorted(km for _, km in ticks)


def test_axis_ticks_fall_back_to_pixels_without_lon_lat():
    from fronts.viz.apps.tiles import panels as F

    class Scene:
        axis_path = np.stack([np.arange(10), np.arange(10)], axis=1)
        # path_metrics returns dist_km=None when it had no coordinates.
        metrics = {"dist_km": None, "dist_px": np.arange(10.0)}

    ticks = F.axis_ticks(Scene(), n=3)
    assert [k for k, _ in ticks] == [0, 4, 9]


def test_chosen_axis_point_overrides_the_automatic_pick():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.schedule_figures()

    scene = page._first_scene()
    assert scene is not None, "stage 1 must leave a scene behind"

    auto = page._resolved_perp_index(scene)
    page.state.perp_index = 0
    assert page._resolved_perp_index(scene) == 0

    # Out of range falls back rather than raising.
    page.state.perp_index = 10 ** 6
    assert page._resolved_perp_index(scene) == auto


def test_plan_view_is_built_from_density_alone():
    """Figure (c) exists before any colour field has been chosen."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]

    asked = []
    provider = page.state.provider
    original = type(provider).tile
    type(provider).tile = lambda self, date, idx, prop, region=None: (
        asked.append(prop) or original(self, date, idx, prop, region))
    try:
        page.build_plan()
    finally:
        type(provider).tile = original

    assert page._planview.object is not None, "the plan view must be drawn"
    assert page._density_scene is not None
    assert set(asked) <= {config.TILE_GEOMETRY_FIELD}, (
        f"only density should be read for the plan view, got {set(asked)}")


def test_sections_use_the_axis_point_the_plan_view_set():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    scene = page._first_scene()
    assert scene is page._density_scene

    page.state.perp_index = 3
    assert page._resolved_perp_index(scene) == 3

    # The slider is kept in step with the axis on screen, so it can always
    # set the point even if the click plumbing misbehaves.
    assert page.w_axis.end == max(len(scene.axis_path) - 1, 1)


def test_evolution_has_no_3d_frame():
    """The 3-D still was the slowest render on the page, so it is gone.

    It cost about as much as the other five figures together and read
    worst of all of them as a movie.  The interactive 3-D scene stays on
    the Tiles page.
    """
    from fronts.viz.apps.evolution import pipeline as EP

    assert "scene3d" not in EP.FRAME_ORDER
    assert "scene3d" not in EP.FRAME_TITLES
    assert not hasattr(EP, "_render_3d_still")


def test_chunk_map_background_is_gradb2():
    """The fronts were detected on gradb2, so that is what they sit on."""
    from fronts.viz.apps.evolution import app as evo
    assert evo.CHUNK_MAP_FIELD == "gradb2"


def test_loading_a_chunk_reports_progress():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionPage

    page = EvolutionPage(provider=sources.get_provider())
    assert page._chunk_progress.visible is False
    page.load_chunk()
    # Hidden again afterwards, and reset so the next load starts at zero.
    assert page._chunk_progress.visible is False
    assert page._chunk_progress.value == 0
    # Two stages, and neither walks the window.  The time series were a
    # third stage until they turned *Load chunk* into a 15-minute wait.
    assert page._chunk_progress.max == 2
    assert page._series.object is None


def test_region_boxes_come_from_the_tile_not_the_configured_centre():
    """The box used to sit at the config centre, not at the tile."""
    from fronts.viz.apps.common import regions

    provider = sources.get_provider()
    date = provider.dates()[0]

    for region in regions.REGIONS[:3]:
        idx = regions.synthetic_tile_idx(region)
        lon0, lat0, lon1, lat1 = regions.tile_extent(provider, date, idx)
        assert lon1 > lon0 and lat1 > lat0, "the box must be a real interval"

    # Different tiles must give different boxes -- one box for all of them
    # is exactly the bug.
    boxes = {regions.tile_extent(provider, date,
                                 regions.synthetic_tile_idx(r))
             for r in regions.REGIONS[:4]}
    assert len(boxes) > 1


def test_tile_extent_refuses_a_tile_off_the_grid():
    from fronts.viz.apps.common import regions

    provider = sources.get_provider()
    with pytest.raises(IndexError):
        regions.tile_extent(provider, provider.dates()[0], 10_000)


def test_the_tile_and_plan_views_do_not_share_axes():
    """Both carry lon/lat dims; HoloViews would otherwise link their zoom."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    assert page._tilemap.object.opts.get().kwargs["shared_axes"] is False

    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()
    # The plan view is a DynamicMap, so its options live on the frame.
    frame = page._planview.object[()]
    assert frame.opts.get("plot").kwargs.get("shared_axes") is False


def test_plan_view_axes_are_degrees_not_pixels():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.tiles.app import TilesPage

    page = TilesPage(provider=sources.get_provider())
    page.draw_tile(force=True)
    page.state.front_label = [int(v) for v in page.w_avail.options][0]
    page.build_plan()

    lon, lat, _ = page._plan_coords(page._first_scene())
    assert -360.0 <= lon.min() and lon.max() <= 720.0
    assert -90.0 <= lat.min() and lat.max() <= 90.0
    # Not a pixel index sequence.
    assert not np.allclose(lon, np.arange(len(lon)))


# --------------------------------------------------------------------------
# Display styles
# --------------------------------------------------------------------------

def test_ri_and_r_ib_share_one_colour_scale():
    """They are the same quantity; separate bars make them incomparable."""
    from fronts.viz import field_styles

    ri = field_styles.get_style("Ri")
    rib = field_styles.get_style("R_ib")
    for attr in ("cmap", "transform", "clim", "clip", "center", "scale"):
        assert getattr(ri, attr) == getattr(rib, attr), attr
    assert ri.title != rib.title, "only the label should differ"
    assert ri.transform == "linear", "Ri is no longer log-scaled"


def test_the_formerly_logged_fields_are_linear_now():
    from fronts.viz import field_styles

    for name in ("Ri", "R_ib", "N2", "vertical_shear", "strain_mag"):
        assert field_styles.get_style(name).transform == "linear", name


def test_all_four_squared_gradients_share_a_colormap():
    from fronts.viz import field_styles

    cmaps = {field_styles.get_style(n).cmap
             for n in ("gradb2", "gradrho2", "gradtheta2", "gradsalt2")}
    assert len(cmaps) == 1, f"expected one colormap, got {cmaps}"
    for name in ("gradb2", "gradrho2", "gradtheta2", "gradsalt2"):
        assert field_styles.get_style(name).transform == "log10", name


def test_global_maps_keep_gradb2_greyscale():
    """A different job: there it is a backdrop for the front overlay."""
    from fronts.viz.apps.common import basemap
    assert basemap._FIELD_CMAPS["gradb2"] == "gray"


def test_wb_uses_a_diverging_ocean_colormap():
    from fronts.viz import field_styles
    from fronts.viz.apps.common import basemap

    style = field_styles.get_style("wB")
    assert style.cmap == "balance"
    assert style.center == 0.0
    # And it must resolve -- balance is cmocean's, not matplotlib's.
    assert len(basemap.bokeh_cmap(style.cmap, 8)) == 8


def test_surface_only_fields_are_not_offered_on_tiles():
    """ug / vg and the frontogenesis split need surface geostrophy."""
    for name in ("ug", "vg", "frontogenesis_geo", "frontogenesis_ageo"):
        assert name not in config.TILE_FIELDS_3D, name
    # The total tendency is still there -- it is defined at depth.
    assert "frontogenesis_tendency" in config.TILE_FIELDS_3D


def test_vertical_shear_is_named_in_words():
    from fronts.viz import field_styles
    title = field_styles.get_style("vertical_shear").title
    assert "vertical shear" in title
    assert "|S|" not in title


def test_w_is_diverging_and_pivoted_on_zero():
    from fronts.viz import field_styles
    style = field_styles.get_style("W")
    assert style.cmap == "balance"
    assert style.center == 0.0
    assert style.transform == "linear"


def test_the_inset_adds_a_second_row_for_a_depth_in_range():
    from fronts.viz.apps.tiles import panels as F

    Z = np.linspace(-1.6, -89.6, 20)
    # In range -> a level index; the surface is index 0.
    assert F._level_for_depth(Z, -50.0) not in (None, 0)
    # Sign-agnostic: users type depths either way.
    assert F._level_for_depth(Z, 50.0) == F._level_for_depth(Z, -50.0)
    # Outside the clipped volume -> no second row, rather than silently
    # showing the deepest level available.
    assert F._level_for_depth(Z, -9999.0) is None
    assert F._level_for_depth(Z, None) is None


def test_inset_depth_only_stales_the_inset():
    from fronts.viz.apps.tiles.app import SECTION_DEPENDS
    assert SECTION_DEPENDS["inset_depth"] == ("inset",)


def test_the_region_field_map_is_configured_at_the_sigma_step():
    from fronts.viz.apps.common.state import TilesState

    state = TilesState(provider=sources.get_provider())
    assert state.region_field == config.TILE_GEOMETRY_FIELD
    assert state.inset_depth < 0, "the second row defaults below the surface"


def test_region_field_map_draws_fronts_over_the_field():
    import matplotlib
    matplotlib.use("Agg")
    from fronts.viz.apps.tiles import pipeline as TP, panels as F

    provider = sources.get_provider()
    date = provider.dates()[0]
    ds = provider.tile(date, 0, "density")
    labels = TP.tile_labels(provider, date, 0, (180, 180), ds=ds)
    scene = TP.build_scene(provider, date, 0, "density",
                           TP.available_fronts(labels)[0])

    surface = np.asarray(ds[ds.attrs.get("tile_var_name") or "sigma0"]
                         .values)[0]
    lon = np.linspace(-10.0, 10.0, surface.shape[1])
    lat = np.linspace(-5.0, 5.0, surface.shape[0])

    out = F.figure_region_field(scene, surface, labels,
                                field_name="sigma0", lon=lon, lat=lat)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tile dimension order
# ---------------------------------------------------------------------------

def test_field_values_puts_depth_first_whatever_the_stored_order():
    """A ``(j, i, k)`` tile reads back as ``(k, j, i)``.

    Tiles whose compute multiplies a 2-D field by a 3-D one come out
    ``(j, i, k)`` -- xarray puts the first operand's dims first -- and get
    written to the store that way.  The page must not care.
    """
    import xarray as xr
    from fronts.viz.apps.tiles import pipeline

    wrong = xr.Dataset(
        {"field": (("j", "i", "k"), np.zeros((4, 5, 3)))})
    right = xr.Dataset(
        {"field": (("k", "j", "i"), np.zeros((3, 4, 5)))})

    assert pipeline.field_values(wrong, "field").shape == (3, 4, 5)
    assert pipeline.field_values(right, "field").shape == (3, 4, 5)


def test_field_values_leaves_two_dimensional_fields_alone():
    import xarray as xr
    from fronts.viz.apps.tiles import pipeline

    ds = xr.Dataset({"XC": (("j", "i"), np.zeros((4, 5)))})
    assert pipeline.field_values(ds, "XC").shape == (4, 5)


def test_remap_to_rect_names_the_shape_it_got():
    """A misordered array must say so, not raise an opaque index error."""
    from fronts.viz.apps.tiles import pipeline

    j_face = np.zeros((4, 5), dtype=int)
    i_face = np.arange(5)[None, :].repeat(4, 0)

    with pytest.raises(ValueError, match=r"\(4, 5, 3\)"):
        pipeline.remap_to_rect(np.zeros((4, 5, 3)), (j_face, i_face))


def test_build_reports_every_column_failing_instead_of_crashing(monkeypatch):
    """All columns failing must land in the status line, not a traceback.

    The status line used to read the per-column ``wanted`` set after the
    loop.  When every column raised, the loop skipped past that
    assignment and the status line -- the one place that would have said
    which field broke and why -- raised ``UnboundLocalError``, hiding the
    real error behind an unretrieved asyncio task.
    """
    from fronts.viz.apps.common import sources
    from fronts.viz.apps.tiles import app as tiles_app

    page = tiles_app.TilesPage(provider=sources.get_provider())

    def boom(*a, **k):
        raise ValueError("no such tile")

    monkeypatch.setattr(tiles_app.pipeline, "build_scene", boom)
    page._scenes.clear()

    page._build(page._token)                                # must not raise

    status = page._build_status.object
    assert "failed" in status
    assert "no such tile" in status
    assert "figures:" not in status                    # nothing was built


# ---------------------------------------------------------------------------
# Evolution: following a front by location rather than by label
# ---------------------------------------------------------------------------

def _drifting_front(shape, j, i, length=12, label=1):
    """A short horizontal front centred at ``(j, i)``, given a label."""
    out = np.zeros(shape, dtype=int)
    i0 = max(0, int(i) - length // 2)
    out[int(j), i0:i0 + length] = int(label)
    return out


def _hourly(n, start_hour=0):
    return [f"2012-07-03T{start_hour + k:02d}_00_00" for k in range(n)]


def test_track_follows_the_front_though_the_label_changes_every_step():
    """The whole point: labels are per-step, position is not."""
    from fronts.viz.apps.evolution import tracking

    shape = (80, 80)
    # Same front, drifting two cells an hour, relabelled every step.
    frames = [_drifting_front(shape, 40, 30 + 2 * k, label=100 + 7 * k)
              for k in range(5)]

    anchor = tracking.anchor_at(frames[0], 0, 100)
    track = tracking.follow(lambda s: frames[s], _hourly(5), anchor)

    assert [track.label_at(s) for s in range(5)] == [100, 107, 114, 121, 128]
    assert track.gaps(5) == []


def test_track_survives_a_daily_gap_that_a_fixed_radius_would_lose():
    """The radius scales with elapsed time, not step count.

    A chunk is daily snapshots around one intensive day, so consecutive
    steps are one hour apart or twenty-four.  Across a day a front travels
    far enough that a radius sized for an hour would drop it.
    """
    from fronts.viz.apps.evolution import tracking

    shape = (80, 160)
    # Step 1 is a day later, and the front has moved 20 cells.
    frames = [_drifting_front(shape, 40, 40, label=5),
              _drifting_front(shape, 40, 60, label=9)]
    times = ["2012-07-03T12_00_00", "2012-07-04T12_00_00"]

    anchor = tracking.anchor_at(frames[0], 0, 5)
    track = tracking.follow(lambda s: frames[s], times, anchor)
    assert track.label_at(1) == 9

    # The same 20-cell jump one hour apart is not plausible drift, so it
    # is reported as a gap rather than linked.
    hourly = ["2012-07-03T12_00_00", "2012-07-03T13_00_00"]
    track = tracking.follow(lambda s: frames[s], hourly, anchor)
    assert track.label_at(1) is None
    assert track.gaps(2) == [1]


def test_track_prefers_the_nearer_front_over_a_neighbour():
    from fronts.viz.apps.evolution import tracking

    shape = (80, 80)
    step0 = _drifting_front(shape, 40, 40, label=1)
    step1 = _drifting_front(shape, 40, 42, label=3)
    step1 += _drifting_front(shape, 46, 44, label=4)    # a decoy nearby

    anchor = tracking.anchor_at(step0, 0, 1)
    track = tracking.follow(lambda s: [step0, step1][s], _hourly(2), anchor)
    assert track.label_at(1) == 3


def test_track_reacquires_after_a_gap():
    """A gap must not end the track -- the reference is kept."""
    from fronts.viz.apps.evolution import tracking

    shape = (80, 80)
    frames = [_drifting_front(shape, 40, 40, label=1),
              np.zeros(shape, dtype=int),               # front absent here
              _drifting_front(shape, 40, 44, label=8)]

    anchor = tracking.anchor_at(frames[0], 0, 1)
    track = tracking.follow(lambda s: frames[s], _hourly(3), anchor)

    assert track.label_at(1) is None
    assert track.label_at(2) == 8


def test_track_walks_backwards_from_the_anchor_too():
    """The front can be picked mid-window, not only at step 0."""
    from fronts.viz.apps.evolution import tracking

    shape = (80, 80)
    frames = [_drifting_front(shape, 40, 30 + 2 * k, label=10 * (k + 1))
              for k in range(5)]

    anchor = tracking.anchor_at(frames[2], 2, 30)
    track = tracking.follow(lambda s: frames[s], _hourly(5), anchor)
    assert track.steps() == [0, 1, 2, 3, 4]
    assert track.label_at(0) == 10


def test_frozen_window_is_padded_so_the_front_can_move_inside_it():
    from fronts.viz.apps.evolution import tracking

    mask = _drifting_front((80, 80), 40, 40, length=12).astype(bool)
    js, iss = tracking.window_for(mask)

    assert js.start < 40 < js.stop                       # contains the front
    assert iss.stop - iss.start > 12                     # with room to move
    assert js.start >= 0 and iss.stop <= 80              # clipped to the array


def test_track_reports_when_the_front_leaves_the_frozen_window():
    """Drifting out of shot is a display fact, not a tracking failure."""
    from fronts.viz.apps.evolution import tracking

    shape = (80, 200)
    frames = [_drifting_front(shape, 40, 40 + 20 * k, label=k + 1)
              for k in range(5)]
    times = [f"2012-07-0{3 + k}T12_00_00" for k in range(5)]

    anchor = tracking.anchor_at(frames[0], 0, 1)
    track = tracking.follow(lambda s: frames[s], times, anchor)

    # Still followed the whole way ...
    assert track.steps() == [0, 1, 2, 3, 4]
    # ... but out of the frozen window early on, and it says so.
    escape = track.first_escape()
    assert escape is not None and escape >= 1


def test_ensure_dbof_names_the_interpreter_when_the_repo_is_missing():
    """The message has to be actionable.

    A bare ``ModuleNotFoundError: No module named 'dbof'`` four frames deep
    in a page build does not say that the cause is usually the wrong conda
    environment.  Naming the interpreter does.
    """
    import importlib.util
    import sys

    from fronts.viz.apps import config

    if importlib.util.find_spec("dbof") is not None:
        pytest.skip("dbof is installed here, so there is no failure to check")

    with pytest.raises(ModuleNotFoundError) as excinfo:
        config.ensure_dbof()

    message = str(excinfo.value)
    assert sys.executable in message
    assert "LLC4320_PREPROC_SRC" in message


def test_picking_a_front_does_not_walk_the_window():
    """Selecting a front must be instant.

    The series need every step -- with real data that is a 0.9 GB label
    plane each -- so they must not be a side effect of a selection.  A
    watcher on ``front_label`` made choosing from the dropdown cost the
    whole window walk.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())

    calls = []
    page.draw_series = lambda *a, **k: calls.append(1)

    page.state.front_label = 1
    page.state.field = "N2"
    assert calls == []                      # neither selection computed


def test_toggling_stat_lines_does_not_compute_series_that_do_not_exist():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())

    calls = []
    page.draw_series = lambda *a, **k: calls.append(1)

    page.state.stat_lines = ["mean"]
    assert calls == []                      # nothing computed yet

    page._series_data = object()            # pretend a build happened
    page.state.stat_lines = ["median"]
    assert calls == [1]                     # now a restyle is cheap


# ---------------------------------------------------------------------------
# Tile / global alignment probe
# ---------------------------------------------------------------------------

def test_alignment_probe_discriminates_between_transforms():
    """The probe must be able to tell a wrong transform from a right one.

    A check that scores everything the same proves nothing, so this pins
    that a deliberately rotated plane loses to the unrotated one.
    """
    from fronts.viz.apps import check_align

    rng = np.random.default_rng(0)
    land = rng.random((60, 60)) < 0.3           # a lumpy fake coastline
    plane = np.where(land, np.nan, 1.0)

    scores = {name: check_align._agreement(~np.isfinite(c), land)
              for name, c in check_align._candidates(plane, None).items()}

    identity = next(v for k, v in scores.items() if k.startswith("identity"))
    assert identity == 1.0
    assert scores["transpose"] < 0.9
    assert scores["rot90"] < 0.9


def test_alignment_probe_runs_end_to_end_on_synthetic(capsys):
    """The whole CLI, on the fake world, where identity is the right answer."""
    from fronts.viz.apps import check_align

    rc = check_align.main(["--date", config.DATES[0],
                           "--region", "California Current System"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "land fraction" in out
    assert "alignment looks correct" in out


def test_prerender_reports_before_the_first_frame(provider, monkeypatch):
    """Progress must start at the first expensive thing, not the first frame.

    shared_settings runs before the frame loop and costs several tile
    compositions.  Reporting only from the loop left the build silent for
    its whole prelude, which is indistinguishable from a hang.

    Rendering is stubbed: what is under test is the reporting contract,
    and real frames make this a minutes-long test.
    """
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]
    monkeypatch.setattr(EP, "shared_settings",
                        lambda *a, progress=None, **k: (
                            progress and progress(1, 1, "sampling step 0"),
                            {"perp_index": 0, "clim": None})[1])
    monkeypatch.setattr(EP, "render_frame",
                        lambda *a, **k: {k2: None for k2 in EP.FRAME_ORDER})

    seen = []
    EP.prerender(provider, chunk, "Ri", EP.build_track(provider, chunk, 0, 1),
                 progress=lambda d, t, w=None: seen.append((d, t, w)))

    assert seen, "prerender reported nothing at all"
    assert "sampling" in (seen[0][2] or "")     # prep, not a frame
    total = seen[0][1]
    assert total > len(provider.chunk_timesteps(chunk))
    assert seen[-1][0] == total


def test_shared_settings_samples_three_steps_not_seven(provider, monkeypatch):
    """Each sampled step is a tile composition, so the count matters."""
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.tiles import panels as F

    chunk = provider.chunks()[0]
    built = []
    original = EP.build_step

    def counted(prov, ch, step, field, label):
        built.append(step)
        return original(prov, ch, step, field, label)

    monkeypatch.setattr(EP, "build_step", counted)
    monkeypatch.setattr(F, "pick_perp_index", lambda *a, **k: 0)

    EP.shared_settings(provider, chunk, "Ri",
                       EP.build_track(provider, chunk, 0, 1))

    assert len(built) <= 3, f"built {len(built)} scenes: {built}"
    assert len(set(built)) == len(built), "the same step was built twice"


# ---------------------------------------------------------------------------
# Chunk grid repair
# ---------------------------------------------------------------------------

def _bare_chunk_grid(with_face=True):
    """A chunk grid.zarr as it comes back: no comodo attrs, face kept."""
    import xarray as xr

    dims = {"j": 4, "i": 4, "j_g": 4, "i_g": 4}
    coords = {d: np.arange(n) for d, n in dims.items()}
    data = {"XC": (("j", "i"), np.zeros((4, 4)))}
    if with_face:
        data["hFacC"] = (("face", "j", "i"), np.ones((1, 4, 4)))
        coords["face"] = np.array([10])
    return xr.Dataset(data, coords=coords)


def test_comodo_attrs_are_restored_on_a_chunk_grid():
    """xgcm finds its axes by attrs, and the transfer drops them.

    Without this every step of every chunk dies with "missing axes
    ['X', 'Y']" from inside tile_utils -- the whole movie, not one frame.
    """
    from fronts.viz.apps.common import s3source

    repaired = s3source._stamp_comodo(_bare_chunk_grid(with_face=False))

    assert repaired.coords["i"].attrs["axis"] == "X"
    assert repaired.coords["j"].attrs["axis"] == "Y"
    assert repaired.coords["i_g"].attrs["axis"] == "X"
    assert repaired.coords["i_g"].attrs["c_grid_axis_shift"] == 0.5
    assert repaired.coords["j_g"].attrs["c_grid_axis_shift"] == 0.5


def test_comodo_matches_the_preprocessing_repos_values():
    """These must not drift from what get_llc_depth_gridfile stamps."""
    from fronts.viz.apps.common import s3source

    assert s3source._COMODO_ATTRS == {
        "j": {"axis": "Y"},
        "j_g": {"axis": "Y", "c_grid_axis_shift": 0.5},
        "i": {"axis": "X"},
        "i_g": {"axis": "X", "c_grid_axis_shift": 0.5},
    }


def test_chunk_grid_keeps_its_face_dimension():
    """The face dimension must survive: compute_tile_property selects on it.

    Dropping it -- which looked like tidying -- broke every step with
    "Dimensions {'face'} do not exist".
    """
    from fronts.viz.apps.common import s3source

    grid = _bare_chunk_grid(with_face=True)
    assert "face" in s3source._stamp_comodo(grid).dims
    assert not hasattr(s3source, "_drop_single_face")

class _LocalFS:
    """The two methods ``_product_window`` uses, over the real filesystem.

    A stub rather than ``fsspec.filesystem("file")`` so the suite does not
    grow a dependency for one test, and so the bytes actually read can be
    counted.
    """

    def __init__(self):
        self.reads = []

    def open(self, path, mode="rb"):
        fh = open(path, mode)
        original = fh.read
        reads = self.reads

        def read(n=-1):
            data = original(n)
            reads.append(len(data))
            return data

        fh.read = read
        return fh


def test_product_window_matches_a_full_read(tmp_path, monkeypatch):
    """The band read must give exactly what slicing the whole array gives.

    The label map is 1.67 GB of int64 and the app wants a 720-cell window,
    so this reads only the window's rows.  It is worth nothing if it does
    not agree with the naive answer.
    """
    from fronts.viz.apps.common import s3source

    rng = np.random.default_rng(1)
    full = rng.integers(0, 5000, size=(300, 400)).astype(np.int64)
    path = tmp_path / "labels.npy"
    np.save(path, full)

    monkeypatch.setattr(s3source, "_filesystems", lambda: (None, _LocalFS()))

    for js, iss in [(slice(0, 720), slice(0, 720)),        # clipped to shape
                    (slice(10, 60), slice(30, 90)),
                    (slice(250, 300), slice(0, 400))]:
        got = s3source._product_window(str(path), js, iss)
        assert np.array_equal(got, full[js, iss]), (js, iss)


def test_product_window_refuses_a_fortran_order_file(tmp_path, monkeypatch):
    """Row-range reads are only valid on C-order data."""
    from fronts.viz.apps.common import s3source

    path = tmp_path / "f.npy"
    np.save(path, np.asfortranarray(np.zeros((20, 20), dtype=np.int64)))
    monkeypatch.setattr(s3source, "_filesystems", lambda: (None, _LocalFS()))

    with pytest.raises(ValueError, match="C-order"):
        s3source._product_window(str(path), slice(0, 10), slice(0, 10))


def test_product_window_reads_only_the_bands_bytes(tmp_path, monkeypatch):
    """Pin the saving: bytes read scale with rows, not with file size."""
    from fronts.viz.apps.common import s3source

    full = np.zeros((1000, 200), dtype=np.int64)
    path = tmp_path / "labels.npy"
    np.save(path, full)

    fs = _LocalFS()
    monkeypatch.setattr(s3source, "_filesystems", lambda: (None, fs))
    s3source._product_window(str(path), slice(0, 50), slice(0, 200))

    assert max(fs.reads) == 50 * 200 * 8               # exactly the band
    assert sum(fs.reads) < full.nbytes / 10            # not the whole file


def test_a_label_valid_at_one_step_still_produces_frames(provider,
                                                          monkeypatch):
    """The failure the user hit: 0/17 frames from a valid dropdown pick.

    The front list offers labels at the chosen step, and labels are
    assigned per date.  Asking for that same label at every step gives
    "label N is absent from step M" for all of them.  Threading the track
    through is what makes the movie possible at all.
    """
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]

    # Relabel the world so no label recurs: step k's fronts are offset by
    # k * 1000, which is what real per-date labelling looks like.
    base = {step: np.asarray(provider.chunk_labels(chunk, step))
            for step in range(len(provider.chunk_timesteps(chunk)))}

    def relabelled(chunk_name, step):
        arr = base[int(step)].copy()
        arr[arr > 0] += int(step) * 1000
        return arr

    monkeypatch.setattr(provider, "chunk_labels", relabelled)

    anchor_label = int(np.max(relabelled(chunk, 0)))
    track = EP.build_track(provider, chunk, 0, anchor_label)

    # The anchor label exists nowhere else, yet the track spans more steps.
    assert track.label_at(0) == anchor_label
    assert len(track.steps()) > 1, "the track collapsed to the anchor step"
    for step in track.steps():
        if step:
            assert track.label_at(step) != anchor_label


# ---------------------------------------------------------------------------
# Stage (b): the region movie
# ---------------------------------------------------------------------------

def test_region_figure_needs_no_front_scene():
    """It takes arrays, not a FrontScene, and numbers every front.

    A scene needs a chosen front, a crop and a mixed-layer clip -- none of
    which you can choose before seeing the fronts.  This is the figure you
    look at to choose, so it must not require the choice.
    """
    from fronts.viz.apps.tiles import panels as F

    surface = 1027.0 + np.random.default_rng(0).random((80, 100))
    labels = np.zeros((80, 100), dtype=int)
    labels[30, 10:60] = 4242
    labels[60, 40:95] = 777
    labels[70, 0:3] = 5                              # speck, below min_pixels

    out = F.figure_region_fronts(surface, labels, field_name="density",
                                 selected=777)
    assert out.exists() and out.stat().st_size > 1000


def test_region_movie_runs_without_a_track(provider, monkeypatch):
    """No front chosen means every front cyan -- not a failure."""
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]
    calls = []
    monkeypatch.setattr(EP, "region_frame",
                        lambda *a, **k: calls.append(k.get("selected")) or "f")

    frames = EP.prerender_region(provider, chunk, "Ri", track=None)

    assert len(frames) == len(provider.chunk_timesteps(chunk))
    assert set(calls) == {0}, "no track must mean nothing is highlighted"


def test_region_movie_highlights_the_tracked_label_per_step(provider,
                                                             monkeypatch):
    """With a track, the red front follows it -- a different label each step."""
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]
    track = EP.build_track(provider, chunk, 0, 1)

    seen = {}
    def fake(prov, ch, step, field, *, selected=0, **k):
        seen[step] = selected
        return "f"

    monkeypatch.setattr(EP, "region_frame", fake)
    EP.prerender_region(provider, chunk, "Ri", track=track)

    for step, label in seen.items():
        assert label == (track.label_at(step) or 0)


def test_every_chunk_drawing_goes_through_one_remap(provider):
    """Both chunk figures must share the frame conversion.

    The bug this pins: draw_chunkmap read the tile directly and skipped
    remap_to_rect, while the labels it overlaid were rect-frame.  On a
    rotated face (10 = California Current) that puts the fronts across the
    features instead of along them, and nothing errors.
    """
    import inspect

    from fronts.viz.apps.evolution import app as ev_app
    from fronts.viz.apps.evolution import pipeline as EP

    src = inspect.getsource(ev_app.EvolutionPage.draw_chunkmap)
    assert "chunk_plane" in src
    assert "chunk_tile" not in src, "draw_chunkmap must not read a tile itself"

    assert "chunk_plane" in inspect.getsource(EP.region_frame)


def test_chunk_plane_returns_the_rect_frame(provider):
    """Surface, coords and labels must all come back the same shape."""
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]
    surface, lon, lat, labels, var = EP.chunk_plane(provider, chunk, 0, "Ri")

    assert surface.shape == lon.shape == lat.shape == labels.shape
    assert var


def test_region_playback_does_no_work_per_frame():
    """Stepping the region player must only swap an image.

    A watcher on `step` called draw_chunkmap, which reads a tile -- so
    playback queued a network read per frame, fell behind the player, and
    showed a frame that did not match the marker.
    """
    import inspect

    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())

    drew = []
    page.draw_chunkmap = lambda *a, **k: drew.append(1)

    # Bytes, not paths: playback must never trigger a fetch.
    page._region_bytes = [b"aaa", b"bbb", b"ccc"]
    for step in (0, 1, 2):
        page._on_region_step(type("E", (), {"new": step})())
        assert page._region_pane.object == page._region_bytes[step]
        assert page.state.step == step

    assert drew == [], "playback must not redraw the chunk map"


def test_region_frames_are_drawn_without_number_chips():
    """Seventeen frames of number chips is unreadable as a movie."""
    import inspect

    from fronts.viz.apps.evolution import pipeline as EP

    assert "annotate=False" in inspect.getsource(EP.region_frame)


def test_chunk_map_draws_fronts_cyan_thick_and_numbered():
    """The still you choose from: one colour, visible, with the numbers on.

    Fronts are often one cell wide, which at 720 cells across a 600-pixel
    figure is a sub-pixel line that does not survive rasterising -- hence
    the dilation.
    """
    from fronts.viz.apps.evolution import app as ev_app

    labels = np.zeros((40, 40), dtype=int)
    labels[20, 5:35] = 88                            # a one-cell-wide front

    r, g, b, a = ev_app._label_rgba(labels, 0, width=2)
    drawn = a > 0
    assert drawn.sum() > (labels > 0).sum(), "the mask was not thickened"

    # All one colour, and it is cyan.
    import matplotlib.colors as mcolors
    cyan = mcolors.to_rgb(ev_app.FRONT_COLOR)
    assert np.allclose(r[drawn], cyan[0])
    assert np.allclose(g[drawn], cyan[1])
    assert np.allclose(b[drawn], cyan[2])

    marks = ev_app._front_number_labels(labels)
    assert len(marks.data) == 1
    assert not hasattr(ev_app, "FRONT_PALETTE"), "the palette is retired"


def test_selected_front_is_still_distinguishable_on_the_chunk_map():
    from fronts.viz.apps.evolution import app as ev_app

    labels = np.zeros((30, 30), dtype=int)
    labels[10, 2:20] = 1
    labels[20, 2:20] = 2

    r, _g, _b, a = ev_app._label_rgba(labels, 2, width=1)
    import matplotlib.colors as mcolors
    assert r[a > 0].max() == pytest.approx(
        mcolors.to_rgb(ev_app.SELECTED_COLOR)[0])


# ---------------------------------------------------------------------------
# Selecting a front by place
# ---------------------------------------------------------------------------

def _lonlat(shape, lon0=230.0, lat0=35.0, span=2.0):
    lon = np.linspace(lon0, lon0 + span, shape[1])[None, :].repeat(shape[0], 0)
    lat = np.linspace(lat0, lat0 + span, shape[0])[:, None].repeat(shape[1], 1)
    return lon, lat


def test_nearest_front_picks_by_place_not_by_label():
    """A point identifies a front; a label identifies it in one step only."""
    from fronts.viz.apps.evolution import tracking as TR

    labels = np.zeros((100, 100), dtype=int)
    labels[20, 10:90] = 55555            # north-ish
    labels[80, 10:90] = 111              # south-ish
    lon, lat = _lonlat(labels.shape)

    # A point next to the southern front must find it, whatever it is
    # called -- the smaller label number must not win by being smaller.
    got, km = TR.nearest_front(labels, lon, lat,
                               float(lon[80, 50]), float(lat[79, 50]))
    assert got == 111 and km < 20

    got, _ = TR.nearest_front(labels, lon, lat,
                              float(lon[20, 50]), float(lat[21, 50]))
    assert got == 55555


def test_nearest_front_refuses_a_point_with_nothing_near_it():
    from fronts.viz.apps.evolution import tracking as TR

    labels = np.zeros((100, 100), dtype=int)
    labels[10, 10:90] = 7
    lon, lat = _lonlat(labels.shape)

    got, km = TR.nearest_front(labels, lon, lat,
                               float(lon[99, 50]), float(lat[99, 50]),
                               max_km=5.0)
    assert got is None and km > 5.0


def test_the_same_place_finds_the_front_under_any_relabelling():
    """The whole point: relabel everything, the place still resolves."""
    from fronts.viz.apps.evolution import tracking as TR

    labels = np.zeros((100, 100), dtype=int)
    labels[40, 10:90] = 12
    lon, lat = _lonlat(labels.shape)
    point = (float(lon[40, 50]), float(lat[41, 50]))

    first, _ = TR.nearest_front(labels, lon, lat, *point)

    relabelled = np.where(labels > 0, 98765, 0)
    second, _ = TR.nearest_front(relabelled, lon, lat, *point)

    assert first == 12 and second == 98765     # different names, same front


def test_clicking_the_map_stores_a_place(monkeypatch):
    """The tap must set lat/lon, not collapse straight to a label."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())
    lon, lat = _lonlat((60, 60))
    page._coords_step = (lon, lat)
    monkeypatch.setattr(page, "_resolve_anchor", lambda: None)

    page._on_tap(x=30.0, y=20.0)

    assert page.state.anchor_lon == pytest.approx(float(lon[20, 30]))
    assert page.state.anchor_lat == pytest.approx(float(lat[20, 30]))


# ---------------------------------------------------------------------------
# A fixed along-front axis across frames
# ---------------------------------------------------------------------------

def test_curtain_panel_honours_a_shared_x_extent():
    """The axis is fixed so the front's length is what visibly changes."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from fronts.viz import curtains

    L, K = 40, 12
    dist_px = np.arange(L, dtype=float)
    Z = -np.arange(K, dtype=float) * 10.0
    color = np.random.default_rng(0).random((K, L))
    sigma = 1027.0 + np.linspace(0, 1, K)[:, None] * np.ones((1, L))

    fig, ax = plt.subplots()
    curtains.plot_curtain_panel(ax, dist_px, Z, color, sigma,
                                add_colorbar=False, xmax=200.0)
    assert ax.get_xlim()[1] == pytest.approx(200.0)
    plt.close(fig)


def test_a_longer_front_is_never_clipped_by_the_shared_extent():
    """xmax extends the axis; it must never shrink it below the data."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from fronts.viz import curtains

    L, K = 80, 8
    dist_px = np.arange(L, dtype=float)
    Z = -np.arange(K, dtype=float) * 10.0
    color = np.zeros((K, L))
    sigma = 1027.0 + np.linspace(0, 1, K)[:, None] * np.ones((1, L))

    fig, ax = plt.subplots()
    # A shared extent smaller than this frame's own front.
    curtains.plot_curtain_panel(ax, dist_px, Z, color, sigma,
                                add_colorbar=False, xmax=10.0)
    assert ax.get_xlim()[1] >= dist_px[-1]
    plt.close(fig)


def test_shared_settings_reports_a_shared_x_extent(provider, monkeypatch):
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.tiles import panels as F

    chunk = provider.chunks()[0]
    monkeypatch.setattr(F, "pick_perp_index", lambda *a, **k: 0)

    track = EP.build_track(provider, chunk, 0, 1)
    shared = EP.shared_settings(provider, chunk, "Ri", track)

    assert "xmax" in shared
    if shared["xmax"] is not None:
        assert shared["xmax"] > 0


# ---------------------------------------------------------------------------
# Pinning things in place rather than in index space
# ---------------------------------------------------------------------------

def _scene_with_coords():
    """A minimal object with the attributes the point helpers read."""
    from types import SimpleNamespace

    lon = np.linspace(230.0, 231.0, 20)[None, :].repeat(15, 0)
    lat = np.linspace(35.0, 36.0, 15)[:, None].repeat(20, 1)
    path = np.column_stack([np.arange(15), np.arange(15)])   # diagonal
    return SimpleNamespace(XC=lon, YC=lat, axis_path=path)


def test_a_geographic_point_maps_into_the_crop():
    from fronts.viz.apps.evolution import pipeline as EP

    scene = _scene_with_coords()
    j, i = EP.crop_index_for_point(scene, float(scene.XC[7, 12]),
                                   float(scene.YC[7, 12]))
    assert (j, i) == (7, 12)


def test_the_transect_index_follows_the_place_not_the_fraction():
    """A fixed place must give different indices as the axis changes.

    That is the point: the front moves through one spot, rather than the
    transect sliding along a front whose length keeps changing.
    """
    from fronts.viz.apps.evolution import pipeline as EP

    scene = _scene_with_coords()
    point = (float(scene.XC[10, 10]), float(scene.YC[10, 10]))
    assert EP.axis_index_for_point(scene, *point) == 10

    # Same place, a shorter axis that starts further along: the index must
    # change to keep the transect over the same water.
    scene.axis_path = np.column_stack([np.arange(5, 15), np.arange(5, 15)])
    assert EP.axis_index_for_point(scene, *point) == 5


def test_the_profile_point_is_the_transect_point():
    """One place, both jobs.

    Two independent pickers asked the user to say twice where they were
    looking, and nothing good came of them differing.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionState

    st = EvolutionState(provider=sources.get_provider())
    assert not hasattr(st, "profile_points"), "the second picker is gone"
    assert "profile_points_text" not in st.param

    st.anchor_lat, st.anchor_lon = 36.0, 237.0
    st.perp_lat, st.perp_lon = 36.4, 237.5
    assert st.perp_point() == (237.5, 36.4)

def test_transect_point_falls_back_to_the_front_point():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionState

    st = EvolutionState(provider=sources.get_provider())
    st.anchor_lat, st.anchor_lon = 36.0, 237.0
    assert st.perp_point() == (237.0, 36.0)

    st.perp_lat, st.perp_lon = 36.4, 237.5
    assert st.perp_point() == (237.5, 36.4)


def test_profiles_are_not_a_movie_frame():
    """One figure with every step on it, not one figure per step.

    A movie of profiles shows one line at a time and throws away the
    comparison that is the whole point.
    """
    from fronts.viz.apps.evolution import pipeline as EP

    assert "profiles" not in EP.FRAME_ORDER
    assert "profiles" not in EP.FRAME_TITLES


def test_movie_downloads_as_a_real_animated_gif():
    """The download must be a playable multi-frame GIF, not a still."""
    import io

    import panel as pn
    pn.extension()
    from PIL import Image

    from fronts.viz.apps.evolution.app import EvolutionPage
    from fronts.viz.apps.tiles import panels as F

    page = EvolutionPage(provider=sources.get_provider())

    frames = []
    for k in range(3):
        surface = 1027.0 + np.random.default_rng(k).random((40, 50))
        labels = np.zeros((40, 50), dtype=int)
        labels[10 + k * 5, 5:45] = 42
        out = F.figure_region_fronts(surface, labels, field_name="density",
                                     selected=42, annotate=False)
        frames.append(out.read_bytes())

    page._region_bytes = frames
    data = page._movie_gif().getvalue()

    assert data[:6] in (b"GIF87a", b"GIF89a")
    assert Image.open(io.BytesIO(data)).n_frames == 3


def test_movie_download_is_empty_rather_than_broken_with_no_frames():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionPage

    page = EvolutionPage(provider=sources.get_provider())
    assert page._movie_gif().getvalue() == b""
    assert page.w_download.disabled is True


# ---------------------------------------------------------------------------
# The stacked profile figure
# ---------------------------------------------------------------------------

def test_profile_stack_draws_every_step_and_survives_gaps():
    from fronts.viz.apps.tiles import panels as F

    Z = -np.arange(20) * 10.0
    cols = [None if k in (2, 5) else 1027.0 + 0.01 * np.arange(20) + 0.02 * k
            for k in range(8)]
    times = [f"2012-07-0{1 + k // 4}T{(k * 3) % 24:02d}_00_00"
             for k in range(8)]

    out = F.figure_profile_stack(cols, Z, times, field_name="Ri",
                                 highlight=4)
    assert out.exists() and out.stat().st_size > 5000


def test_profile_stack_says_so_when_there_is_nothing_to_draw():
    from fronts.viz.apps.tiles import panels as F

    out = F.figure_profile_stack([None, None], -np.arange(5) * 10.0,
                                 ["a", "b"], field_name="Ri")
    assert out.exists()


def test_the_profile_highlight_follows_the_player():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())
    page._profile_bytes = [b"zero", b"one", b"two"]

    page._on_step(type("E", (), {"new": 2})())
    assert page._profile_pane.object == b"two"

    page._on_step(type("E", (), {"new": 0})())
    assert page._profile_pane.object == b"zero"


def test_show_frame_ignores_the_data_carried_with_a_frame():
    """A frame carries the profile column too; it is not a picture."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())
    page._frames = [{"inset": b"img", "_profile": np.zeros(5), "_Z": None}]
    page.show_frame(0)

    assert "_profile" not in page._panes
    assert page._panes["inset"].object == b"img"


# ---------------------------------------------------------------------------
# One longitude convention, one point, surface-only fields
# ---------------------------------------------------------------------------

def test_scene_longitudes_are_positive(provider, date):
    """0..360 everywhere: the plan view read -124 where the movie read 236."""
    from fronts.viz.apps.common import regions
    from fronts.viz.apps.tiles import pipeline

    region = regions.REGIONS[0]
    idx = regions.synthetic_tile_idx(region)
    labels = pipeline.tile_labels(provider, date, idx, None)
    label = pipeline.available_fronts(labels)[0]

    scene = pipeline.build_scene(provider, date, idx, "Ri", label,
                                 region=region.name)
    assert np.nanmin(scene.XC) >= 0.0
    assert np.nanmax(scene.XC) <= 360.0


def test_the_transect_follows_the_chosen_front_point():
    """Choosing a front should not mean retyping its coordinates."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution.app import EvolutionState

    st = EvolutionState(provider=sources.get_provider())
    st.anchor_lat, st.anchor_lon = 36.25, 237.5
    assert st.perp_lat == pytest.approx(36.25)
    assert st.perp_lon == pytest.approx(237.5)

    # An explicit transect still sticks until the front point moves again.
    st.perp_lat, st.perp_lon = 36.9, 238.1
    assert st.perp_point() == (238.1, 36.9)

    st.anchor_lat = 35.0
    assert st.perp_lat == pytest.approx(35.0)


def test_surface_only_fields_are_offered_but_cannot_be_sectioned(provider):
    from fronts.viz.apps import config
    from fronts.viz.apps.evolution import pipeline as EP
    from fronts.viz.apps.evolution.app import EvolutionState

    import panel as pn
    pn.extension()

    st = EvolutionState(provider=sources.get_provider())
    assert "oceTAUX" in st.param.field.objects, "wind fields must be offered"
    assert config.is_surface_only("oceTAUX")
    assert not config.is_surface_only("Ri")

    chunk = provider.chunks()[0]
    track = EP.build_track(provider, chunk, 0, 1)
    with pytest.raises(EP.SurfaceOnlyField, match="surface only"):
        EP.prerender(provider, chunk, "oceTAUX", track)


def test_every_movie_figure_has_its_own_gif_download():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app
    from fronts.viz.apps.evolution import pipeline as EP

    page = ev_app.EvolutionPage(provider=sources.get_provider())
    assert set(page._downloads) == set(EP.FRAME_ORDER)
    assert all(w.disabled for w in page._downloads.values())


def test_clicking_the_plan_view_places_the_transect(monkeypatch):
    import panel as pn
    pn.extension()
    from fronts.viz.apps.evolution import app as ev_app

    page = ev_app.EvolutionPage(provider=sources.get_provider())
    monkeypatch.setattr(page, "_draw_preview", lambda: None)

    page._on_preview_tap(x=-123.5, y=36.4)
    assert page.state.perp_lat == pytest.approx(36.4)
    # Normalised, so a click never reintroduces the negative convention.
    assert page.state.perp_lon == pytest.approx(236.5)


# ---------------------------------------------------------------------------
# Front tracking: shape, prediction, confidence
# ---------------------------------------------------------------------------

def _line(shape, j, i0, i1, label=1, thickness=1):
    out = np.zeros(shape, dtype=int)
    out[int(j):int(j) + thickness, int(i0):int(i1)] = int(label)
    return out


def test_shape_terms_stop_the_jump_to_a_wrongly_shaped_neighbour():
    """The bug this fixes: distance alone picks the nearer wrong front.

    Our front has moved on; a short stubby neighbour has not, so it is
    closer.  Position ranks the neighbour first and length/orientation
    have to overrule it.
    """
    from fronts import front_tracking as FT

    shape = (120, 200)
    step0 = _line(shape, 60, 20, 180, label=1)          # long, east-west

    # Six cells in an hour: ~4 km/h, fast for a front but plausible.
    step1 = _line(shape, 60, 26, 186, label=7)          # ours, moved east
    step1 += _line(shape, 62, 95, 115, label=8)         # stubby, stayed put

    anchor = FT.anchor_at(step0, 0, 1)
    track = FT.follow(lambda s: [step0, step1][s],
                      ["2012-07-03T00_00_00", "2012-07-03T01_00_00"], anchor)

    assert track.label_at(1) == 7, "followed the wrong front"

    # And position alone would indeed have preferred the decoy.
    ours = FT.describe(step1, 7)
    decoy = FT.describe(step1, 8)
    ref = FT.describe(step0, 1)
    assert (np.hypot(*(np.array(decoy.centre) - np.array(ref.centre)))
            < np.hypot(*(np.array(ours.centre) - np.array(ref.centre))))


def test_orientation_term_rejects_a_front_that_turned_too_far():
    from fronts import front_tracking as FT

    shape = (120, 120)
    ours = FT.describe(_line(shape, 60, 10, 110, label=1), 1)   # east-west

    crossing = np.zeros(shape, dtype=int)
    crossing[10:110, 60] = 2                                    # north-south
    turned = FT.describe(crossing, 2)

    score, terms = FT.score_candidate(turned, ours, ours.centre, 20.0)
    assert terms["orientation"] > 2.0, terms


def test_position_veto_beats_a_perfect_shape_match():
    """Identical shape must not license implausible motion.

    A candidate with exactly the same shape far away is more likely a
    different front that looks similar than the same one teleporting.
    """
    from fronts import front_tracking as FT

    shape = (120, 220)
    ours = FT.describe(_line(shape, 60, 10, 110, label=1), 1)
    far = FT.describe(_line(shape, 60, 100, 200, label=2), 2)

    score, terms = FT.score_candidate(far, ours, ours.centre, radius=5.0)
    assert terms["length"] == pytest.approx(0.0, abs=1e-6)
    assert terms["orientation"] == pytest.approx(0.0, abs=1e-6)
    assert not np.isfinite(score), "the veto did not fire"


def test_prediction_extrapolates_steady_motion():
    """The long daily links work because a steady front keeps moving."""
    from datetime import datetime, timedelta

    from fronts import front_tracking as FT

    t0 = datetime(2012, 7, 3, 0)
    history = [(0, (10.0, 10.0), t0),
               (1, (10.0, 20.0), t0 + timedelta(hours=1))]

    ahead = FT._predict(history, t0 + timedelta(hours=2))
    assert ahead == pytest.approx((10.0, 30.0))

    # With one sighting there is nothing to extrapolate from.
    assert FT._predict(history[:1], t0 + timedelta(hours=2)) == (10.0, 10.0)


def test_opposite_tilts_are_not_mistaken_for_the_same_orientation():
    """+40 and -40 are 80 degrees apart, not identical.

    The display convention folds orientation to 0-90, which is right for
    a histogram and wrong for comparing two fronts -- it makes mirror
    images look the same.  Tracking keeps the sign.
    """
    from fronts import front_tracking as FT

    shape = (120, 120)
    rows = np.arange(20, 100)
    plus = np.zeros(shape, dtype=int)
    plus[rows, rows] = 1                       # tilted one way
    minus = np.zeros(shape, dtype=int)
    minus[rows, 119 - rows] = 1                # mirrored

    assert FT.orientation_deg(plus) == pytest.approx(
        FT.orientation_deg(minus), abs=1.0)    # folded: indistinguishable

    a = FT.orientation_signed_deg(plus)
    b = FT.orientation_signed_deg(minus)
    assert FT._angle_gap(a, b) > 60.0          # signed: clearly different

    # An axis still has no direction, so near-parallel stays near-parallel.
    assert FT._angle_gap(89.0, -89.0) == pytest.approx(2.0)


def test_track_records_how_confident_each_link_was():
    """No ground truth, so the next best thing is saying which were close."""
    from fronts import front_tracking as FT

    shape = (100, 160)
    frames = [_line(shape, 50, 10 + 4 * k, 120 + 4 * k, label=100 + k)
              for k in range(5)]
    times = [f"2012-07-03T{k:02d}_00_00" for k in range(5)]

    track = FT.follow(lambda s: frames[s], times,
                      FT.anchor_at(frames[0], 0, 100))

    assert track.steps() == [0, 1, 2, 3, 4]
    assert all(l.score >= 0 for l in track.links.values())
    assert len(track.weakest(2)) == 2


def test_tracking_lives_outside_viz():
    """It takes label maps and returns labels; it is not a page.

    Under viz/apps/evolution it could only be reached by importing the
    app, so analysis code could not use it.
    """
    import importlib
    import sys

    for name in list(sys.modules):
        if name.startswith(("panel", "holoviews", "bokeh")):
            break
    else:
        mod = importlib.import_module("fronts.front_tracking")
        assert "panel" not in sys.modules, "tracking pulled in Panel"
        assert hasattr(mod, "follow")

    # The old import path still works.
    from fronts.viz.apps.evolution import tracking
    assert tracking.follow is importlib.import_module(
        "fronts.front_tracking").follow


def test_a_long_front_that_grows_at_one_end_is_not_lost():
    """The reported failure, reproduced.

    A 400-cell front extends 110 cells at one end.  It has not moved --
    but its *centroid* has shifted 55 cells, which centroid-distance
    scoring reads as implausible motion and vetoes.  A short unrelated
    front sitting just below it is then the only candidate left.

    Mask-to-mask distance says what we actually mean: could this be the
    same water?  For a front that grew at one end the answer is zero
    cells away.
    """
    from fronts import front_tracking as FT

    shape = (200, 600)
    step0 = np.zeros(shape, dtype=int)
    step0[100, 50:450] = 1
    step1 = np.zeros(shape, dtype=int)
    step1[100, 50:560] = 7                      # the same front, extended
    step1[112, 240:270] = 8                     # a short decoy below it

    track = FT.follow(lambda s: [step0, step1][s],
                      ["2012-07-03T00_00_00", "2012-07-03T01_00_00"],
                      FT.anchor_at(step0, 0, 1))

    assert track.label_at(1) == 7, "followed the short decoy"
    assert track.links[1].terms["position"] == pytest.approx(0.0)

    # The centroid really did move far enough to have been vetoed.
    c0 = FT.describe(step0, 1).centre
    c1 = FT.describe(step1, 7).centre
    assert abs(c1[1] - c0[1]) > 3 * FT.MIN_RADIUS_PX


def test_the_area_term_penalises_a_much_smaller_front():
    """'WAYY shorter and smaller' should cost a candidate, not be free."""
    from fronts import front_tracking as FT

    shape = (200, 600)
    big = FT.describe(np.pad(np.ones((1, 400), dtype=int),
                             ((100, 99), (50, 150))), 1)
    small = np.zeros(shape, dtype=int)
    small[100, 240:270] = 1
    tiny = FT.describe(small, 1)

    _score, terms = FT.score_candidate(tiny, big, big.centre, radius=20.0)
    assert terms["area"] > 3.0, terms
    assert terms["length"] > 3.0, terms


def test_mask_distance_is_measured_from_the_predicted_position():
    """Prediction shifts the reference before the distance is taken."""
    from fronts import front_tracking as FT

    mask = np.zeros((50, 50), dtype=bool)
    mask[25, 10:20] = True

    moved = FT._shift(mask, 5, 3)
    js, iss = np.nonzero(moved)
    assert js.min() == 30 and iss.min() == 13

    field = FT._distance_field(mask)
    assert field[25, 15] == 0.0
    assert field[28, 15] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Bivariate: a signed field centres on zero, and the JPDF panel
# ---------------------------------------------------------------------------

def test_a_field_spanning_both_signs_splits_at_zero():
    """Zero is the meaningful centre whether or not the field was named.

    Splitting a signed field at its median puts the boundary at an
    arbitrary value and makes 'positive' and 'negative' straddle one
    colour -- the exact distinction the reader is looking for.
    """
    from fronts.viz import bivariate as BV

    rng = np.random.default_rng(0)
    signed = rng.normal(0.3, 1.0, 20000)          # both signs, offset mean
    edges = BV.bin_edges(signed, 2, field_name="not_in_the_table")

    assert edges[1] == pytest.approx(0.0)
    assert edges[1] != pytest.approx(float(np.median(signed)))


def test_a_positive_definite_field_still_uses_quantiles():
    from fronts.viz import bivariate as BV

    rng = np.random.default_rng(1)
    positive = rng.lognormal(0.0, 0.5, 20000)
    edges = BV.bin_edges(positive, 2, field_name="gradb2")

    assert edges[1] > 0.0
    assert edges[1] == pytest.approx(float(np.median(positive)), rel=0.05)


def test_bivariate_jpdf_draws_the_section_boundaries():
    from fronts.viz import bivariate as BV

    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    scheme = BV.build_scheme(a, b, n=2, name_a="a", name_b="b")

    fig, ax = BV.figure_jpdf(a, b, name_a="a", name_b="b", scheme=scheme)
    # One dashed line per interior edge, on each axis.
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2
    assert ax.get_xlabel() == "a" and ax.get_ylabel() == "b"


def test_bivariate_jpdf_says_so_when_there_is_nothing_to_plot():
    from fronts.viz import bivariate as BV

    fig, ax = BV.figure_jpdf([np.nan, np.nan], [1.0, 2.0])
    assert not ax.get_images()


def test_the_bivariate_page_has_a_jpdf_for_each_section():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.bivariate.app import BivariatePage

    page = BivariatePage(provider=sources.get_provider())
    assert page._grid_jpdf is not None
    assert page._front_jpdf is not None


# ---------------------------------------------------------------------------
# The region map on Field Characteristics
# ---------------------------------------------------------------------------

def test_region_map_is_empty_until_a_region_is_chosen():
    """Nothing selected means nothing to show -- not the whole globe again."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    page.draw_regionmap()
    assert page._regionmap.object is None


def test_region_map_is_drawn_for_a_selection():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    page.state.set_bounds((200.0, -10.0, 240.0, 20.0))

    page.draw_regionmap()
    assert page._regionmap.object is not None


def test_the_region_map_is_tight_to_the_box_while_the_navigation_map_pads():
    """Two maps, two jobs: one to navigate with, one to read the panels by."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    page = CharacteristicsPage(provider=sources.get_provider())
    page.state.set_bounds((200.0, -10.0, 240.0, 20.0))

    (px0, px1), (py0, py1) = page._zoom_limits()          # padded
    (tx0, tx1), (ty0, ty1) = page._zoom_limits(pad=0.0)   # exact

    assert px0 < tx0 and px1 > tx1
    assert py0 < ty0 and py1 > ty1
    assert (tx0, tx1) == pytest.approx((200.0, 240.0))


# ---------------------------------------------------------------------------
# One colour range per field, across a whole movie
# ---------------------------------------------------------------------------

def test_region_frames_honour_a_shared_colour_range():
    """Per-frame limits make a movie unreadable.

    A colour that changes because the *scale* moved is indistinguishable
    on screen from one that changed because the ocean did.
    """
    import inspect

    from fronts.viz.apps.tiles import panels as F

    src = inspect.getsource(F.figure_region_fronts)
    assert "if clim is None:" in src, "no shared range accepted"

    surface = 1027.0 + np.random.default_rng(0).random((40, 50))
    labels = np.zeros((40, 50), dtype=int)
    labels[20, 5:45] = 3

    a = F.figure_region_fronts(surface, labels, field_name="density",
                               clim=(1027.0, 1028.0), annotate=False)
    assert a.exists()


def test_region_clim_pools_across_the_window(provider, monkeypatch):
    from fronts.viz.apps.evolution import pipeline as EP

    chunk = provider.chunks()[0]
    seen = []
    original = EP.chunk_plane

    def counted(prov, ch, step, field):
        seen.append(step)
        return original(prov, ch, step, field)

    monkeypatch.setattr(EP, "chunk_plane", counted)
    clim = EP.region_clim(provider, chunk, "Ri")

    assert len(seen) <= 3, f"sampled {len(seen)} steps"
    assert len(set(seen)) == len(seen), "sampled the same step twice"
    if clim is not None:
        assert clim[0] < clim[1]


def test_the_perpendicular_shows_the_mixed_layer():
    """The transect crosses the axis, so the along-axis curtain will not do."""
    import inspect

    from fronts.viz import curtains
    from fronts.viz.apps.tiles import panels as F

    assert "mld_curtain" in inspect.signature(
        curtains.figure_perpendicular).parameters
    assert "scene.mld_field" in inspect.getsource(F.figure_perpendicular)


def test_surface_only_fields_can_colour_the_tiles_map(provider):
    """A plan view needs one level; a curtain needs a profile."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps import config
    from fronts.viz.apps.common.state import TilesState

    st = TilesState(provider=provider)
    for name in ("oceTAUX", "oceQnet", "mixed_layer_depth",
                 "ml_heat_content", "KE"):
        assert name in st.param.region_field.objects, name
        assert name not in st.param.fields.objects, \
            f"{name} has no depth to section"
        assert config.is_surface_only(name)


# ---------------------------------------------------------------------------
# Depth channel resolution
# ---------------------------------------------------------------------------

class _DepthStore(sources.DataProvider):
    """A store named exactly as run_v5_depth.yaml produces.

    Compute channels carry a depth suffix; ``extra_channels`` and the
    surface-only subsets do not.  Both kinds have to work.
    """

    mode, synthetic = "test", True

    def dates(self): return ["2012-05-16T06_00_00"]
    def dates_3d(self): return self.dates()
    def coords(self, d): ...
    def field(self, d, n): ...
    def front_binary(self, d): ...
    def labels(self, d): ...
    def geometry(self, d): ...
    def colocation(self, d): ...
    def tile(self, *a, **k): ...

    def field_names(self, date):
        out = []
        for root in ("N2", "Ri", "gradb2", "relative_vorticity"):
            out += [f"{root}_{s}"
                    for s in ("sfc", "z25m", "mld", "mld_mean")]
        out += ["mixed_layer_depth", "ml_heat_content",
                "oceTAUX", "oceTAUY", "oceQnet", "coriolis_f", "SIarea"]
        return sorted(out)


def test_depth_field_list_offers_roots_not_every_suffix():
    """N2 once, not four times -- and resolve must not double-suffix."""
    store = _DepthStore()
    date = store.dates()[0]

    roots = store.field_roots(date)
    assert roots.count("N2") == 1
    assert "N2_mld" not in roots
    # Bare channels are their own roots.
    assert "mixed_layer_depth" in roots and "oceTAUX" in roots


def test_a_bare_channel_is_not_given_a_depth_suffix():
    """mixed_layer_depth_mld does not exist; mixed_layer_depth does."""
    store = _DepthStore()
    date = store.dates()[0]

    assert store.channel_in(date, "N2", "Mixed layer depth") == "N2_mld"
    assert store.channel_in(date, "N2", "25 m") == "N2_z25m"
    assert store.channel_in(
        date, "mixed_layer_depth", "Mixed layer depth") == "mixed_layer_depth"
    assert store.channel_in(date, "oceTAUX", "25 m") == "oceTAUX"


def test_a_field_that_was_not_built_says_which_names_it_looked_for():
    store = _DepthStore()
    with pytest.raises(KeyError, match="Fr_mld"):
        store.channel_in(store.dates()[0], "Fr", "Mixed layer depth")


def test_surface_mode_channel_resolution_is_untouched():
    """The Surface page must not notice any of this."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, SURFACE

    page = CharacteristicsPage(SURFACE, provider=sources.get_provider())
    assert page.resolve("gradb2") == "gradb2"
    assert page.mode.has_depth is False


def test_the_depth_page_is_served():
    from fronts.viz.apps import config, serve

    assert "depth" in config.ENABLED_PAGES
    assert "/depth" in serve.ROUTES


def test_a_surface_only_field_can_be_read_from_a_tile():
    """'expected exactly one 3-D variable, found []' was the wind failing.

    A plan view needs one level and nothing more, so resolving the field
    variable must not insist on a depth axis -- wind stress, heat flux and
    the mixed-layer quantities are genuinely two-dimensional.
    """
    import xarray as xr

    from fronts.viz.apps.tiles import pipeline

    flat = xr.Dataset({"oceTAUX": (("j", "i"), np.zeros((4, 5))),
                       "XC": (("j", "i"), np.zeros((4, 5))),
                       "YC": (("j", "i"), np.zeros((4, 5)))})
    assert pipeline.sole_field(flat) == "oceTAUX"

    deep = xr.Dataset({"Ri": (("k", "j", "i"), np.zeros((3, 4, 5))),
                       "XC": (("j", "i"), np.zeros((4, 5)))})
    assert pipeline.sole_field(deep) == "Ri"          # 3-D still preferred

    # Curtains still refuse: a surface field has no profile to section.
    with pytest.raises(KeyError, match="surface-only"):
        pipeline._sole_3d(flat)


def test_the_isopycnal_figure_is_the_depth_axis_one():
    """The along-surface-length version was tried and dropped."""
    from fronts.viz import curtains
    from fronts.viz.apps.tiles import panels as F

    assert not hasattr(curtains, "figure_isopycnal_length")
    assert not hasattr(F, "figure_isopycnal_legacy")
    assert "isopycnal_legacy" not in F.FIGURE_ORDER
    assert "figure_isopycnal_surface" in inspect.getsource(F.figure_isopycnal)


# ---------------------------------------------------------------------------
# Resolution of a zoomed view
# ---------------------------------------------------------------------------

def test_the_interactive_maps_keep_their_cell_budget(monkeypatch):
    """Detail on demand belongs behind a button, not on every pan.

    Counting only the *visible* cells would let each zoomed map jump to
    the finest pyramid level -- more correct in principle, much heavier on
    a map that redraws as you navigate.  The static region figure is
    where native resolution lives instead.
    """
    from fronts.viz.apps import config
    from fronts.viz.apps.common import basemap

    monkeypatch.setattr(basemap, "HAVE_DATASHADER", False)

    box = ((230.0, 240.0), (33.0, 40.0))
    wanted = basemap.width_for_extent(box)
    assert wanted == max(config.PYRAMID_WIDTHS)

    # Budgeted the same way with or without an extent: the interactive
    # path does not get heavier just because you zoomed in.
    assert basemap._affordable_width(wanted) < wanted
    assert basemap._affordable_width(wanted, box) == \
        basemap._affordable_width(wanted)


def test_the_global_view_is_still_capped(monkeypatch):
    """The budget must still apply where the whole raster really is sent."""
    from fronts.viz.apps import config
    from fronts.viz.apps.common import basemap

    monkeypatch.setattr(basemap, "HAVE_DATASHADER", False)

    globe = ((0, 360), config.PYRAMID_LAT_RANGE)
    assert basemap._affordable_width(max(config.PYRAMID_WIDTHS),
                                     globe) < max(config.PYRAMID_WIDTHS)


def test_the_region_map_is_a_static_figure_from_native_data():
    """Not an interactive map, and not from the display pyramid."""
    import inspect

    from fronts.viz.apps.characteristics import panels as P
    from fronts.viz.apps.characteristics.page import CharacteristicsPage

    src = inspect.getsource(CharacteristicsPage.draw_regionmap)
    assert "figure_region_map" in src
    assert "global_map" not in src, "that is the interactive path"

    builder = inspect.getsource(P.figure_region_map)
    assert "provider.field(" in builder       # the native array
    # The call, not the word: the docstring explains why the pyramid is
    # the wrong source here, so a bare substring check reads its own prose.
    assert "pyramid.level(" not in builder
    assert "basemap." not in builder


def test_the_static_region_map_draws_fronts_on_the_same_window():
    """One window slice for the field and the fronts, so they must align."""
    import inspect

    from fronts.viz.apps.characteristics import panels as P

    src = inspect.getsource(P.figure_region_map)
    assert "provider.field(date, channel)[win]" in src
    assert "provider.front_binary(date)[win]" in src


# ---------------------------------------------------------------------------
# The region map on the Depth page
# ---------------------------------------------------------------------------

def test_the_depth_page_asks_for_a_depth_provider():
    """A SURF provider here would show surface fields without complaint.

    The depth fields are in their own S3 prefix under suffixed channel
    names, so a shared SURF provider finds only bare names -- and shows
    them, under a depth selector that changes nothing.  Wrong with no
    error anywhere, which is the kind worth a test.
    """
    import inspect

    from fronts.viz.apps.characteristics import depth

    src = inspect.getsource(depth.page)
    assert 'get_provider("DEPTH")' in src


def test_get_provider_keeps_the_two_pipelines_apart(monkeypatch):
    from fronts.viz.apps.common import sources as S

    monkeypatch.setenv("FRONTS_APP_DATA", "s3")
    S.get_provider.cache_clear()
    monkeypatch.setattr(S, "_OVERRIDE", None)
    try:
        surf = S.get_provider("SURF")
        deep = S.get_provider("DEPTH")
        assert surf.folder != deep.folder
        assert deep.folder == config.DEPTH_FOLDER
        # Fronts are shared: there is one set of labels.
        assert surf.fronts_folder == deep.fronts_folder
    finally:
        S.get_provider.cache_clear()


def test_the_region_map_uses_the_resolved_channel_in_depth_mode(monkeypatch):
    """gradb2 at 'Mixed layer depth' must request gradb2_mld, not gradb2."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics import panels as P
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    page.state.set_bounds((200.0, -10.0, 240.0, 20.0))

    asked = {}
    monkeypatch.setattr(
        P, "figure_region_map",
        lambda prov, date, channel, box, **k: asked.setdefault(
            "channel", channel) or "fig")
    # Force a store whose channels really are suffixed.
    monkeypatch.setattr(page, "resolve", lambda f: f"{f}_mld")

    page.draw_regionmap()
    assert asked["channel"] == "gradb2_mld" or asked["channel"].endswith("_mld")


def test_both_pages_build_the_same_kind_of_region_figure():
    """Depth inherits the feature; it must not diverge quietly."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    made = {}
    for mode in (SURFACE, DEPTH):
        page = CharacteristicsPage(mode, provider=sources.get_provider())
        page.state.set_bounds((200.0, -10.0, 240.0, 20.0))
        page.draw_regionmap()
        made[mode.key] = (type(page._regionmap).__name__,
                          type(page._regionmap.object).__name__)

    assert made["surface"] == made["depth"]
    assert made["surface"] == ("Matplotlib", "Figure")


# ---------------------------------------------------------------------------
# Preprocessing-branch API mismatch
# ---------------------------------------------------------------------------

def test_a_branch_with_a_different_tile_api_says_so():
    """"no attribute 'resolve_property'" names a symptom, not a cause.

    The app reproduces steps 1-7 of tile_utils.run rather than calling it
    (run writes a NetCDF and returns a path -- there is no way to ask it
    for a Dataset), so it reaches into module internals, and those differ
    between branches of the preprocessing repo.
    """
    from types import SimpleNamespace

    from fronts.viz.apps.common import s3source

    # The API as it stands on 'transfer-depth-seasons'.
    partial = SimpleNamespace(
        _build_output_dataset=1, _load_grid_for_tile=1,
        _load_tracers_for_tile=1, compute_tile_property=1,
        mit_date_to_iteration=1, rect_ij_to_tile=1,
        __file__="/x/src/dbof/tiles/tile_utils.py")

    with pytest.raises(RuntimeError) as excinfo:
        s3source._check_tile_api(partial)

    message = str(excinfo.value)
    assert "resolve_property" in message
    assert "_build_tile_context" in message
    assert "tiles-viz" in message                 # the branch it wants
    assert "stored tiles" in message              # why only some fields fail


def test_the_full_tile_api_passes_the_check():
    from types import SimpleNamespace

    from fronts.viz.apps.common import s3source

    complete = SimpleNamespace(**{n: 1 for n in s3source._TILE_API})
    s3source._check_tile_api(complete)            # must not raise


def test_depth_levels_come_from_the_store_not_from_config():
    """A partial depth build must not offer levels it did not produce.

    run_v5_depth.yaml can be run with depth_suffixes: [sfc], and step 4
    can still be in progress -- offering 'Mixed layer depth' then puts the
    failure after the click instead of before it.
    """
    store = _DepthStore()
    date = store.dates()[0]

    # Everything built: all four offered.
    assert store.depth_levels(date) == list(config.DEPTH_LEVELS)

    # A surface-only build offers only the surface.
    class _SfcOnly(_DepthStore):
        def field_names(self, date):
            return ["N2_sfc", "Ri_sfc", "mixed_layer_depth", "oceTAUX"]

    levels = _SfcOnly().depth_levels(date)
    assert levels == ["Surface"]


def test_depth_levels_never_come_back_empty():
    """A store with no suffixed channels at all still needs a control."""
    class _Bare(_DepthStore):
        def field_names(self, date):
            return ["mixed_layer_depth", "oceTAUX", "oceQnet"]

    store = _Bare()
    assert store.depth_levels(store.dates()[0]) == list(config.DEPTH_LEVELS)


def test_a_partly_built_depth_store_still_lists_its_fields():
    """Subsets land one at a time; the page grows with them.

    _cached_index skips a subset that is not there yet, so kinematic,
    icearea and native_fields arriving later add fields rather than
    breaking the ones already present.
    """
    class _Partial(_DepthStore):
        def field_names(self, date):
            # stratification + vertical_shear only.
            return sorted([f"{r}_{s}" for r in ("N2", "Ri", "vertical_shear")
                           for s in ("sfc", "z25m", "mld", "mld_mean")]
                          + ["mixed_layer_depth", "ml_heat_content"])

    store = _Partial()
    date = store.dates()[0]
    roots = store.field_roots(date)

    assert roots == ["N2", "Ri", "mixed_layer_depth", "ml_heat_content",
                     "vertical_shear"]
    assert store.channel_in(date, "Ri", "Mixed layer depth") == "Ri_mld"
    assert store.depth_levels(date) == list(config.DEPTH_LEVELS)


# ---------------------------------------------------------------------------
# Depth: display styles, and the two-stage flow
# ---------------------------------------------------------------------------

def test_a_depth_channel_is_drawn_like_its_surface_twin():
    """gradb2_mld is gradb2. It was falling back to a linear default.

    Style lookup is by channel name, and a DEPTH channel carries a suffix,
    so *every* depth field missed its registered style -- log scale,
    colours and all -- and got percentile-linear instead.
    """
    from fronts.viz import field_styles
    from fronts.viz.apps.common import basemap

    for root in ("gradb2", "Ri", "N2"):
        base = field_styles.get_style(root)
        for suffix in ("sfc", "z25m", "mld", "mld_mean"):
            got = field_styles.get_style(f"{root}_{suffix}")
            assert got.transform == base.transform, f"{root}_{suffix}"
            assert got.cmap == base.cmap, f"{root}_{suffix}"

    # The map has its own display path; it must agree.
    arr = np.abs(np.random.default_rng(0).normal(0, 1e-14, (20, 30)))
    for name in ("gradb2", "gradb2_mld", "gradb2_mld_mean"):
        _v, _clim, label = basemap.field_display(arr, name)
        assert label.startswith("log10("), name
        assert basemap._FIELD_CMAPS.get(
            name, basemap._FIELD_CMAPS.get(basemap._root(name))) == "gray"


def test_mld_mean_is_not_mistaken_for_mld():
    """'mld' is a prefix of 'mld_mean'.

    Stripping the shorter one first leaves N2_mean, which is registered
    nowhere and falls back silently -- so the suffixes are tried longest
    first.
    """
    from fronts.viz import field_styles

    assert field_styles.strip_depth_suffix("N2_mld_mean") == "N2"
    assert field_styles.strip_depth_suffix("N2_mld") == "N2"
    assert field_styles.strip_depth_suffix("N2") == "N2"
    assert field_styles.strip_depth_suffix("mixed_layer_depth") == \
        "mixed_layer_depth"


def test_the_depth_page_waits_for_the_map_button():
    """Opening the page must not regrid a 0.9 GB plane."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    deep = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    assert DEPTH.manual_map is True
    assert deep._map.object is None, "the map drew itself on open"

    # Surface is unchanged: its map is there immediately.
    surf = CharacteristicsPage(SURFACE, provider=sources.get_provider())
    assert SURFACE.manual_map is False
    assert surf._map.object is not None


def test_the_depth_page_has_two_buttons_and_they_do_different_things():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    calls = []
    page.schedule_stats = lambda: calls.append("stats")
    page.schedule_front_props = lambda: calls.append("props")

    page.rebuild()                      # map only -- no region chosen yet
    assert page._map.object is not None
    assert calls == []

    page.state.set_bounds((200.0, -10.0, 240.0, 20.0))
    page.run_statistics()
    assert calls == ["stats", "props"]


def test_surface_still_computes_everything_from_one_button():
    """The Surface flow is unchanged."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, SURFACE)

    page = CharacteristicsPage(SURFACE, provider=sources.get_provider())
    calls = []
    page.schedule_stats = lambda: calls.append("stats")
    page.schedule_front_props = lambda: calls.append("props")

    page.rebuild()
    assert calls == ["stats", "props"]


def test_depth_fronts_come_from_the_surface_products():
    """One set of labels, found at the surface, used at every level."""
    assert config.DEPTH_FRONTS_FOLDER == config.SURFACE_FRONTS_FOLDER
    assert config.DEPTH_FRONTS_RUN_ID == config.SURFACE_FRONTS_RUN_ID
    # ... while the fields themselves are separate.
    assert config.DEPTH_FOLDER != config.SURFACE_FOLDER


# ---------------------------------------------------------------------------
# Drawing a box must not destroy the map
# ---------------------------------------------------------------------------

def test_a_box_on_the_depth_map_does_not_refetch_the_field():
    """Selecting a region must not rebuild the raster.

    The zoomed extent asks width_for_extent for a finer pyramid level than
    the global view did.  For a depth channel that level may not exist, so
    the rebuild failed, the handler returned an empty overlay, and the map
    collapsed -- taking the box-select tool with it.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    page.rebuild()
    dmap = page._map.object
    dmap[()]                       # a DynamicMap is lazy; Panel renders it

    builds = []
    original = page._base_overlay
    page._base_overlay = lambda extent: (builds.append(extent)
                                         or original(extent))

    before = len(dmap[()])
    page._bounds.event(bounds=(200.0, -10.0, 240.0, 20.0))
    after = dmap[()]

    assert builds == [], "the field was re-fetched for the box"
    assert len(after) > before, "the outline was not added"
    assert page.state.box.label() != "global"


def test_a_failed_redraw_keeps_the_map_that_worked():
    """An empty overlay loses the picture *and* the box-select tool."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    page.rebuild()
    page._map.object[()]                        # render it once
    good = page._map_base
    assert good is not None

    def boom(extent):
        raise RuntimeError("pyramid level not built for this channel")

    page._base_overlay = boom
    page._map_base = None                       # force the rebuild path
    frame = page._map_for_bounds(bounds=None)
    assert "Map unavailable" in (page._status.object or "")
    assert len(frame) <= 1                      # nothing to fall back to

    page._map_base = good                       # with a fallback available
    page._map_stale = True
    frame = page._map_for_bounds(bounds=None)
    assert len(frame) > 1, "a working map was thrown away"


def test_a_failed_rebuild_does_not_collapse_the_map():
    """Pressing Rebuild must not be able to destroy a working map.

    redraw_map used to clear the cached base so the next frame would
    refetch -- which defeated the fallback at exactly the moment it was
    needed: a failed refetch then had nothing to fall back to, and
    replaced the map with an empty frame.  Staleness is now a separate
    flag, so the old base survives until a new one succeeds.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    page.rebuild()
    working = len(page._map.object[()])
    assert working > 1

    def boom(extent):
        raise RuntimeError("pyramid level not built for this channel")

    good = page._base_overlay
    page._base_overlay = boom
    page.rebuild()

    assert len(page._map.object[()]) > 1, "a failed Rebuild collapsed the map"
    assert "Map unavailable" in (page._status.object or "")

    # And it recovers: the map is still marked stale, so the next
    # successful attempt refetches rather than serving the old base.
    assert page._map_stale is True
    page._base_overlay = good
    page.rebuild()
    page._map.object[()]                # a DynamicMap clears it on render
    assert page._map_stale is False


def test_rebuild_map_actually_refetches():
    """The cached base must not outlive the field that made it."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    page.rebuild()
    page._map.object[()]
    assert page._map_base is not None

    page.redraw_map()
    assert page._map_stale is True, "the base would survive a field change"
    page._map.object[()]
    assert page._map_stale is False, "the refetch never happened"


# ---------------------------------------------------------------------------
# The kinematic roles behind the joint PDFs
# ---------------------------------------------------------------------------

class _RoleStore(sources.DataProvider):
    mode, synthetic = "test", True

    def __init__(self, names):
        self._names = sorted(names)

    def dates(self): return ["2012-05-16T06_00_00"]
    def dates_3d(self): return self.dates()
    def field_names(self, date): return self._names
    def coords(self, d): ...
    def field(self, d, n): ...
    def front_binary(self, d): ...
    def labels(self, d): ...
    def geometry(self, d): ...
    def colocation(self, d): ...
    def tile(self, *a, **k): ...


def test_roles_resolve_to_roots_so_the_level_is_applied_once():
    """Returning a channel name meant a second suffix got appended.

    resolve_channels used to answer 'relative_vorticity_sfc', which the
    page then resolved again -- 'relative_vorticity_sfc_mld'.
    """
    date = "2012-05-16T06_00_00"
    store = _RoleStore(
        [f"{r}_{s}" for r in ("relative_vorticity", "strain_mag")
         for s in ("sfc", "z25m", "mld", "mld_mean")] + ["coriolis_f"])

    roles = store.resolve_channels(date)
    assert roles == {"vorticity": "relative_vorticity",
                     "strain": "strain_mag", "coriolis": "coriolis_f"}

    at_mld = {k: store.channel_in(date, v, "Mixed layer depth")
              for k, v in roles.items()}
    assert at_mld["vorticity"] == "relative_vorticity_mld"
    assert at_mld["strain"] == "strain_mag_mld"
    # Coriolis has no depth variant, and needs no special case.
    assert at_mld["coriolis"] == "coriolis_f"


def test_a_store_built_at_one_level_only_still_has_its_roles():
    """Matching exact channel names reported every role missing."""
    date = "2012-05-16T06_00_00"
    store = _RoleStore(["relative_vorticity_mld", "strain_mag_mld",
                        "coriolis_f"])

    roles = store.resolve_channels(date)
    assert all(roles.values()), roles
    assert store.channel_in(date, roles["vorticity"],
                            "Mixed layer depth") == "relative_vorticity_mld"


def test_surface_roles_are_unchanged():
    date = "2012-05-16T06_00_00"
    store = _RoleStore(["relative_vorticity", "strain_mag", "coriolis_f"])
    assert store.resolve_channels(date) == {
        "vorticity": "relative_vorticity", "strain": "strain_mag",
        "coriolis": "coriolis_f"}


def test_a_missing_kinematic_subset_says_what_to_build():
    """Panel (a) draws fine from the same region, so "empty" is baffling."""
    from fronts.viz.apps.characteristics import panels as P
    from fronts.viz.apps.characteristics.stats import RegionSamples

    date = "2012-05-16T06_00_00"
    store = _RoleStore(["N2_mld", "Ri_mld", "coriolis_f"])
    roles = store.resolve_channels(date)
    assert roles["vorticity"] is None and roles["strain"] is None

    samples = RegionSamples(values=np.zeros(3), zeta_f=np.empty(0),
                            sigma_f=np.empty(0), n_cells=10,
                            missing=("vorticity", "strain"))
    message = P._kinematics_message(samples)
    assert "relative_vorticity" in message
    assert "strain_mag" in message
    assert "not the selected field" in message


# ---------------------------------------------------------------------------
# The map keeps its frame, and Rebuild re-reads the store
# ---------------------------------------------------------------------------

def test_composing_an_overlay_drops_its_options():
    """The reason the map squashed, stated as the library behaviour.

    ``base * outline`` builds a *new* Overlay and the Overlay-level
    options set on ``base`` do not come with it -- so the height was lost
    and Bokeh fell back to its default frame.  ``.opts(xlim=...)`` does
    not bring them back either.
    """
    import holoviews as hv
    hv.extension("bokeh")

    base = hv.Overlay([hv.Image((np.arange(4), np.arange(3),
                                 np.zeros((3, 4))))])
    base = base.opts(hv.opts.Overlay(height=720))
    assert base.opts.get("plot").kwargs.get("height") == 720

    composed = (base * hv.Rectangles([(0, 0, 2, 2)])).opts(
        hv.opts.Overlay(xlim=(0, 4)))
    assert composed.opts.get("plot").kwargs.get("height") is None


def test_the_map_keeps_its_height_through_every_interaction():
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    for mode in (DEPTH, SURFACE):
        page = CharacteristicsPage(mode, provider=sources.get_provider())
        if mode.manual_map:
            page.rebuild()

        def height():
            return page._map.object[()].opts.get("plot").kwargs.get("height")

        assert height() == page.MAP_HEIGHT, mode.key
        page._bounds.event(bounds=(200.0, -10.0, 240.0, 20.0))
        assert height() == page.MAP_HEIGHT, f"{mode.key}: box"
        page.rebuild()
        assert height() == page.MAP_HEIGHT, f"{mode.key}: rebuild"


def test_the_box_select_survives_a_redraw():
    """The tool has to still be wired after *Rebuild*.

    HoloViews links box-select to a stream through
    ``Stream.registry[stream.source]``, and ``source`` is pinned to the
    first DynamicMap the stream was given to.  Replacing the DynamicMap on
    every redraw -- even while keeping the same stream -- built the new
    plot with no BoundsCallback, so from the first Rebuild onwards the box
    drew and reported nothing and the region silently stayed put.  That is
    what made the selection work sometimes and not others.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    for mode in (SURFACE, DEPTH):
        page = CharacteristicsPage(mode, provider=sources.get_provider())
        if mode.manual_map:
            page.rebuild()

        assert page._bounds.source is page._map.object, \
            f"{mode.key}: never wired"

        # Every way the map gets redrawn.
        page.rebuild()
        page.redraw_map()
        page.state.field = page.state.param.field.objects[-1]
        page._reset_region()

        assert page._bounds.source is page._map.object, \
            f"{mode.key}: the box-select was left on a discarded map"

        # And the box drawn after all that has to land.
        page._bounds.event(bounds=(200.0, -10.0, 240.0, 20.0))
        assert page.state.box.label() != "global", f"{mode.key}: box lost"
        assert page._bounds.source is page._map.object, \
            f"{mode.key}: unwired by the box it just handled"


def test_holoviews_pins_a_stream_to_its_first_dynamicmap():
    """Why the map above is created once and then only nudged.

    This is HoloViews behaviour, not ours; if it ever changes, the
    machinery in ``redraw_map`` can be simplified away.
    """
    import numpy as np
    import holoviews as hv
    from holoviews.plotting.bokeh import BokehRenderer

    hv.extension("bokeh")
    stream = hv.streams.BoundsXY(bounds=None)
    frame = lambda bounds=None: hv.Image(np.zeros((4, 4)))   # noqa: E731

    first = hv.DynamicMap(frame, streams=[stream])
    assert [type(c).__name__ for c in BokehRenderer.get_plot(first).callbacks] \
        == ["BoundsCallback"]

    second = hv.DynamicMap(frame, streams=[stream])
    assert stream.source is first, "source moved to the new DynamicMap"
    assert BokehRenderer.get_plot(second).callbacks == [], \
        "the second plot was wired after all -- redraw_map can be simpler"


def test_a_redraw_still_refetches_the_field():
    """Nudging must not become a no-op: a new field has to be read."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, SURFACE)

    page = CharacteristicsPage(SURFACE, provider=sources.get_provider())
    page._map.object[()]

    builds = []
    original = page._base_overlay
    page._base_overlay = lambda extent: (builds.append(extent)
                                         or original(extent))
    page.redraw_map()
    page._map.object[()]
    assert builds, "the redraw drew the stale base again"


def test_bokeh_fixes_axis_limits_when_the_plot_is_built():
    """Why a zoom or an un-zoom has to be a new plot.

    A DynamicMap frame carrying different ``xlim`` does not move the axes
    of a plot that already exists -- it is silently ignored.  This is
    HoloViews/Bokeh behaviour, not ours; if it changes, ``redraw_map`` can
    stop replacing the plot.
    """
    import numpy as np
    import holoviews as hv
    from holoviews.plotting.bokeh import BokehRenderer

    hv.extension("bokeh")
    Nudge = hv.streams.Stream.define("Nudge", n=0)
    nudge = Nudge()

    def frame(n=0):
        lim = (0, 360) if n == 0 else (100, 140)
        return hv.Image(np.zeros((4, 4))).opts(hv.opts.Image(xlim=lim))

    figure = BokehRenderer.get_plot(hv.DynamicMap(frame, streams=[nudge])).state
    assert (figure.x_range.start, figure.x_range.end) == (0, 360)
    nudge.event(n=1)
    assert (figure.x_range.start, figure.x_range.end) == (0, 360), \
        "the frame moved the axes after all -- redraw_map can be simpler"


def test_reset_region_puts_the_map_back_on_the_globe():
    """The button has to un-zoom, which means a new plot."""
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    for mode in (SURFACE, DEPTH):
        page = CharacteristicsPage(mode, provider=sources.get_provider())
        if mode.manual_map:
            page.rebuild()

        page._bounds.event(bounds=(200.0, -10.0, 240.0, 20.0))
        assert page._map_for_bounds().opts.get().kwargs["xlim"] != (0, 360)

        page._reset_region()
        assert page.state.box.is_global, f"{mode.key}: box not cleared"
        assert page._map_for_bounds().opts.get().kwargs["xlim"] == (0, 360), \
            f"{mode.key}: still zoomed in"
        assert page._selection_outline() is None, \
            f"{mode.key}: the outline outlived the selection"


def test_drawing_a_box_reads_no_data():
    """Selection is navigation; only *Rebuild* pays for data.

    The zoomed extent asks width_for_extent for a finer pyramid level than
    the global view did, and for a depth channel that level may not exist
    -- so the refetch failed and the map collapsed.  Cropping the field
    already in hand cannot fail and cannot be slow.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import (
        CharacteristicsPage, DEPTH, SURFACE)

    for mode in (SURFACE, DEPTH):
        page = CharacteristicsPage(mode, provider=sources.get_provider())
        if mode.manual_map:
            page.rebuild()
        page._map.object[()]                 # the first frame reads, once

        reads = []
        original = page._base_overlay
        page._base_overlay = lambda extent: (reads.append(extent)
                                             or original(extent))

        page._bounds.event(bounds=(200.0, -10.0, 240.0, 20.0))
        page._map.object[()]
        assert reads == [], f"{mode.key}: the box re-read the field"

        page.rebuild()                       # this one is allowed to
        page._map.object[()]
        assert reads, f"{mode.key}: Rebuild did not re-read the field"


def test_rebuild_re_reads_the_store():
    """A subset written after the page first looked must become visible.

    The channel listing is cached per date, so during a build in progress
    a newly written subset stayed invisible for the life of the process --
    and the panels that need it stayed blank with no way to tell that the
    data had arrived.
    """
    import panel as pn
    pn.extension()
    from fronts.viz.apps.characteristics.page import CharacteristicsPage, DEPTH

    page = CharacteristicsPage(DEPTH, provider=sources.get_provider())
    refreshed = []
    page.state.provider.refresh = lambda: refreshed.append(1)

    page.rebuild()
    assert refreshed == [1], "Rebuild did not re-read the store"


def test_the_s3_provider_forgets_listings_but_not_field_arrays():
    """Listings go stale; a cached field array keyed on content does not."""
    import inspect

    from fronts.viz.apps.common import s3source

    src = inspect.getsource(s3source.S3Provider.refresh)
    for name in ("_cached_dates", "_cached_index", "_cached_reader",
                 "_product_path"):
        assert f"{name}.cache_clear()" in src, name
    # The disk cache of the arrays themselves must survive.
    assert "cache.array" not in src
    assert "trim" not in src
