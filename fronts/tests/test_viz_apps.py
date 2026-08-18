"""Headless tests for the visualisation pages.

No browser, no server, no display.  The pages keep their whole selection
in ``param.Parameterized`` classes, so the state machine and every
computation behind it can be driven directly.

The Panel layout itself is not tested, which is the correct thing to
leave untested.
"""

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

def test_six_regions_with_unique_keys():
    assert len(regions.REGIONS) == 6
    assert len({r.key for r in regions.REGIONS}) == 6
    assert len(set(regions.names())) == 6


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
    assert st.field in st.fields


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


def test_persistent_labels_exclude_flickering_ones(provider):
    from fronts.viz.apps.evolution import timeseries as TS

    chunk = provider.chunks()[0]
    labels = TS.common_labels(provider, chunk, min_steps=4)
    assert 1 in labels

    strict = TS.common_labels(provider, chunk, min_steps=24)
    assert set(strict) <= set(labels)


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
    from fronts.viz.apps.evolution import timeseries as TS

    chunk = provider.chunks()[0]
    series = TS.build(provider, chunk, 1, "Ri")

    assert series.n == config.EVOLUTION_N_STEPS
    assert len(series.length_km) == series.n
    assert len(series.orientation) == series.n
    for name in config.DEFAULT_EVOLUTION_STAT_LINES:
        assert name in series.stats
        assert len(series.stats[name]) == series.n

    assert series.present().sum() > series.n // 2
    assert np.nanmax(series.length_km) > np.nanmin(series.length_km)


def test_evolution_series_orientation_is_in_range(provider):
    from fronts.viz.apps.evolution import timeseries as TS

    series = TS.build(provider, provider.chunks()[0], 1, "Ri")
    finite = series.orientation[np.isfinite(series.orientation)]
    assert finite.size and finite.min() >= 0.0 and finite.max() <= 90.0


def test_shared_settings_fix_the_transect_and_colour_scale(provider):
    """Every frame must share these, or the movie appears to pulse."""
    from fronts.viz.apps.evolution import pipeline as EP

    shared = EP.shared_settings(provider, provider.chunks()[0], "Ri", 1)
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

    st.built = True
    st.front_label = 3
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

    markers = TilesPage._label_markers(None, labels, [41953, 44615])
    rows = {row[2]: (row[0], row[1]) for row in markers.data.itertuples(index=False)} \
        if hasattr(markers.data, "itertuples") else None

    text = list(markers.dimension_values("text"))
    assert set(text) == {"41953", "44615"}
    i_vals = markers.dimension_values("i")
    j_vals = markers.dimension_values("j")
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

    def limits():
        return page._tilemap.object.opts.get().kwargs

    whole = limits()
    available = [int(v) for v in page.w_avail.options]
    assert available, "fixture tile needs at least one front"

    page.state.select_front(available[0])
    zoomed = limits()

    labels = page._labels_tile
    js, iss = np.nonzero(labels == available[0])

    xlim, ylim = zoomed["xlim"], zoomed["ylim"]
    assert xlim[0] <= iss.min() and xlim[1] >= iss.max()
    assert ylim[0] <= js.min() and ylim[1] >= js.max()

    nj, ni = labels.shape
    assert (xlim[1] - xlim[0]) < ni or (ylim[1] - ylim[0]) < nj, \
        "selecting a front must tighten the view, not keep the whole tile"
    assert (whole["xlim"], whole["ylim"]) != (xlim, ylim)


def test_deselecting_returns_the_whole_tile():
    page = _tiles_page()
    available = [int(v) for v in page.w_avail.options]
    page.state.select_front(available[0])
    page.state.front_label = 0

    opts = page._tilemap.object.opts.get().kwargs
    nj, ni = page._labels_tile.shape
    assert opts["xlim"] == (0, ni - 1)
    assert opts["ylim"] == (0, nj - 1)


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
