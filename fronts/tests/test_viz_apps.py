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


def test_s3_provider_reports_what_it_needs():
    """The unwired provider must say what is missing, not fail obscurely."""
    p = sources.S3Provider()
    with pytest.raises(sources.NotWiredUp) as exc:
        p.field_names("2012-05-16T06_00_00")
    assert "channel list" in str(exc.value)
    assert "WIRING" in str(exc.value)


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
