"""Unit tests for fronts/viz/curtains.py (geometry + sampling) and
fronts/viz/field_styles.py (transforms + color limits), plus smoke tests
for the figure assemblers."""

import numpy as np
import pytest

from fronts.viz import curtains
from fronts.viz.field_styles import FieldStyle, apply_transform, default_clim


# ---------------------------------------------------------------------------
# Skeleton fixtures (thinned, 1-px wide)
# ---------------------------------------------------------------------------

def _straight_horizontal(H=20, W=40, j=10, i0=5, i1=34):
    m = np.zeros((H, W), dtype=bool)
    m[j, i0:i1 + 1] = True
    return m


def _with_side_branch():
    # A long horizontal spine with a short vertical stub branching off.
    m = _straight_horizontal()
    # stub from the spine at i=20 going up 4 pixels
    for d in range(1, 5):
        m[10 - d, 20] = True
    return m


def _curved():
    # A quarter-circle-ish arc traced pixel by pixel.
    H = W = 60
    m = np.zeros((H, W), dtype=bool)
    t = np.linspace(0, np.pi / 2, 80)
    j = (10 + 40 * np.sin(t)).round().astype(int)
    i = (10 + 40 * np.cos(t)).round().astype(int)
    m[j, i] = True
    return m


# ---------------------------------------------------------------------------
# extract_main_axis
# ---------------------------------------------------------------------------

def test_main_axis_straight():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    # Should recover all 30 spine pixels (5..34 inclusive).
    assert axis.shape[0] == 30
    # All on row 10.
    assert np.all(axis[:, 0] == 10)
    # Monotonic in i (ordered along the line).
    di = np.diff(axis[:, 1])
    assert np.all(np.abs(di) == 1)


def test_main_axis_ignores_side_branch():
    m = _with_side_branch()
    axis = curtains.extract_main_axis(m)
    # The main axis is the 30-px spine; the 4-px stub must be excluded.
    assert axis.shape[0] == 30
    assert np.all(axis[:, 0] == 10)  # no stub pixels (which are at j<10)


def test_main_axis_curved_endpoints():
    m = _curved()
    axis = curtains.extract_main_axis(m)
    # Endpoints of the arc should be the two extremes.
    pts = set(map(tuple, np.argwhere(m)))
    assert tuple(axis[0]) in pts and tuple(axis[-1]) in pts
    # Path length should cover most of the arc pixels.
    assert axis.shape[0] >= 0.8 * len(pts)


def test_main_axis_single_pixel():
    m = np.zeros((10, 10), dtype=bool)
    m[5, 5] = True
    axis = curtains.extract_main_axis(m)
    assert axis.shape[0] == 1
    assert tuple(axis[0]) == (5, 5)


def test_main_axis_empty_raises():
    with pytest.raises(ValueError):
        curtains.extract_main_axis(np.zeros((5, 5), dtype=bool))


# ---------------------------------------------------------------------------
# path_metrics: distances + normals, smoothing only affects direction
# ---------------------------------------------------------------------------

def test_path_metrics_straight_normals():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    # Tangent of a horizontal line is +/- i-direction; normal is +/- j.
    assert np.allclose(np.abs(met["tangents"][:, 0]), 0, atol=1e-9)
    assert np.allclose(np.abs(met["normals"][:, 1]), 0, atol=1e-9)
    # dist_px increments by 1 per pixel.
    assert np.allclose(np.diff(met["dist_px"]), 1.0)


def test_path_metrics_smoothing_keeps_columns():
    m = _curved()
    axis = curtains.extract_main_axis(m)
    raw = curtains.path_metrics(axis, smooth=False)
    sm = curtains.path_metrics(axis, smooth=True, smooth_window=5)
    # Distances (column positions) are identical -- smoothing never moves them.
    assert np.allclose(raw["dist_px"], sm["dist_px"])
    # Normals differ (direction field was smoothed).
    assert not np.allclose(raw["normals"], sm["normals"])
    # Smoothed normals are less jittery: smaller mean angular step.
    def _ang_step(n):
        ang = np.arctan2(n[:, 0], n[:, 1])
        return np.mean(np.abs(np.diff(np.unwrap(ang))))
    assert _ang_step(sm["normals"]) <= _ang_step(raw["normals"]) + 1e-9


def test_path_metrics_km_monotonic():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    # Fake lon/lat: 0.01 deg per pixel in i.
    H, W = m.shape
    XC = np.tile(np.arange(W) * 0.01, (H, 1))
    YC = np.full((H, W), 30.0)
    met = curtains.path_metrics(axis, XC, YC)
    assert met["dist_km"] is not None
    assert np.all(np.diff(met["dist_km"]) > 0)
    # ~0.01 deg lon at 30N ~ 0.96 km; 29 steps -> ~28 km total, sane range.
    assert 20 < met["dist_km"][-1] < 35


# ---------------------------------------------------------------------------
# offsets + overlap detection
# ---------------------------------------------------------------------------

def test_offset_paths_shapes_and_sides():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    a, b = curtains.offset_paths(axis, met["normals"], 3)
    assert len(a) == 3 and len(b) == 3
    # Side A and B are mirror images about the axis.
    for k in range(3):
        assert np.allclose(a[k] + b[k], 2 * axis, atol=1e-9)
    # Offset 1 of a horizontal line is shifted by 1 px in j.
    assert np.allclose(np.abs(a[0][:, 0] - axis[:, 0]), 1.0)


def test_offset_quality_straight_no_overlap():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    a, _ = curtains.offset_paths(axis, met["normals"], 3)
    # A straight line's offsets never self-overlap.
    for p in a:
        assert not curtains.offset_quality_flags(p).any()


def test_offset_quality_flags_tight_curve():
    # Tight curve -> inner offsets collide; expect some flagged columns.
    m = _curved()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis, smooth=False)
    a, b = curtains.offset_paths(axis, met["normals"], 8)
    # The concave side should show overlaps at the largest offset.
    flagged_any = any(curtains.offset_quality_flags(p).any() for p in (a + b))
    assert flagged_any


# ---------------------------------------------------------------------------
# sample_curtain + perpendicular + extremum
# ---------------------------------------------------------------------------

def test_sample_curtain_constant_and_nan():
    K, H, W = 5, 20, 40
    field = np.full((K, H, W), 3.0)
    m = _straight_horizontal(H, W)
    axis = curtains.extract_main_axis(m)
    cur = curtains.sample_curtain(field, axis)
    assert cur.shape == (K, axis.shape[0])
    assert np.allclose(cur, 3.0)
    # A path leaving the window -> NaN.
    off = axis.astype(float).copy()
    off[:, 0] = -5  # above the grid
    cur2 = curtains.sample_curtain(field, off)
    assert np.all(np.isnan(cur2))


def test_perpendicular_path_geometry():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    idx = axis.shape[0] // 2
    perp = curtains.perpendicular_path(axis, met["normals"], idx, 5)
    assert perp.shape == (11, 2)
    # Centre row equals the axis point.
    assert np.allclose(perp[5], axis[idx])
    # For a horizontal axis, the perpendicular runs in j (column i constant).
    assert np.allclose(perp[:, 1], axis[idx, 1], atol=1e-9)


def test_trim_offset_loops_straight_keeps_all():
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    a, _ = curtains.offset_paths(axis, met["normals"], 2)
    for p in a:
        assert curtains.trim_offset_loops(p).all()  # no loops -> keep everything


def test_trim_offset_loops_removes_crossing():
    # Hand-built polyline with an explicit loop (figure-eight pinch).
    p = np.array([
        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 3], [1, 2],            # fold back (crosses the outgoing run)
        [0, 5], [0, 6], [0, 7],
    ], dtype=float)
    keep = curtains.trim_offset_loops(p)
    kept = p[keep]
    # The trimmed line must have no self-intersections among its segments.
    n = kept.shape[0]
    for a in range(n - 1):
        for b in range(a + 2, n - 1):
            assert not curtains._segments_intersect(
                kept[a], kept[a + 1], kept[b], kept[b + 1])
    # And it stays shorter than or equal to the original.
    assert kept.shape[0] <= p.shape[0]


def test_transect_front_crossings_straight():
    # A single straight front: a perpendicular at mid-span crosses it once.
    m = _straight_horizontal()
    axis = curtains.extract_main_axis(m)
    met = curtains.path_metrics(axis)
    counts = curtains.transect_front_crossings(axis, met["normals"], m, 6)
    mid = axis.shape[0] // 2
    assert counts[mid] == 1


def test_transect_front_crossings_double():
    # Two parallel horizontal fronts 4 px apart: a tall vertical transect
    # through the middle crosses both.
    H = W = 30
    m = np.zeros((H, W), bool)
    m[12, 5:25] = True
    m[16, 5:25] = True
    # Build a single-front axis on the first line; normals point in +/- j.
    axis = np.column_stack([np.full(20, 12), np.arange(5, 25)]).astype(int)
    met = curtains.path_metrics(axis)
    counts = curtains.transect_front_crossings(axis, met["normals"], m, 8)
    mid = 10
    assert counts[mid] >= 2


def test_pick_extremum_index():
    K, L = 4, 12
    cur = np.full((K, L), 1.0)
    cur[2, 7] = -3.0  # a deep minimum at column 7
    cur[1, 3] = 9.0   # a maximum at column 3
    assert curtains.pick_extremum_index(cur, "min") == 7
    assert curtains.pick_extremum_index(cur, "max") == 3
    with pytest.raises(ValueError):
        curtains.pick_extremum_index(np.full((K, L), np.nan), "min")


# ---------------------------------------------------------------------------
# field_styles.apply_transform
# ---------------------------------------------------------------------------

def test_apply_transform_log10_nonpositive_to_nan():
    style = FieldStyle(transform="log10")
    vals = np.array([-1.0, 0.0, 1.0, 100.0, np.nan, np.inf])
    out = apply_transform(vals, style)
    assert np.isnan(out[0])          # negative -> NaN
    assert np.isnan(out[1])          # zero -> NaN
    assert np.isnan(out[4]) and np.isnan(out[5])  # non-finite -> NaN
    assert out[2] == pytest.approx(0.0)   # log10(1)
    assert out[3] == pytest.approx(2.0)   # log10(100)


def test_apply_transform_log10_clips_before_log():
    style = FieldStyle(transform="log10", clip=(1e-2, 1e2))
    out = apply_transform(np.array([1e-6, 1e6]), style)
    assert out[0] == pytest.approx(-2.0)  # clipped up to 1e-2
    assert out[1] == pytest.approx(2.0)   # clipped down to 1e2


def test_apply_transform_symlog_signed():
    style = FieldStyle(transform="symlog", linthresh=1.0)
    vals = np.array([-99.0, 0.0, 99.0])
    out = apply_transform(vals, style)
    # sign(x) * log10(1 + |x| / linthresh)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(2.0)       # log10(1 + 99)
    assert out[0] == pytest.approx(-out[2])   # odd function


def test_apply_transform_linear_clip_and_scale():
    style = FieldStyle(transform="linear", clip=(-2.0, 2.0), scale=1e-1)
    out = apply_transform(np.array([-5.0, 0.1, 5.0]), style)
    assert out[0] == pytest.approx(-20.0)  # clip to -2, then / 0.1
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(20.0)


def test_apply_transform_overrides():
    style = FieldStyle(transform="log10", clip=(1e-2, 1e4))
    vals = np.array([-10.0, 10.0])
    # Override to symlog: negative values survive instead of going NaN.
    out = apply_transform(vals, style, transform_override="symlog",
                          clip_override=(-5.0, 5.0))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(-out[1])  # clipped to +/-5, symmetric


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError):
        apply_transform(np.array([1.0]), FieldStyle(transform="sqrt"))


# ---------------------------------------------------------------------------
# field_styles.default_clim (pinned clim vs center-symmetric vs percentiles)
# ---------------------------------------------------------------------------

def test_default_clim_pinned():
    style = FieldStyle(clim=(-1.0, 2.0))
    assert default_clim(np.linspace(-9, 9, 100), style) == (-1.0, 2.0)


def test_default_clim_pinned_skipped_on_override():
    # A pinned clim is calibrated to the style's own display space, so a CLI
    # transform/clip override must fall back to percentiles.
    style = FieldStyle(clim=(-1.0, 2.0))
    vals = np.linspace(-9, 9, 1000)
    lo, hi = default_clim(vals, style, transform_override="symlog")
    assert (lo, hi) != (-1.0, 2.0)
    assert lo == pytest.approx(np.nanpercentile(vals, 2.0))
    assert hi == pytest.approx(np.nanpercentile(vals, 98.0))
    lo2, hi2 = default_clim(vals, style, clip_override=(-3.0, 3.0))
    assert (lo2, hi2) == (lo, hi)


def test_default_clim_center_symmetric():
    style = FieldStyle(center=0.0)
    vals = np.linspace(-1.0, 10.0, 1000)  # asymmetric spread
    lo, hi = default_clim(vals, style)
    assert lo == pytest.approx(-hi)  # symmetric about 0
    assert hi == pytest.approx(np.nanpercentile(vals, 98.0))  # larger excursion


def test_default_clim_percentiles():
    style = FieldStyle()
    vals = np.concatenate([np.linspace(0, 1, 1000), [np.nan] * 50])
    lo, hi = default_clim(vals, style)
    assert lo == pytest.approx(0.02, abs=1e-2)
    assert hi == pytest.approx(0.98, abs=1e-2)


# ---------------------------------------------------------------------------
# Figure assemblers (smoke tests: each writes a non-empty PNG)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scene():
    """Tiny (K, J, I) fields + a straight front axis, enough to render."""
    rng = np.random.default_rng(42)
    K, H, W = 6, 20, 40
    color = rng.normal(size=(K, H, W))
    sigma0 = np.sort(rng.normal(25.0, 0.5, size=(K, H, W)), axis=0)
    Z = -np.linspace(1.0, 60.0, K)
    m = _straight_horizontal(H, W)
    axis = curtains.extract_main_axis(m)
    metrics = curtains.path_metrics(axis)
    return color, sigma0, Z, axis, metrics


def test_figure_main_axis_smoke(synthetic_scene, tmp_path):
    color, sigma0, Z, axis, metrics = synthetic_scene
    out = curtains.figure_main_axis(
        color, sigma0, Z, axis, metrics, tmp_path / "main.png",
        clim=(-2.0, 2.0), color_title="test",
    )
    assert out.exists() and out.stat().st_size > 0


def test_figure_offsets_smoke(synthetic_scene, tmp_path):
    color, sigma0, Z, axis, metrics = synthetic_scene
    out = curtains.figure_offsets(
        color, sigma0, Z, axis, metrics, 2, tmp_path / "offsets.png",
        clim=(-2.0, 2.0),
    )
    assert out.exists() and out.stat().st_size > 0


def test_figure_perpendicular_smoke(synthetic_scene, tmp_path):
    color, sigma0, Z, axis, metrics = synthetic_scene
    idx = axis.shape[0] // 2
    perp = curtains.perpendicular_path(axis, metrics["normals"], idx, 5)
    # Without lon/lat: pixel axis only.
    out = curtains.figure_perpendicular(
        color, sigma0, Z, perp, 5, tmp_path / "perp.png",
    )
    assert out.exists() and out.stat().st_size > 0
    # With lon/lat: adds the km twin axis.
    H, W = 20, 40
    XC = np.tile(np.arange(W) * 0.01, (H, 1))
    YC = np.full((H, W), 30.0)
    out2 = curtains.figure_perpendicular(
        color, sigma0, Z, perp, 5, tmp_path / "perp_km.png",
        XC_rect=XC, YC_rect=YC,
    )
    assert out2.exists() and out2.stat().st_size > 0
