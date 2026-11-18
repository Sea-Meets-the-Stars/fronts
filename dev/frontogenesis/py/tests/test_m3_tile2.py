""" Tests for ``m3_tile2.py`` (M3-digging task 5), offline: the tile resolver
round-trips for 330 and 329 and the provenance pixel is the slice centre
(tile 330's constants unchanged); the masks on a synthetic grid without
land (coast distance +inf, offshore all-true, analysis = edge & ocean);
the dry run lists 72 timestamps and the store names; the QA band."""
import numpy as np

import m3_tile2 as t2
import masking as mk
import osn_tiles as ot
import synthetic as sy


def test_resolver_round_trips_330_and_329():
    r330 = t2.resolve((13320, 9720))
    assert r330['tile_idx'] == 330 and r330['face'] == 10 and r330['j_face'] == [0, 720] and r330['i_face'] == [2880, 3600]
    r329 = t2.resolve(t2.RECT)
    assert r329['tile_idx'] == 329 and r329['face'] == 7 and r329['j_face'] == [3600, 4320] and r329['i_face'] == [2880, 3600]
    assert r329['rect_i'] == [12240, 12960] and r329['rect_j'] == [9360, 10080]
    assert ot.rect_pixel(ot.tile_spec()) == (ot.TILE_RECT_I, ot.TILE_RECT_J) == (13320, 9720)
    assert ot.rect_pixel(r329['tile']) == (12600, 9720)


def test_masks_on_a_grid_without_land():
    g, grid, pos = sy.synthetic_cgrid(nj=40, ni=50, dx=1800.0, dy=1900.0, land=None)
    g['hFacW'] = (('face', 'j', 'i_g'), np.ones((1, 40, 50)))
    g['hFacS'] = (('face', 'j_g', 'i'), np.ones((1, 40, 50)))
    m = mk.build_masks(g, halo_cells=7, min_km=100.0, edge_cells=7)
    assert m['mask_ocean'].values.all() and m['mask_offshore'].values.all() and m['mask_halo'].values.all()
    d = m['coast_distance_km'].values
    assert np.isposinf(d).all()
    assert np.array_equal(m['mask_analysis'].values, mk.edge_mask(g, 7))
    assert int(m['mask_analysis'].values.sum()) == (40 - 14) * (50 - 14)


def test_dry_run_lists_the_window_and_the_stores(capsys):
    assert t2.main(['--step', 'dry-run']) == 0
    out = capsys.readouterr().out
    assert '72 timestamps: 2012-07-02 00:00:00 .. 2012-07-04 23:00:00' in out
    assert 'tile 329: face 7' in out and 'tile329_raw_20120702T00_72h.zarr' in out
    assert 'tile329_grid.zarr' in out and 'tile329_masks.nc' in out


def test_qa_band_and_declaration():
    assert t2.QA_BAND == (0.95, 1.05)
    assert t2.DECLARED['tile'] == 'tile329' and t2.DECLARED['rect_pixel'] == [12600, 9720]
    assert 'two-term catch-all' in t2.DECLARED['criterion1']
    assert t2.TAG == 'tile329' and str(t2.paths()['raw']).endswith('tile329_raw_20120702T00_72h.zarr')
