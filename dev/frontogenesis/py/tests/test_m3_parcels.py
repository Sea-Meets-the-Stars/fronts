""" Tests for ``m3_parcels.py`` (M3-digging task 6): the geographic rotation on
face 10, the along/cross-front decomposition, and the period fit."""
import numpy as np
import xarray as xr

import m3_parcels as mp


def test_geographic_rotation_on_face_10_is_u_east_V_v_north_minus_U():
    g = xr.Dataset({'CS': (('j', 'i'), np.zeros((4, 4))), 'SN': (('j', 'i'), -np.ones((4, 4)))})
    U = xr.DataArray(np.full((4, 4), 2.0), dims=('j', 'i'))     # model-x (along i, southward)
    V = xr.DataArray(np.full((4, 4), 3.0), dims=('j', 'i'))     # model-y (along j, eastward)
    ue, vn = mp.geographic(U, V, g)
    assert np.allclose(ue, 3.0) and np.allclose(vn, -2.0)
    g1 = xr.Dataset({'CS': (('j', 'i'), np.ones((4, 4))), 'SN': (('j', 'i'), np.zeros((4, 4)))})
    ue, vn = mp.geographic(U, V, g1)
    assert np.allclose(ue, 2.0) and np.allclose(vn, 3.0)        # the unrotated identity


def test_along_cross_decomposition():
    bx, by = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])
    u, v = np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])
    vn, vt = mp.along_cross(u, v, bx, by)
    assert np.allclose(vn, [1.0, 0.0, np.sqrt(2)]) and np.allclose(np.abs(vt), [0.0, 1.0, 0.0])
    assert np.allclose(np.hypot(vn, vt), np.hypot(u, v))         # the two components recompose the speed
    vn0, _ = mp.along_cross(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(vn0[0])                                       # no gradient, no normal


def test_fit_periods_recovers_amplitudes():
    t = np.arange(50.0)
    y = 0.2 + 0.05 * np.cos(2 * np.pi * t / 12.42) + 0.08 * np.sin(2 * np.pi * t / 20.0 + 0.3)
    f = mp.fit_periods(t, y, (12.42, 20.0))
    assert abs(f['mean'] - 0.2) < 1e-6
    assert abs(f['amplitudes']['12.42h'] - 0.05) < 1e-6 and abs(f['amplitudes']['20.00h'] - 0.08) < 1e-6
    assert f['residual_rms'] < 1e-9
