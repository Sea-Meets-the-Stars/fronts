""" Tests for ``m3_case_sflux.py`` (M3-digging task 9): the integral from the seed
outwards is the case's own, and the parts sum to the term on a synthetic split."""
import numpy as np

import m3_case_sflux as cs


def test_integrate_from_the_seed_both_ways():
    r = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-18          # 4 intervals, 5 times, seed at index 2
    out = cs.integrate(r, G0=1e-14, n_back=2, dt=3600.0)
    assert out[2] == 1e-14
    assert np.isclose(out[3], 1e-14 + 3e-18 * 3600) and np.isclose(out[4], out[3] + 4e-18 * 3600)
    assert np.isclose(out[1], 1e-14 - 2e-18 * 3600) and np.isclose(out[0], out[1] - 1e-18 * 3600)
    assert cs.PARTS == ('nonsolar', 'shortwave', 'freshwater')
