"""Track analysis, Surface-flux task 1: the multiplier and its block bootstrap."""
import numpy as np
import pandas as pd

import surface_flux_tracks as sft


def _table(a, n=69, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=n)
    F = rng.normal(size=n)
    V = 0.1 * rng.normal(size=n)
    r = F + V + a * S + noise * rng.normal(size=n)
    return pd.DataFrame(dict(step=np.arange(1, n + 1), npix=100, two_F=F, V_mean=V, S_1m_mean=S,
                             S_kpp_mean=0.05 * S, r_store=r, r_matched=r))


def test_multiplier_recovers_the_weight():
    for a in (1.0, 0.0, -0.5):
        p = sft.projection([_table(a)], 'S_1m', 'r_store')
        assert abs(p['multiplier'] - a) < 1e-12
    p = sft.projection([_table(0.05)], 'S_kpp', 'r_store')       # the kpp form is 0.05 S: weight +1
    assert abs(p['multiplier'] - 1.0) < 1e-9


def test_bootstrap_interval_covers_and_pools():
    dfs = [_table(1.0, noise=0.5, seed=k) for k in range(4)]
    p = sft.projection(dfs, 'S_1m', 'r_matched')
    assert p['n'] == 4 * 69 and p['ci95'][0] < 1.0 < p['ci95'][1] and p['ci95'][0] > 0.5
    assert p['rms_I_minus_aS'] <= p['rms_I_minus_S'] + 1e-12
