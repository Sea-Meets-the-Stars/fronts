""" The M1 gates run and report (coding doc §5, prompt 2 tasks 5-7).

``validate.py``'s ``test_*`` functions are the gates of coding §4.9, not
pytest tests (``__test__ = False`` there; the module is imported, never
star-imported, so pytest cannot collect them).  Each test here runs one
V-function without its PNG and asserts the acceptance threshold:

* V1 (``test_cartesian_deformation``): ``G ~ exp(2 a t)`` to < 1% at the
  reference width, both orientations; the semi-Lagrangian step alone < 1%
  at every width; ``measured_DGDt`` is the same arithmetic.
* V2 (``test_native_metric``, ``needs_grid``): analytic gradients on the
  tile grid to < 1% on ``mask_analysis``; the metric alone to 0.1%.
* V3: the slot for task 6 (``test_discrete_null``), skipped until written.
* V4 (``test_interpolation_bias``): no threshold in the criteria -- the bias
  is *recorded* -- so the assertions are the order hierarchy, exactness at
  integer shifts, and a loose regression bound on the headline.
* V5 / V6: the two supporting figures run and reproduce their logged
  numbers (the split of ``validate.py`` in task 5 must not change them).
"""

import numpy as np
import pytest

import validate as va


def test_V1_cartesian_deformation():
    r = va.test_cartesian_deformation(png=False)
    print(f"\nV1: series max err at ell = {r['ref_ell_cells']:g} dx over {r['n_steps']} h: "
          f"{100 * r['series_max_err_ref']:.3f}% (gate < 1%); semilag-only max {100 * r['semilag_err_max_all']:.3f}%; "
          f"rate rms err vs width {dict(zip(r['widths_cells'], [round(100 * e, 2) for e in r['rate']['CS=1']['rms_err']]))} %; "
          f"convergence order {r['convergence_order']:.2f}")
    assert r['gate']['series_pass'] and r['series_max_err_ref'] < 0.01
    for key in ('CS=1', 'face10'):
        assert r['series'][key]['max_err'] < 0.01
        assert r['series'][key]['measured_DGDt_max_rel_diff'] < 1e-12
        assert r['series'][key]['n'] > 500
    # the semi-Lagrangian step alone (departure + interpolation) at every width
    assert max(r['semilag_err_max']['order3']) < 0.01
    assert max(r['semilag_err_max']['order5']) < 0.01
    # the growth rate's error is the stencil's truncation: second order in dx/ell
    assert 1.5 < r['convergence_order'] < 2.5
    # face 10 is an exact axis swap: identical numbers
    assert r['orientation_max_diff'] < 1e-10


@pytest.mark.needs_grid
def test_V2_native_metric(grid_ds):
    r = va.test_native_metric(grid_ds, png=False)
    print(f"\nV2: b_x max {100 * r['err_x_max']:.4f}% rms {100 * r['err_x_rms']:.4f}%; b_y max "
          f"{100 * r['err_y_max']:.4f}% rms {100 * r['err_y_rms']:.4f}% (gate < 1%); metric alone "
          f"{100 * r['lin_err_x_max']:.4f}% / {100 * r['lin_err_y_max']:.4f}%; R {r['R_from_dxC'] / 1e3:.1f} / "
          f"{r['R_from_dyC'] / 1e3:.1f} km; worst b_x at {r['worst_x']['lat']:.2f}N {-r['worst_x']['lon']:.2f}W")
    assert r['gate']['passed'] and r['err_x_max'] < 0.01 and r['err_y_max'] < 0.01
    assert r['finite_on_analysis'] and r['n_analysis'] == 262925
    # the metric alone (linear functions) is right to 0.1%; a swapped component would be O(1)
    assert r['lin_err_x_max'] < 1e-3 and r['lin_err_y_max'] < 1e-3
    assert r['err_if_components_swapped'] > 0.5
    # the sinusoid's error is the stencil's truncation: second order in the wavelength
    assert 1.7 < r['sweep_order'] < 2.3
    assert r['err_x_max'] < 2.0 * r['truncation_x'] and r['err_y_max'] < 2.0 * r['truncation_y']
    # the metric's sphere is MITgcm's rSphere
    assert abs(r['R_from_dxC'] - 6370e3) < 5e3 and abs(r['R_from_dyC'] - 6370e3) < 5e3


@pytest.mark.skip(reason='V3 test_discrete_null is prompt 2 task 6 (not written yet)')
def test_V3_discrete_null():
    pass


def test_V4_interpolation_bias():
    r = va.test_interpolation_bias(png=False)
    h = r['headline']
    print(f"\nV4 headline (order 3, sigma_G 1.5, real-hour displacements): rms {100 * h['rms']:.3f}% of G/h "
          f"(all cross-front {100 * h['rms_all_cross_front']:.3f}%), at the maximum {100 * h['at_max_mean']:+.3f}%; "
          f"= {100 * h['signal_fraction'][0]:.1f}-{100 * h['signal_fraction'][1]:.1f}% of the 7-20% signal; "
          f"order 1 {100 * r['headline_order1']['rms']:.2f}%, order 5 {100 * r['headline_order5']['rms']:.3f}%; "
          f"half cell: {100 * r['vs_fraction']['order3']['rms'][10]:.3f}% rms, "
          f"{100 * r['vs_fraction']['order3']['at_max'][10]:+.3f}% at the max")
    # integer shifts are exact (whole-cell translation, test_semilag)
    assert r['integer_shift_rms'] < 1e-12
    # the order hierarchy, at the half-cell shift and over the real-hour distribution
    for o1, o3, o5 in ((r['headline_order1'], r['headline_order3'], r['headline_order5']),):
        assert o1['rms'] > 5 * o3['rms'] > 0 and o3['rms'] > 3 * o5['rms'] > 0
    assert r['vs_fraction']['order1']['rms'][10] > 0.02          # the order-1 bias: several % of G per hour
    assert r['vs_fraction']['order3']['rms'][10] < 0.006
    # the fabricated tendency is positive at the maximum for every order
    assert r['vs_fraction']['order1']['at_max'][10] > 0 and r['vs_fraction']['order3']['at_max'][10] > 0
    # the bias falls steeply with the front width (order 3: ~sigma_G^-3.5 to -4 for sigma_G >= 1.5)
    assert r['width_slope']['order3'] < -3.0
    # a regression bound on the recorded error bar, not a tuned threshold: order 3
    # must stay below 1% of G per hour, i.e. under 15% of the smallest (7%) signal
    assert h['rms'] < 0.01 and h['rms_all_cross_front'] < 0.01


def test_V5_demo_interp_half_cell():
    r = va.demo_interp_half_cell(png=False)
    # the M1 task 3 numbers: bilinear G -4.94%, bilinear b -4.99%, cubic -0.54%, quintic -0.10%
    assert abs(r['bias_bilinear_G'] + 0.0494) < 0.002
    assert abs(r['bias_b_order1'] + 0.0499) < 0.002
    assert abs(r['bias_b_order3'] + 0.0054) < 0.0005
    assert abs(r['bias_b_order5'] + 0.0010) < 0.0003
    assert r['sweep_sigma_G_cells'][2] == 1.5 and abs(r['sweep_bias']['b_order3'][2] - r['bias_b_order3']) < 1e-12


@pytest.mark.needs_grid
def test_V6_qa_land_halo(grid_ds):
    r = va.qa_land_halo(grid_ds, png=False)
    # the M1 task 1 counts
    assert r['n_ocean'] == 356877 and r['n_halo'] == 341960 and r['n_analysis'] == 262925
    assert r['halo_cells'] == 7 and r['edge_cells'] == 7 and r['edge_margin_covers_rim']
    assert r['gulf']['gulf_n_analysis'] == 0
