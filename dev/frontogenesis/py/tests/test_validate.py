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


def _v3_report(r):
    f = r['fits']
    ch = r['changes_tried']
    print(f"\nV3 {r['velocities']}: gate ({r['gate_form']}) OLS {r['slope']:.4f} CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] "
          f"n {r['n_front']} | chain {f['chain']['ols']:.4f} | first attempt "
          f"{ch.get('chain, bilinear departure velocity (first attempt)', float('nan')):.4f} | "
          f"orth {f[r['gate_form']]['orthogonal']:.4f} inv {f[r['gate_form']]['ols_inverse']:.4f} "
          f"gm {f[r['gate_form']]['geometric_mean']:.4f} ratio {f[r['gate_form']]['ratio']:.4f}")


def test_V3_discrete_null_strain():
    """The hard gate, strain variant: slope = 1 +/- 0.05 on front pixels
    with the consistent F (the default form); the chain-rule form is
    recorded below 1 and degrades with front sharpness (the finding)."""
    r = va.test_discrete_null(velocities='strain', png=False, n_boot=300)
    _v3_report(r)
    assert r['gate_form'] == 'discrete' and r['gate']['passed']
    assert abs(r['slope'] - 1.0) <= 0.05
    assert r['ci'][0] > 0.95 and r['ci'][1] < 1.05
    assert r['n_front'] > 10_000
    # every width in the pool within the gate on its own, and the out-of-pool
    # 1 and 1.5 dx fronts too (the consistent form is right to O((dx/ell)^4))
    for w, v in r['per_width']['discrete'].items():
        assert abs(v['ols'] - 1.0) < 0.02, (w, v)
    for w, v in r['diag_widths']['per_width']['discrete'].items():
        assert abs(v['ols'] - 1.0) < 0.03, (w, v)
    # the chain-rule form: below 1, monotone in the width, ~1 - (2/3)(dx/ell)^2 softened by the flanks
    c = [r['per_width']['chain'][str(w)]['ols'] for w in r['widths_cells']]
    assert all(np.diff(c) > 0) and c[0] < 0.95 and c[-1] > 0.985
    assert r['diag_widths']['per_width']['chain']['1.0']['ols'] < 0.85
    # the 2nd-order neighbour gradient over-corrects
    assert r['per_width']['discrete_o2']['2.0']['ols'] > 1.02
    # the other estimators agree with the gate to a few %
    g = r['fits']['discrete']
    assert abs(g['orthogonal'] - g['ols']) < 0.03 and abs(g['ols_origin'] - g['ols']) < 0.01
    assert g['corr'] > 0.97


@pytest.mark.needs_grid
def test_V3_discrete_null_llc(grid_ds):
    """The hard gate, real-velocity variant (M0's two hours, hour 0 b,
    mask_analysis): slope = 1 +/- 0.05 with the consistent F and the cubic
    departure velocity; the first attempt (chain F, bilinear velocity) is
    recorded near 0.76; the null is blind to the Jacobian attenuation."""
    r = va.test_discrete_null(velocities='llc', png=False, grid_ds=grid_ds, n_boot=300)
    _v3_report(r)
    print('strain seen (departure / Jacobian / flux form):', r['strain_seen'])
    assert r['gate_form'] == 'discrete' and r['gate']['passed']
    assert abs(r['slope'] - 1.0) <= 0.05
    assert r['ci'][0] > 0.95 and r['ci'][1] < 1.05
    assert r['n_front'] == 26_293 and r['n_valid'] == 262_925
    assert r['fits']['discrete']['corr'] > 0.97
    ch = r['changes_tried']
    assert ch['chain, bilinear departure velocity (first attempt)'] < 0.85       # the first attempt
    assert ch['chain, cubic departure velocity'] < 0.85
    assert ch['discrete, bilinear departure velocity'] < ch['discrete, cubic departure velocity']
    assert abs(ch['discrete, velocity low-passed L = 8 (b raw)'] - 1.0) < 0.02   # a smooth velocity: exact
    # the departure map's strain is the Jacobian's (both are D_h u_c), and both
    # see the flux-form divergence attenuated ~0.85: the null cannot detect it
    s = r['strain_seen']
    assert s['departure_vs_jacobian'] > 0.98 and 0.8 < s['jacobian_vs_fluxform'] < 0.9
    assert 0.3 < r['displacement_cells']['median'] < 0.5


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


# ---------------------------------------------------------------------------
# V3b: the finite-volume null (task 6b; M1-Q2 -- a recorded bias, NOT a gate)
# ---------------------------------------------------------------------------
def _v3b_report(r):
    for s, t in r['table'].items():
        print(f"\nV3b {r['velocities']} {s:8s}", {f: (round(v['slope'], 4), [round(c, 4) for c in v['ci']])
                                                  for f, v in t.items()})
    print('V3b bias:', r['bias']['scheme'], r['bias']['form'], round(r['bias']['slope'], 4),
          [round(c, 4) for c in r['bias']['ci']])


def test_V3b_fv_step_basics():
    """The flux-form step itself: the OS7 weights' limits, a uniform tracer
    under a divergent flow (advective form) and the conservative form's
    spurious -b delta, exactness at c = 1."""
    import fvadvect as fv
    import synthetic as sy
    A = fv._OS7_A
    assert np.allclose(A[:, 0] * 420, [-3, 25, -101, 319, 214, -38, 4])      # c -> 0: 7th-order upwind
    assert np.allclose(A.sum(axis=1), [0, 0, 0, 1, 0, 0, 0], atol=1e-12)    # c = 1: the exact shift
    case = sy.null_strain_case(ell_cells=4, n=48)
    one = case['b_t'] * 0 + 3.0
    for sch in fv.SCHEMES:
        o = fv.fv_advect(one, case['U'], case['V'], case['g'], scheme=sch, dt_sub=600.0)
        assert o.dims == one.dims and np.isfinite(o.values).all()
        assert np.abs(o.values - 3.0).max() < 1e-12, sch
    o = fv.fv_advect(one, case['U'], case['V'], case['g'], scheme='os7', dt_sub=600.0, form='conservative')
    assert np.abs(o.values - 3.0).max() > 0.01           # -b delta over an hour: percent level
    # a whole-cell shift in one sub-step is exact for the one-step schemes
    g, grid, pos = sy.synthetic_cgrid(nj=8, ni=64, dx=1800.0, dy=1800.0)
    x, y = pos('c')
    b = sy.da(np.exp(-((x - 32 * 1800.0) / (3 * 1800.0)) ** 2), sy.C_DIMS)
    U = sy.da(np.full((1, 8, 64), 0.5), sy.U_DIMS)
    V = sy.da(np.zeros((1, 8, 64)), sy.V_DIMS)
    ex = np.exp(-((x - 33 * 1800.0) / (3 * 1800.0)) ** 2)
    for sch in ('dst3', 'os7', 'os7mp'):
        o = fv.fv_advect(b, U, V, g, scheme=sch, dt_sub=3600.0)
        assert np.abs(o.values - ex)[0, :, 8:-8].max() < 1e-12, sch


def test_V3b_fv_null_strain():
    """V3b, strain variant: runs, finite, reproduces V3 when the truth is the
    semi-Lagrangian step, and the physical expectations that hold; no
    assertion on the headline number (a recorded bias)."""
    r = va.test_fv_null('strain', schemes=('semilag', 'centred', 'os7', 'os7mp', 'dst3'), png=False, n_boot=300)
    _v3b_report(r)
    for s, t in r['table'].items():
        for f, v in t.items():
            assert np.isfinite(v['slope']) and np.isfinite(v['ci']).all() and v['n'] > 10_000, (s, f)
    # the semi-Lagrangian truth IS V3 (same code path): the logged 1.0044 / 0.9512
    v3 = va.test_discrete_null('strain', png=False, forms=('chain', 'discrete'), n_boot=50, diag_widths=None,
                               changes=False)
    assert abs(r['table']['semilag']['discrete']['slope'] - v3['slope']) < 1e-12
    assert abs(r['table']['semilag']['chain']['slope'] - v3['fits']['chain']['ols']) < 1e-12
    assert abs(r['table']['semilag']['discrete']['slope'] - 1.0044) < 0.001
    # centred FV on the 8 dx front: within 2% of 1 (the stencil effect vanishes for a resolved front)
    assert abs(r['per_scheme']['centred']['per_width']['discrete']['8.0']['ols'] - 1.0) < 0.02
    # the limiter (all but) never engages on a smooth monotone front: os7 == os7mp to 1e-4
    # (it touches a few flank cells of the sheared 60-degree cases: 4e-6 in the slope)
    assert abs(r['table']['os7']['discrete']['slope'] - r['table']['os7mp']['discrete']['slope']) < 1e-4
    # the seventh-order truth: per width monotone towards 1 and within 3% in the pool; the
    # third-order truth below it at 2 dx (its larger implicit diffusion)
    w7 = [r['per_scheme']['os7mp']['per_width']['discrete'][str(w)]['ols'] for w in r['widths_cells']]
    assert all(np.diff(w7) > 0) and all(abs(v - 1) < 0.03 for v in w7)
    assert r['per_scheme']['dst3']['per_width']['discrete']['2.0']['ols'] < w7[0]
    assert r['per_scheme']['dst3']['diffusion']['slope_shift']['discrete'] < r['per_scheme']['os7']['diffusion']['slope_shift']['discrete']
    # the chain form sits below the discrete form for every truth (the chain-rule violation)
    for s in r['schemes']:
        assert r['table'][s]['chain']['slope'] < r['table'][s]['discrete']['slope']
    assert r['bias']['scheme'] == 'os7mp' and r['bias']['form'] == 'discrete' and np.isfinite(r['bias']['slope'])


@pytest.mark.needs_grid
def test_V3b_fv_null_llc(grid_ds):
    """V3b, real-velocity variant (hour-0 b, midpoint velocity, mask_analysis):
    runs, finite, reproduces V3 with the semi-Lagrangian truth; the headline
    is recorded, not gated."""
    r = va.test_fv_null('llc', schemes=('semilag', 'os7', 'os7mp'), png=False, grid_ds=grid_ds, n_boot=300)
    _v3b_report(r)
    for s, t in r['table'].items():
        for f, v in t.items():
            assert np.isfinite(v['slope']) and np.isfinite(v['ci']).all() and v['n'] == 26_293, (s, f)
    assert r['n_analysis'] == 262_925
    assert abs(r['table']['semilag']['discrete']['slope'] - 0.9806) < 0.001      # V3's logged llc slope
    assert abs(r['table']['semilag']['chain']['slope'] - 0.7914) < 0.001
    b = r['bias']
    assert b['scheme'] == 'os7mp' and b['form'] == 'discrete'
    assert b['ci'][0] < b['slope'] < b['ci'][1]
    assert r['per_scheme']['os7mp']['limiter']['reference'] == 'os7'
    assert np.isfinite(r['per_scheme']['os7mp']['limiter']['median_dt_over_G'])
