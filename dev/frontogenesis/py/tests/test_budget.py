""" Tests for ``budget.py`` (M3 task 3; coding §3.4, §4.7, §5).

Offline on ``synthetic.py``'s exact advection solutions, as
``test_coarsegrain.py``; one ``needs_grid`` smoke on the real stores.

The synthetic hour pair is built **in buoyancy space**: a field ``b_t`` is
chosen, advected one hour by ``semilag``'s own step (so our discrete
advection is satisfied by construction -- the V3 setup), and both hours are
turned back into ``Theta`` by inverting JMD95 at uniform ``Salt`` on a fine
monotone table.  ``operators.buoyancy`` then recovers the intended ``b`` to
~1e-9 relative, which matters because ``compute_budget`` reaches ``b`` only
through the EOS: advecting ``Theta`` instead would make the truth
``f(interp Theta)`` while the measurement expects ``interp f(Theta)``, and
the EOS's curvature would enter the residual as a fake numerical term.

Guards: the V3 identity (residual < 2 % of measured on front pixels for the
pure-deformation case, and the V3 strain mix's own 8 % floor recorded);
``compute_budget`` reproduces ``validate.null_step`` term by term; the loud
degradation without ``chunk_ds`` (``closed=None``, the warning in the
report's text, ``terms_missing``); a synthetic chunk dataset with
``W_k1 = 0`` and uniform fluxes giving ``vertical`` and ``surface_flux``
**exactly** zero and ``terms_missing = []``; the subfilter identically zero
at ``L = 0``, non-zero at ``L = 2`` and different without ``tau_delta``;
``two_F_chain != two_F``; the wider order-5 NaN rim; the §3.4 var list,
dims, dtype and attrs; the front pool selected on ``G_mid`` alone; ``valid``
excluding NaN; the Laplacian exact on a quadratic; ``front_width`` on tanh
fronts of ``ell`` 1-4 dx with its stated resolution floor; and
``write_derived``'s resume / no-op / clobber through ``zarr_series``.
"""

import numpy as np
import pytest
import xarray as xr

import budget as bg
import coarsegrain as cg
import inputs as inp
import operators as op
import semilag as sl
import synthetic as sy
import validate as va
import vertical as vt

C, U_D, V_D = sy.C_DIMS, sy.U_DIMS, sy.V_DIMS
N, DX, DT = 128, 1800.0, 3600.0
S0, T_REF = 33.6, 17.0
MARGIN = 10
#: the sinusoidal modes switched off -- ``synthetic.null_strain_case`` merges
#: its ``modes`` into ``NULL_MODES``, so zero amplitudes, not ``{}``
NO_MODES = dict(U1=0.0, L1_cells=40.0, U3=0.0, L3_cells=36.0, V2=0.0, L2_cells=48.0)
#: front_width is a resolution-limited proxy: measured width^2 = ell^2 + C_W dx^2
C_WIDTH = 3.5


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, dtype='float64') ** 2)))


# ---------------------------------------------------------------------------
# the synthetic hour pair, built in buoyancy space
# ---------------------------------------------------------------------------
def _theta_of_b(b, S=S0, n=40001):
    """Invert ``operators.buoyancy`` at uniform ``S`` on a monotone table."""
    T = np.linspace(4.0, 30.0, n)
    ds = xr.Dataset({'Theta': (('face', 'j', 'i'), T.reshape(1, 1, -1)),
                     'Salt': (('face', 'j', 'i'), np.full((1, 1, n), S))})
    bt = op.buoyancy(ds).values[0, 0]
    o = np.argsort(bt)
    return np.interp(b, bt[o], T[o])


def _b_ref():
    """``b`` of the reference state (``T_REF``, ``S0``), added so the
    inversion table is used near its middle."""
    one = np.ones((1, 4, 4))
    return float(op.buoyancy(xr.Dataset({'Theta': (('face', 'j', 'i'), one * T_REF),
                                         'Salt': (('face', 'j', 'i'), one * S0)})).values.mean())


def pair(ell_cells=8.0, modes=NO_MODES, order=3, dt=DT, n=N):
    """One hour pair as a ``(time: 2, j, i)`` Dataset shaped like the §3.2
    store, plus the case: ``b_t`` from ``synthetic.null_strain_case`` and
    ``b_tp1`` its semi-Lagrangian image."""
    case = sy.null_strain_case(n=n, ell_cells=ell_cells, modes=modes, dx=DX)
    g, grid, U, V, b_t = case['g'], case['grid'], case['U'], case['V'], case['b_t']
    u_c, v_c = sl.centre_velocities(U, V, g, grid)
    di, dj = sl.departure_index(u_c, v_c, g, dt=dt)
    b_tp1 = sl.interp_to_departure(b_t, di, dj, order)
    b0 = _b_ref()
    hours = []
    for b in (b_t, b_tp1):
        T = sy.da(_theta_of_b((b0 + b).values), C).squeeze('face')
        hours.append(xr.Dataset({'Theta': T, 'Salt': xr.full_like(T, S0),
                                 'U': U.squeeze('face'), 'V': V.squeeze('face'),
                                 'KPPhbl': xr.full_like(T, 30.0)}))
    ds = xr.concat(hours, 'time').assign_coords(
        time=np.array(['2012-07-02T00:00', '2012-07-02T01:00'], 'datetime64[ns]'))
    case.update(ds=ds, b_t=b_t, b_tp1=b_tp1, mask=sy.inner((n, n), MARGIN))
    return case


def chunk_for(case, w=0.0, qnet=100.0, qsw=200.0, fw=-1e-5, dT=0.0):
    """A §3.3-shaped chunk dataset for ``case``: three levels of
    ``Theta_k``/``Salt_k`` (``k = 0`` **identical** to the OSN hour, as the
    store's invariant requires), ``W_k`` uniform ``w`` on three interfaces,
    uniform fluxes with the stored downward-positive convention, and
    ``drF``.  ``dT`` warms level 0 relative to 1 and 2 (a stratified top
    cell); ``dT = 0`` makes ``b_k1 == b_k0``."""
    ds = case['ds']
    nt, nj, ni = ds['Theta'].shape
    drF = np.array([1.0, 1.14, 1.30])
    Th = np.stack([ds['Theta'].values + (0.0 if k == 0 else -dT) for k in range(3)], axis=1)
    Sa = np.full_like(Th, S0)
    W = np.full((nt, 3, nj, ni), float(w))
    dims4 = ('time', 'k', 'j', 'i')
    out = xr.Dataset(
        {'Theta_k': (dims4, Th), 'Salt_k': (dims4, Sa),
         'W_k': (('time', 'k_l', 'j', 'i'), W),
         'oceQnet': (('time', 'j', 'i'), np.full((nt, nj, ni), qnet)),
         'oceQsw': (('time', 'j', 'i'), np.full((nt, nj, ni), qsw)),
         'oceFWflx': (('time', 'j', 'i'), np.full((nt, nj, ni), fw)),
         'drF': (('k',), drF)},
        coords={'time': ds['time'], 'k': np.arange(3), 'k_l': np.arange(3),
                'Zl': ('k_l', np.array([0.0, -1.0, -2.14]))})
    for v in ('oceQnet', 'oceQsw', 'oceFWflx'):
        out[v].attrs.update(sign_convention='positive downward (stored; M2 task 5)')
    # the OSN side must carry W for the k = 0 identity inputs.hour_pair checks
    ds = ds.assign(W=(('time', 'j', 'i'), W[:, 0]))
    case = dict(case, ds=ds)
    return case, out


def budget_of(case, L=0, chunk=None, **kw):
    return bg.compute_budget(case['ds'], case['g'], case['grid'], case['mask'], L,
                             chunk_ds=chunk, **kw)


@pytest.fixture(scope='module')
def case8():
    return pair(ell_cells=8.0)


# ---------------------------------------------------------------------------
# the discrete operators budget.py adds
# ---------------------------------------------------------------------------
def test_laplacian_is_exact_on_a_quadratic():
    """``div grad`` of ``a x^2 + c y^2`` is ``2(a + c)`` to round-off: the
    flux-form stencil telescopes exactly on a uniform grid."""
    g, grid, pos = sy.synthetic_cgrid(nj=40, ni=56, dx=DX, dy=DX)
    x, y = pos('c')
    f = sy.da(3e-9 * x ** 2 + 1e-9 * y ** 2, C)
    lap = bg.laplacian(f, g, grid).values[0]
    m = sy.inner(lap.shape, 2)
    assert np.allclose(lap[m], 2 * (3e-9 + 1e-9), rtol=1e-10)
    assert np.isnan(lap[0, 0]) and np.isnan(lap[-1, -1])      # the rim, both ends


def test_laplacian_of_a_sine_matches_minus_k2():
    """``lap sin(kx) -> -k^2 sin(kx)`` to the compact stencil's O(k^2 dx^2)."""
    g, grid, pos = sy.synthetic_cgrid(nj=32, ni=96, dx=DX, dy=DX)
    x, _ = pos('c')
    k = 2 * np.pi / (24 * DX)
    f = sy.da(np.sin(k * x), C)
    lap = bg.laplacian(f, g, grid).values[0]
    m = sy.inner(lap.shape, 2)
    expect = (-k ** 2 * np.sin(k * x))[0]
    assert rms(lap[m] - expect[m]) / rms(expect[m]) < 0.02


@pytest.mark.parametrize('ell_cells', [1.0, 1.5, 2.0, 3.0, 4.0])
def test_front_width_recovers_ell_with_a_stated_resolution_floor(ell_cells):
    """M3-Q5 (a): ``2 sqrt(G/|lap G|)`` at the ridge of ``b = b0 tanh(x/ell)``.

    **Stated tolerance.**  The proxy does not return ``ell`` but
    ``sqrt(ell^2 + C_W dx^2)`` with ``C_W = 3.5``, to within 6 % over
    ``ell = 1..8 dx`` -- the 2 dx centred gradient stencil inside ``G``
    cannot represent a front narrower than about 1.9 dx, so the proxy has a
    floor there (2.17 dx measured at ``ell = 1``).  It is therefore within
    11 % of ``ell`` itself only for ``ell >= 4 dx``; narrower fronts are
    reported wider, monotonically, which is what matters for M3-Q5's
    binning.  A finding of the discretisation, not a tuning.
    """
    g, grid, pos = sy.synthetic_cgrid(nj=40, ni=120, dx=DX, dy=DX)
    x, _ = pos('c')
    b = sy.da(1e-2 * np.tanh((x - x.mean()) / (ell_cells * DX)), C)
    G = op.gradb2(b, g, grid)
    W = bg.front_width(G, g, grid).values[0]
    Gv = G.values[0]
    m = sy.inner(Gv.shape, 3)
    j, i = np.unravel_index(np.nanargmax(np.where(m, Gv, np.nan)), Gv.shape)
    w = W[j, i]
    assert w == pytest.approx(np.sqrt(ell_cells ** 2 + C_WIDTH), rel=0.06)
    if ell_cells >= 4.0:
        assert w == pytest.approx(ell_cells, rel=0.11)
    assert bg.front_width(G, g, grid).attrs['bins_dx'] == list(bg.WIDTH_BINS)


def test_front_width_is_monotone_in_ell():
    g, grid, pos = sy.synthetic_cgrid(nj=32, ni=160, dx=DX, dy=DX)
    x, _ = pos('c')
    got = []
    for ell in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        b = sy.da(1e-2 * np.tanh((x - x.mean()) / (ell * DX)), C)
        G = op.gradb2(b, g, grid)
        Gv, W = G.values[0], bg.front_width(G, g, grid).values[0]
        m = sy.inner(Gv.shape, 3)
        j, i = np.unravel_index(np.nanargmax(np.where(m, Gv, np.nan)), Gv.shape)
        got.append(W[j, i])
    assert np.all(np.diff(got) > 0)


# ---------------------------------------------------------------------------
# the V3 identity and the agreement with validate.null_step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('ell_cells', [4.0, 8.0])
def test_v3_identity_residual_under_two_percent(ell_cells):
    """With the semi-Lagrangian step as truth and ``chunk_ds=None``, the
    residual (= measured - 2F, the subfilter being identically zero at
    ``L = 0``) is under 2 % of measured on front pixels for the pure
    deformation -- the V3 identity, now through the whole ``compute_budget``
    plumbing rather than ``validate``'s."""
    c = pair(ell_cells=ell_cells)
    bd = budget_of(c, L=0)
    rep = bg.closure_report(bd)
    assert rep['front_and_valid']['residual_over_measured'] < 0.02
    assert rep['front_and_valid']['euler_on_semilag']['corr'] > 0.99


def test_v3_strain_mix_floor_is_recorded():
    """The V3 'strain mix' (shear + vorticity + divergence at 36-48 dx) is
    the realistic null, and its residual floor is ~8 % of measured, not 2 %:
    the discrete null's own scatter, which the pure-deformation case above
    does not show.  Recorded here so the 2 % number is never mistaken for
    what the operators do on a varying flow."""
    bd = budget_of(pair(ell_cells=4.0, modes=None), L=0)
    r = bg.closure_report(bd)['front_and_valid']['residual_over_measured']
    assert 0.04 < r < 0.10


def test_compute_budget_reproduces_validate_null_step(case8):
    """Term by term against ``validate.null_step`` on the same inputs:
    ``two_F`` (both forms) and ``DGDt_semilag`` agree to the EOS
    round-trip's ~1e-9, and ``G`` is ``operators.gradb2`` of the midpoint."""
    bd = budget_of(case8, L=0)
    out = va.null_step(case8['b_t'], case8['U'], case8['V'], case8['g'], case8['grid'],
                       order=3, forms=('discrete', 'chain'))
    m = case8['mask']
    for name, ref in (('two_F', out['two_F_discrete']), ('two_F_chain', out['two_F_chain']),
                      ('DGDt_semilag', out['measured']), ('G', out['G_mid'])):
        got = bd[name].values[0]
        assert rms(got[m] - ref[m]) / rms(ref[m]) < 1e-6, name
    b_mid = inp.midpoint(*[op.buoyancy(case8['ds'].isel(time=t).expand_dims('face'))
                           for t in (0, 1)])
    assert np.allclose(bd['G'].values[0][m],
                       op.gradb2(b_mid, case8['g'], case8['grid']).values[0][m], rtol=1e-12)


# ---------------------------------------------------------------------------
# loud degradation, and the chunk terms when they are there
# ---------------------------------------------------------------------------
def test_without_chunk_the_terms_are_absent_and_the_report_says_so(case8):
    """Absent, not zero: the variables are missing from the Dataset, the
    residual names them, and the verdict is ``None`` with a warning that is
    in the printed text."""
    bd = budget_of(case8, L=0)
    for v in bg.CHUNK_TERMS:
        assert v not in bd
    assert bd['residual'].attrs['terms_missing'] == list(bg.CHUNK_TERMS)
    assert 'CATCH-ALL' in bd['residual'].attrs['interpretation']
    assert bd.attrs['chunk_terms'].startswith('ABSENT')
    rep = bg.closure_report(bd)
    assert rep['closed'] is None
    assert rep['terms_missing'] == list(bg.CHUNK_TERMS)
    assert 'NOT A CLOSED BUDGET' in rep['warning']
    text = bg.format_closure(rep)
    assert 'NO VERDICT' in text and 'NOT A CLOSED BUDGET' in text
    assert 'vertical' in text and 'surface_flux' in text


def test_zero_W_and_zero_fluxes_give_exactly_zero_chunk_terms(case8):
    """``W_k1 = 0`` kills the vertical term identically (the tendency is
    ``-W b_z``) and zero fluxes kill the surface-flux term identically
    (``B_sfc`` is then exactly 0).  Both are *present* and zero, and
    ``terms_missing`` is empty -- the opposite of the previous test."""
    c, ch = chunk_for(case8, w=0.0, dT=0.02, qnet=0.0, qsw=0.0, fw=0.0)
    bd = budget_of(c, L=0, chunk=ch)
    m = c['mask']
    assert bd['residual'].attrs['terms_missing'] == []
    assert np.all(bd['vertical'].values[0][m] == 0.0)
    assert np.all(bd['vertical_factorised'].values[0][m] == 0.0)
    assert np.all(bd['surface_flux'].values[0][m] == 0.0)
    assert bg.closure_report(bd)['closed'] is not None
    assert bg.closure_report(bd)['warning'] is None


def test_uniform_fluxes_still_give_a_term_through_alpha_of_T(case8):
    """**A spatially uniform heat flux is not a zero surface-flux term.**
    ``B_sfc`` carries JMD95's ``alpha(Theta, Salt)``, and ``alpha`` rises
    about 1.1e-5 K^-2 at 17 degC, so across a temperature front a uniform
    ``oceQnet``/``oceQsw`` still has a buoyancy-tendency *gradient*: here
    100 / 200 W m^-2 over this front give **0.4 % of 2F** in rms, with
    ``W = 0`` so nothing else can contribute.  Zero fluxes (the test above)
    give exactly zero, so this is the EOS coefficient and not a stencil
    artefact.  Worth knowing before Figure 6 is read as "the flux gradient":
    part of the term is the flux field, part is the front's own temperature.
    """
    c, ch = chunk_for(case8, w=0.0, dT=0.02, qnet=100.0, qsw=200.0, fw=-1e-5)
    bd = budget_of(c, L=0, chunk=ch)
    m = c['mask']
    ratio = rms(bd['surface_flux'].values[0][m]) / rms(bd['two_F'].values[0][m])
    assert 1e-4 < ratio < 5e-2
    assert np.all(bd['vertical'].values[0][m] == 0.0)


def test_nonzero_W_gives_a_nonzero_vertical_term_and_a_sane_b_z(case8):
    """A uniform ``W`` with a stratified top cell: ``b_z < 0`` for a warm
    (light) layer over cooler water in code ``b``, and the vertical term is
    finite and non-zero."""
    c, ch = chunk_for(case8, w=-2e-5, dT=0.05)
    bd = budget_of(c, L=0, chunk=ch)
    m = c['mask']
    assert np.nanmedian(bd['b_z'].values[0][m]) < 0.0
    assert rms(bd['vertical'].values[0][m]) > 0.0
    assert np.isfinite(bd['vertical'].values[0][m]).all()
    assert bd['b_z'].attrs['order'] == 1


def test_a_flux_gradient_gives_a_nonzero_surface_flux_term(case8):
    """Uniform fluxes give zero; a flux that varies across the front does
    not.  ``oceQsw`` enters with ``f_sw``, so a shortwave-only gradient is
    still a term."""
    c, ch = chunk_for(case8, w=0.0, dT=0.02)
    x = c['xc']
    ch = ch.copy()
    ramp = np.broadcast_to(100.0 * np.tanh((x - x.mean()) / (8 * DX)), ch['oceQsw'].shape)
    ch['oceQsw'] = (ch['oceQsw'].dims, ch['oceQsw'].values + ramp, ch['oceQsw'].attrs)
    bd = budget_of(c, L=0, chunk=ch)
    m = c['mask']
    assert rms(bd['surface_flux'].values[0][m]) > 0.0


def test_an_upward_positive_flux_store_is_refused(case8):
    c, ch = chunk_for(case8, qsw=-200.0)
    with pytest.raises(ValueError, match='upward-positive|refusing'):
        budget_of(c, L=0, chunk=ch)


# ---------------------------------------------------------------------------
# the subfilter column and the two F forms
# ---------------------------------------------------------------------------
def test_subfilter_is_identically_zero_at_L0_and_labelled(case8):
    bd = budget_of(case8, L=0)
    m = case8['mask']
    assert np.all(bd['subfilter'].values[0][m] == 0.0)
    assert 'IDENTICALLY ZERO' in bd['subfilter'].attrs['L0_note']
    assert 'identically zero' in bd.attrs['subfilter_note']


def test_subfilter_is_nonzero_at_L2_and_needs_tau_delta():
    """At ``L = 2`` the explicit term is non-zero, and dropping
    ``tau_delta`` changes it -- M1 task 4.  The **V3 strain mix** is used
    here, not the pure deformation: ``u = -a x, v = a y`` is exactly
    non-divergent, and ``tau_delta = mean(b div u) - bbar div ubar`` is
    identically zero for a non-divergent (and for a uniformly divergent)
    flow, so the pure case cannot show the difference at all."""
    case8 = pair(ell_cells=8.0, modes=None)
    bd = budget_of(case8, L=2)
    m = case8['mask']
    sub = bd['subfilter'].values[0]
    assert rms(sub[m]) > 0.0
    assert 'included' in bd['subfilter'].attrs['tau_delta']
    g, grid = case8['g'], case8['grid']
    h0, h1 = (case8['ds'].isel(time=t).expand_dims('face').astype('float64') for t in (0, 1))
    b_r, U_r, V_r = (inp.midpoint(a, b) for a, b in zip(inp.filtered(h0, 0), inp.filtered(h1, 0)))
    b_bar = inp.midpoint(*[inp.filtered(h, 2)[0] for h in (h0, h1)])
    tx, ty = cg.subfilter_flux(b_r, U_r, V_r, 2, g, grid)
    no_delta = 2.0 * cg.subfilter_term(b_bar, tx, ty, g, grid).values[0]
    assert rms((sub - no_delta)[m]) > 0.01 * rms(sub[m])


def test_two_F_chain_differs_from_the_discrete_form(case8):
    """M1-Q1: both forms are run; they are not the same field."""
    bd = budget_of(case8, L=0)
    m = case8['mask']
    a, b = bd['two_F'].values[0], bd['two_F_chain'].values[0]
    assert rms((a - b)[m]) > 1e-3 * rms(a[m])
    assert bd.attrs['forms'] == ['discrete', 'chain']


def test_forms_must_contain_discrete(case8):
    with pytest.raises(ValueError, match='discrete'):
        budget_of(case8, L=0, forms=('chain',))


def test_order5_has_the_wider_nan_rim(case8):
    """M1-Q6: the order-5 sensitivity costs a wider NaN rim per axis than
    the order-3 primary."""
    bd = budget_of(case8, L=0)
    n3 = np.isnan(bd['DGDt_semilag'].values[0]).sum()
    n5 = np.isnan(bd['DGDt_semilag_o5'].values[0]).sum()
    assert n5 > n3


# ---------------------------------------------------------------------------
# the schema, the masks and the front pool
# ---------------------------------------------------------------------------
def test_schema_dims_dtype_and_attrs(case8):
    c, ch = chunk_for(case8, w=-1e-5, dT=0.02)
    bd = budget_of(c, L=2, chunk=ch)
    for v in bg.DERIVED_VARS:
        assert v in bd, v
        assert bd[v].dims == ('time', 'j', 'i'), v
    assert bd.sizes['time'] == 1
    for v in bd.data_vars:
        assert bd[v].dtype in (np.dtype('float64'), np.dtype('bool')), v
    assert str(bd['time_mid'].values[0])[:16] == '2012-07-02T00:30'
    for k in ('L_cells', 't0', 'front_pct', 'edge_cells', 'order', 'order_sens', 'measured',
              'terms_present', 'terms_missing', 'n_valid', 'n_front', 'n_lost', 'git_commit',
              'dbof_commit', 'budget', 'operators'):
        assert k in bd.attrs, k
    assert bd.attrs['L_cells'] == 2 and bd.attrs['edge_cells'] == bg.EDGE_CELLS
    assert bd.attrs['measured'] == 'DGDt_semilag'
    assert np.array_equal(bg.measured(bd).values, bd['DGDt_semilag'].values,
                          equal_nan=True)
    for v in ('two_F', 'subfilter', 'vertical', 'surface_flux', 'residual'):
        assert bd[v].attrs.get('units') == 's-5', v


def test_valid_excludes_nan_and_counts_what_it_lost(case8):
    """``valid`` is ``mask_analysis & isfinite(every budget field)``; a NaN
    planted inside the analysis mask removes exactly that cell and is
    counted in ``n_lost``."""
    c = dict(case8)
    ds = c['ds'].copy()
    Th = ds['Theta'].values.copy()
    Th[:, 64, 64] = np.nan
    ds['Theta'] = (ds['Theta'].dims, Th, ds['Theta'].attrs)
    c['ds'] = ds
    bd = budget_of(c, L=0)
    base = budget_of(case8, L=0)
    v, v0 = bd['valid'].values[0], base['valid'].values[0]
    assert not v[64, 64] and v0[64, 64]
    assert bd.attrs['n_lost'] == int(v0.sum() - v.sum()) + base.attrs['n_lost']
    assert bd.attrs['n_valid'] < base.attrs['n_valid']
    assert np.all(np.isfinite(bd['two_F'].values[0][v]))
    assert np.all(np.isfinite(bg.measured(bd).values[0][v]))


def test_front_is_selected_on_G_at_the_midpoint_only(case8):
    """Planning §11: the pool is ``G_mid >= p90`` over ``valid``, so it is
    independent of either endpoint.  Scaling ``b`` at ``t + 1`` alone moves
    ``G_tp1`` and the measured tendency but **not** the front pool."""
    bd = budget_of(case8, L=0)
    n = int(bd['front'].values.sum())
    assert n == pytest.approx(0.1 * bd.attrs['n_valid'], rel=0.02)
    assert bd['front'].attrs['front_pct'] == 90.0
    assert bool(np.all(bd['valid'].values[bd['front'].values]))      # front is inside valid
    p80 = budget_of(case8, L=0, front_pct=80.0)
    assert int(p80['front'].values.sum()) > n


def test_residual_is_measured_minus_the_terms_present(case8):
    c, ch = chunk_for(case8, w=-1e-5, dT=0.02)
    bd = budget_of(c, L=2, chunk=ch)
    r = bg.measured(bd).values[0].copy()
    for t in bg.TERMS:
        r = r - bd[t].values[0]
    assert np.allclose(bd['residual'].values[0], r, equal_nan=True)
    assert bd['residual'].attrs['terms_present'] == list(bg.TERMS)


# ---------------------------------------------------------------------------
# the derived store
# ---------------------------------------------------------------------------
def test_write_derived_resume_noop_and_clobber(tmp_path, case8):
    """float32 on disk, one chunk per pair, resume is a no-op, and clobber
    truncates from that pair and re-appends (``zarr_series``)."""
    out = tmp_path / 'derived_L0.zarr'
    bd = budget_of(case8, L=0)
    bg.write_derived(bd, 0, out=out)
    ds = xr.open_zarr(out)
    assert ds.sizes['time'] == 1
    assert ds['two_F'].dtype == np.dtype('float32')
    assert ds['two_F'].encoding['chunks'][0] == 1
    assert ds.attrs['n_pairs'] == 1 and ds.attrs['L_cells'] == 0
    ds.close()

    bg.write_derived(bd, 0, out=out)                       # resume: already present
    assert xr.open_zarr(out).sizes['time'] == 1

    bd1 = budget_of(case8, L=0, t0=0)                      # a second pair, shifted in time
    bd1 = bd1.assign_coords(time=bd1['time'] + np.timedelta64(1, 'h'),
                            time_mid=('time', bd1['time_mid'].values + np.timedelta64(1, 'h')))
    bg.write_derived(bd1, 0, out=out)
    assert xr.open_zarr(out).sizes['time'] == 2

    bg.write_derived(bd1, 0, out=out, clobber=True)        # rewrite the last pair
    ds = xr.open_zarr(out)
    assert ds.sizes['time'] == 2
    ds.close()
    with pytest.raises(ValueError, match='L ='):
        bg.write_derived(bd, 8, out=out)


def test_derived_path_default():
    assert str(bg.derived_path(4)).endswith('tile330_derived_L4.zarr')


# ---------------------------------------------------------------------------
# the report's arithmetic
# ---------------------------------------------------------------------------
def test_closure_report_pools_and_gates(case8):
    """Both pools are reported; the gates read the module's pre-declared
    constants; ``mask`` restricts both."""
    c, ch = chunk_for(case8, w=0.0, dT=0.02)
    bd = budget_of(c, L=2, chunk=ch)
    rep = bg.closure_report(bd)
    assert set(rep['front_and_valid']['terms']) == set(bg.TERMS)
    assert rep['front_and_valid']['n'] < rep['valid']['n']
    assert rep['tolerances']['resid_ratio_max'] == bg.Q1_RESID_RATIO_MAX
    assert rep['gates']['Q2']['gated'] is True                     # L = 2
    assert bg.closure_report(budget_of(c, L=0, chunk=ch))['gates']['Q2']['gated'] is False
    half = np.zeros(case8['mask'].shape, bool)
    half[:64] = True
    assert bg.closure_report(bd, half)['n_valid'] < rep['n_valid']
    assert 'CLOSURE REPORT' in bg.format_closure(rep)


def test_closure_verdict_follows_the_predeclared_tolerances(case8):
    """A residual made large by hand must flip the verdict to False."""
    c, ch = chunk_for(case8, w=0.0, dT=0.02)
    bd = budget_of(c, L=2, chunk=ch)
    bad = bd.copy()
    # ten times measured: a residual that is both large and perfectly
    # correlated with 2F, so every M3-Q1 gate fails for a stated reason
    bad['residual'] = 10.0 * bg.measured(bd)
    rep = bg.closure_report(bad)
    assert rep['closed'] is False
    assert 'NOT CLOSED' in bg.format_closure(rep)


# ---------------------------------------------------------------------------
# the real stores
# ---------------------------------------------------------------------------
@pytest.mark.needs_grid
def test_real_pair0_bitforbit_identities_and_the_V3_pool():
    """Pair 0 (07-02 00-01 UTC) at ``L = 0``: ``two_F`` **bit-for-bit**
    ``validate.two_F(..., form='discrete')`` and ``DGDt_semilag``
    bit-for-bit ``validate.null_step``'s ``measured`` with the **real**
    ``b_tp1`` substituted -- same functions, same inputs, so any difference
    is a bug in the plumbing -- plus the V3 / M2 pool, 26,293 front pixels
    of 262,925 valid."""
    for p in (inp.RAW_ZARR, inp.CHUNK_ZARR, inp.GRID_ZARR, inp.MASKS_NC):
        if not p.exists():
            pytest.skip(f'{p} not on disk')
    ds, g, grid, masks = inp.open_inputs()
    bd = bg.compute_budget(ds, g, grid, masks, 0, t0=0)
    h0, h1 = inp.hour_pair(ds, 0)
    b0, U0, V0 = inp.filtered(h0, 0)
    b1, U1, V1 = inp.filtered(h1, 0)
    b_mid = inp.midpoint(b0, b1)
    U_mid, V_mid = inp.midpoint(U0, U1), inp.midpoint(V0, V1)
    assert np.array_equal(bd['two_F'].values[0],
                          va.two_F(b_mid, U_mid, V_mid, g, grid, form='discrete'), equal_nan=True)
    assert np.array_equal(bd['DGDt_semilag'].values[0],
                          sl.measured_DGDt(b0, b1, U_mid, V_mid, g, grid,
                                           order=3).values[0], equal_nan=True)
    assert bd.attrs['n_valid'] == 262_925 and bd.attrs['n_front'] == 26_293
    assert bd.attrs['n_lost'] == 0
    assert bd['residual'].attrs['terms_missing'] == []
    rep = bg.closure_report(bd)
    assert rep['closed'] in (True, False) and rep['warning'] is None
