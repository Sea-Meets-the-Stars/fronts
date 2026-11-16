"""The three case-study figures for §5.1 of the interim report.

Everything is drawn from ``data/m3_case_fields.npz`` and ``data/m3_case_summary.json``
(written by ``py/m3_case.py``) and ``data/m3_closure_summary.json`` (task 6).  **No data store
is opened here**, which is the report's provenance rule (M3-Q23 (a)) applied to the case study:
the analysis runs once and writes its numbers down; the figures and the prose read those.

``figs/case_front_maps.png``  the front over four hourly snapshots, with the followed parcels,
                              and the predicted tendency on the same frames
``figs/case_strength.png``    its strength against the prediction, and the hourly rates
``figs/case_budget.png``      its five-term budget beside the 71-pair aggregate

Run:  <frontogenesis env python> make_case_figs.py
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'data'
OUT = HERE / 'figs'
OUT.mkdir(exist_ok=True)

MIDNIGHT, DEEP, TEAL = '#21295C', '#065A82', '#1C7293'
AMBER, GREEN, RED, MUTED = '#A8501B', '#2C6E49', '#B22222', '#62737F'

matplotlib.rcParams.update({'font.size': 10, 'figure.dpi': 170,
                            'axes.spines.top': False, 'axes.spines.right': False})

Z = np.load(DATA / 'm3_case_fields.npz', allow_pickle=False)
C = json.loads((DATA / 'm3_case_summary.json').read_text())
S = json.loads((DATA / 'm3_closure_summary.json').read_text())


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'{name}  {p.stat().st_size / 1e3:.0f} kB')


# ====================================================== 1. the front, and the prediction
def _lonlat(pos, box, XC, YC):
    """Parcel positions (absolute ``j, i``) as longitude and latitude, bilinear in the box."""
    jj = np.clip(pos[:, 0] - box[0], 0, XC.shape[0] - 1.001)
    ii = np.clip(pos[:, 1] - box[2], 0, XC.shape[1] - 1.001)
    out = []
    for A in (XC, YC):
        j0, i0 = np.floor(jj).astype(int), np.floor(ii).astype(int)
        fj, fi = jj - j0, ii - i0
        out.append((A[j0, i0] * (1 - fj) * (1 - fi) + A[j0 + 1, i0] * fj * (1 - fi)
                    + A[j0, i0 + 1] * (1 - fj) * fi + A[j0 + 1, i0 + 1] * fj * fi))
    return out


def front_maps():
    G, F2, pos, box = Z['G_box'], Z['two_F_box'], Z['pos'], Z['box']
    idx, t0 = Z['map_index'], C['case']['t0']
    XC, YC = Z['XC'], Z['YC']
    ext = [XC.min(), XC.max(), YC.min(), YC.max()]

    def geo(a):                       # stored (j, i); i runs north -> south on this face
        return a.T[::-1]

    lo, hi = -15.2, -12.6
    fmax = float(np.nanpercentile(np.abs(F2), 99.5))
    fig, axs = plt.subplots(2, 4, figsize=(14.2, 7.4), sharex=True, sharey=True)
    for c in range(4):
        hrs = int(Z['map_pairs'][c]) - t0
        im0 = axs[0, c].imshow(geo(np.log10(G[c])), extent=ext, vmin=lo, vmax=hi, cmap='magma',
                               origin='lower', aspect='auto', interpolation='bilinear')
        lon, lat = _lonlat(pos[idx[c]], box, XC, YC)
        axs[0, c].plot(lon, lat, '.', ms=1.6, color='#34D3EB', alpha=0.9, mew=0, zorder=4)
        axs[0, c].set_title('seed  ($t = 0$ h)' if hrs == 0 else f'$t = {hrs:+d}$ h',
                            fontsize=11.5, color=MIDNIGHT)
        im1 = axs[1, c].imshow(geo(F2[c]), extent=ext, norm=TwoSlopeNorm(0.0, -fmax, fmax),
                               cmap='RdBu_r', origin='lower', aspect='auto',
                               interpolation='bilinear')
        axs[1, c].plot(lon, lat, '.', ms=1.0, color='#1A1A1A', alpha=0.45, mew=0, zorder=4)
        axs[1, c].set_xlabel('longitude', fontsize=9.5)
    for a in axs.ravel():
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        a.tick_params(labelsize=8.5)
    axs[0, 0].set_ylabel('front strength\n$\\log_{10} G$   [s$^{-4}$]', fontsize=10)
    axs[1, 0].set_ylabel('predicted tendency\n$2F$   [s$^{-5}$]', fontsize=10)
    fig.subplots_adjust(right=0.90)
    for row, im, lab in ((0, im0, '$\\log_{10} G$'), (1, im1, '$2F$')):
        pos_ax = axs[row, 3].get_position()
        cax = fig.add_axes([0.915, pos_ax.y0, 0.013, pos_ax.height])
        fig.colorbar(im, cax=cax).set_label(lab, fontsize=9)
        cax.tick_params(labelsize=8)
    sl = C['case']['seed_lat']
    fig.suptitle(
        f"One front, followed.  The filament at {np.mean(sl):.1f}$^\\circ$N, "
        f"{abs(C['case']['lon'][1]):.0f}-{abs(C['case']['lon'][0]):.0f}$^\\circ$W, "
        f"{C['case']['coast_km']:.0f} km offshore   ·   "
        f"{C['case']['n_followed']:,} parcels, seeded on its front pixels at $t = 0$ "
        '(cyan above, dark below)', fontsize=12, color=MIDNIGHT, y=0.975)
    save(fig, 'case_front_maps.png')


# ====================================================== 2. strength against the prediction
def strength():
    h, rh, nb = Z['hours'], Z['rate_hours'], int(Z['n_back'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    a.fill_between(h, Z['G_q25'], Z['G_q75'], color=MIDNIGHT, alpha=0.09, lw=0,
                   label='followed parcels, inter-quartile range')
    a.plot(h, Z['G_from_budget'], '--', color=TEAL, lw=1.8,
           label=r'all four terms:  $G_0 + \int (2F{+}\tau{+}V{+}S)\,dt$')
    a.plot(h, Z['G_from_2F'], '-', color=GREEN, lw=2.0,
           label=r'predicted:  $G_0 + \int 2F\,dt$')
    a.plot(h, Z['G_ctrl'], ':', color=MUTED, lw=1.8,
           label=f"unselected control ({C['selection']['n_control']:,} parcels)")
    a.plot(h, Z['G_obs'], '-o', color=MIDNIGHT, ms=3.2, lw=2.0, label='observed  $G$', zorder=5)
    a.plot([0], [Z['G_obs'][nb]], 'o', ms=8, mfc='none', mec=RED, mew=1.8, zorder=6)
    a.axvline(0, color=MUTED, lw=0.9, alpha=0.5)
    a.set_xlabel('hours from the seed time'); a.set_ylabel('$G$  [s$^{-4}$]')
    a.set_title('(a)  front strength along the trajectory', fontsize=11, color=MIDNIGHT)
    a.set_ylim(0, float(np.nanmax(Z['G_q75'])) * 1.30)
    a.legend(fontsize=8.0, loc='upper left', framealpha=0.95)
    a.text(0.975, 0.60,
           'the parcels are selected at $t=0$,\nso $G$ falls away on both sides of it:\n'
           'that is the selection, and the\nunselected control shows it',
           transform=a.transAxes, ha='right', va='top', fontsize=8.4, color=MUTED,
           linespacing=1.5)

    b.axhline(0, color=MUTED, lw=0.9)
    b.plot(rh, Z['med_surface_flux'], '-', color=AMBER, lw=1.3, alpha=0.85,
           label='surface flux  $S$')
    b.plot(rh, Z['med_residual'], '-', color=RED, lw=1.6, alpha=0.9, label='residual  $R$')
    b.plot(rh, Z['med_two_F'], '-', color=GREEN, lw=2.0, label='predicted  $2F$')
    b.plot(rh, Z['med_measured'], '-o', color=MIDNIGHT, ms=3.0, lw=2.0,
           label='measured  $D_hG/Dt$', zorder=5)
    b.axvline(0, color=MUTED, lw=0.9, alpha=0.5)
    b.set_xlabel('hours from the seed time'); b.set_ylabel('tendency  [s$^{-5}$]')
    b.set_title('(b)  the hourly rates along the same parcels', fontsize=11, color=MIDNIGHT)
    lo = min(float(np.nanmin(Z[k])) for k in
             ('med_measured', 'med_two_F', 'med_residual', 'med_surface_flux'))
    hi = max(float(np.nanmax(Z[k])) for k in
             ('med_measured', 'med_two_F', 'med_residual', 'med_surface_flux'))
    pad = 0.12 * (hi - lo)
    b.set_ylim(lo - pad, hi + 2.2 * pad)
    b.legend(fontsize=8.2, ncol=2, loc='upper left', framealpha=0.95)
    r = C['rates']
    b.text(0.985, 0.74,
           f"after the seed: $2F > 0$ at {r['hours_2F_positive']}/{r['n_intervals']} intervals,\n"
           f"measured $> 0$ at {r['hours_measured_positive']}/{r['n_intervals']}",
           transform=b.transAxes, fontsize=8.6, color=RED, va='top', ha='right')
    fig.tight_layout()
    save(fig, 'case_strength.png')


# ====================================================== 3. one front vs all 71 pairs
def budget_bars():
    names = [str(x) for x in Z['term_names']]
    label = {'two_F': '$2F$\nkinematic', 'subfilter': r'$\tau$' + '\nsub-filter',
             'vertical': '$V$\nvertical', 'surface_flux': '$S$\nsurface flux',
             'residual': '$R$\nresidual'}
    case = [C['terms'][k]['rms_over_measured'] for k in names]
    agg_t = S['per_L']['0']['closure']['terms']
    agg = [agg_t[k]['over_measured'] if k in agg_t
           else S['per_L']['0']['closure']['residual']['over_measured'] for k in names]
    x = np.arange(len(names)); w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - w / 2, case, w, color=DEEP, label=f"this one front "
           f"({C['case']['n_followed']:,} parcels, "
           f"{C['case']['n_back'] + C['case']['n_fwd']} h)")
    ax.bar(x + w / 2, agg, w, color=MUTED, alpha=0.75,
           label='all 71 pairs, 1,866,803 front pixels')
    ax.axhline(S['declared']['criterion1']['resid_ratio_max'], color=RED, ls='--', lw=1.6)
    ax.text(len(names) - 0.52, S['declared']['criterion1']['resid_ratio_max'] + 0.02,
            'closure tolerance 0.50  (it applies to $R$)', color=RED, fontsize=9, ha='right')
    for xi, (c, a) in enumerate(zip(case, agg)):
        ax.text(xi - w / 2, c + 0.012, f'{c:.3f}', ha='center', fontsize=8.4, color=DEEP)
        ax.text(xi + w / 2, a + 0.012, f'{a:.3f}', ha='center', fontsize=8.4, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([label[k] for k in names], fontsize=9.5)
    ax.set_ylabel('r.m.s., as a fraction of the measured tendency')
    ax.set_title('The same budget, one front and the whole pool  ($L = 0$)',
                 fontsize=11.5, color=MIDNIGHT)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, max(max(case), max(agg)) * 1.22)
    fig.tight_layout()
    save(fig, 'case_budget.png')


# ====================================================== 4. the surface-flux term on this front
def sflux():
    """M3-digging task 9 (2026-11-05): the surface-flux term on the followed front, from
    ``data/m3_case_sflux.{json,npz}`` (``py/m3_case_sflux.py``) -- no store opened."""
    X = np.load(DATA / 'm3_case_sflux.npz', allow_pickle=False)
    J = json.loads((DATA / 'm3_case_sflux.json').read_text())
    h, rh, nb = X['hours'], X['rate_hours'], int(X['n_back'])
    XC, YC, box = X['XC'], X['YC'], X['box']
    ext = [XC.min(), XC.max(), YC.min(), YC.max()]

    def geo(a):
        return a.T[::-1]
    fig = plt.figure(figsize=(14.2, 10.6))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15), hspace=0.42, wspace=0.36)
    # (a) the maps at the seed: S, its parts, the imbalance
    lon, lat = _lonlat(X['pos_seed'], box, XC, YC)
    smax = float(np.nanpercentile(np.abs(X['map_S']), 99.0))
    panels = (('map_S', '$S$ (all fluxes)'), ('map_S_nonsolar', 'non-solar heat part'),
              ('map_S_shortwave', 'shortwave part'), ('map_I', 'imbalance $I = D_hG/Dt - 2F - \\tau - V$'))
    for c, (k, lab) in enumerate(panels):
        a = fig.add_subplot(gs[0, c])
        vmax = smax if k != 'map_I' else float(np.nanpercentile(np.abs(X['map_I']), 99.0))
        im = a.imshow(geo(X[k]), extent=ext, norm=TwoSlopeNorm(0.0, -vmax, vmax), cmap='RdBu_r',
                      origin='lower', aspect='auto', interpolation='bilinear')
        a.plot(lon, lat, '.', ms=1.3, color='#1A1A1A', alpha=0.5, mew=0, zorder=4)
        a.set_title(('(a)  ' if c == 0 else '') + lab, fontsize=10, color=MIDNIGHT)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.tick_params(labelsize=8)
        if c == 0:
            a.set_ylabel('latitude', fontsize=9)
        a.set_xlabel('longitude', fontsize=9)
        cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    # (b) the rates along the parcels
    b = fig.add_subplot(gs[1, 0:2])
    b.axhline(0, color=MUTED, lw=0.9)
    b.plot(rh, X['med_DGDt_semilag'], '-o', color=MIDNIGHT, ms=2.6, lw=1.8, label='measured  $D_hG/Dt$', zorder=5)
    b.plot(rh, X['med_I'], '-', color=MUTED, lw=1.6, label='imbalance  $I$')
    b.plot(rh, X['med_surface_flux'], '-', color=AMBER, lw=2.2, label='$S$ (top cell, 1 m)')
    b.plot(rh, X['med_top_nonsolar'], '--', color=RED, lw=1.2, label='non-solar part')
    b.plot(rh, X['med_top_shortwave'], '--', color='#E0A458', lw=1.2, label='shortwave part')
    b.plot(rh, X['med_top_freshwater'], '--', color=TEAL, lw=1.2, label='freshwater part')
    b.plot(rh, X['med_surface_flux_kpp'], ':', color=AMBER, lw=1.6, label='$S$ with $h_{\\rm bl}$ (§6.3)')
    b.axvline(0, color=MUTED, lw=0.9, alpha=0.5)
    b.set_xlabel('hours from the seed time'); b.set_ylabel('parcel-median tendency  [s$^{-5}$]')
    b.set_title('(b)  the surface-flux term and its parts along the parcels', fontsize=11, color=MIDNIGHT)
    b.legend(fontsize=7.6, ncol=2, loc='lower left', framealpha=0.95)
    sg = J['sign']
    b.text(0.985, 0.97, f"$S < 0$ (frontolytic) at {sg['S_negative_intervals_fwd']}/{sg['n_intervals']} forward intervals",
           transform=b.transAxes, fontsize=8.4, color=AMBER, va='top', ha='right')
    # (c) the integrals with and without S
    c_ = fig.add_subplot(gs[1, 2])
    c_.plot(h, X['G_obs'], '-o', color=MIDNIGHT, ms=2.8, lw=2.0, label='observed  $G$', zorder=5)
    c_.plot(h, X['int_two_F'], '-', color=GREEN, lw=1.8, label=r'$G_0 + \int 2F\,dt$')
    c_.plot(h, X['int_two_F_tau_V'], '-.', color=TEAL, lw=1.4, label=r'$G_0 + \int (2F{+}\tau{+}V)\,dt$')
    c_.plot(h, X['int_all_four'], '--', color=AMBER, lw=2.0, label=r'$G_0 + \int (2F{+}\tau{+}V{+}S)\,dt$')
    c_.plot(h, X['int_all_four_kpp'], ':', color=AMBER, lw=1.6, label=r'the same with $S$ on $h_{\rm bl}$')
    c_.axvline(0, color=MUTED, lw=0.9, alpha=0.5)
    c_.set_xlabel('hours from the seed time'); c_.set_ylabel('$G$  [s$^{-4}$]')
    c_.set_title('(c)  the integral with and without $S$', fontsize=10.5, color=MIDNIGHT)
    c_.set_ylim(0, float(np.nanmax(X['G_obs'])) * 1.35)
    c_.legend(fontsize=7.4, loc='upper left', framealpha=0.95)
    # (d) the pointwise relation
    d = fig.add_subplot(gs[1, 3])
    x, y = X['pw_S_x'].astype('f8'), X['pw_S_y'].astype('f8')
    ok = np.isfinite(x) & np.isfinite(y)
    d.plot(x[ok], y[ok], '.', ms=1.2, color=MUTED, alpha=0.25, mew=0)
    qs = np.nanpercentile(x[ok], np.linspace(0, 100, 13))
    cx, cy, lo, hi = [], [], [], []
    for q0, q1 in zip(qs[:-1], qs[1:]):
        m = ok & (x >= q0) & (x < q1)
        if m.sum() > 20:
            cx.append(np.median(x[m])); cy.append(np.median(y[m]))
            lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
    d.errorbar(cx, cy, yerr=[np.array(cy) - np.array(lo), np.array(hi) - np.array(cy)], fmt='o-', color=AMBER,
               ms=4, lw=1.6, capsize=2, label='binned median, inter-quartile', zorder=5)
    pw = J['pointwise']['S']
    xx = np.array([np.nanpercentile(x[ok], 1), np.nanpercentile(x[ok], 99)])
    d.plot(xx, xx, '-', color=MUTED, lw=1.0, label='unit line (closure)')
    d.plot(xx, pw['ols'] * xx, '--', color=RED, lw=1.2, label=f"OLS slope {pw['ols']:+.2f}")
    d.axhline(0, color=MUTED, lw=0.6); d.axvline(0, color=MUTED, lw=0.6)
    d.set_xlim(xx * 1.15); d.set_ylim(np.nanpercentile(y[ok], [1, 99]) * 1.15)
    d.set_xlabel('$S$ on the parcel  [s$^{-5}$]'); d.set_ylabel('imbalance $I$ on the parcel  [s$^{-5}$]')
    d.set_title(f"(d)  pointwise, corr {pw['corr']:+.2f}", fontsize=10.5, color=MIDNIGHT)
    d.text(0.02, 0.02, f"{pw['n']:,} parcel-hours", transform=d.transAxes, fontsize=8, color=MUTED)
    d.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
    d.legend(fontsize=7.6, loc='upper left', framealpha=0.95)
    r = J['rms_over_measured']
    fig.suptitle(f"The surface-flux term on the followed front ($L = 0$): r.m.s. {r['S']:.3f} of the measured "
                 f"tendency, {J['parts_over_S']['nonsolar']:.2f} of it non-solar; its integral over the forward "
                 f"day is {J['integrals']['S_integral_over_dG']:+.2f} of the observed change",
                 fontsize=11.5, color=MIDNIGHT, y=0.985)
    save(fig, 'case_sflux.png')


if __name__ == '__main__':
    front_maps()
    strength()
    budget_bars()
    sflux()
