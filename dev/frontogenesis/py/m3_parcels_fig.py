""" The two figures for M3-digging task 6 -- the velocity field at the seed and the
parcels' motion relative to the front."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LogNorm                          # noqa: E402

import figures as fg                                           # noqa: E402

CYAN, DARK = '#00e5ff', '#1b1b3a'


def _box(a, XC, YC, box):
    sj, si = slice(box['j0'], box['j1']), slice(box['i0'], box['i1'])
    return a[sj, si], XC[sj, si], YC[sj, si]


def _lonlat(pos, XC, YC):
    jj = np.clip(pos[:, 0], 0, XC.shape[0] - 1.001)
    ii = np.clip(pos[:, 1], 0, XC.shape[1] - 1.001)
    j0, i0 = jj.astype(int), ii.astype(int)
    fj, fi = jj - j0, ii - i0
    def bil(A):
        return ((1 - fj) * (1 - fi) * A[j0, i0] + fj * (1 - fi) * A[j0 + 1, i0]
                + (1 - fj) * fi * A[j0, i0 + 1] + fj * fi * A[j0 + 1, i0 + 1])
    return bil(XC), bil(YC)


def figure_velocity(S, f0, geo, speed0, vn0, XC, YC, pos0, ctrl0, box, path, dpi=160):
    sd = S['seed']
    G, X, Y = _box(f0['G'], XC, YC, box)
    ue, vn = (_box(a, XC, YC, box)[0] for a in geo)
    sp, _, _ = _box(speed0, XC, YC, box)
    vc, _, _ = _box(vn0, XC, YC, box)
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    # (a) log10 G + quiver + seeds
    a = ax[0, 0]
    m = fg.draw_map(a, X, Y, np.log10(np.where(G > 0, G, np.nan)), cmap='viridis')
    k = 5
    a.quiver(X[::k, ::k], Y[::k, ::k], ue[::k, ::k], vn[::k, ::k], color='w', scale=8, width=0.0022,
             alpha=0.9)
    a.quiverkey(a.quiver([], [], [], []), 0.88, 0.07, 0.5, '0.5 m/s', color='w', labelcolor='w')
    lon, lat = _lonlat(pos0, XC, YC)
    a.scatter(lon, lat, s=5, color=CYAN, lw=0, zorder=5, label='seed parcels')
    a.set_title(f'log10 G at the seed ({S["case"]["t_seed"]} UTC) and the velocity at that hour\n'
                f'seed parcels: speed {sd["speed_ms"]["median"]:.2f} m/s = {sd["cells_per_hour_median"]:.2f} cells/h (median)',
                fontsize=10)
    plt.colorbar(m, ax=a, label='log10 G [s$^{-4}$]', shrink=0.8)
    # (b) speed with streamlines
    a = ax[0, 1]
    m = fg.draw_map(a, X, Y, sp, cmap='magma', vmin=0, vmax=np.nanpercentile(sp, 99))
    try:
        a.streamplot(X[0, :], Y[:, 0], ue, vn, color='w', density=1.4, linewidth=0.6, arrowsize=0.8)
    except Exception:                                         # a non-rectilinear lon/lat grid
        a.quiver(X[::4, ::4], Y[::4, ::4], ue[::4, ::4], vn[::4, ::4], color='w', scale=8, width=0.002)
    a.scatter(lon, lat, s=4, color=CYAN, lw=0, zorder=5)
    a.set_title('speed at the seed hour with streamlines', fontsize=10)
    plt.colorbar(m, ax=a, label='|u| [m/s]', shrink=0.8)
    # (c) cross-front speed map on front pixels
    a = ax[1, 0]
    fr, _, _ = _box(f0['front'] & f0['valid'], XC, YC, box)
    m = fg.draw_map(a, X, Y, np.where(fr, vc, np.nan), cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    a.scatter(lon, lat, s=4, color=CYAN, lw=0, zorder=5)
    a.set_title('cross-front velocity on front pixels (+ towards denser water)\n'
                f'seed parcels: |cross| {sd["cross_ms"]["median"]:.3f} m/s, |along| {sd["along_ms"]["median"]:.3f} m/s (medians); '
                f'front width {sd["front_width_km"]:.1f} km -> crossing time {sd["crossing_time_h"]:.0f} h',
                fontsize=10)
    plt.colorbar(m, ax=a, label='u . n [m/s]', shrink=0.8)
    # (d) hodograph of the box-mean velocity
    a = ax[1, 1]
    bm = S['box_mean']
    h = np.asarray(bm['hours_from_seed'])
    ue_b, vn_b = np.asarray(bm['u_east']), np.asarray(bm['v_north'])
    sc = a.scatter(ue_b, vn_b, c=h, cmap='coolwarm', s=25, zorder=5)
    a.plot(ue_b, vn_b, '-', color='0.5', lw=0.8)
    a.plot(ue_b.mean(), vn_b.mean(), 'k+', ms=14, mew=2, label='mean')
    for i in range(0, len(h), 12):
        a.annotate(f'{h[i]:+.0f} h', (ue_b[i], vn_b[i]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    a.axhline(0, color='0.8', lw=0.8)
    a.axvline(0, color='0.8', lw=0.8)
    a.set_aspect('equal')
    a.set_xlabel('u_east [m/s]')
    a.set_ylabel('v_north [m/s]')
    fu, fv = bm['fit_u']['amplitudes'], bm['fit_v']['amplitudes']
    keys = list(fu)
    a.set_title(f'box-mean velocity, hours {h[0]:+.0f}..{h[-1]:+.0f} from the seed (hodograph)\n'
                f'mean {bm["mean_speed_ms"]:.3f} m/s; rms about it {bm["rms_about_mean_ms"]:.3f}; '
                f'amplitudes u/v: {keys[0]} {fu[keys[0]]:.3f}/{fv[keys[0]]:.3f}, '
                f'{keys[1]} (inertial) {fu[keys[1]]:.3f}/{fv[keys[1]]:.3f} m/s',
                fontsize=9)
    plt.colorbar(sc, ax=a, label='hours from the seed', shrink=0.8)
    a.legend(fontsize=8)
    fig.suptitle('M3 digging task 6 -- the velocity field at the seed of the §5.1 case, and its parcels',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fg.caption(fig, 'L = 0; the case box of §5.1 (37 N, 161 km offshore); geographic components on face 10: '
                    'u_east = V, v_north = -U; the front normal is grad b / |grad b| at the midpoint field.')
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path


def figure_motion(S, f0, XC, YC, pos, cpos, box, path, dpi=160):
    se, ct = S['selected'], S['control']
    h = np.asarray(S['hours_from_seed'])
    n0 = int(S['case']['n_back'])
    G, X, Y = _box(f0['G'], XC, YC, box)
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    # (a) trajectories over log G at the seed
    a = ax[0, 0]
    m = fg.draw_map(a, X, Y, np.log10(np.where(G > 0, G, np.nan)), cmap='Greys')
    cmap = plt.get_cmap('coolwarm')
    for n in range(n0, len(h) - 1):
        lon0, lat0 = _lonlat(pos[n], XC, YC)
        lon1, lat1 = _lonlat(pos[n + 1], XC, YC)
        a.plot(np.stack([lon0, lon1]), np.stack([lat0, lat1]), '-', color=cmap((h[n] - h[n0]) / (h[-1] - h[n0])),
               lw=0.5, alpha=0.6)
    for n, col, lab in ((n0, CYAN, 'seed (0 h)'), (n0 + 16, '#ff7f0e', '+16 h'), (len(h) - 1, '#d62728', '+24 h')):
        lon, lat = _lonlat(pos[n], XC, YC)
        a.scatter(lon, lat, s=6, color=col, lw=0, zorder=5, label=lab)
    a.legend(fontsize=8, loc='lower left')
    a.set_title('the 450 parcels over the forward day, on log10 G at the seed', fontsize=10)
    plt.colorbar(m, ax=a, label='log10 G at the seed', shrink=0.8)
    # (b) on-front fraction
    a = ax[0, 1]
    a.plot(h, se['on_front'], 'k-o', ms=3, label='selected: on a front pixel (G >= p90)')
    a.plot(h, se['on_p80'], 'k--', lw=1, label='selected: G >= p80')
    a.plot(h, ct['on_front'], '-o', ms=3, color='0.6', label='control: on a front pixel')
    a.plot(h, ct['on_p80'], '--', lw=1, color='0.6', label='control: G >= p80')
    a.axhline(0.1, color='#b22222', ls=':', lw=1, label='10 % (a random pixel)')
    a.axvline(0, color='0.8')
    a.set_ylim(0, 1.02)
    a.set_xlabel('hours from the seed')
    a.set_ylabel('fraction of parcels')
    a.set_title('how many parcels sit on a front pixel', fontsize=10)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    # (c) distance to the nearest front pixel and the front strength
    a = ax[1, 0]
    a.plot(h, se['dist_front_km'], 'k-o', ms=3, label='selected: distance to nearest front pixel')
    a.plot(h, ct['dist_front_km'], '-o', ms=3, color='0.6', label='control')
    a.set_xlabel('hours from the seed')
    a.set_ylabel('median distance [km]')
    a.axvline(0, color='0.8')
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(h, np.asarray(se['G_median']) / S['seed']['G_seed'], 'b-', lw=1.2, label='median G / G at seed (selected)')
    a2.set_ylabel('G / G_seed', color='b')
    a2.set_ylim(0, 1.3)
    a.set_title(f'drift off the front: distance to the nearest front pixel; front width at the seed '
                f'{S["seed"]["front_width_km"]:.1f} km', fontsize=10)
    lines = a.get_legend_handles_labels()
    l2 = a2.get_legend_handles_labels()
    a.legend(lines[0] + l2[0], lines[1] + l2[1], fontsize=7, loc='upper left')
    # (d) speeds along the trajectory
    a = ax[1, 1]
    a.plot(h, se['speed_ms'], 'k-', label='speed (selected, median)')
    a.plot(h, se['cross_abs_ms'], 'r-', label='|cross-front|')
    a.plot(h, se['along_abs_ms'], 'b-', label='|along-front|')
    a.plot(h, ct['speed_ms'], 'k:', label='speed (control)')
    a.plot(h[:-1] + 0.5, np.asarray(se['step_cells_median']) * S['km_per_cell'] * 1e3 / 3600, 'g--',
           lw=1, label='hourly step / 1 h (selected)')
    a.axvline(0, color='0.8')
    a.set_xlabel('hours from the seed')
    a.set_ylabel('m/s')
    fd = se['forward_day']
    a.set_title(f'the parcels\' speed and its components along the path\nforward day: path {fd["path_cells_median"]:.1f} cells, '
                f'net {fd["net_cells_median"]:.1f} cells ({fd["net_km_median"]:.0f} km), straightness {fd["straightness"]:.2f}; '
                f'cross-front path {fd["cross_path_km_median"]:.1f} km, net {fd["cross_net_km_median_abs"]:.1f} km',
                fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    fig.suptitle('M3 digging task 6 -- what the followed parcels do relative to the front', fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fg.caption(fig, 'L = 0; the §5.1 case (seed 2012-07-03 06 UTC, pair 30); front pixel = G >= p90 at the midpoint over '
                    'the analysis mask; the control is the case\'s 2,500 unselected parcels, recomputed with its seed.')
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
    return path
