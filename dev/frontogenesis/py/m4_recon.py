"""M4 task 3 runner: the reconciliation on Lagrangian-matched pixels (criterion 4) for the
ten flow-informed tracks of task 2.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python m4_recon.py

For every link of every track (``tracking.reconcile_track``): the front's pixels at step
``t`` are parcels, carried to ``t + 1`` by the flow store (``tracking.forward_positions``,
the inverse of the stored arrival map); the matched rate ``[mean G(t+1 at x') - mean G(t at
x)] / dt`` is compared with the per-pixel ``DGDt_semilag`` averaged over the same parcels,
against the pre-declared tolerance (``tracking.RECON_RMS_TOL``, ``RECON_CORR_MIN``).  The
thresholded-mask and endpoint-label rates are reported as contrasts.  Before the links, one
check: ``forward_positions`` against the case study's forward integrator
(``m3_case.step_forward``) on the raw velocities.

Four constructions are run (``VARIANTS``): the pre-registered one-sided forward difference
on the label's pixels is the criterion-4 verdict; the time-centred difference and the 1-cell
band, singly and together, are the diagnosis of the registration artefact (see ``tracking.py``).

Outputs: ``data/m4_recon.json`` (every row of every construction and the summaries),
``data/m4_recon.log``.
"""
import datetime as dt
import json
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget                                                       # noqa: E402
import inputs as inp                                                # noqa: E402
import m3_case                                                      # noqa: E402
import tracking as tr                                               # noqa: E402
from m4_find import Log                                             # noqa: E402
from m4_track import FLOW_PATH, TRACKS_PATH                         # noqa: E402

DATA_DIR = tr.DATA_DIR
# (name, centred, band).  The FIRST is the pre-registered construction and the criterion-4
# verdict; the others are the diagnosis (tracking.py, "The pre-registered construction").
VARIANTS = [('skeleton_forward', False, 0), ('skeleton_centred', True, 0),
            ('band1_forward', False, 1), ('band1_centred', True, 1)]
LOG_PATH = DATA_DIR / 'm4_recon.log'
OUT_PATH = DATA_DIR / 'm4_recon.json'


def check_forward(ds, grid_ds, grid, flow, labels, k, say, n=2000):
    lab = labels[list(labels)[k]]
    jj, ii = np.nonzero(lab > 0)
    pos = np.stack([jj, ii], axis=1).astype('f8')[:n]
    di, dj = flow.displacement(k, +1)
    x = tr.forward_positions(pos, di, dj)
    h = ds.isel(time=k + 1).load()
    ui, vj = m3_case.index_velocity(h['U'], h['V'], grid_ds, grid)
    x_case, _ = m3_case.step_forward(pos, ui, vj)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(x_case).all(axis=1)
    err = np.hypot(*(x - x_case)[ok].T)
    # and the round trip: the departure of the arrival point is the start
    back = x - np.stack([tr.sample_at(dj, x), tr.sample_at(di, x)], axis=1)
    rt = np.hypot(*(back - pos)[ok].T)
    out = dict(transition=int(k), n=int(ok.sum()), vs_case_median_cells=float(np.median(err)),
               vs_case_p99_cells=float(np.percentile(err, 99)),
               round_trip_median_cells=float(np.median(rt)), round_trip_max_cells=float(rt.max()))
    say(f'forward_positions check, transition {k}: vs m3_case.step_forward median '
        f'{out["vs_case_median_cells"]:.2e} p99 {out["vs_case_p99_cells"]:.2e} cells (n {out["n"]}); '
        f'round trip median {out["round_trip_median_cells"]:.1e} max {out["round_trip_max_cells"]:.1e}')
    return out


def main() -> int:
    say = Log(LOG_PATH)
    started, t_start = dt.datetime.now(dt.timezone.utc), time.time()
    say(f'start: tolerance rms_ratio <= {tr.RECON_RMS_TOL}, corr >= {tr.RECON_CORR_MIN} '
        f'(pre-declared, tracking.py)')
    status, error, out = 'ok', None, {}
    try:
        ds, grid_ds, grid, masks = inp.open_inputs()
        flow = tr.Flow(ds, grid_ds, grid, store=FLOW_PATH)
        labels = tr.open_fronts_series(tr.fronts_path(0))
        T = json.load(open(TRACKS_PATH))
        tracks = tr.tracks_from_json(T['flow'])
        say(f'{len(tracks)} flow-informed tracks from {TRACKS_PATH.name}; flow store {FLOW_PATH.name}')
        out['forward_check'] = check_forward(ds, grid_ds, grid, flow, labels, 30, say)
        fields = tr.DerivedFields(budget.derived_path(0))
        out['variants'] = {}
        out['primary'] = VARIANTS[0][0]
        for vname, centred, band in VARIANTS:
            say(f'--- construction {vname}: centred={centred}, band={band}'
                + ('  [PRIMARY, pre-registered]' if vname == out['primary'] else '  [variant]'))
            dfs = []
            for n, trk in enumerate(tracks):
                t0 = time.time()
                df = tr.reconcile_track(trk, labels, flow, fields, centred=centred, band=band)
                df.insert(0, 'track', n)
                dfs.append(df)
                s = tr._stats(df['r_matched'], df['r_store'])
                say(f'track {n} (label {trk.anchor.label}): {len(df)} links, parcels '
                    f'{int(df["n_parcels"].sum())} lost {int(df["n_lost"].sum())}; matched vs store '
                    f'corr {s["corr"]:.3f} rms_ratio {s["rms_ratio"]:.3f} amp {s["amp_ratio"]:.2f} '
                    f'sign {s.get("sign_agree", float("nan")):.2f}; label vs matched rms_ratio '
                    f'{tr._stats(df["r_label"], df["r_matched"])["rms_ratio"]:.2f}; {time.time() - t0:.1f} s')
            df = pd.concat(dfs, ignore_index=True)
            summ = tr.reconcile_summary(df)
            out['variants'][vname] = dict(summary=summ, rows=df.to_dict('records'))
            m = summ['matched_vs_store']
            say(f'POOLED {vname} over {summ["n_links"]} links of {summ["n_tracks"]} tracks: matched vs '
                f'store corr {m["corr"]:.3f}, rms_ratio {m["rms_ratio"]:.3f}, amp_ratio '
                f'{m["amp_ratio"]:.3f}, sign agreement {m["sign_agree"]:.3f} -> criterion 4 '
                f'{"PASSES" if summ["passes"] else "FAILS"} ({summ["n_tracks_passing"]}/{summ["n_tracks"]} '
                f'tracks pass individually); frac r_matched > 0 {np.nanmean(df["r_matched"] > 0):.2f}, '
                f'r_store > 0 {np.nanmean(df["r_store"] > 0):.2f}')
            for name in ('mask_vs_store', 'label_vs_store', 'label_vs_matched', 'two_F_vs_matched'):
                s = summ[name]
                say(f'  {name}: corr {s["corr"]:.3f} rms_ratio {s["rms_ratio"]:.3f} amp {s["amp_ratio"]:.3f}')
            say(f'  set-evolution term rms / matched rms: {summ["set_evolution_over_matched"]:.3f}; '
                f'split/merge links {summ["n_split_merge"]}; parcels lost '
                f'{summ["parcels"]["lost"]} of {summ["parcels"]["total"]}')
        out['summary'] = out['variants'][out['primary']]['summary']
        out['params'] = dict(L=0, dt=tr.DT, sample_order=3, forward_n_iter=8,
                             variants=[dict(name=v, centred=c, band=b) for v, c, b in VARIANTS],
                             tracks=str(TRACKS_PATH), flow=str(FLOW_PATH),
                             derived=str(budget.derived_path(0)))
    except Exception as e:                                   # noqa: BLE001
        status, error = 'error', traceback.format_exc()
        say(f'ERROR: {e!r}')
    out.update(status=status, started=started.isoformat(timespec='seconds'),
               finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(time.time() - t_start, 1), error=error)
    tr.dump_json(out, OUT_PATH)
    say(f'done: {status}, {time.time() - t_start:.0f} s; {OUT_PATH.name} written')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
