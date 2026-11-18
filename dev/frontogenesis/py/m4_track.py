"""M4 task 2 runner: flow-informed tracking of the 10 largest fronts, against plain
``front_tracking.follow`` on the same anchors (coding §4.10 ``track_top_n``).

Run detached (~25 min: 2.5 min for the flow store, ~45 s per plain track, ~85 s per flow
track):

    cd dev/frontogenesis/py && nohup /home/xavier/miniconda3/envs/frontogenesis/bin/python \\
        m4_track.py > ../data/m4_track.nohup 2>&1 &

Steps, each logged to ``data/m4_track.log``:

1. the flow -- ``tracking.Flow`` over the raw store's ``U``, ``V`` at every integer hour,
   through ``semilag.departure_index`` both ways, cached in ``data/tile330_flow.zarr``
   (reused as is when present);
2. the flow's informativeness, transition by transition: the dilated IoU of the advected
   pair-``k`` front mask with the pair-``k+1`` fronts, against persistence (no advection)
   and against the wrong sign (``-d``), and the pixel retention of the advected skeleton;
3. ``track_top_n`` with the flow and without it, on the ``N`` largest fronts of the anchor
   pair (default the middle one, 35), with ``km_per_px`` from the grid for both;
4. the comparison -- which links differ -- and the per-track table (span, gaps, scores,
   split/merge flags).

Outputs: ``data/m4_tracks.json`` (every track, both runs, every link's terms) and
``data/m4_track_summary.json`` (the tables and the numbers the log entry quotes).
"""
import argparse
import datetime as dt
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import inputs as inp                                                # noqa: E402
import tracking as tr                                               # noqa: E402
from m4_find import Log                                             # noqa: E402

DATA_DIR = tr.DATA_DIR
LOG_PATH = DATA_DIR / 'm4_track.log'
FLOW_PATH = DATA_DIR / 'tile330_flow.zarr'
TRACKS_PATH = DATA_DIR / 'm4_tracks.json'
SUMMARY_PATH = DATA_DIR / 'm4_track_summary.json'


def flow_informativeness(flow, labels, keys, say):
    rows = []
    for k in range(len(keys) - 1):
        m0, m1 = labels[keys[k]] > 0, labels[keys[k + 1]] > 0
        di, dj = flow.displacement(k, +1)
        adv, wrong = tr.advect_mask_by(m0, di, dj), tr.advect_mask_by(m0, -di, -dj)
        f = np.isfinite(di) & m0
        mag = np.hypot(di[f], dj[f])
        rows.append(dict(transition=k, iou_persist=tr.iou_dilated(m0, m1),
                         iou_advected=tr.iou_dilated(adv, m1),
                         iou_wrong_sign=tr.iou_dilated(wrong, m1),
                         retention=float(adv.sum() / m0.sum()),
                         disp_median_cells=float(np.median(mag)) if mag.size else np.nan,
                         disp_p99_cells=float(np.percentile(mag, 99)) if mag.size else np.nan))
    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != 'transition'}
    out = {k: dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
           for k, v in a.items()}
    out['n_transitions_advected_beats_persistence'] = int((a['iou_advected'] > a['iou_persist']).sum())
    out['n_transitions_advected_beats_wrong_sign'] = int((a['iou_advected'] > a['iou_wrong_sign']).sum())
    out['per_transition'] = rows
    say(f'flow informativeness over {len(rows)} transitions: IoU1 persist median '
        f'{out["iou_persist"]["median"]:.3f}, advected {out["iou_advected"]["median"]:.3f}, '
        f'wrong sign {out["iou_wrong_sign"]["median"]:.3f}; advected > persistence on '
        f'{out["n_transitions_advected_beats_persistence"]}, > wrong sign on '
        f'{out["n_transitions_advected_beats_wrong_sign"]}; retention median '
        f'{out["retention"]["median"]:.3f}; front-pixel |d| median of medians '
        f'{out["disp_median_cells"]["median"]:.3f} cells')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', type=int, default=0)
    ap.add_argument('--config', default='D')
    ap.add_argument('--n', type=int, default=tr.N_TOP)
    ap.add_argument('--anchor-step', type=int, default=None, help='default: the middle pair')
    ap.add_argument('--flow-weight', type=float, default=tr.FLOW_WEIGHTS['flow_iou'])
    ap.add_argument('--clobber-flow', action='store_true')
    args = ap.parse_args(argv)

    say = Log(LOG_PATH)
    started, t_start = dt.datetime.now(dt.timezone.utc), time.time()
    say(f'start: L = {args.L}, config {args.config}, n = {args.n}, anchor step '
        f'{args.anchor_step if args.anchor_step is not None else "middle"}, '
        f'flow_iou weight {args.flow_weight}')
    status, error, summary = 'ok', None, {}
    try:
        ds, grid_ds, grid, masks = inp.open_inputs()
        labels = tr.open_fronts_series(tr.fronts_path(args.L, args.config))
        keys = list(labels)
        say(f'{len(keys)} label arrays from {tr.fronts_path(args.L, args.config).name}')
        if args.clobber_flow and FLOW_PATH.exists():
            import shutil
            shutil.rmtree(FLOW_PATH)
        flow = tr.Flow(ds, grid_ds, grid, store=FLOW_PATH)
        if flow._zarr is None:
            t0 = time.time()
            flow.precompute(log=say)
            say(f'flow store written: {FLOW_PATH} in {time.time() - t0:.0f} s')
        else:
            say(f'flow store reused: {FLOW_PATH}')
        say(f'km_per_px {flow.km_per_px:.4f} (front_tracking default {tr.ft.DEFAULT_KM_PER_PX})')
        summary['flow'] = flow_informativeness(flow, labels, keys, say)

        weights = {**tr.FLOW_WEIGHTS, 'flow_iou': args.flow_weight}
        anchor_step = len(keys) // 2 if args.anchor_step is None else args.anchor_step
        say(f'anchor pair {anchor_step} ({keys[anchor_step]}): largest labels '
            f'{tr.largest_labels(labels[keys[anchor_step]], args.n)}')
        t0 = time.time()
        plain = tr.track_top_n(labels, keys, n=args.n, flow=None, anchor_step=anchor_step,
                               km_per_px=flow.km_per_px, log=say)
        t_plain = time.time() - t0
        t0 = time.time()
        flowed = tr.track_top_n(labels, keys, n=args.n, flow=flow, anchor_step=anchor_step,
                                km_per_px=flow.km_per_px, weights=weights, log=say)
        t_flow = time.time() - t0
        tab_p, tab_f = tr.track_table(plain, keys), tr.track_table(flowed, keys)
        cmp = tr.compare_tracks(plain, flowed)
        for name, tab in (('plain', tab_p), ('flow', tab_f)):
            say(f'{name}: spans h {tab["span_h"].tolist()}; found {tab["n_found"].tolist()}; '
                f'gaps in span {tab["n_gaps_in_span"].tolist()}; worst score '
                f'{[round(x, 2) for x in tab["worst_score"]]}; split/merge flags '
                f'{tab["n_split_merge"].tolist()}')
        say(f'comparison: same {cmp["same"].tolist()}, different {cmp["different"].tolist()}, '
            f'only plain {cmp["only_a"].tolist()}, only flow {cmp["only_b"].tolist()}')
        n_links = int(sum(len(t.labels) - 1 for t in flowed))
        summary.update(
            params=dict(L=args.L, config=args.config, n=args.n, anchor_step=anchor_step,
                        anchor_time=keys[anchor_step], km_per_px=flow.km_per_px,
                        weights=weights, max_score=tr.ft.MAX_SCORE, min_pixels=5,
                        mask_order=tr.MASK_ORDER, flow_iou_dilate=tr.FLOW_IOU_DILATE,
                        max_drift_ms=tr.ft.MAX_DRIFT_MS, min_radius_px=tr.ft.MIN_RADIUS_PX),
            wall_s=dict(plain=round(t_plain, 1), flow=round(t_flow, 1)),
            tracks_plain=tab_p.to_dict('records'), tracks_flow=tab_f.to_dict('records'),
            comparison=cmp.to_dict('records'),
            n_tracks=len(flowed),
            n_tracks_ge_12h_flow=int((tab_f['span_h'] >= 12).sum()),
            n_tracks_ge_12h_plain=int((tab_p['span_h'] >= 12).sum()),
            min_span_h_flow=float(tab_f['span_h'].min()),
            n_links_flow=n_links,
            n_links_with_flow_term=int(tab_f['n_flow_links'].sum()),
            n_links_different=int(cmp['different'].sum()),
            n_links_only_plain=int(cmp['only_a'].sum()),
            n_links_only_flow=int(cmp['only_b'].sum()),
            n_split_merge_flags=int(tab_f['n_split_merge'].sum()))
        tr.dump_json(dict(times=keys, plain=tr.tracks_to_json(plain, keys),
                          flow=tr.tracks_to_json(flowed, keys), params=summary['params']),
                     TRACKS_PATH)
        say(f'wrote {TRACKS_PATH.name}')
    except Exception as e:                                   # noqa: BLE001
        status, error = 'error', traceback.format_exc()
        say(f'ERROR: {e!r}')
    summary.update(status=status, started=started.isoformat(timespec='seconds'),
                   finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                   wall_total_s=round(time.time() - t_start, 1), error=error)
    tr.dump_json(summary, SUMMARY_PATH)
    say(f'done: {status}, {time.time() - t_start:.0f} s; {SUMMARY_PATH.name} written')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
