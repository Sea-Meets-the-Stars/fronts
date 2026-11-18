"""M4 task 4 runner: the tracking-quality diagnostic (criterion 3) for the task-2 tracks,
flow-informed and plain, with split / merge flags, and the figure.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python m4_quality.py

Outputs: ``data/m4_quality.json`` (every row of both track sets, the two summaries, the figure's
numbers), ``figs/fig_tq_tracking_quality.png``, ``data/m4_quality.log``.
"""
import datetime as dt
import json
import pathlib
import sys
import time
import traceback

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tracking as tr                                               # noqa: E402
import figures_m4 as f4                                             # noqa: E402
from m4_find import Log                                             # noqa: E402
from m4_track import FLOW_PATH, TRACKS_PATH                         # noqa: E402

DATA_DIR = tr.DATA_DIR
LOG_PATH = DATA_DIR / 'm4_quality.log'
OUT_PATH = DATA_DIR / 'm4_quality.json'


def main() -> int:
    say = Log(LOG_PATH)
    started, t_start = dt.datetime.now(dt.timezone.utc), time.time()
    say(f'start: split/merge threshold {tr.SPLIT_MIN_PIXELS} px, radius {tr.RADIUS_CELLS} cells')
    status, error, out = 'ok', None, {}
    try:
        flow = tr.Flow(store=FLOW_PATH)
        labels = tr.open_fronts_series(tr.fronts_path(0))
        T = json.load(open(TRACKS_PATH))
        qs = {}
        for name in ('flow', 'plain'):
            t0 = time.time()
            tracks = tr.tracks_from_json(T[name])
            q = tr.tracking_quality(tracks, flow, labels, name=name)
            s = tr.quality_summary(q)
            qs[name] = q
            out[name] = dict(summary=s, rows=q.to_dict('records'))
            say(f'{name}: {s["n_links"]} links of {s["n_tracks"]} tracks in {time.time() - t0:.0f} s; '
                f'|chosen - predicted| centroid median {s["centroid_diff_cells"]["median"]:.2f} cells '
                f'({s["centroid_diff_km"]["median"]:.2f} km), p90 {s["centroid_diff_cells"]["p90"]:.1f}, '
                f'max {s["centroid_diff_cells"]["max"]:.1f}; > radius on {s["frac_diff_gt_radius"]:.3f}, '
                f'> 5 cells on {s["frac_diff_gt_5"]:.3f}; mask distance median '
                f'{s["mask_dist_median"]["median"]:.2f} p90 {s["mask_dist_median"]["p90"]:.2f}, '
                f'<= 1 cell on {s["frac_mask_dist_le_1"]:.3f}; IoU median {s["flow_iou"]["median"]:.2f}; '
                f'splits {s["n_split"]}, merges {s["n_merge"]}, either {s["n_split_or_merge"]}; '
                f'chosen is the majority overlap on {s["frac_chosen_is_majority"]:.3f}; '
                f'centroid diff median on unflagged {s["centroid_diff_no_flag"]["median"]:.2f} / '
                f'flagged {s["centroid_diff_flagged"].get("median", float("nan")):.2f}')
            for k, p in s['per_track'].items():
                say(f'  track {k} (label {p["anchor_label"]}): diff median {p["diff_median"]:.2f} max '
                    f'{p["diff_max"]:.1f}; mask dist median {p["mask_dist_median"]:.2f}; IoU '
                    f'{p["flow_iou_median"]:.2f}; splits {p["n_split"]} at {p["split_steps"]}; merges '
                    f'{p["n_merge"]} at {p["merge_steps"]}')
        out['figure'] = f4.fig_tracking_quality(qs['flow'], qs['plain'])
        say(f'figure: {out["figure"]["path"]}')
    except Exception as e:                                   # noqa: BLE001
        status, error = 'error', traceback.format_exc()
        say(f'ERROR: {e!r}')
    out.update(status=status, started=started.isoformat(timespec='seconds'),
               finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(time.time() - t_start, 1), error=error,
               params=dict(split_min_pixels=tr.SPLIT_MIN_PIXELS, radius_cells=tr.RADIUS_CELLS,
                           tracks=str(TRACKS_PATH), flow=str(FLOW_PATH)))
    tr.dump_json(out, OUT_PATH)
    say(f'done: {status}, {time.time() - t_start:.0f} s; {OUT_PATH.name} written')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
