"""M4 task 1 runner: finding config D on every pair of a derived store ->
``data/tile330_fronts_D_L{L}.zarr`` (coding §4.10 ``find_fronts_series``), with a progress
log and a done-file.

Run detached (the 71 pairs take ~12 min at 8 workers; ``nohup`` inside ``tmux``):

    cd dev/frontogenesis/py && nohup /home/xavier/miniconda3/envs/frontogenesis/bin/python \\
        m4_find.py --L 0 > ../data/m4_find.nohup 2>&1 &

Progress goes to ``data/m4_find.log`` (one line per pair: fronts, pixels, wall time);
``data/m4_find_done.json`` is written when the run ends, with the per-pair counts
(``tracking.series_stats``) the log entry quotes.  Config D's ``'pool'`` threshold is a
``ProcessPoolExecutor``, hence the ``__main__`` guard.

``--mask analysis`` (default) is ``mask_analysis`` -- the M3 pool, applied as NaN.
``--mask halo`` is ``mask_halo`` (ocean minus the 7-cell coastal halo, 100 km offshore
included) -- a sensitivity, not the primary.
"""
import argparse
import datetime as dt
import pathlib
import sys
import time
import traceback

import numpy as np
import xarray as xr

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget                                                       # noqa: E402
import masking as mk                                                # noqa: E402
import tracking as tr                                               # noqa: E402

DATA_DIR = tr.DATA_DIR
LOG_PATH = DATA_DIR / 'm4_find.log'
DONE_PATH = DATA_DIR / 'm4_find_done.json'
MASKS = {'analysis': 'mask_analysis', 'halo': 'mask_halo'}


class Log:
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, msg):
        line = f'{dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")} {msg}'
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _parse_pairs(s):
    if ':' in s:
        a, b = s.split(':')
        return list(range(int(a or 0), int(b)))
    return [int(x) for x in s.split(',')]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', type=int, default=0, help='filter scale of the derived store (0)')
    ap.add_argument('--config', default='D')
    ap.add_argument('--mask', choices=sorted(MASKS), default='analysis')
    ap.add_argument('--n-workers', type=int, default=tr.N_WORKERS)
    ap.add_argument('--pairs', type=_parse_pairs, default=None, help='e.g. 0:71 or 0,35,70')
    ap.add_argument('--out', default=None, help='override the output zarr path')
    ap.add_argument('--clobber', action='store_true')
    args = ap.parse_args(argv)

    say = Log(LOG_PATH)
    started = dt.datetime.now(dt.timezone.utc)
    t_start = time.time()
    src = budget.derived_path(args.L)
    out = tr.fronts_path(args.L, args.config, args.out)
    if out.exists() and not args.clobber:
        say(f'{out} exists and --clobber not given; nothing done')
        return 1
    say(f'start: L = {args.L}, config {args.config}, mask {args.mask}, n_workers {args.n_workers}, '
        f'source {src.name} -> {out.name}')
    cfg = tr.load_finding_config(args.config)
    say(f'config {args.config} (read from {tr.config_path(args.config).name}): {cfg}')
    status, error, rows = 'ok', None, []
    try:
        ds = xr.open_zarr(str(src))
        masks = mk.open_masks()
        mask = masks[MASKS[args.mask]].values
        say(f'mask {MASKS[args.mask]}: {int(mask.sum())} cells retained; {ds.sizes["time"]} pairs '
            f'in the store')
        labels = tr.find_fronts_series(ds, mask, config=args.config, n_workers=args.n_workers,
                                       pairs=args.pairs, log=say)
        idx = range(ds.sizes['time']) if args.pairs is None else args.pairs
        fronts_p90 = [ds['front'].isel(time=int(t)).values for t in idx]
        rows = tr.series_stats(labels, fronts_p90)
        # every labelled pixel must be finite G and inside the mask, at every pair
        bad = 0
        for t, (k, lab) in zip(idx, labels.items()):
            G = ds['G'].isel(time=int(t)).values
            bad += int(((lab > 0) & ~(np.isfinite(G) & mask)).sum())
        say(f'labelled pixels off the finite masked field: {bad}')
        attrs = dict(config=args.config, config_values=str(cfg), mask=MASKS[args.mask],
                     n_workers=args.n_workers, L_cells=args.L, source=str(src),
                     finder='fronts.finding.algorithms.fronts_from_gradb2 + caller-side task-7a '
                            'rules (explicit n_workers; despur only on a non-empty skeleton; '
                            'fronts &= isfinite(G))',
                     mask_applied_as='NaN (never zero: the local percentile is over finite cells)',
                     field='G at the midpoint time of each pair (derived store, §3.4)',
                     n_bad_pixels=bad)
        tr.write_fronts_series(labels, out, like=ds, attrs=attrs, clobber=args.clobber)
        say(f'wrote {out} ({len(labels)} times)')
    except Exception as e:                                   # noqa: BLE001
        status, error = 'error', traceback.format_exc()
        say(f'ERROR: {e!r}')
    wall = time.time() - t_start
    n = np.array([r['n_fronts'] for r in rows]) if rows else np.zeros(0)
    px = np.array([r['n_pixels'] for r in rows]) if rows else np.zeros(0)
    fr = np.array([r['frac_on_p90'] for r in rows]) if rows else np.zeros(0)
    summary = dict(status=status, started=started.isoformat(timespec='seconds'),
                   finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                   wall_s=round(wall, 1), L=args.L, config=args.config, config_values=cfg,
                   mask=MASKS[args.mask], n_workers=args.n_workers, out=str(out),
                   n_times=len(rows),
                   n_fronts=dict(min=int(n.min()), median=float(np.median(n)), max=int(n.max()))
                   if n.size else None,
                   n_pixels=dict(min=int(px.min()), median=float(np.median(px)), max=int(px.max()))
                   if px.size else None,
                   frac_on_p90=dict(min=round(float(fr.min()), 4), median=round(float(np.median(fr)), 4),
                                    max=round(float(fr.max()), 4)) if fr.size else None,
                   per_pair=rows, error=error)
    tr.dump_json(summary, DONE_PATH)
    say(f'done: {status}, {wall:.0f} s; {DONE_PATH.name} written')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
