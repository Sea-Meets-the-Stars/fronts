""" M2 task 5: pull k = 0..2 of the 72 chunk-store hours into ``tile330_chunk_20120702T00_72h.zarr``.

A thin driver around ``vertical.load_chunk_levels``, modelled on ``m2_pull.py`` (task 2)
and meant to run **detached** (~6.3 h at the ~0.55 MB/s this machine gets from Nautilus,
~174 MB fetched per hour)::

    cd dev/frontogenesis/py && nohup ~/miniforge3/envs/frontogenesis/bin/python \
        m2_chunk_pull.py > ../data/m2_chunk_pull.nohup 2>&1 &

Progress goes to ``data/m2_chunk_pull.log`` (appended, one timestamped line per event:
hours present, each large object's size and rate, each hour's wall time with an ETA,
retries, failures, repairs, totals) and to stdout.  When the run ends -- complete, stopped
at a failed hour, or by an exception -- ``data/m2_chunk_pull_done.json`` is written with
the report, the wall time and a status; a stale one is removed at start.

Each hour is checked against the OSN store's ``Eta`` (time alignment) and the chunk
``grid.zarr`` against M0's ``tile330_grid.zarr``.  Resumable: re-running continues where
an interrupted run stopped, and is a no-op on a complete store.

``--dry-run`` lists the 72 timestamps and what is already present, and exits without
touching the network or the store.
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vertical as vt                                          # noqa: E402
import zarr_series as zs                                       # noqa: E402
from m2_pull import Progress, timestamps_72, N_HOURS           # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402
from dbof.llc4320_ingestion.date_iterations import DATE_FMT    # noqa: E402

OUT_ZARR = vt.CHUNK_ZARR
LOG_PATH = DATA_DIR / 'm2_chunk_pull.log'
DONE_PATH = DATA_DIR / 'm2_chunk_pull_done.json'
GRID_PATH = DATA_DIR / 'tile330_grid.zarr'
OSN_PATH = vt.OSN_RAW_ZARR
K_MAX = 2


def write_done(status: str, report: dict, wall_s: float, started: str, n_present: int,
               error: str = None):
    by = report.get('bytes', {})
    out = dict(status=status, started=started,
               finished=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(wall_s, 1), out_zarr=str(OUT_ZARR), n_requested=N_HOURS,
               n_present=int(n_present), bytes_fetched=int(sum(by.values())),
               pid=os.getpid(), host=socket.gethostname(), report=report, error=error)
    tmp = DONE_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(out, indent=1, default=str))
    tmp.replace(DONE_PATH)                            # atomic: a watcher never sees a partial file
    return out


def dry_run(ts: list) -> int:
    present = zs.present_times(OUT_ZARR)
    want = np.array([np.datetime64(datetime.strptime(t, DATE_FMT), 's') for t in ts])
    here = np.isin(want, present)
    print(f'store   : {OUT_ZARR} ({"exists" if OUT_ZARR.exists() else "absent"}, '
          f'{len(present)} hours present)')
    print(f'source  : s3://{vt.CHUNK_PREFIX}/{{YYYYMMDDTHH}}.zarr at {vt.CHUNK_ENDPOINT}')
    print(f'log     : {LOG_PATH}\ndone    : {DONE_PATH}')
    for name, p in (('grid', GRID_PATH), ('osn', OSN_PATH)):
        print(f'{name:8s}: {p} ({"exists" if p.exists() else "ABSENT"})')
    print(f'{len(ts)} timestamps requested, {int(here.sum())} present, '
          f'{int((~here).sum())} to pull (~174 MB fetched each):')
    for k, (t, h) in enumerate(zip(ts, here)):
        print(f'  {k:2d}  {t}  {vt._store_name(t):18s} {"present" if h else "missing"}')
    extra = present[~np.isin(present, want)]
    if len(extra):
        print(f'NOTE: store holds {len(extra)} hours outside the window: {extra.astype(str).tolist()}')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true',
                    help='list the timestamps and what is already present; no network, no writes')
    args = ap.parse_args(argv)
    ts = timestamps_72()
    if args.dry_run:
        return dry_run(ts)

    if DONE_PATH.exists():
        DONE_PATH.unlink()
    rep = {}
    say = Progress(LOG_PATH, rep, len(ts))
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    t_start = time.time()
    status, err, n_present = 'error', None, 0
    say(f'===== m2_chunk_pull start (pid {os.getpid()}, host {socket.gethostname()}) =====')
    say(f'window {ts[0]} .. {ts[-1]} ({len(ts)} hours), k=0..{K_MAX} -> {OUT_ZARR}')
    try:
        vt.load_chunk_levels(ts, K_MAX, OUT_ZARR, osn_store=OSN_PATH, local_grid=GRID_PATH,
                             log=say, report=rep)
        n_present = len(zs.present_times(OUT_ZARR))
        status = 'ok' if not rep['failed'] and n_present == len(ts) else 'failed'
    except BaseException as e:                              # noqa: BLE001 -- done-file first
        err = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        say(f'EXCEPTION: {type(e).__name__}: {e}')
        try:
            n_present = len(zs.present_times(OUT_ZARR))
        except Exception:                                   # noqa: BLE001
            pass
        status = 'interrupted' if isinstance(e, KeyboardInterrupt) else 'error'
    wall = time.time() - t_start
    walls = list(rep.get('wall_s', {}).values())
    mb = sum(rep.get('bytes', {}).values()) / 1e6
    say(f'totals: {len(rep.get("pulled", []))} pulled, {len(rep.get("skipped", []))} skipped, '
        f'{len(rep.get("failed", []))} failed, {len(rep.get("not_attempted", []))} not attempted, '
        f'{rep.get("repaired", 0)} repaired; {n_present}/{len(ts)} hours on disk; '
        f'{mb:.0f} MB fetched')
    if walls:
        say(f'per-hour wall: median {np.median(walls):.0f} s, min {min(walls):.0f} s, '
            f'max {max(walls):.0f} s over {len(walls)} hours; mean rate '
            f'{mb / max(sum(walls), 1e-3):.2f} MB/s')
    say('report: ' + json.dumps({k: v for k, v in rep.items() if k not in ('wall_s', 'bytes')},
                                default=str))
    say(f'===== m2_chunk_pull end: status={status}, wall {wall:.0f} s ({wall / 3600:.2f} h) =====')
    write_done(status, rep, wall, started, n_present, err)
    say(f'wrote {DONE_PATH}')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
