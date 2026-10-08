""" M2 task 2: pull the 72 OSN hours into ``tile330_raw_20120702T00_72h.zarr``.

A thin driver around ``osn_tiles.pull_series`` (M2 task 1), meant to run
**detached**::

    cd dev/frontogenesis/py && nohup ~/miniforge3/envs/frontogenesis/bin/python \
        m2_pull.py > ../data/m2_pull.nohup 2>&1 &

Progress goes to ``data/m2_pull.log`` (appended, one timestamped line per
event: hours present, each hour's wall time, retries, failures, totals) and
to stdout.  When the run ends -- complete, stopped at a failed hour, or by an
exception -- ``data/m2_pull_done.json`` is written with the report, the wall
time and an exit status; the main session watches for that file.  Any stale
done-file is removed at start.

Resumable: ``pull_series`` skips the hours already in the store, so a
re-run after an interruption continues, and a re-run on a complete store is
a no-op that writes a done-file showing 0 pulled.

``--dry-run`` lists the 72 timestamps and what is already present, and exits
without touching the network or the store.
"""

import argparse
import json
import socket
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from osn_tiles import DATA_DIR, open_grid, pull_series        # noqa: E402
import zarr_series as zs                                      # noqa: E402
from dbof.llc4320_ingestion.date_iterations import DATE_FMT   # noqa: E402

# The window (prompt 3): 2012-07-02 00:00 -> 2012-07-04 23:00 UTC, 72 hours
T_START = '2012-07-02 00:00:00'
N_HOURS = 72
OUT_ZARR = DATA_DIR / 'tile330_raw_20120702T00_72h.zarr'
LOG_PATH = DATA_DIR / 'm2_pull.log'
DONE_PATH = DATA_DIR / 'm2_pull_done.json'
GRID_PATH = DATA_DIR / 'tile330_grid.zarr'


def timestamps_72() -> list:
    """The 72 hourly timestamps in dbof format (``'%Y-%m-%d %H:%M:%S'``)."""
    t0 = datetime.strptime(T_START, DATE_FMT)
    ts = [(t0 + timedelta(hours=h)).strftime(DATE_FMT) for h in range(N_HOURS)]
    assert len(ts) == 72, len(ts)
    assert ts[0] == '2012-07-02 00:00:00' and ts[-1] == '2012-07-04 23:00:00', (ts[0], ts[-1])
    return ts


class Progress:
    """Timestamped lines to ``m2_pull.log`` (append) and stdout, flushed on
    every line so a tail of the file is current.  Appends an ETA to the
    per-hour lines from the live report dict."""

    def __init__(self, path: Path, report: dict, n_total: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, 'a')
        self.rep = report
        self.n_total = n_total

    def __call__(self, msg: str):
        if 'pulled and appended' in msg and self.rep.get('wall_s'):
            walls = list(self.rep['wall_s'].values())
            on_disk = len(self.rep['skipped']) + len(self.rep['pulled'])
            left = self.n_total - on_disk
            msg += f' | mean {np.mean(walls):.0f} s/h, ETA ~{left * np.mean(walls) / 60:.0f} min'
        line = f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}  {msg}'
        self.fh.write(line + '\n')
        self.fh.flush()
        print(line, flush=True)

    def close(self):
        self.fh.close()


def write_done(status: str, report: dict, wall_s: float, started: str, n_present: int,
               error: str = None):
    out = dict(status=status, started=started,
               finished=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(wall_s, 1), out_zarr=str(OUT_ZARR), n_requested=N_HOURS,
               n_present=int(n_present), pid=os.getpid(), host=socket.gethostname(),
               report=report, error=error)
    tmp = DONE_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(out, indent=1, default=str))
    tmp.replace(DONE_PATH)                            # atomic: watcher never sees a partial file
    return out


def dry_run(ts: list) -> int:
    present = zs.present_times(OUT_ZARR)
    want = np.array([np.datetime64(datetime.strptime(t, DATE_FMT), 's') for t in ts])
    here = np.isin(want, present)
    print(f'store   : {OUT_ZARR} ({"exists" if OUT_ZARR.exists() else "absent"}, '
          f'{len(present)} hours present)')
    print(f'log     : {LOG_PATH}\ndone    : {DONE_PATH}\ngrid    : {GRID_PATH} '
          f'({"exists" if GRID_PATH.exists() else "ABSENT"})')
    print(f'{len(ts)} timestamps requested, {int(here.sum())} present, '
          f'{int((~here).sum())} to pull:')
    for k, (t, h) in enumerate(zip(ts, here)):
        print(f'  {k:2d}  {t}  {"present" if h else "missing"}')
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
    say(f'===== m2_pull start (pid {os.getpid()}, host {socket.gethostname()}) =====')
    say(f'window {ts[0]} .. {ts[-1]} ({len(ts)} hours) -> {OUT_ZARR}')
    try:
        grid = open_grid(GRID_PATH, with_face=False)          # XC/YC source (stored layout)
        say(f'grid opened from {GRID_PATH.name}: XC/YC {tuple(grid.XC.shape)}')
        pull_series(ts, OUT_ZARR, grid_ds=grid, log=say, report=rep)
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
    say(f'totals: {len(rep.get("pulled", []))} pulled, {len(rep.get("skipped", []))} skipped, '
        f'{len(rep.get("failed", []))} failed, {len(rep.get("not_attempted", []))} not attempted, '
        f'{rep.get("repaired", 0)} repaired; {n_present}/{len(ts)} hours on disk')
    if walls:
        say(f'per-hour wall: median {np.median(walls):.0f} s, min {min(walls):.0f} s, '
            f'max {max(walls):.0f} s over {len(walls)} hours')
    say(f'report: ' + json.dumps({k: v for k, v in rep.items() if k != 'wall_s'}, default=str))
    say(f'===== m2_pull end: status={status}, wall {wall:.0f} s ({wall / 60:.1f} min) =====')
    write_done(status, rep, wall, started, n_present, err)
    say(f'wrote {DONE_PATH}')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
