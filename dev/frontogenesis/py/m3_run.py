""" M3 task 4: the ``L_cells`` sweep -- ``compute_budget`` over every hour pair, every
filter scale, into ``data/tile330_derived_L{L}.zarr``.

A thin driver around ``budget.compute_budget`` / ``write_derived`` /
``closure_report``, modelled on ``m2_chunk_pull.py`` (M2 task 5) and meant to run
**detached** (on this workstation, ``nohup`` inside ``tmux``; ``caffeinate`` is macOS-only
and nothing here suspends -- task 2b)::

    cd dev/frontogenesis/py && nohup /home/xavier/miniconda3/envs/frontogenesis/bin/python \
        m3_run.py > ../data/m3_run.nohup 2>&1 &

For each ``L`` in ``{0, 2, 4, 8}`` (the M3-Q3 (a) contract) and each pair ``t0`` in
``0..70``: one budget, one ``write_derived`` append, one ``closure_report`` cached to
``data/m3_closure_L{L}.json``.  **One pair in memory at a time** -- the pair is loaded,
reduced to the ~23 stored fields and dropped before the next, so RSS stays near the
120 MB a single pair needs rather than growing with the sweep.

**Resumable at pair granularity, in both places.**  ``write_derived`` skips a pair already
in the store (``zarr_series.present_times``), and the closure JSON is keyed by pair index,
so an interrupted run is *relaunched, not debugged* -- and a complete run re-launched is a
no-op that computes nothing.  The two can disagree only if a run is killed between the
append and the JSON write, which the next run notices and repairs by recomputing that one
pair's report (the store is the authority).

Progress goes to ``data/m3_run.log`` (appended, one timestamped line per pair: wall time,
``n_valid``/``n_lost``, the five rms fractions and the verdict, with an ETA) and to stdout.
``data/m3_run_done.json`` is written when the run ends -- complete, stopped or by an
exception -- with the per-``L`` counts, the wall-time statistics and a status; a stale one
is removed at start.

Flags: ``--L 0,2,4,8``, ``--pairs a:b`` (python slice bounds on ``t0``), ``--dry-run``,
``--no-chunk`` (the catch-all comparison of task 6 -- writes to
``tile330_derived_noChunk_L{L}.zarr`` so it can never overwrite the real product),
``--clobber``.

**M3-digging task 1 (2026-10-31):** ``--raw``, ``--chunk`` (a path, or ``none`` for a
surface-only run through ``inputs.open_inputs(chunk=None)``), ``--grid``, ``--masks`` and
``--tag``.  A tag writes ``data/{tag}_derived_L{L}.zarr``, ``m3_closure_{tag}_L{L}.json``,
``m3_run_{tag}.log`` and ``m3_run_{tag}_done.json``; with no tag every name is the product's.
``--clobber`` is **refused** on the product (no ``--tag``, no ``--no-chunk``): the M3 product
is never rewritten from here (the prompt's "Do not").
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

import budget as bg                                            # noqa: E402
import inputs as inp                                           # noqa: E402
import zarr_series as zs                                       # noqa: E402
from m2_pull import timestamps_72                              # noqa: E402
from osn_tiles import DATA_DIR                                 # noqa: E402

L_CONTRACT = bg.L_CONTRACT                                     # (0, 2, 4, 8), M3-Q3 (a)
N_PAIRS = 71                                                   # t0 = 0..70 of the 72 hours
LOG_PATH = DATA_DIR / 'm3_run.log'
DONE_PATH = DATA_DIR / 'm3_run_done.json'
#: the four input paths the sweep opens; ``chunk=None`` is a surface-only run
PRODUCT_INPUTS = dict(raw=inp.RAW_ZARR, chunk=inp.CHUNK_ZARR, grid=inp.GRID_ZARR,
                      masks=inp.MASKS_NC)


def _tagged(name: str, tag=None) -> str:
    """``m3_run.log`` -> ``m3_run_{tag}.log`` etc.; unchanged without a tag."""
    if not tag:
        return name
    stem, _, ext = name.partition('.')
    return f'{stem}_{tag}.{ext}' if ext else f'{stem}_{tag}'


def log_path(tag=None):
    return DATA_DIR / _tagged('m3_run.log', tag)


def done_path(tag=None):
    return DATA_DIR / (f'm3_run_{tag}_done.json' if tag else 'm3_run_done.json')


def closure_path(L, no_chunk=False, tag=None):
    if tag:
        return DATA_DIR / f'm3_closure_{tag}_L{int(L)}.json'
    return DATA_DIR / f'm3_closure{"_noChunk" if no_chunk else ""}_L{int(L)}.json'


def store_path(L, no_chunk=False, tag=None):
    if tag:
        return bg.derived_path(L, tag=tag)
    return (DATA_DIR / f'tile330_derived_noChunk_L{int(L)}.zarr' if no_chunk
            else bg.derived_path(L))


class Progress:
    """Timestamped lines to ``m3_run.log`` (append) and stdout, flushed on
    every line so a tail of the file is current.  Appends a mean rate and an
    ETA to the per-pair lines from the live wall-time list."""

    def __init__(self, path: Path, walls: list, n_total: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, 'a')
        self.walls, self.n_total, self.n_done = walls, n_total, 0

    def __call__(self, msg: str, pair: bool = False):
        if pair:
            self.n_done += 1
            if self.walls:
                mean = float(np.mean(self.walls))
                left = self.n_total - self.n_done
                msg += f' | mean {mean:.1f} s, ETA ~{left * mean / 60:.0f} min'
        line = f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}  {msg}'
        self.fh.write(line + '\n')
        self.fh.flush()
        print(line, flush=True)

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# the per-pair closure cache
# ---------------------------------------------------------------------------
def load_closure(L, no_chunk=False, tag=None) -> dict:
    """``{pair index (str): report}`` from ``m3_closure_L{L}.json``, or empty."""
    p = closure_path(L, no_chunk, tag)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:                      # a run killed mid-write
        return {}


def save_closure(L, reports: dict, no_chunk=False, tag=None):
    """Atomic rewrite, so a watcher (or the next run) never reads a partial file."""
    p = closure_path(L, no_chunk, tag)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(reports, indent=1, default=str))
    tmp.replace(p)


def pair_times(ts, pairs):
    """``(t0, timestamp of t0)`` for the pair indices requested."""
    return [(t0, ts[t0]) for t0 in pairs]


def present_pairs(L, ts, no_chunk=False, tag=None) -> set:
    """Pair indices already in the store, by the ``t0`` timestamp each pair
    is written under."""
    out = store_path(L, no_chunk, tag)
    if not out.exists():
        return set()
    present = set(zs.present_times(out).astype('datetime64[s]').tolist())
    want = [np.datetime64(t.replace(' ', 'T'), 's').astype(object) for t in ts]
    return {k for k, w in enumerate(want) if w in present}


# ---------------------------------------------------------------------------
# one pair
# ---------------------------------------------------------------------------
def run_pair(ds, g, grid, masks, L, t0, no_chunk, clobber, say, tag=None):
    """One budget: compute, append, report.  Returns ``(report, wall_s)``.
    Nothing from the pair survives the call but the report -- the Dataset is
    dropped before the next pair is loaded."""
    t = time.time()
    # chunk_ds stays None throughout: the sweep opens the task-1 *merge*, which already
    # carries the §3.3 variables, so "with the chunk terms" is the default and --no-chunk
    # has to strip them from the dataset (_strip_chunk) rather than withhold an argument
    bd = bg.compute_budget(ds, g, grid, masks, L, t0=t0, chunk_ds=None)
    bg.write_derived(bd, L, out=store_path(L, no_chunk, tag), clobber=clobber)
    rep = bg.closure_report(bd)
    wall = time.time() - t
    p = rep['front_and_valid']
    terms = p['terms']
    say(f'L={L:<2d} pair {t0:2d} ({rep["time_mid"]}, {rep["local_solar_hour"]:4.1f} LST) '
        f'{wall:5.1f} s  n_valid {rep["n_valid"]:,} n_lost {rep["n_lost"]:2d}  '
        + '  '.join(f'{k[:5]} {terms[k]["over_measured"]:.3f}' for k in bg.TERMS if k in terms)
        + f'  resid {p["residual_over_measured"]:.3f}  '
        f'{"CLOSED" if rep["closed"] else ("no-verdict" if rep["closed"] is None else "open")}',
        pair=True)
    return rep, wall


def _strip_chunk(ds):
    """The merged dataset without the §3.3 variables -- what ``--no-chunk``
    feeds ``compute_budget``, so the vertical and surface-flux terms are
    genuinely absent rather than quietly zero."""
    return ds.drop_vars([v for v in bg.CHUNK_VARS if v in ds])


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep(Ls, pairs, no_chunk=False, clobber=False, say=print, walls=None, *, inputs=None,
          tag=None):
    """Every ``(L, pair)`` requested, outer loop over ``L`` so each store is
    finished before the next is opened.  Returns the per-``L`` report dict.
    ``inputs``: the four store paths (default :data:`PRODUCT_INPUTS`;
    ``chunk=None`` opens the surface store alone); ``tag``: the run's store
    and cache names (M3-digging task 1)."""
    ts = timestamps_72()
    inputs = dict(PRODUCT_INPUTS if inputs is None else inputs)
    ds, g, grid, masks = inp.open_inputs(**inputs)
    if ds.attrs.get('chunk') == 'absent':
        say('chunk = none: the surface store alone -- the vertical and surface-flux terms '
            'are ABSENT and every report a no-verdict catch-all (inputs.open_inputs(chunk=None))')
    if no_chunk:
        ds = _strip_chunk(ds)
        say(f'--no-chunk: dropped {list(bg.CHUNK_VARS)} -- the vertical and surface-flux '
            'terms will be ABSENT and every report a no-verdict catch-all')
    say(f'inputs: {ds.sizes["time"]} hours, {len(ds.data_vars)} vars; masks '
        f'{int(masks["mask_analysis"].values.sum()):,} analysis cells')
    out = {}
    for L in Ls:
        reports = load_closure(L, no_chunk, tag)
        have = present_pairs(L, ts, no_chunk, tag)
        todo = [t0 for t0 in pairs if clobber or t0 not in have]
        say(f'--- L = {L}: {len(have)} pair(s) on disk, {len(reports)} cached report(s), '
            f'{len(todo)} to compute -> {store_path(L, no_chunk, tag).name}')
        res = dict(L=int(L), computed=[], skipped=[t0 for t0 in pairs if t0 not in todo],
                   failed=None, wall_s={})
        for t0 in todo:
            rep, wall = run_pair(ds, g, grid, masks, L, t0, no_chunk, clobber, say, tag)
            reports[str(t0)] = rep
            save_closure(L, reports, no_chunk, tag)   # after every pair: resumable
            res['computed'].append(t0)
            res['wall_s'][str(t0)] = round(wall, 1)
            if walls is not None:
                walls.append(wall)
        # a pair in the store with no cached report (killed between the two writes)
        missing = [t0 for t0 in pairs if t0 in present_pairs(L, ts, no_chunk, tag)
                   and str(t0) not in reports]
        if missing:
            say(f'L = {L}: {len(missing)} pair(s) on disk without a cached report {missing} '
                '-- recomputing their reports only (the store is the authority)')
            for t0 in missing:
                rep, _ = run_pair(ds, g, grid, masks, L, t0, no_chunk, False, say, tag)
                reports[str(t0)] = rep
                save_closure(L, reports, no_chunk, tag)
        res['n_present'] = len(present_pairs(L, ts, no_chunk, tag))
        res['n_reports'] = len(reports)
        res['store'] = str(store_path(L, no_chunk, tag))
        res['bytes'] = _du(store_path(L, no_chunk, tag))
        out[str(L)] = res
        say(f'--- L = {L}: {res["n_present"]}/{len(pairs)} pairs on disk, '
            f'{res["n_reports"]} reports, {res["bytes"] / 1e9:.2f} GB')
    return out


def _du(path) -> int:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def _inputs_of(args) -> dict:
    """The four store paths from the flags (``--chunk none`` -> ``None``)."""
    chunk = None if str(args.chunk).lower() == 'none' else Path(args.chunk)
    return dict(raw=Path(args.raw), chunk=chunk, grid=Path(args.grid), masks=Path(args.masks))


def write_done(status, per_L, wall_s, started, args, walls, error=None):
    tag = getattr(args, 'tag', None)
    out = dict(status=status, started=started,
               finished=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(wall_s, 1), L=list(args.L), pairs=[args.pairs[0], args.pairs[-1]],
               n_pairs_requested=len(args.pairs), no_chunk=bool(args.no_chunk),
               clobber=bool(args.clobber), tag=tag,
               inputs={k: (None if v is None else str(v)) for k, v in _inputs_of(args).items()},
               per_L=per_L,
               n_computed=int(sum(len(v['computed']) for v in per_L.values())),
               bytes_total=int(sum(v['bytes'] for v in per_L.values())),
               wall_per_pair=(dict(n=len(walls), median=round(float(np.median(walls)), 2),
                                   min=round(min(walls), 2), max=round(max(walls), 2),
                                   mean=round(float(np.mean(walls)), 2)) if walls else None),
               pid=os.getpid(), host=socket.gethostname(), error=error)
    dp = done_path(tag)
    tmp = dp.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(out, indent=1, default=str))
    tmp.replace(dp)                                   # atomic
    return out


def dry_run(args) -> int:
    ts = timestamps_72()
    tag = args.tag
    ins = _inputs_of(args)
    print(f'L        : {list(args.L)}  (contract {list(L_CONTRACT)}, M3-Q3 (a))')
    print(f'pairs    : {args.pairs[0]}..{args.pairs[-1]} ({len(args.pairs)} of {N_PAIRS})')
    print(f'tag      : {tag or "(none: the product)"}')
    print(f'chunk    : ' + ('ABSENT (--no-chunk: the merge, stripped)' if args.no_chunk else
                           'ABSENT (--chunk none: the surface store alone)' if ins['chunk'] is None
                           else 'present'))
    print(f'log      : {log_path(tag)}\ndone     : {done_path(tag)}')
    for k, p in ins.items():
        print(f'input    : {k:5s} ' + ('none' if p is None else
                                       f'{p} ({"exists" if Path(p).exists() else "ABSENT"})'))
    total = 0
    for L in args.L:
        have = present_pairs(L, ts, args.no_chunk, tag)
        todo = [t0 for t0 in args.pairs if args.clobber or t0 not in have]
        n = _du(store_path(L, args.no_chunk, tag))
        total += len(todo)
        print(f'  L = {L:<2d} {store_path(L, args.no_chunk, tag).name:34s} '
              f'{len(have):3d} present, {len(todo):3d} to compute, '
              f'{len(load_closure(L, args.no_chunk, tag)):3d} cached reports, {n / 1e9:.2f} GB, '
              f'closure {closure_path(L, args.no_chunk, tag).name}')
    print(f'{total} pair-budgets to compute')
    return 0


def _parse_pairs(s):
    a, _, b = s.partition(':')
    lo = int(a) if a else 0
    hi = int(b) if b else N_PAIRS
    out = list(range(max(lo, 0), min(hi, N_PAIRS)))
    if not out:
        raise argparse.ArgumentTypeError(f'--pairs {s!r} selects nothing of 0..{N_PAIRS - 1}')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--L', default=','.join(str(L) for L in L_CONTRACT),
                    type=lambda s: [int(x) for x in s.split(',') if x != ''],
                    help='filter scales, comma separated (default the 0,2,4,8 contract)')
    ap.add_argument('--pairs', default=f'0:{N_PAIRS}', type=_parse_pairs,
                    help=f'pair range as a:b, python slice bounds on t0 in 0..{N_PAIRS - 1}')
    ap.add_argument('--dry-run', action='store_true',
                    help='say what would be computed and exit; no reads, no writes')
    ap.add_argument('--no-chunk', action='store_true',
                    help='drop the chunk variables: the vertical and surface-flux terms are '
                         'ABSENT and every report a no-verdict catch-all (task 6 comparison). '
                         'Writes to tile330_derived_noChunk_L{L}.zarr, never the real store')
    ap.add_argument('--clobber', action='store_true',
                    help='recompute pairs already on disk (truncates the store from each). '
                         'REFUSED on the product (no --tag, no --no-chunk)')
    ap.add_argument('--raw', default=str(inp.RAW_ZARR), help='the §3.2 surface store')
    ap.add_argument('--chunk', default=str(inp.CHUNK_ZARR),
                    help='the §3.3 chunk store, or "none" for a surface-only run (the vertical '
                         'and surface-flux terms ABSENT; M3-digging task 1)')
    ap.add_argument('--grid', default=str(inp.GRID_ZARR), help='the §3.1 static grid')
    ap.add_argument('--masks', default=str(inp.MASKS_NC), help='the §3.5 masks')
    ap.add_argument('--tag', default=None,
                    help='name this run: writes data/{tag}_derived_L{L}.zarr, '
                         'm3_closure_{tag}_L{L}.json, m3_run_{tag}.log, m3_run_{tag}_done.json. '
                         'Without it every name is the product\'s')
    args = ap.parse_args(argv)
    if args.tag is not None and (not args.tag or any(c in args.tag for c in '/ .')):
        ap.error(f'--tag {args.tag!r}: use a plain name (no slash, space or dot)')
    if args.clobber and not args.tag and not args.no_chunk:
        ap.error('--clobber on the product is refused (no --tag): the M3 product stores are '
                 'never rewritten from here (prompt M3-digging, "Do not"). Give the run a --tag.')
    if args.dry_run:
        return dry_run(args)

    tag = args.tag
    if done_path(tag).exists():
        done_path(tag).unlink()
    walls = []
    say = Progress(log_path(tag), walls, len(args.L) * len(args.pairs))
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    t_start = time.time()
    per_L, status, err = {}, 'error', None
    say(f'===== m3_run start (pid {os.getpid()}, host {socket.gethostname()}) =====')
    ins = _inputs_of(args)
    say(f'L = {args.L}, pairs {args.pairs[0]}..{args.pairs[-1]} '
        f'({len(args.L) * len(args.pairs)} pair-budgets), '
        f'chunk {"ABSENT" if (args.no_chunk or ins["chunk"] is None) else "present"}, '
        f'clobber {args.clobber}, tag {tag or "(none: the product)"}')
    say('inputs: ' + ', '.join(f'{k}={v}' for k, v in ins.items()))
    try:
        per_L = sweep(args.L, args.pairs, args.no_chunk, args.clobber, say, walls,
                      inputs=ins, tag=tag)
        done = all(v['n_present'] >= len(args.pairs) for v in per_L.values())
        status = 'ok' if done else 'incomplete'
    except BaseException as e:                              # noqa: BLE001 -- done-file first
        err = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        say(f'EXCEPTION: {type(e).__name__}: {e}')
        status = 'interrupted' if isinstance(e, KeyboardInterrupt) else 'error'
    wall = time.time() - t_start
    if walls:
        say(f'per-pair wall: median {np.median(walls):.1f} s, min {min(walls):.1f} s, '
            f'max {max(walls):.1f} s over {len(walls)} budgets')
    say(f'totals: {sum(len(v["computed"]) for v in per_L.values())} computed, '
        f'{sum(len(v["skipped"]) for v in per_L.values())} skipped, '
        f'{sum(v["bytes"] for v in per_L.values()) / 1e9:.2f} GB on disk')
    say(f'===== m3_run end: status={status}, wall {wall:.0f} s ({wall / 3600:.2f} h) =====')
    write_done(status, per_L, wall, started, args, walls, err)
    say(f'wrote {done_path(tag)}')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
