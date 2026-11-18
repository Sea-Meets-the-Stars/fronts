""" M5 task 1: the one command that regenerates every figure of the study.

    cd dev/frontogenesis/py
    PY=/home/xavier/miniconda3/envs/frontogenesis/bin/python
    $PY make_figs.py                     # all 20 registry figures -> figs/
    $PY make_figs.py --report            # ... and the report's own -> report/figs/
    $PY make_figs.py --only fig08,figV6  # a subset
    $PY make_figs.py --recompute         # re-run the M4 runners from the stores first (~71 min)
    $PY make_figs.py --list

**What it draws.**  Every entry of :data:`figures.FIGURES`, in order: Figures
1, 2, 2b, 3, 3b, 4, 5, 6, 7, 8, 9, 10, the tracking-quality diagnostic, and
V1, V2, V3, V3b, V4, V5, V6.  Each returns the numbers it drew, with ``png``,
``L`` and ``mask`` keys; they are collected in ``data/figs_summary.json`` in
``m3_figs_summary.json``'s form.  ``--report`` then runs
``report/make_report_figs.py`` (two schematics) and
``report/make_case_figs.py`` (the case-study figures).  Deck figures stay
with the deck builds (M5-Q3 (a)).

**What it reads (M5-Q3 (a), JXP 2026-11).**  By default the *products*: the
four derived stores ``tile330_derived_L{0,2,4,8}.zarr`` and
``m3_closure_summary.json`` (Figures 1-7, 10); ``m4_strength.json``,
``m4_quality.json``, ``m4_tracks.json``, the label zarr and the ``L = 0``
store (Figures 8, 9, the diagnostic); the grid, the masks and the 2 h raw
store (V2, V3, V3b, V6); nothing at all for V1, V4, V5.  Nothing is
recomputed except the V-figures, whose gates *are* the computation.

``--recompute`` first re-runs the five M4 runners from the stores, in order
-- ``m4_find.py --clobber`` (551 s as logged), ``m4_track.py --clobber-flow``
(1,481 s), ``m4_recon.py`` (1,276 s), ``m4_quality.py`` (221 s),
``m4_strength.py`` (729 s) -- so Figures 8, 9 and the diagnostic are
regenerated from the stores alone.  **The M3 closure is not re-run by this
command**: ``m3_closure_summary.json`` and the derived stores are M3's
products (``m3_run.py``, ``m3_closure.py``), and ``m3_figs.py`` remains the
provenance of ``m3_figs_summary.json`` (prompt 4 task 8's audit).

Each figure runs in its own try/except (``m3_figs.render``); the exit status
is non-zero if any figure, runner or report script fails.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import figures as fg                                           # noqa: E402
import m3_figs                                                 # noqa: E402

OUT_JSON = fg.DATA_DIR / 'figs_summary.json'
REPORT_DIR = HERE.parent / 'report'

#: the M4 runners ``--recompute`` re-runs, in dependency order, with the
#: flags that let them overwrite their own products and the wall time logged
#: in prompt 5 (s)
M4_RUNNERS = (('m4_find.py', ('--clobber',), 551),
              ('m4_track.py', ('--clobber-flow',), 1481),
              ('m4_recon.py', (), 1276),
              ('m4_quality.py', (), 221),
              ('m4_strength.py', (), 729))
#: ``--report``: the report's own figure scripts, run where they live
REPORT_SCRIPTS = ('make_report_figs.py', 'make_case_figs.py')


def run_scripts(scripts, cwd, say=print, run=subprocess.run) -> list:
    """Run ``(script, args, expected_s)`` rows in order with this interpreter;
    stop at the first failure (each M4 runner needs the one before it).
    Returns ``[{script, returncode, wall_s}]``."""
    rows = []
    for script, args, expected in scripts:
        t = time.time()
        say(f'  {script} {" ".join(args)}'.rstrip()
            + (f'   (logged: {expected} s)' if expected else ''))
        r = run([sys.executable, script, *args], cwd=str(cwd))
        rows.append(dict(script=script, args=list(args), returncode=int(r.returncode),
                         wall_s=round(time.time() - t, 1)))
        say(f'    -> exit {r.returncode}, {time.time() - t:.0f} s')
        if r.returncode:
            break
    return rows


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', type=lambda s: [x for x in s.split(',') if x],
                    help='comma-separated registry names (see --list)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--report', action='store_true',
                    help='also run report/make_report_figs.py and report/make_case_figs.py')
    ap.add_argument('--recompute', action='store_true',
                    help='re-run the five M4 runners from the stores first (~71 min)')
    ap.add_argument('--out', default=str(OUT_JSON))
    ap.add_argument('--fig-dir', default=None, help='default: dev/frontogenesis/figs')
    args = ap.parse_args(argv)
    if args.list:
        for n, f in fg.FIGURES.items():
            print(f'{n:7s} {f.__name__:26s} {(f.__doc__ or "").strip().splitlines()[0]}')
        return 0
    unknown = [n for n in (args.only or []) if n not in fg.FIGURES]
    if unknown:
        ap.error(f'unknown figure(s) {unknown}; see --list')
    t0, bad = time.time(), False
    res = {}
    if args.recompute:
        print('recomputing the M4 products from the stores')
        res['_recompute'] = run_scripts(M4_RUNNERS, HERE, run=run)
        if res['_recompute'][-1]['returncode']:
            print(f'FAILED: {res["_recompute"][-1]["script"]}; no figure drawn')
            return 1
    n = len(args.only or fg.FIGURES)
    print(f'rendering {n} figure(s) into {args.fig_dir or fg.FIG_DIR}')
    res.update(m3_figs.render(args.only, fig_dir=args.fig_dir, names=list(fg.FIGURES)))
    bad |= bool(res['_failed'])
    if args.report:
        print(f'report figures into {REPORT_DIR / "figs"}')
        res['_report'] = run_scripts([(s, (), None) for s in REPORT_SCRIPTS], REPORT_DIR, run=run)
        bad |= any(r['returncode'] for r in res['_report']) or len(res['_report']) < len(REPORT_SCRIPTS)
    res['_wall_s'] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f'wrote {args.out}: {res["_n_ok"]}/{n} ok in {res["_wall_s"]:.0f} s'
          + (f', FAILED {res["_failed"]}' if res['_failed'] else '')
          + (', a report script FAILED' if bad and not res['_failed'] else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
