""" M3 task 7's driver: render the ten figures and record what they drew.

    cd dev/frontogenesis/py && python m3_figs.py            # all ten
    python m3_figs.py --only fig02,fig06                    # a subset
    python m3_figs.py --list

Each ``figures.fig*`` function returns the numbers it drew; they are
collected into ``data/m3_figs_summary.json`` so task 8's audit can check the
figures against ``m3_closure_summary.json`` without re-opening the PNGs.
Nothing here computes physics.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import figures as fg                                           # noqa: E402

OUT_JSON = fg.DATA_DIR / 'm3_figs_summary.json'


def render(only=None, data_dir=None, fig_dir=None, summary=None, say=print) -> dict:
    """Every figure in :data:`figures.FIGURES` (or the subset named), each in
    its own try/except so one broken panel does not cost the other nine."""
    names = list(fg.FIGURES) if not only else [n for n in fg.FIGURES if n in set(only)]
    out, failed = {}, []
    for name in names:
        fn = fg.FIGURES[name]
        t = time.time()
        kw = dict(data_dir=data_dir, fig_dir=fig_dir)
        if 'summary' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            kw['summary'] = summary
        try:
            res = fn(**kw)
            out[name] = dict(res, wall_s=round(time.time() - t, 1))
            say(f'  {name:6s} {time.time() - t:5.1f} s  -> {Path(res["png"]).name}')
        except Exception as e:                               # noqa: BLE001 -- one bad panel
            failed.append(name)
            out[name] = dict(error=f'{type(e).__name__}: {e}',
                             traceback=traceback.format_exc())
            say(f'  {name:6s} FAILED: {type(e).__name__}: {e}')
    out['_failed'] = failed
    out['_n_ok'] = len(names) - len(failed)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', type=lambda s: [x for x in s.split(',') if x])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--out', default=str(OUT_JSON))
    args = ap.parse_args(argv)
    if args.list:
        for n, f in fg.FIGURES.items():
            print(f'{n:6s} {f.__name__:24s} {(f.__doc__ or "").strip().splitlines()[0]}')
        return 0
    print(f'rendering {len(args.only or fg.FIGURES)} figure(s) into {fg.FIG_DIR}')
    res = render(args.only)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f'wrote {args.out}: {res["_n_ok"]}/{len(args.only or fg.FIGURES)} ok'
          + (f', FAILED {res["_failed"]}' if res['_failed'] else ''))
    return 1 if res['_failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
