"""Fetch and regrid ahead of the first page view.

    python -m fronts.viz.apps.warm
    python -m fronts.viz.apps.warm --date 2012-05-16T06_00_00
    python -m fronts.viz.apps.warm --chunk monterey_bay

Every page draws from the same few grid-sized planes, and each one is a
single zarr chunk -- 0.83 GB, with no such thing as a partial read.  Doing
that download while you wait at a prompt is better than doing it while the
browser waits, so run this once before ``serve`` on a new date.

Everything it touches lands in the disk cache, so the pages then start
from local memory-mapped files.

``--chunk`` warms an Evolution window, which is the worst cold start in
the tool.  Front labels are one grid-sized product **per date**, and a
chunk spans ~17 dates, so a first movie build pulls ~15 GB before it can
render anything.  The app caches the 720x720 slice of each, so this is a
one-time cost -- but it is much better spent at a prompt that prints
progress than inside a build that looks hung.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.common import basemap, pyramid, sources

#: What the pages ask for before you have touched a single control:
#: ``gradb2`` is the default field on Surface and the overview map on
#: Tiles and Evolution; the other three are the kinematic roles the joint
#: PDFs need.
DEFAULT_ROLES = ("vorticity", "strain", "coriolis")


def warm(provider, date: str, fields, width: int) -> None:
    started = time.perf_counter()

    def step(label, fn):
        t = time.perf_counter()
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001
            print(f"  skip  {label}: {type(exc).__name__}: {exc}")
            return
        print(f"  ok    {label}  ({time.perf_counter() - t:.0f}s)")

    print(f"{date}  ->  {config.CACHE_DIR}")
    step("coordinates", lambda: provider.coords(date))
    step("land mask", lambda: pyramid.land_level(provider, date, width))
    for name in fields:
        step(name, lambda n=name: pyramid.level(provider, date, n, width))

    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")


def warm_chunk(provider, chunk: str) -> None:
    """Cache the front labels for every step of a chunk window.

    One 0.83 GB label product per date, reduced to a ~2 MB window that is
    kept.  Printed step by step, because the whole point is that the cost
    is visible here instead of inside a movie build.
    """
    times = provider.chunk_timesteps(chunk)
    print(f"{chunk}: {len(times)} steps with fronts")
    print(f"  first pass pulls ~{0.83 * len(times):.0f} GB of label maps; "
          "after this they are local")

    started = time.perf_counter()
    for n, date in enumerate(times, start=1):
        t = time.perf_counter()
        try:
            labels = provider.chunk_labels(chunk, n - 1)
            n_fronts = len(set(int(v) for v in np.unique(labels) if v))
            print(f"  [{n}/{len(times)}] {date}  "
                  f"{n_fronts} fronts  {time.perf_counter() - t:.1f}s")
        except Exception as exc:                            # noqa: BLE001
            print(f"  [{n}/{len(times)}] {date}  "
                  f"FAILED {type(exc).__name__}: {exc}")
    print(f"done in {time.perf_counter() - started:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Timestamp to warm.  Default: the first available.")
    ap.add_argument("--field", action="append", default=None,
                    help="Extra field to warm.  Repeatable.")
    ap.add_argument("--chunk", default=None,
                    help="Warm an Evolution chunk's front labels instead.")
    ap.add_argument("--data", choices=("synthetic", "s3"), default=None,
                    help="Override FRONTS_APP_DATA for this run.")
    args = ap.parse_args(argv)

    if args.data:
        os.environ["FRONTS_APP_DATA"] = args.data
        sources.get_provider.cache_clear()

    provider = sources.get_provider()
    print(f"provider={provider.mode} synthetic={provider.synthetic}")

    if args.chunk:
        warm_chunk(provider, args.chunk)
        return 0

    date = args.date or provider.dates()[0]

    fields = ["gradb2"]
    try:
        fields += list(provider.resolve_channels(date).values())
    except Exception:                                       # noqa: BLE001
        pass
    fields += args.field or []

    seen, ordered = set(), []
    for name in fields:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    warm(provider, date, ordered, basemap._affordable_width(
        config.PYRAMID_WIDTHS[1]))


if __name__ == "__main__":
    sys.exit(main())
