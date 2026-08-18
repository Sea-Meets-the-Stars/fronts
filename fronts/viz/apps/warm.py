"""Fetch and regrid ahead of the first page view.

    python -m fronts.viz.apps.warm
    python -m fronts.viz.apps.warm --date 2012-05-16T06_00_00

Every page draws from the same few grid-sized planes, and each one is a
single zarr chunk -- 0.83 GB, with no such thing as a partial read.  Doing
that download while you wait at a prompt is better than doing it while the
browser waits, so run this once before ``serve`` on a new date.

Everything it touches lands in the disk cache, so the pages then start
from local memory-mapped files.
"""

from __future__ import annotations

import argparse
import sys
import time

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
    step("land mask", lambda: pyramid.level(provider, date, "__land__",
                                            width, reduce="any"))
    for name in fields:
        step(name, lambda n=name: pyramid.level(provider, date, n, width))

    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Timestamp to warm.  Default: the first available.")
    ap.add_argument("--field", action="append", default=None,
                    help="Extra field to warm.  Repeatable.")
    args = ap.parse_args(argv)

    provider = sources.get_provider()
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
