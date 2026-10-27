"""Walk the Tiles page's data path in a terminal, with real tracebacks.

    python -m fronts.viz.apps.check_tiles
    python -m fronts.viz.apps.check_tiles --region "Gulf Stream" --field Ri

The page catches every exception and turns it into a small card, which is
the right thing for a browser and the wrong thing for debugging.  This runs
the same steps in order and stops at the first one that fails, printing the
traceback and the timing.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.common import regions as regions_mod
from fronts.viz.apps.common import sources
from fronts.viz.apps.tiles import pipeline as TP


def step(label, fn):
    t = time.perf_counter()
    try:
        out = fn()
    except Exception:                                       # noqa: BLE001
        print(f"  FAIL  {label}  ({time.perf_counter() - t:.0f}s)\n")
        traceback.print_exc()
        sys.exit(1)
    print(f"  ok    {label}  ({time.perf_counter() - t:.0f}s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--field", default="Ri")
    args = ap.parse_args(argv)

    provider = sources.get_provider()
    print(f"provider={provider.mode}")

    dates = provider.dates_with_fronts(provider.dates_3d())
    print(f"3-D dates: {provider.dates_3d()}")
    print(f"with fronts: {dates}")
    date = args.date or (dates or provider.dates_3d())[0]

    region = (regions_mod.by_name(args.region) if args.region
              else regions_mod.REGIONS[0])
    print(f"\n{region.name}  {date}  field={args.field}")

    from fronts.viz.apps.common.state import TilesState
    state = TilesState(provider=provider, region=region.name, date=date)
    idx = step("resolve region -> tile index", state.tile_index)
    print(f"        tile {idx}")

    ds = step(f"generate density tile {idx}",
              lambda: provider.tile(date, idx, "density", region.name))
    print(f"        dims {dict(ds.sizes)}")
    print(f"        attrs {[k for k in ds.attrs]}")

    print(f"        rect origin {TP.rect_origin(ds)}")
    lookup = step("build face lookup",
                  lambda: TP.tile_lookup(ds, synthetic=provider.synthetic))
    print("        lookup none -- synthetic tile, already in the rect frame"
          if lookup is None else f"        lookup {lookup[0].shape}")

    labels = step("labels for this window",
                  lambda: TP.tile_labels(provider, date, idx,
                                         (config.TILE_SIZE,) * 2, ds=ds))
    n = int((labels > 0).sum())
    print(f"        {n:,} front pixels, {len(np.unique(labels)) - 1} labels")

    available = step("fronts with 25+ pixels",
                     lambda: TP.available_fronts(labels))
    print(f"        {available[:10]}")
    if not available:
        print("\nNo front in this tile is big enough to draw.  Try another "
              "region -- the figures need a front with a real main axis.")
        return 1

    label = available[0]
    scene = step(f"build scene for label {label}",
                 lambda: TP.build_scene(provider, date, idx, args.field,
                                        label))
    print(f"        sigma0 {scene.sigma0.shape}  Z {scene.Z.shape}  "
          f"axis {len(scene.axis_path)} points")
    print("\nEverything the Tiles page needs is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
