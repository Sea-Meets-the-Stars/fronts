"""Pre-generate 3-D tiles into the S3 tile store.

    python -m fronts.viz.apps.build_tiles
    python -m fronts.viz.apps.build_tiles --date 2012-05-16T06_00_00
    python -m fronts.viz.apps.build_tiles --field Ri --field N2 --region "Gulf Stream"
    python -m fronts.viz.apps.build_tiles --all-fields

Run it on profx, next to the raw depth stores.  Each tile is one dask pass
over ``LLC4320_RAW/DEPTH`` -- about 15 s there -- and lands at

    s3://dbof/tiles/{YYYYMMDD_HHMMSS}/{region}/{field}.zarr

Tiles that already exist are skipped, so the command is safe to re-run and
safe to interrupt.  ``density`` is always built: the 3-D geometry comes
from it whatever field is being coloured.
"""

from __future__ import annotations

import argparse
import sys
import time

from fronts.viz.apps import config
from fronts.viz.apps.common import regions as regions_mod
from fronts.viz.apps.common import sources, tilestore


def _plan(provider, dates, regions, fields):
    """Every (date, region, field) still missing from the store."""
    todo, have = [], 0
    for date in dates:
        for region in regions:
            for field in fields:
                if tilestore.exists(date, region.name, field):
                    have += 1
                else:
                    todo.append((date, region, field))
    return todo, have


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", action="append", default=None,
                    help="Timestamp to build.  Repeatable.  "
                         "Default: every 3-D date.")
    ap.add_argument("--region", action="append", default=None,
                    help="Region name.  Repeatable.  Default: all six.")
    ap.add_argument("--field", action="append", default=None,
                    help="Field name.  Repeatable.")
    ap.add_argument("--all-fields", action="store_true",
                    help="Every field in TILE_FIELDS_3D (~39, hours).")
    ap.add_argument("--clobber", action="store_true",
                    help="Rebuild tiles that already exist.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be built, then stop.")
    args = ap.parse_args(argv)

    provider = sources.get_provider()
    if provider.synthetic:
        print("provider is synthetic -- set FRONTS_APP_DATA=s3", file=sys.stderr)
        return 2

    dates = args.date or provider.dates_3d()
    regions = ([regions_mod.by_name(r) for r in args.region]
               if args.region else list(regions_mod.REGIONS))

    if args.all_fields:
        fields = list(config.TILE_FIELDS_3D)
    else:
        fields = list(args.field or config.TILE_STORE_DEFAULT_FIELDS)
    if config.TILE_GEOMETRY_FIELD not in fields:
        fields.insert(0, config.TILE_GEOMETRY_FIELD)

    print(f"dates   {dates}")
    print(f"regions {[r.name for r in regions]}")
    print(f"fields  {fields}")
    print(f"store   s3://{config.S3_BUCKET}/{config.TILE_STORE_FOLDER}/\n")

    if args.clobber:
        todo = [(d, r, f) for d in dates for r in regions for f in fields]
        have = 0
    else:
        todo, have = _plan(provider, dates, regions, fields)

    print(f"{len(todo)} to build, {have} already there "
          f"(~{len(todo) * 15 // 60} min at 15 s each)\n")
    if args.dry_run or not todo:
        for date, region, field in todo:
            print(f"  would build {date} {region.name} {field}")
        return 0

    started = time.perf_counter()
    built = failed = 0
    for n, (date, region, field) in enumerate(todo, 1):
        tag = f"[{n}/{len(todo)}] {date} {region.name} {field}"
        t = time.perf_counter()
        try:
            idx = regions_mod.tile_index_for(provider, date, region)
            ds = provider.tile(date, idx, field, region.name)
            written = tilestore.write(ds, date, region.name, field,
                                      clobber=args.clobber)
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f"{tag}\n    FAIL {type(exc).__name__}: {exc}")
            continue
        built += 1
        where = "already stored" if written is None else "stored"
        print(f"{tag}  {where}  ({time.perf_counter() - t:.0f}s)")

    print(f"\n{built} built, {failed} failed, "
          f"{(time.perf_counter() - started) / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
