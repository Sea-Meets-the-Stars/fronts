"""Check the real-data path piece by piece.

    python -m fronts.viz.apps.check_s3

Each line is one thing the app needs.  Run it before serving with
FRONTS_APP_DATA=s3 -- a failure here says which store is wrong, rather
than surfacing later as an empty page.
"""

from __future__ import annotations

import sys
import traceback

from fronts.viz.apps import config


def check(label, fn, pending_ok=False):
    from fronts.viz.apps.common.sources import NotWiredUp
    try:
        result = fn()
    except NotWiredUp as exc:
        tag = "pend" if pending_ok else "FAIL"
        print(f"  {tag}  {label}: {str(exc).splitlines()[0]}")
        return None
    except Exception as exc:
        print(f"  FAIL  {label}\n          {type(exc).__name__}: {exc}")
        return None
    print(f"  ok    {label}: {result}")
    return result


def _ls(folder):
    from fronts.viz.apps.common.s3source import _filesystems
    _, fs_sync = _filesystems()
    return sorted(p.rsplit("/", 1)[-1]
                  for p in fs_sync.ls(f"{config.S3_BUCKET}/{folder}"))


def main(argv=None):
    verbose = "-v" in (argv or sys.argv[1:])
    from fronts.viz.apps.common.s3source import S3Provider

    print(f"endpoint {config.S3_ENDPOINT}  bucket {config.S3_BUCKET}")

    print("\n2-D grid "
          f"(s3://{config.S3_BUCKET}/{config.GRID_FOLDER}/{config.GRID_STORE})")
    surf = S3Provider("SURF")
    check("XC/YC shape", lambda: surf.coords("")[0].shape)
    check("land fraction", lambda: round(float(surf.land_mask("").mean()), 3))

    for pipeline, label in (("SURF", "surface"), ("DEPTH", "depth")):
        p = S3Provider(pipeline)
        print(f"\n{label} fields "
              f"(s3://{config.S3_BUCKET}/{p.folder}/{p.run_id})")
        dates = check("dates", lambda p=p: len(p.dates()))
        if not dates:
            check(f"run ids under {p.folder}", lambda p=p: _ls(p.folder))
            continue
        d = p.dates()[0]
        names = check(f"channels at {d}", lambda p=p, d=d: len(p.field_names(d)))
        if names:
            if verbose:
                print(f"          {p.field_names(d)}")
            check("roles resolved",
                  lambda p=p, d=d: p.resolve_channels(d))
            first = p.field_names(d)[0]
            check(f"read {first}",
                  lambda p=p, d=d, f=first: p.field(d, f).shape)

    print("\nfront products (build_v5 steps 2-4)")
    probe_date = config.DATES_3D[0] if config.DATES_3D else config.DEFAULT_DATE
    for method in ("front_binary", "labels", "geometry", "colocation"):
        check(method, lambda m=method: getattr(surf, m)(probe_date),
              pending_ok=True)

    print("\nchunks "
          f"(s3://{config.S3_BUCKET}/{config.CHUNK_FOLDER})")
    names = check("chunk names", lambda: surf.chunks())
    if names:
        c = names[0]
        steps = check(f"timesteps in {c}",
                      lambda c=c: len(surf.chunk_timesteps(c)))
        if steps:
            check("first timestep",
                  lambda c=c: surf.chunk_timesteps(c)[0])
            check("tile from chunk (density)",
                  lambda c=c: dict(surf.chunk_tile(c, 0, "density").sizes))

    print("\n3-D tile from RAW/DEPTH")
    if config.DATES_3D:
        d3 = config.DATES_3D[0]
        check(f"tile 330 density at {d3}",
              lambda: dict(surf.tile(d3, 330, "density").sizes))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
