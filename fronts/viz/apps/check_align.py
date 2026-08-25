"""Does the tile data line up with the global front products?

A tile is stored **face-local**; the front products are on the **rect**
grid.  The app bridges them with ``build_tile_lookup`` + ``remap_to_rect``
-- remap the tile into rect, then slice the global labels by the rect
window.  If that bridge is wrong on a rotated face the fronts land beside
the features they were detected on, and nothing downstream notices: you
get a plausible-looking figure with the fronts in the wrong place.

So measure it, on land.

Land is the ideal probe.  The tile carries NaNs where the ocean model has
no wet cells, and so does the global field, and both describe the same
coastline.  Get the transform right and the two masks agree almost
exactly.  Get it wrong and they do not -- and *how* they disagree says
which transform was needed.

Every candidate transform is scored, not just the one the app uses, so
the output either confirms the convention or names the fix:

    python -m fronts.viz.apps.check_align --date 2012-05-16T06_00_00 \\
        --region "California Current System"

Agreement is the fraction of cells where "is land" matches.  Read it as:

* **> 0.98** -- aligned.
* **~0.5-0.8** -- suspicious; could be a coincidence on a mostly-ocean
  tile, so try a tile with more coastline before concluding anything.
* **the app's transform scoring below another one** -- that is the bug,
  and the winner names the correction.

A tile that is almost entirely ocean cannot discriminate: with 2% land
every transform scores ~0.96.  The output says so rather than reporting a
confident meaningless number.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


#: Candidates.  The app uses "lookup"; the rest are the transforms a
#: face-rotation bug would produce, so a winner other than "lookup" is
#: both the diagnosis and the fix.
def _candidates(tile_plane, lookup):
    from fronts.viz.apps.tiles import pipeline

    # Whichever of these the app actually uses is labelled as such, so
    # the verdict below never has to guess which row is the app's.
    if lookup is None:
        out = {"identity -- what the app does (no remap here)": tile_plane}
    else:
        out = {"identity (no remap)": tile_plane,
               "lookup -- what the app does": pipeline.remap_to_rect(
                   tile_plane, lookup)}
    out["transpose"] = tile_plane.T
    out["flip j"] = tile_plane[::-1, :]
    out["flip i"] = tile_plane[:, ::-1]
    out["rot90"] = np.rot90(tile_plane)
    out["rot180"] = np.rot90(tile_plane, 2)
    out["rot270"] = np.rot90(tile_plane, 3)
    return out


def _agreement(a_land: np.ndarray, b_land: np.ndarray) -> float:
    if a_land.shape != b_land.shape:
        return float("nan")
    return float(np.mean(a_land == b_land))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--field", default="density",
                    help="tile property to probe (default: density)")
    ap.add_argument("--global-field", default="gradb2",
                    help="global channel whose NaNs mark land")
    # Same flag as serve.py.  Without it this defaults to the synthetic
    # world, where there are no faces and so nothing to get wrong -- i.e.
    # a confident "alignment looks correct" about data nobody asked about.
    ap.add_argument("--data", choices=("synthetic", "s3"), default=None,
                    help="Override FRONTS_APP_DATA for this run.")
    args = ap.parse_args(argv)

    from fronts.viz.apps.common import regions, sources
    from fronts.viz.apps.tiles import pipeline

    if args.data:
        os.environ["FRONTS_APP_DATA"] = args.data
        sources.get_provider.cache_clear()

    provider = sources.get_provider()
    print(f"provider={provider.mode} synthetic={provider.synthetic}")
    if provider.synthetic:
        print("  !! SYNTHETIC DATA -- this says nothing about the real "
              "tiles.\n     The fake grid has no faces, so identity is "
              "correct by construction.\n     Re-run with --data s3.")

    region = regions.by_name(args.region)
    if provider.synthetic:
        idx = regions.synthetic_tile_idx(region)
    else:
        idx = regions.tile_index_for(provider, args.date, region)
    print(f"region={region.name!r} tile={idx}")

    ds = provider.tile(args.date, idx, args.field, region.name)
    var = ds.attrs.get("tile_var_name") or pipeline.sole_field(ds)

    print("\ntile provenance")
    for key in ("face_index", "rect_i_start", "rect_j_start",
                "tile_i_rect", "tile_j_rect", "tile_idx", "resolved_face",
                "i_start", "j_start"):
        if key in ds.attrs:
            print(f"  {key:>14} = {ds.attrs[key]}")

    try:
        lookup = pipeline.tile_lookup(ds, synthetic=provider.synthetic)
    except Exception as exc:                                # noqa: BLE001
        lookup = None
        print(f"\n  !! tile_lookup failed: {type(exc).__name__}: {exc}")
        print("     the app would silently fall back to no remap")

    # The tile's surface plane, and the global field over the same window.
    plane = pipeline.field_values(ds, var)
    plane = plane[0] if plane.ndim == 3 else plane

    js, iss = pipeline.tile_window(ds)
    window = np.asarray(provider.field(args.date, args.global_field)[js, iss])

    ref_land = ~np.isfinite(window)
    frac = float(ref_land.mean())
    print(f"\nglobal {args.global_field} over rect window "
          f"j={js.start}:{js.stop} i={iss.start}:{iss.stop}")
    print(f"  land fraction = {frac:.3f}")

    if frac < 0.05 or frac > 0.95:
        print("\n  !! this tile is almost all one thing, so land cannot "
              "discriminate\n     between transforms -- pick a tile with "
              "coastline in it (a coastal\n     region, not open ocean) "
              "and run again.")

    print(f"\nagreement of tile {args.field!r} land with global land:")
    scores = {}
    for name, candidate in _candidates(plane, lookup).items():
        score = _agreement(~np.isfinite(candidate), ref_land)
        scores[name] = score
        shape = "x".join(str(n) for n in candidate.shape)
        print(f"  {score:6.3f}  {name}   [{shape}]")

    usable = {k: v for k, v in scores.items() if not np.isnan(v)}
    if usable:
        best = max(usable, key=usable.get)
        app_key = next((k for k in usable if "what the app does" in k), None)
        app = usable.get(app_key) if app_key else None
        print(f"\nbest: {best} ({usable[best]:.3f})")
        if app is None:
            print("the app's transform could not be evaluated (see above)")
        elif best == app_key:
            print("the app's convention wins -- alignment looks correct")
        elif usable[best] - app > 0.02:
            print(f"the app scores {app:.3f}, so {best!r} is the correction "
                  "needed -- alignment is WRONG")
        else:
            print("no candidate is clearly better; the probe is not "
                  "discriminating here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
