# Frontogenesis prompt doc

## Objective

Explore frontogenesis for the LLC4320 model.  In particular, compare the measured strengthening of fronts compared to the frontogenesis rate calculated from the model's velocity and buoyancy fields.

## Context

- All of the code in this repository 
- All of the code in Oceanography/python/llc4320-native-grid-preprocessing

Lauren tells me: 

"I would do it by applying the tiles code to the surface-only fields using the 'OSN' fields (OSN referring to the ones in Spencer's store).

I had started doing this a bit in the branch tiles-surface-only in the llc4320-preprocessing repo, which modified the tiles slightly so that we could make a surface only tile (yes descriptive branch name :wink:).

Then working in the viz_tools branch in the fronts repo should let you make the curtains, etc

In the fronts repo I think running build_v5.py with the run_v5_gulf_stream_chunks.yaml config (might need to change the timestamps) should run the front finding on the gulf stream chunk, although front finding may need to be altered to run on a chunk instead of full global map
"""

## Prompts

### Planning

1. Examine the context files and objective.  Have a conversation with me on how to plan our effort.  The analysis should include figures to help me understand the results.  Use Fable if you can.  Log your work below.

2. I have read your comments and then answered the questions you posed.  Please read my answers and then let's have another round of conversation.  Put your comments and questions in the Q&A/Planning section. Log your work below. Use Fable if you can.

3. I have read your comments and then answered the questions you posed.  Please read my answers and decide if we need another round of conversation.  If so, put your comments and questions in the Q&A/Planning section. 
Log your work below. Use Fable if you can.

4. Ok, I think we are ready to create a planning doc.  Let's name it `frontogenesis_planning.md`.  Do so.
Log your work below. Use Fable if you can.

5. Ok, that planning doc looks great.  Now generate a separate coding doc named `frontogenesis_coding.md`.  With key milestones (presumably matching the Phases).  We will then use that to create a series of prompt docs for execution.
Log your work below. Use Fable if you can.

6. Can you now generate a Google Slide deck that summarizes the planning?  
Put it on the AIOcean Drive under `data/HIINet/Frontogenesis/`.  Name it `Frontogenesis_Planning`.  
Use Fable if you can.  Log your work

7. Lauren has provided comments on the planning. I have put them in the file `lauren_comments_on_planning.txt`.  Please read them and then let's have a conversation.  Put your comments and questions in the Q&A/Planning section.  Log your work below. Use Fable if you can.

8. I have answered your qeustions Q13-Q15.  Please read my answers, update the planning docs and then proceed to create a series of prompt docs for execution.
Name them `frontogenesis_prompt_1.md`, `frontogenesis_prompt_2.md`, etc.  Log your work below. Use Fable if you can.

9. Modify the `Frontogenesis_Planning.pptx` to:
    - Have no fonts with size less than 20pt.
    - Add a glossary that defines the primary terms used in the document.
    - Include a slide describing how we will track a given front from one hour to the next.
    - Add a version number and bump it to v2
Use Opus 5.5 and log your work below.

## Q&A

### Planning

#### Claude, 2026-09-11 (prompt Planning-1)

**Where we stand.** The *predicted* side is already built and built well. The
*measured* side does not exist yet. That is the whole of the new work.

Already present:

- `frontogenesis_tendency` / `_geo` / `_ageo` on the native grid —
  `llc4320-native-grid-preprocessing/src/dbof/preprocessing/calculate_additional_fields.py:543`,
  with true `dxC`/`dyC` metrics and `CS`/`SN` rotation via
  `src/dbof/utils/native_gradient.py`.
- Front tracking across time — `fronts/front_tracking.py` (branch `viz_tools`), `follow()`.
- Front-following cross-sections — `fronts/viz/curtains.py`
  (`extract_main_axis`, `perpendicular_path`, `transect_front_crossings`).
- Hourly surface tile series on OSN — `run_series(..., pipeline="OSN")` on
  preprocessing branch `tiles-surface-only`.

Missing: `D(gradb2)/Dt` from consecutive snapshots. That is it.

**Convention to fix now.** `F = (1/2) D(gradb2)/Dt`. We compare **2F** against the
measured `D(gradb2)/Dt`. Easy factor of two to lose.

---

##### Four things that can burn us

**C1. `F` is a material derivative and the Gulf Stream advects fast.**
At u ~ 1 m/s a parcel moves ~3.6 km in an hour, about two grid cells. Computing
`dG/dt + u.grad(G)` as two Eulerian terms (G = gradb2) gives two large, nearly
cancelling numbers; the residual we care about drowns in the cancellation.
Proposal: semi-Lagrangian difference,
`[G(x, t+dt) - G(x - u*dt, t)] / dt`, midpoint departure point. One
well-conditioned difference. This is the single choice that most determines
whether the comparison means anything.

Corollary: `run_v5_gulf_stream_chunks.yaml` is 3-hourly (~6 cells of
displacement) — too coarse. The hourly OSN store is the right source.


**C2. Surface-only cannot close the budget.** The full tendency is

```
(1/2) D|grad_h b|^2 / Dt  =  F_horiz  -  b_z (w_x b_x + w_y b_y)  +  grad_h b . grad_h D
                             ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
                             have it     tilting: needs b_z          mixing/diabatic
```

Tilting needs `b_z`, which k=0 alone does not give (OSN does carry `W` at k=0, but
w ~ 0 there). So with surface-only the residual lumps **tilting + mixing** and we
cannot attribute it. I do not think this kills the project — the dissipation
efficiency is interesting either way — but it should be a deliberate choice, not a
discovery at figure time. See Q2.

**C3. Scale.** `F` is quadratic in `grad b`, so it is *more* grid-noise dominated
than `gradb2`. Filtering `b` differently on the two sides manufactures a mismatch.
Proposal: identical filter on both sides, sweep the scale (none, 2, 4, 8 cells),
and report slope/correlation *as a function of scale*. The scale at which the
balance closes is itself a result.

**C4. A staggering sign that is load-bearing.** `c_grid_axis_shift` was `+0.5` on
preprocessing `main`, fixed to `-0.5` on `tiles-surface-only`; `fronts/llc/tiles.py`
(`viz_tools`) uses `-0.5` and notes that a wrong sign lands velocity a full cell
away *with the wrong sign*. The branches now agree, but this contaminates every
term of `F` and nothing downstream would look obviously wrong. Verify empirically
on a real tile in Phase 0.

Separately: `wrangler/preproc/pp_ogcm.py:270 calc_F_s` is dimensionally wrong
(velocity derivatives never divided by grid spacing; fixed `dx=2 km`); its docstring
says `1/s^2 (maybe)`. Not usable quantitatively — the native-grid version supersedes it.

---

##### Proposed phases

- **Phase 0 — trust before scale.** Branch strategy (the `frontogenesis` branch lacks
  `front_tracking.py`, `llc/tiles.py`, region configs — those are on `viz_tools`).
  Verify C4 on real data. Validate the operators against an analytic case: Gaussian
  front in pure deformation `u = -a x, v = a y`, where `|grad b|^2` grows as
  `exp(2 a t)` exactly. Cheap, and catches metric errors, filter artifacts and
  semi-Lagrangian scheme error in one shot.
- **Phase 1 — data.** N consecutive hourly OSN surface tiles over the Gulf Stream:
  Theta, Salt, U, V, Eta. `run_series` writes one file per timestamp with no time
  dim — needs a concat step into a time-stacked zarr.
- **Phase 2 — field-level budget**, no front finding. Semi-Lagrangian
  `D(gradb2)/Dt` vs `2F`, per pixel. The rigorous core; independent of how we
  define a "front".
- **Phase 3 — front-level.** Tile front finding per hour -> `front_tracking.follow()`
  -> per-front strength time series vs `integral(2F dt)`. `curtains.perpendicular_path`
  gives cross-front `db` and width directly.
- **Phase 4 — bound the missing terms** with the full-depth `CHUNKS/gulf_stream`
  store (3-hourly) to measure how large tilting actually is, hence how much of the
  Phase-2 residual is really mixing.
- **Phase 5 — synthesis and writeup.**

---

##### Figures

The ones that would actually change your mind about something:

1. Two-panel map, shared colorscale: measured `D(gradb2)/Dt` beside `2F`, plus
   residual. If these do not look alike, nothing else matters.
2. **Joint PDF, measured vs predicted**, on front pixels, 1:1 line, regression
   slope. Slope < 1 is the mixing signature. The money plot.
3. Slope and correlation **vs filter scale** (see C3).
4. **Alignment PDF**: angle between `grad b` and the strain compressional axis.
   `F = -(1/2) div |grad b|^2 - (1/2) |sigma| |grad b|^2 cos(2 theta)`, so this is
   the classic frontogenesis signature and a strong physical check.
5. Sharpening-timescale map `tau = gradb2 / (2F)` with the `dt = 1 h` contour drawn
   — shows where hourly sampling is even adequate.
6. Tracked-front case study: 4-6 panel time sequence of one Gulf Stream front, with
   observed strength and the `F`-predicted curve overlaid.
7. Population statistics over all tracked fronts: frontogenetic vs frontolytic
   fractions, lifetime vs mean `F`.
8. Term budget: `F_geo` vs `F_ageo` vs tilting vs residual.
9. (Appendix) Analytic deformation validation from Phase 0.

---

##### Questions

**Q1 — Order of attack.** I would do Phase 2 (field-level budget, no tracking)
before Phase 3 (tracked fronts). The tracked-front figure is the compelling one,
but if the per-pixel budget does not close we would be chasing our tails in object
space. Agree, or do you want the tracking result early for a talk/deadline?

> **JXP:** I agree

**Q2 — The vertical terms.** Accept surface-only and report the residual as
"tilting + mixing"? Or spend Phase-1 effort on 2-3 levels (k=0,1,2) so we can
actually close the budget? OSN's kerchunk refs are k=0 only, so multi-level means a
different transfer — real cost, real payoff.

> **JXP:** Yes, surface-only for this analysis.  We will consider going to depth later

**Q3 — Region and window.** Default: Gulf Stream tile, **March 2012, 48-72
consecutive hours** — winter, deep mixed layer, vigorous submesoscale, and it
overlaps the existing `gulf_stream` chunk transfer so Phase 4 lines up. Reason to
prefer a different season or region?

> **JXP:**  The point you make about the very rapid velocities in the Gulf Stream is well taken.  I suggest we instead examine the California Current.  Any show stoppers here?

**Q4 — Output destination.** The `sharpen` effort used an Overleaf project
(`Front_properties`) with a standalone planning `.tex` and PNGs pushed there. Same
pattern with a new Overleaf project, or keep everything in
`dev/frontogenesis/{py,data,figs}` for now?

> **JXP:** Keep everything in `dev/frontogenesis/{py,data,figs}` for now.  Use markdown files for any reports, plans, etc.

**Q5 — Lauren's route.** She suggests `build_v5.py` with the Gulf Stream config, but
that config finds fronts on the **full global field** at 3-hourly cadence and uses
the chunk only for co-location afterward. For frontogenesis I would use the hourly
OSN tile path (`build.tile_find`), which already exists on `viz_tools`. Is there a
reason for the global route I am not seeing — worth checking with her?

> **JXP:** I agree.  Let's not do the global route.

#### Claude, 2026-09-11 (prompt Planning-2)

Decisions locked from Q1-Q5: field-level budget first; **surface-only**; **California
Current**; everything in `dev/frontogenesis/{py,data,figs}` with markdown reports; no
global route. Below: the answer to Q3, a correction to C2 that works in our favour, the
new blockers, and Q6-Q10.

---

##### Q3 answered: California Current is not just viable, it is the better-supported path

No show-stoppers. Concretely:

- `configs/tiles/tile330_gradb2_osn.yaml` (preprocessing, `tiles-surface-only`) is
  *already* the California Current: tile 330, lon -121.9 / lat 36.8, **504 consecutive
  hourly** snapshots 2012-06-29 00:00 -> 2012-07-19 23:00 UTC. Resolves to **face 10**,
  face-local `j 0:720, i 2880:3600`; box approx **lon -128..-113, lat 26.7..38.2**,
  720x720 native cells.
- **C1 is substantially de-risked by the move.** CC surface speeds ~0.1-0.4 m/s vs Gulf
  Stream 1-2 m/s. At ~1.9 km native spacing that is **0.2-0.4 grid cells of displacement
  per hour** instead of ~2. The semi-Lagrangian difference is still the right scheme, but
  a plain Eulerian `dG/dt + u.grad(G)` now becomes a usable independent cross-check
  rather than a numerically hopeless one. Your instinct here was right.
- Strain rates are also gentler: CC submesoscale `alpha ~ 1e-5 - 5e-5 /s` gives an
  e-folding time of 6-28 h, comfortably resolved by hourly sampling. `f` at 37N is
  8.8e-5 /s (inertial period 19.9 h); M2 is 12.4 h. Hourly resolves both.
- Bonus for later: `LLC4320_RAW/CHUNKS/monterey_bay` (17 dates, **full depth**) and
  `run_v5_tiles_monterey.yaml` (27 properties incl. `frontogenesis_tendency/_geo/_ageo`)
  already exist, so the Phase-4 depth cross-check has data waiting.

---

##### Correction to C2 — surface-only is cleaner than I said

In round 1 I wrote that the residual would lump **tilting + mixing**. That was too
pessimistic, and the distinction changes how we read the money plot.

The tilting term is `-b_z (w_x b_x + w_y b_y)`. At the surface `w -> 0` (LLC4320 uses a
z* free surface, so the top is very nearly a material surface), which kills the term
regardless of how large `b_z` is. The surface budget is therefore

```
D/Dt ( (1/2)|grad_h b|^2 )  =  F_horiz  +  grad_h b . grad_h(B_diabatic)
                               ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               we have     the residual = what we measure
```

So the residual is **purely diabatic** — air-sea flux plus vertical mixing/entrainment
through the base of the top cell — not an unattributable mixture. That makes
"frontogenesis efficiency" a clean result from surface fields alone, and it makes your
Q2 call better than I credited it.

Two caveats. (a) `k=0` is a finite cell (~1 m), not the mathematical surface, so the
residual includes entrainment through that cell's base — which under KPP is exactly the
mixing we want to quantify, so this is fine, but it should be stated in the writeup.
(b) The premise is checkable: **OSN serves `W` at `k_l=0`**. A Phase-0 one-liner
confirming `W[k_l=0] ~ 0` over the tile settles it empirically instead of by argument.

---

##### C5 (NEW) — land contamination. The real blocker for a coastal tile.

Nothing masks land before differencing. MITgcm stores land as 0, so `b(Theta=0, S=0)` is
finite and every ocean cell adjacent to a coast inherits the full land/ocean buoyancy
jump — a gradient far larger than any real front.

- `utils/native_gradient.py:29-35` does `grid.diff(b,'X')/dxC` then
  `grid.interp(..., boundary='fill')` with **no** `hFacC` applied to Theta/Salt first.
- `tile_utils.compute_tile_property` (`tiles-surface-only:613-697`) masks *afterwards*:
  `field.where(hfac > 0)`. That NaNs the land cells and leaves the poisoned ocean
  neighbours intact. Its own docstring admits derived fields are pathological "over
  **and next to** land".
- `generate_tile_gradb2` (`fronts/preproc/gradb2.py:89-98`) does not even expose
  `mask_land`.

On an open-ocean Gulf Stream tile this never mattered. Tile 330 has the California and
Baja coastline through its interior, so untreated we would "discover" that the strongest
front in the domain is the shoreline.

**Good news on the downstream half:** the finding chain is already NaN-safe —
`pyboa.front_thresh` uses `nanpercentile` with `cval=np.nan` and `frnt = array > window_qt`
(`finding/pyboa.py:708-824`), so NaN never becomes a front and local thresholds are built
from ocean values only; `sharpen`, `cropping`, `despur` are pure binary ops. **Zero-filled
land would be actively harmful** (zeros drag the 85th-percentile window down near the
coast), NaN-filled land is handled correctly.

**Fix:** apply a *dilated* land mask to `b` (and U, V) before any differencing. The helper
already exists and is simply not wired into the tile path —
`preprocessing/static_masks.generate_halo_land_mask(ds_grid, target_km_res, ...)` ->
`halo_mask.llc_native_grid_halo_mask(..., halo_km=...)`. The halo must cover the full
stencil reach (3 cells for the Jacobian+interp chain, matching `edge_margin=3`) **plus**
the C3 filter half-width. Small change, but everything downstream is worthless without it.

---

##### C6 (NEW) — tracking is single-front, and we need populations

`front_tracking.follow(labels_at, times, anchor, ...)` takes an **anchor** and returns one
`Track`. Good news: `dt` is a genuine parameter parsed from the timestamps (uneven cadence
is a stated design goal), and `MAX_DRIFT_MS=0.6` feeds
`_radius_px = max(2, 0.6*dt/1000/km_per_px)`, which at hourly floors at 2 px ~ 4.6 km ~
1.3 m/s equivalent — ample for CC speeds. So hourly CC tracking works as-is.

But Figure 7 (population statistics over all fronts) needs many-to-many linking. Either
loop `follow()` over anchors on the largest N fronts (cheap, some double-counting) or
write a proper linker on top of the existing `score_candidate`. See Q10.

##### C7 (NEW, small) — per-front strength is nearly free

`properties/colocation.py:101` already computes per-front mean/std/median/percentiles of
**any** tile property with an optional `dilation_radius` — including `gradb2` *and*
`frontogenesis_tendency`. So "front-mean gradb2 at each hour" + a `Track` gives the
measured per-front strengthening with very little new code. What does **not** exist:
cross-front `delta b`, front width (FWHM across a transect), peak gradient along the axis,
or any time-differencing. Those are new, built on `curtains.perpendicular_path` /
`path_metrics`, which are reusable as-is.

##### C8 (NEW) — environment is a hard blocker today

`dbof` is installed in **no** conda env; `xgcm` is missing from every env. `fronts` lives
in `ocean14` (Python 3.14.6), but `dbof`'s `pyproject.toml` pins `torch==2.8.0` (ocean14
has 2.12.1), `timm==0.3.2`, `xarray==2025.10.1`, `dask==2025.10.0`, `s3fs==2025.9.0`, plus
`ecco_v4_py`. Installing `dbof` into `ocean14` would downgrade torch and several core libs
underneath `fronts`. **A dedicated env is needed** (Python 3.12/3.13; 3.14 is likely too
new for some of these). `kerchunk` the package is *not* required — the OSN refs are read
through fsspec's built-in `reference://` filesystem with `engine="zarr"`,
`consolidated=False` (`get_raw_data.py:109-118`).

##### C9 (NEW, small) — the code path we depend on has never been run

There is no OSN test in the preprocessing repo and no test of the `tile_find` branch,
`generate_tile_gradb2`, or `coords_file` in `fronts` (`test_build_v5.py` covers the global
path only). No test anywhere exercises `fronts_from_gradb2` on a field containing NaN.
Phase 0 should add at least the NaN-input finding test, since C5's fix makes NaN land the
normal case.

---

##### The one piece of unambiguous good news

**Velocities already flow through the OSN tile path and F needs zero new code.**
`_load_tracers_for_tile` is misnamed but variable-agnostic; the kerchunk file is
`llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json` and
`OSN_SURFACE_VARS = {Theta, Salt, Eta, U, V, W}`. `frontogenesis_tendency` is registered
(`field_registry.py:307-310`, vars U/V/Theta/Salt, `edge_margin=3`) and reachable as

```
generate-tile --property frontogenesis_tendency --pipeline OSN \
    --dates-config configs/tiles/tile330_gradb2_osn.yaml ...
```

Caveat: one property per run/file (`_build_output_dataset` assumes a single data var), so
this is N series or a ~20-line multi-var registry entry. See Q9.

---

##### Revised plan (decisions locked)

- **Phase 0 — trust.** Dedicated env (C8). Verify the `c_grid_axis_shift` sign on real
  data (C4). Verify `W[k_l=0] ~ 0` (corrected C2). Wire the halo land mask into the tile
  path (C5). Add a NaN-input finding test (C9). Validate operators against the analytic
  deformation flow `u=-ax, v=ay` where `|grad b|^2` grows as `exp(2at)`.
- **Phase 1 — data.** Tile 330, N consecutive hours: `b`/`gradb2`, `U`, `V`, and `F`.
  Concat to a single time-dimensioned zarr — **no concat step exists anywhere**, we write it.
- **Phase 2 — field budget.** Semi-Lagrangian `D(gradb2)/Dt` vs `2F`, offshore as primary
  (Q7). Figures 1-5, plus the hour-of-day composite of the residual.
- **Phase 3 — fronts.** `tile_find` per hour -> tracking -> per-front series. Figures 6-7.
- **Phase 4 — later**, against `CHUNKS/monterey_bay` at depth.

Figure list from round 1 stands, with one addition promoted to primary: **residual
composited by hour of day**. Given the corrected C2 the residual is purely diabatic, so
the diurnal cycle is not a contaminant to be filtered — it is the signal.

---

##### Questions

**Q6 — Season.** The configured window is summer (2012-06-29 -> 07-19): peak upwelling,
sharp filament fronts, but the thinnest mixed layer and strongest diurnal restratification
of the year. Under the corrected C2 that diurnal signal is measurable science rather than
noise, so I lean toward **using the configured summer window as-is**. Winter (Jan-Feb 2012
is inside OSN coverage; nothing has been run there) would give a deeper ML and a smaller
diabatic residual — a cleaner test of the pure kinematic balance. Summer only, or summer
now and a winter window later as contrast?

> **JXP:**  Summer only

**Q7 — Coastal cut and halo width.** Two separate numbers. (a) Halo width for the
land mask: I propose `edge_margin` 3 cells + filter half-width, i.e. ~5-8 cells (~10-15
km), applied before any differencing. (b) A stricter cut for the *primary statistics*:
exclude everything within some distance of the coast or shallower than some depth, and
report the result as a function of distance offshore so the choice is visible rather than
buried. Do you have a preferred criterion, or shall I pick (say, `Depth > 500 m`) and show
the sensitivity?

> **JXP:** (a) use your recommendation; (b) Let's work at least 100km offshore.

**Q8 — How many hours in the first pass?** I propose **72 consecutive hours** out of the
504 — three full diurnal cycles, three inertial periods, enough for meaningful front
tracking, and small enough to iterate quickly. Scale up once Phase 2 closes. Agree, or do
you want the full 3 weeks from the start?

> **JXP:** 72 hours is fine

**Q9 — One gradient operator or two?** C3 says measured and predicted must share an
identical stencil and identical filtering. The registry's `frontogenesis_tendency` applies
no filtering. So I would like to **compute both sides ourselves** in
`dev/frontogenesis/py` from saved `b`, `U`, `V` — one operator, one filter, applied
identically — and use the registry's `frontogenesis_tendency` as an unfiltered regression
test rather than as the science product. Cost: we save `U`, `V`, `b` per hour instead of
just `F`. Any objection to that, or would you rather lean on the registry output?

> **JXP:** No objection

**Q10 — Population tracking (Figure 7).** Loop the existing anchor-based `follow()` over
the largest N fronts (cheap, some double-counting, available immediately), or write a
many-to-many linker on top of the existing `score_candidate` (cleaner statistics, more
work)? I would start with the loop and only upgrade if the double-counting distorts the
distributions.

> **JXP:** Yes, let's do the largest N fronts, with N=10


#### Claude, 2026-09-12 (prompt Planning-3)

**Verdict: one short round, and essentially one real decision — branch strategy (Q11).**
Q6-Q10 are all settled. Everything else below is either something I resolved myself (no
input needed, recorded for the log) or a confirm-or-veto.

Locked: summer only; halo = my recommendation; **>=100 km offshore** for primary
statistics; **72 h**; we compute both sides with one shared operator/filter; largest
**N=10** fronts via looped `follow()`.

---

##### Two of my earlier concerns were wrong. Retracting both.

**C4 retracted — the comodo sign is already resolved.**
`prompts/tile_surface_series.md:146-152` (preprocessing, `tiles-surface-only`) records:
"RESOLVED 2026-09-01: `-0.5` is right ... it is a signed direction (xgcm maps -0.5 ->
'left'), MITgcm puts U on the west face, and xmitgcm and the OSN grid both stamp -0.5;
measured directly from the corner geometry too." It now lives once, in
`llc4320_ingestion.grid.COMODO_COORD_META`. I was flagging settled work. Downgraded from
a Phase-0 investigation to a single assertion in our test.

**C8 downgraded — no dedicated env needed after all.**
`dbof`'s `torch==2.8.0` / `timm==0.3.2` pins are only reachable through
`cutout_dataset_creation/dask_pipeline.py` and `spatial_cutouts.py`, neither of which the
library route touches. `pip install -e . --no-deps` (or just `PYTHONPATH=.../src`) skips
them entirely. What we actually need to add: **`xgcm<0.10`** (0.10 removed `periodic` and
renamed `boundary`), `scikit-fmm` (for the halo mask), plus `s3fs`, `ujson`, `xmitgcm`,
`zarr`, `dask` — most already in `ocean14`. Remaining risk: `ocean14` is Python 3.14 and
`xgcm<0.10` / `scikit-fmm` may not have 3.14 wheels; fallback is a py3.13 env. Much
cheaper than the torch downgrade I described in round 2.

---

##### The library route works — no changes to Lauren's branch

Q9's answer (one shared operator) means we do not want the repo's *derived* products at
all. Verified that every piece is an ordinary module-level function with no config/CLI
coupling, so we can pull raw fields and do everything ourselves:

```python
EP = "https://mghp.osn.xsede.org"
tile   = rect_ij_to_tile(13320, 9720)          # -> face 10, j 0:720, i 2880:3600
g      = process_llc4320_grid(get_remote_gridfile(EP))          # static, pull once
g_tile = _ensure_comodo_attrs(g.isel(face=[tile.face_idx], **_tile_indexer(g, tile)).compute())
grid   = set_xgcm_grid(g_tile, use_connections=False)
for ts in timestamps:                                            # hourly
    ds = get_remote_llc_data(EP, osn_date_to_iteration(ts), [tile.face_idx])
    ds_tile = ds.isel(**_tile_indexer(ds, tile))                 # Theta/Salt/U/V/W/Eta, lazy
```

`get_remote_llc_data` returns a lazy dask Dataset with `Eta/Theta/Salt` on `(face,j,i)`,
`U` on `(face,j,i_g)`, `V` on `(face,j_g,i)`. The grid carries
`XC,YC,dxC,dyC,dxG,dyG,rAz,rA,Depth,hFacC,SN,CS` — everything the gradient and Jacobian
helpers need. **Consequence: we modify neither `tiles-surface-only` nor the CLI.** The
only genuinely entangled function is `latlon_to_rect_ij` (needs network + an `s3_cfg`),
and we can skip it since the config records rect (13320, 9720) already. See Q12.

---

##### C5 refined — and the halo helper has two bugs we will hit

Whether OSN stores land as 0 or NaN is **unverified** (I could not settle it from code).
One-line Phase-0 check; the halo is the right fix either way, since NaN land would
otherwise propagate through the stencil uncontrolled.

Two real defects in the helper I proposed using:

- `halo_mask.llc_native_grid_halo_mask` returns a 2D `mask_f` early if a face is
  **entirely land** (halo_mask.py:74-75) — wrong shape, silent.
- If OSN's `hFacC` carries a `k` dim, the mask goes 4D and `skfmm` fails.
  `compute_tile_property` already defends against this (`tiles-surface-only:689-690`),
  which suggests it does happen.

Also note the returned convention: **True = retained**, land = False, and it is a plain
numpy array, not a DataArray.

---

##### Settled myself — recorded, not asking

1. **The 72 hours: 2012-07-02 00:00 -> 2012-07-04 23:00 UTC.** Not the start of the
   window. The full-depth `monterey_bay` chunk store is daily at 12:00 **plus a dense
   3-hourly day on 2012-07-03** — evidently the day already chosen as interesting. This
   window puts **11 of the 17** full-depth snapshots inside it, including all 8 of the
   dense day. Starting at 2012-06-29 would capture only 3. Free, and it makes the Phase-4
   depth cross-check far more useful.
2. **Halo width = 7 cells (~13 km):** 3 for the Jacobian+interp stencil (matching
   `edge_margin=3`) + 4 for the widest filter half-width.
3. **Gulf of California:** the tile's eastern edge (lon -113.0) clips the head of the
   Gulf. The >=100 km offshore cut removes it automatically — I will verify rather than
   add a special-case polygon.
4. **Front strength (Phase 3): primary = front-mean `gradb2`** from `colocation.py`, so
   Phase 2 and Phase 3 measure the *same* quantity and are directly comparable. Secondary
   = cross-front `delta b` and width from `curtains.perpendicular_path`.
5. **Filter scales:** none, 2, 4, 8 cells.
6. **Validation splits in two:** a Cartesian test of the semi-Lagrangian scheme against
   `u=-ax, v=ay` (exact `exp(2at)` growth), and a *separate* native-grid test using an
   analytic function of `XC/YC` to exercise `dxC/dyC/CS/SN`. One test cannot do both.

**One comment on your Q7(b).** A >=100 km cut excludes the inner upwelling zone, which is
where CC frontogenesis is most vigorous — filament roots rather than tips. I think that is
the right call for a *clean* measurement (it is also where the diabatic residual and any
remaining land contamination are worst), but it does mean our headline number describes
the offshore regime, not the CC as a whole. I will keep the distance-stratified plot so
what we excluded stays visible, and we can revisit after seeing it.

---

##### Questions

**Q11 — Branch strategy. The one thing I cannot decide for you.**

*fronts:* `frontogenesis` (yours, 9 commits: pipeline-aware build_v5, ice mask,
meta/publish) and `origin/viz_tools` (65 commits: Panel apps, `front_tracking.py`,
`llc/tiles.py`, `tile_find`) have **diverged** — 14 commits on `frontogenesis` are not in
`viz_tools`, and both branches independently created `llc/meta.py` and `llc/publish.py`
and both modified `build_v5.py` and `properties/run.py`. We need `tile_find` and
`front_tracking.py`, which exist only on `viz_tools`. Options: (a) merge `viz_tools` into
`frontogenesis` and resolve the add/add conflicts once, (b) branch fresh off `viz_tools`
and give up the 9 `frontogenesis` commits for now, (c) something you had already planned.

*preprocessing:* the checked-out branch `llc4320_v2` is **102 commits behind main**
(HEAD 2026-08-31; main moved 2026-09-11 with PR #36). Everything we need — `run_series`,
the OSN pipeline, the tile330 config — is in just **2 unmerged commits** on
`tiles-surface-only`, which is itself 19 behind main. Cleanest is to merge current main
into `tiles-surface-only` (2 commits to replay, low risk) and work from there. Your call,
and also: is `tiles-surface-only` yours to merge, or Lauren's?

> **JXP:**

**Q12 — Confirm the library route.** I intend to import `dbof` as a library from
`dev/frontogenesis/py` and pull raw `Theta/Salt/U/V` + grid myself, rather than driving
`generate-tile`. Upside: one gradient operator and one filter on both sides (what Q9
requires), no modifications to Lauren's branch, no CLI/config coupling. Downside: our
pipeline diverges from the repo's tile conventions, so it is less reusable by others and
Lauren would not get the land-halo fix for free. Object, or proceed?

> **JXP:**


#### Claude, 2026-09-26 (prompt Planning-7 — responding to Lauren's comments)

Lauren's comments are adopted essentially in full. One of them (the tracking point) is more
important than she framed it, one data offer unlocks something I had written off as
impossible, and two of her branch assumptions are now out of date in our favour.

---

##### Answers to her three direct questions

**"Are you doing hourly here?"** Yes. Hourly, 72 consecutive steps,
**2012-07-02 00:00 -> 2012-07-04 23:00 UTC**. The window was chosen *because* the existing
`monterey_bay` set already covers 11 hours inside it (07-02 T12; 07-03 T00/03/06/09/12/15/18/21;
07-04 T00 and T12), so hourly full-depth would complete a window that is already half-covered.

**"Was frontogenesis branched off of viz_tools?"** **No.** It was branched off `build_v5`.
Precisely: `frontogenesis` = `origin/build_v5` minus one commit (`fcf9cb1 reorganize config
info`) plus 5 commits — and those 5 touch **only** files under `dev/frontogenesis/`
(the prompts doc, the planning doc, the coding doc, the deck generator). Nothing outside it.
`origin/viz_tools` and `frontogenesis` share no recent history: 65 commits on viz_tools are
absent from frontogenesis and 20 the other way.

**"I can do this if you let me know."** Yes please — see Q15 for the exact sequence. But
two of the premises have already moved:

- **`tiles-surface-only` does not need a rebase any more.** It is now **0 behind / 4 ahead**
  of `origin/main` — a clean fast-forward. (On 2026-09-12 it was 19 behind; someone has
  already brought it up.) `COMODO_COORD_META` on `origin/main` does carry
  `c_grid_axis_shift: -0.5`, confirming your note; and `origin/main` has moved to
  `80048ac Merge PR #37 face-stitch`.
- **The thing actually blocking the chain is PR #24 ("Build v5"), still OPEN** since
  2026-09-01. `origin/build_v5` is 10 ahead of and **0 behind** `main`, so it merges clean.
  Everything else queues behind it.

Also worth knowing: `run_series` and the OSN pipeline are **still not on `origin/main`** —
they live only on `tiles-surface-only`. And the physics file is `calculate_fields.py` on
`main` but `calculate_additional_fields.py` on the stale local `llc4320_v2`, which is where
some earlier notes' line numbers came from.

---

##### The data offer: yes, and it changes the budget from inferred to measured

I had written in planning §4 that heat fluxes are in neither OSN store, so the diabatic term
could only ever be inferred as a residual. **That is true of OSN but not of the CHUNKS
store**, whose variable list already includes **`oceQnet`** alongside 3D
`Theta, Salt, U, V, W` and 2D `Eta, oceTAUX, oceTAUY, SIarea`.

That matters a lot, because the adversarial review's most serious objection (planning §2.3)
is that we cannot separate implicit numerical diffusion from genuine air-sea forcing — both
land in the same residual. With hourly chunks we get, at every timestep:

- `k = 1, 2` Theta/Salt -> `b_z` -> the finite-top-cell vertical term **measured** rather
  than bounded (planning §2.2 currently promises only a bound from 11 snapshots);
- `W` below the surface -> the vertical terms properly;
- `oceQnet` -> most of `grad b . grad B` computed **directly**.

The residual then reduces to *numerical diffusion + interior KPP*, which is a far stronger
statement than "everything we could not compute". This upgrades Figure 2b from a
circumstantial discriminator to a real budget term.

**One concrete ask beyond the transfer itself** (Q13): `transfer.variables` is configurable,
so please **add `oceQsw`**, and `oceFWflx` if the MIT source has it. Without `oceQsw` we
cannot separate penetrating shortwave from the non-penetrating part, and a large fraction of
shortwave is absorbed inside the ~1 m top cell — which is exactly where our `b` lives. Net
flux alone gets the dominant signal but blurs the term that peaks at local noon, i.e. the
one Figure 6 is about. `oceFWflx` matters less in the summer CC but `b` does depend on salt.

**Cost, so the decision is informed.** All 51 levels, float32: 105.75 MB per 3D variable per
timestep; 5 3D + 5 2D ≈ **539 MB/timestep**. 61 new stores (11 already held) ≈ **33 GB**
(38.8 GB for all 72). Confirmed incidentally: the source has **51 levels to ~968 m with
`drF[0] = 1.0 m`, `Z[0] = -0.5 m`** — which pins the "~1 m top cell" assumption in
planning §2.2 that was previously marked *to confirm*.

**A depth subset is not currently possible without a small patch.** The level count is taken
straight from the source (`transfer/zarr_io.py:326`, `nk = ds.sizes[vdim]`) and nothing in
the config schema or CLI selects `k`. A `k = 0..4` subset would be ~4.5 GB instead of 33 GB,
and needs roughly ten lines in `TransferConfig` plus an `isel` in `pipeline._resolve_target`.
**Footgun worth flagging:** `config._only_known()` *silently drops* unknown YAML keys, so
adding a `k_max:` to the yaml would be ignored rather than rejected.

---

##### S5.3 / S6 — validation figures. Agreed, all of it, and promoted from optional

Every one of these is cheap and makes a decision that is currently abstract into something
visible. Adopted as **M1 acceptance deliverables, not nice-to-haves**:

- `validate.py` writes **one PNG per test** (four), not just a dict of numbers.
- **The interpolation demo exactly as you describe it:** a synthetic front shifted by half a
  cell; truth vs `G` from bilinear-`G` vs `G` from cubic-`b`, with the negative bias at the
  maximum annotated. This is the clearest possible statement of why §5.3 item 2 exists, and
  it doubles as the regression test.
- **The discrete-null slope drawn on Figure 2 as an explicit baseline line**, not merely
  quoted in the caption. Better than what I had written.
- **A new main figure for the filter sweep**: rows `{b, G, 2F, tau-term}` x columns
  `{L = 0, 2, 4, 8}`. Promoted to a main figure rather than an appendix, because §5.4 is the
  part of the method that is hardest to believe on prose alone, and showing where `tau`
  lives is the whole argument. This is Figure 3b.

Figures are therefore restructured: main Figures 1-10 (with 2b and 3b), plus a validation
set V1-V5 written by `validate.py` itself.

---

##### S5.7 — the tracking point. You are right, and it is load-bearing

Your read of the two mechanisms is exactly correct: `follow()` matches labels on
position/size/orientation, and the departure-point machinery is per-pixel and currently only
feeds `DG/Dt`. They should talk, and here is why it is not optional.

**Your integer-label worry dissolves.** We never interpolate labels. We advect the
**boolean front mask** through the same `semilag` machinery (as a float, thresholded at 0.5),
producing a *flow-predicted mask* at `t+dt`. Labels stay integers throughout.

**Then the prediction goes in non-invasively.** `score_candidate` already takes a `weights`
dict, so `IoU(flow-predicted mask, candidate label)` is an additional scored term, not a
rewrite. It is also strictly better physics than what is there now: `_predict()` extrapolates
*centroid* velocity from the last two sightings, and a front whose centroid moves because it
grew asymmetrically is not a front that moved with the fluid.

**Where I would push further than you did.** You wrote that this "seems relevant" to the M4
criterion. It is stronger than that — **M4's reconciliation is meaningless without it.** If
`follow()` ever links a front at `t` to a different physical front at `t+dt`, then the
per-front `d(front-mean G)/dt` is not a material derivative and comparing it to
`integral 2F dt` compares nothing.

And even a *perfect* flow-following track is not enough, which I had missed: front-mean `G`
is a mean over a **changing pixel set** (fronts lengthen, split, merge), so `d/dt` of that
mean carries an extra term from the set's evolution. So M4's reconciliation must be done on
the **advected pixel set** — Lagrangian-matched pixels — not "pixels labelled front at `t`"
against "pixels labelled front at `t+dt`". That is a real change to the M4 criterion and it
came out of your comment.

**Two things we get for free.** A principled **split/merge detector** (the predicted mask
overlapping two candidate labels), which `follow()` has no notion of and which will certainly
happen over 72 h; and a **quality metric for all of Phase 3**: the distribution of
(`follow()`-chosen displacement − flow-predicted displacement). If those disagree often, the
tracking is not following the fluid and we know it before interpreting anything. That becomes
a figure.

---

##### Housekeeping consequence

The coding doc §2 API table is pinned to `tiles-surface-only` and `viz_tools` line numbers.
Both are about to move. **§2 must be re-verified after the merges** — before M0 runs against
it, not after something silently misbehaves.

---

##### Questions

**Q13 — Transfer scope, for Lauren.** (a) All 51 levels (61 new stores, ~33 GB) or patch the
transfer for `k = 0..4` (~4.5 GB)? I lean **all 51 levels** if the storage is unremarkable:
it touches no transfer code, and it makes M6 (the subsurface budget) real work rather than
aspiration. (b) Either way, please add **`oceQsw`** to `transfer.variables`, and `oceFWflx`
if the source carries it — that is the difference between measuring the diabatic term and
inferring it.

> **Lauren / JXP:** Yes, all 51 levels.  And yes, please add `oceQsw` and `oceFWflx`.

**Q14 — Does flow-informed tracking become an M4 requirement?** My recommendation: yes,
including the Lagrangian-matched-pixel reconciliation above. It is modest work (the `semilag`
machinery already exists and `score_candidate` already takes weights) and without it M4's
acceptance criterion does not test what it claims to. The alternative is to keep `follow()`
as-is and report the disagreement statistic as a caveat only.

> **JXP:** Yes, it becomes an M4 requirement.

**Q15 — Branch sequencing.** Proposed order, which I think needs only Lauren's hand on steps
2 and 3:
1. **Merge PR #24** (`build_v5` -> `main`). Clean: 0 behind.
2. **Lauren rebases `viz_tools` onto the new `main`.** This is where the `llc/meta.py` /
   `llc/publish.py` add/add conflicts get resolved — once, by the person who wrote viz_tools,
   rather than by us guessing.
3. **Fast-forward `tiles-surface-only`** into `main` in the llc repo (0 behind — no rebase
   needed, contrary to the original concern).
4. **Rebase `frontogenesis`** onto the result. Its 5 commits are documentation-only under
   `dev/frontogenesis/`, so they replay with zero conflicts onto anything.

Does that match what you had in mind, and do you want to pick up `fcf9cb1 reorganize config
info` directly or let it arrive via `main`?

> **JXP / Lauren:**  Yes, that looks very good.


## Logs
### 2026-09-11 — Planning prompt 1 (Claude Opus 5, surveys by Fable)

**Method.** Two Fable subagents surveyed `fronts` and `llc4320-native-grid-preprocessing`;
I read the physics modules directly.

**What already exists (more than expected).**

- `llc4320-native-grid-preprocessing/src/dbof/preprocessing/calculate_additional_fields.py:543`
  `_frontogenesis_formula` — `F = -(u_x b_x^2 + (u_y+v_x) b_x b_y + v_y b_y^2)`, plus
  `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`. The 3-D twins are in
  `calculated_fields_at_depth.py:1426+`.
- `src/dbof/utils/native_gradient.py` — gradients/Jacobian use true `dxC`/`dyC` metrics and
  rotate to geographic via `CS`/`SN`. This is the correct native-grid machinery.
- `ertel_pv_terms_3d` already computes `b_z`, `w_x`, `w_y` — the ingredients of the tilting term.
- `fronts/front_tracking.py` (branch `viz_tools`) — `follow()` tracks labelled fronts across
  time by centroid extrapolation + IoU/orientation scoring.
- `fronts/viz/curtains.py` — `extract_main_axis`, `perpendicular_path`,
  `transect_front_crossings`, `recenter_curtain`: front-following cross-sections.
- Tile front-finding on chunks is IMPLEMENTED on `viz_tools`
  (`build.tile_find` -> `preproc/gradb2.generate_tile_gradb2` -> `dbof.tiles.tile_utils.run_series`,
  pipeline `OSN`).

**Data.** OSN = Spencer Jones' archive, `https://mghp.osn.xsede.org`, anonymous,
kerchunk refs `cnh-bucket-1/llc_surf/.../llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json`:
surface only (k=0), **hourly**, 2011-09-13 -> 2012-11-15, with per-face grid metrics
(`CS`, `SN`, `dxC`, `dyC`, `hFacC`). The alternative full-depth `CHUNKS/gulf_stream` store is
3-hourly, 32 steps, 2012-03-01 -> 03-04 (transfer not confirmed).

**Things that must be got right (flagged, not yet resolved).**

1. `F` is Lagrangian. Eulerian differencing at 1 h in the Gulf Stream (u ~ 1 m/s -> ~2 grid
   cells/hr) makes `d/dt` and `u.grad` two large, nearly cancelling terms. Use a
   semi-Lagrangian departure-point difference instead.
2. Surface-only cannot close the budget: the tilting term `-b_z (w_x b_x + w_y b_y)` needs
   `b_z`, which k=0 alone does not give. The residual will lump tilting + mixing.
3. `F` is quadratic in `grad b`, so it is grid-noise dominated. The same spatial filter must be
   applied to `b` on both sides of the comparison; report results vs filter scale.
4. `c_grid_axis_shift` was `+0.5` on preprocessing `main` and is fixed to `-0.5` on
   `tiles-surface-only`; `fronts/llc/tiles.py` (viz_tools) also uses `-0.5` with a note that the
   sign is load-bearing for every velocity-derived field. Verify empirically before trusting `F`.
5. `wrangler/preproc/pp_ogcm.py:270 calc_F_s` is dimensionally wrong (velocity derivatives not
   divided by grid spacing; fixed `dx=2 km`). Do not use it quantitatively.
6. Sign/factor convention: `F = (1/2) D(gradb2)/Dt`. Compare `2F` against `D(gradb2)/Dt`.
7. The existing `run_v5_gulf_stream_chunks.yaml` is 3-hourly — too coarse for submesoscale
   frontogenesis; prefer the hourly OSN path.

**Proposed plan** (presented to JXP for discussion; phases 0-5, ~10 figures). Open decisions
put to JXP: scope order (field-level budget vs front tracking first), whether to invest in
multi-level data for the tilting term, region/date window, and output destination.

### 2026-09-11 — Planning prompt 2 (Claude Opus 5, surveys by Fable)

**Method.** Read JXP's answers to Q1-Q5. Launched two Fable subagents to test the
California Current switch for show-stoppers: one on the OSN surface tile path
(`llc4320-native-grid-preprocessing`, `origin/tiles-surface-only`), one on land/coast
handling in the front-finding chain (`fronts`, `origin/viz_tools`). Checked the Python
environment myself.

**Verdict on Q3 (California Current): no show-stopper.** `configs/tiles/tile330_gradb2_osn.yaml`
is already the CC — tile 330, 504 consecutive hourly steps 2012-06-29 -> 07-19, face 10,
box approx lon -128..-113 / lat 26.7..38.2. CC speeds (~0.1-0.4 m/s) give 0.2-0.4 grid
cells of displacement per hour vs ~2 in the Gulf Stream, which materially de-risks C1.

**Correction to C2 from round 1.** I claimed the surface-only residual would lump tilting
with mixing. Wrong: the tilting term carries a factor `w`, and `w -> 0` at the surface
(z* free surface), so the surface budget is `D/Dt((1/2)|grad b|^2) = F_horiz + diabatic`
with **no tilting term**. The residual is purely diabatic (air-sea flux + entrainment
through the base of the ~1 m top cell). Surface-only is therefore well-posed for this
question, and the hour-of-day composite of the residual becomes a primary figure rather
than a nuisance diagnostic. Checkable in Phase 0: OSN serves `W` at `k_l=0`; confirm it is
~0 over the tile.

**New findings.**

- *Good:* velocities already flow through the OSN tile path (`_load_tracers_for_tile` is
  misnamed but variable-agnostic; `OSN_SURFACE_VARS = {Theta,Salt,Eta,U,V,W}`), and
  `frontogenesis_tendency` is registered (`field_registry.py:307-310`) and reachable via
  `generate-tile --property frontogenesis_tendency --pipeline OSN`. **Zero new code for F.**
  One property per output file, though.
- *Good:* `properties/colocation.py:101` already gives per-front stats of any property
  incl. `gradb2` and `frontogenesis_tendency`; `front_tracking.follow()` takes `dt` as a
  real parameter and its search radius floors at ~4.6 km at hourly — fine for CC.
- *Good:* the finding chain is NaN-safe (`pyboa.front_thresh` uses `nanpercentile`,
  `cval=np.nan`), so NaN land is handled correctly; zero-filled land would not be.
- **C5 (blocker):** no land mask before differencing. MITgcm land = 0, so `b(0,0)` is
  finite and ocean cells adjacent to coast inherit the land/ocean jump;
  `compute_tile_property` masks only *after* (`field.where(hfac>0)`). Fix exists but is
  unwired: `static_masks.generate_halo_land_mask` / `halo_mask.llc_native_grid_halo_mask`.
  Halo must cover stencil reach (3) + filter half-width.
- **C6:** `follow()` is anchor-based / single-front; population statistics need
  many-to-many linking.
- **C7:** cross-front `delta b`, front width, peak-along-axis do not exist — new work on
  top of `curtains.perpendicular_path`.
- **C8 (blocker):** `dbof` installed in no env; `xgcm` missing everywhere; `dbof` pins
  `torch==2.8.0` / `xarray==2025.10.1` / `dask==2025.10.0` / `s3fs==2025.9.0` against
  `ocean14`'s newer stack (py3.14, torch 2.12.1). Dedicated env required. `kerchunk` the
  package is not needed (fsspec `reference://`).
- **C9:** no OSN test, no `tile_find` test, no NaN-input finding test. The path we depend
  on has never been exercised.

**Posed Q6-Q10** in the Q&A section: season (summer as configured vs adding winter);
coastal halo width and the offshore cut for primary statistics; 72 h vs the full 3 weeks
for the first pass; whether to compute both sides of the comparison with one shared
operator/filter in `dev/frontogenesis/py` (recommended) or lean on the registry's
unfiltered `frontogenesis_tendency`; and loop-anchors vs a proper many-to-many tracker.

### 2026-09-12 — Planning prompt 3 (Claude Opus 5, survey by Fable)

**Method.** Read JXP's answers to Q6-Q10 (all five settled). Launched one Fable subagent
to test whether `dbof` can be used as a *library* — which Q9's answer (one shared gradient
operator and filter on both sides) makes the natural shape for Phase 1. Checked branch
topology in both repos myself.

**Decision: one more short round is warranted, for one real question (Q11, branch
strategy) plus a confirm-or-veto (Q12, library route).** Everything else is settled.

**Two of my own earlier concerns retracted.**

- **C4 (comodo sign) was already resolved** on 2026-09-01 and recorded in
  `prompts/tile_surface_series.md:146-152`: `-0.5` verified from corner geometry and
  centralized in `grid.COMODO_COORD_META`. I was flagging finished work. Downgraded to a
  single test assertion.
- **C8 (environment) is much smaller than I said.** `dbof`'s `torch==2.8.0` / `timm==0.3.2`
  pins are reachable only via `cutout_dataset_creation/dask_pipeline.py` and
  `spatial_cutouts.py`, which the library route never imports. `pip install -e . --no-deps`
  sidesteps them. Real needs: `xgcm<0.10` (0.10 removed `periodic`), `scikit-fmm`, plus
  s3fs/ujson/xmitgcm/zarr/dask. Residual risk: `ocean14` is py3.14 and those wheels may not
  exist; fallback a py3.13 env.

**Library route verified feasible.** `rect_ij_to_tile` -> `_tile_indexer` ->
`process_llc4320_grid(get_remote_gridfile(EP))` -> `set_xgcm_grid(..., use_connections=False)`
-> `get_remote_llc_data(EP, osn_date_to_iteration(ts), [face])`. All module-level, no CLI
or config coupling. Grid carries `XC,YC,dxC,dyC,dxG,dyG,rAz,rA,Depth,hFacC,SN,CS`.
Consequence: **no modifications needed to `tiles-surface-only` or the CLI.**

**C5 refined.** Whether OSN stores land as 0 or NaN remains unverified — a one-line Phase-0
check. Two real defects found in the halo helper we planned to use:
`llc_native_grid_halo_mask` returns a 2D array early if a face is entirely land
(`halo_mask.py:74-75`), and a `k`-carrying `hFacC` makes the mask 4D and breaks `skfmm`.
Convention: True = retained, land = False, plain numpy.

**Branch topology (new, and consequential).**

- `fronts`: `frontogenesis` and `origin/viz_tools` have diverged — 14 commits on the former
  are absent from the latter; both independently created `llc/meta.py` and `llc/publish.py`
  and both edited `build_v5.py` / `properties/run.py`. We need `tile_find` and
  `front_tracking.py`, which exist only on `viz_tools`.
- `llc4320-native-grid-preprocessing`: the checked-out `llc4320_v2` is **102 commits behind
  main** (HEAD 2026-08-31 vs main 2026-09-11). Everything needed sits in 2 unmerged commits
  on `tiles-surface-only`, itself 19 behind main.

**Settled without asking:** the 72-hour window is **2012-07-02 00:00 -> 2012-07-04 23:00
UTC** — chosen because the full-depth `monterey_bay` store is daily at 12:00 plus a dense
3-hourly day on 2012-07-03, so this window contains 11 of its 17 snapshots (vs 3 if we
started at the beginning of the series), maximising the later Phase-4 cross-check for free.
Also: halo = 7 cells (3 stencil + 4 filter half-width); Gulf of California excluded
automatically by the >=100 km cut (verify, do not special-case); front strength primary =
front-mean `gradb2` so Phases 2 and 3 measure the same quantity; filter sweep none/2/4/8;
validation split into a Cartesian scheme test and a separate native-grid metric test.

**Flagged, not re-asked:** the >=100 km offshore cut excludes the inner upwelling zone
where CC frontogenesis is most vigorous, so the headline number will describe the offshore
regime rather than the CC as a whole. Keeping the distance-stratified plot so the exclusion
stays visible.

### 2026-09-12 — Planning prompt 4 (Claude Opus 5, adversarial review by Fable)

**Deliverable:** `dev/frontogenesis/frontogenesis_planning.md` (605 lines, 12 sections).

**Note:** Q11 (branch strategy) and Q12 (library route) were left blank. The doc records
Q12 as proposed-and-unobjected and carries **Q11 forward as the single blocking open item**
(§10) rather than deciding it silently.

**Method.** Drafted the plan from rounds 1-3, then had a Fable subagent review it
adversarially as a physical oceanographer / numerical analyst. The review was substantive
and **overturned three claims in the draft**. All corrections are folded in.

**What the review broke, and what changed:**

1. **"Residual = purely diabatic" was wrong.** LLC4320 carries no explicit horizontal
   tracer diffusion; its high-order monotonicity-preserving advection has implicit
   dissipation that switches on exactly at the grid-scale gradients defining our fronts.
   `kappa_num ~ 6-60 m^2/s` damps `G` at a `4dx` feature at `~0.1-1 f` — the same order as
   the strain. **A slope of 0.5-0.8 is fully explicable by numerics with zero air-sea
   flux.** §2.3 rewritten to list four residual contributions; new Figure 2b (residual vs
   `grad^4 b` and vs `KPPhbl`) now gates any physical reading of the money plot.
2. **"The unfiltered limit isolates the diabatic part" was wrong.** The coarse-grained
   budget has a subfilter term `-grad(bbar).grad(div tau)` — a third derivative of the
   subfilter flux — that is `O(1)` at every `L` and, as `L -> dx`, *becomes* the numerical
   diffusion term rather than vanishing. §5.4 rewritten; `tau` is now computed explicitly
   (Germano/Aluie) and the sweep is reframed as a scale-transfer diagnostic.
3. **Semi-Lagrangian interpolation error is not negligible.** Bilinear interpolation of `G`
   at half-cell offset errs by `dx^2 G_xx/8` ~ 5.5% of `G`, *systematically negative at
   maxima* — i.e. it fabricates frontogenesis — against a per-hour signal of only 7-20%.
   My "0.2-0.4 cells" was the median, not the tail: strong filaments plus tides exceed 1.5
   cells. §5.3 now specifies cubic+ interpolation of `b` (not `G`) and evaluation at the
   **trajectory midpoint time** (M2 strain rotates 29 deg/hr).
4. **New, and the most consequential addition: a discrete null test.** `F = (1/2) DG/Dt`
   relies on the chain rule, which centred differences violate at `O((k dx)^2) ~ 2.5` at a
   `4dx` feature; and C-grid staggering attenuates `G` (two interpolations) and `F` (three)
   *differently*, biasing the slope 0.7-1.4 with no physics. Phase 0 now has **four**
   validation tests, and Phase 2's exit criterion is **budget closure, not a slope**.
5. **Statistics hardened** (§11): selection-on-outcome / regression-to-the-mean added to
   attenuation bias; midpoint selection and evaluation; binned conditional means split by
   sign of `X`; ratio estimators; bootstrap over frontal features, not pixels.
6. **EOS:** use JMD95 at ~0.5 dbar (what the model advected), not TEOS-10 — ~1% but
   systematic.

**Data addition found while revising.** The OSN `llc_wind` store
(`KPPhbl, PhiBot, oceTAUX, oceTAUY, SIarea`, hourly, k=0, coverage 2011-11-01 ->
2012-07-15) **contains our window**. `KPPhbl` is the key interpretive variable for the
diurnal residual. Heat fluxes are in neither OSN store, so the diabatic term must be
inferred as a residual — which is precisely why separating it from numerical diffusion
matters. Also: `CHUNKS/monterey_bay` supplies `k=1` at 11 times inside the window, so the
finite-top-cell vertical term (~30% of `F` by day) can be **bounded** in Phase 2 rather
than deferred to Phase 4.

**Decisions respected, consequences recorded.** The review argued for >=30 km rather than
>=100 km offshore, and for a longer window than 72 h. Both are JXP's settled calls (Q7b,
Q8), so they stand; the doc records the consequences honestly — the offshore cut biases the
sample toward wave-dominated weak fronts, and 72 h (3 diurnal / 3.6 inertial / 5.8 M2
cycles) cannot separate those bands, making Figure 6 indicative rather than conclusive and
making a 504 h extension the natural first follow-on.

### 2026-09-12 — Planning prompt 5 (Claude Opus 5, API survey by Fable)

**Deliverable:** `dev/frontogenesis/frontogenesis_coding.md` (533 lines, 8 sections,
milestones M0-M6).

**Structure.** §1 conventions (units, the factor of two, mask polarity, dtype, filter
scale) — most failure modes in this project are convention slips, so they are pinned first.
§2 external APIs, read from source. §3 data contracts (four zarr/nc schemas). §4 module
specifications as exact signatures — the contract the prompt docs will be written against.
§5 test strategy. §6 milestones M0-M6, **one per execution prompt doc**. §7 critical path.
§8 a pitfall checklist distilled from the adversarial review.

**Milestones.** M0 access/reconnaissance; **M1 operators + validation (HARD GATE)**;
M2 data pull; **M3 field budget (HARD GATE)**; M4 fronts/tracking; M5 figures/report;
M6 depth (deferred). M1 and M2 can run in parallel — the data pull needs no physics.
M1's gate is the discrete null test at slope = 1 +/- 0.05; M3's is budget closure, not a
slope.

**Method note.** Used a Fable subagent to read exact signatures from source rather than
writing plausible-looking calls into a doc that will be executed. This paid for itself
three times over — it caught three errors in documents I had already written:

1. **Wrong filename throughout.** On `tiles-surface-only` the physics lives in
   `calculate_fields.py`; `calculate_additional_fields.py` is the name on the **stale
   `llc4320_v2` branch** I had been reading earlier. Every citation in the planning doc was
   pointing at a file that does not exist on the branch we will work from. Fixed.
2. **Buoyancy convention was wrong.** I wrote `b = -g rho/rho0`. The repo's
   `calculate_fields.buoyancy_of_field` is `b = +g sigma0/rho0` with `g=9.81`,
   `rho0=1000.0` — it *increases with density*, the negative of the textbook definition.
   Harmless downstream (`G` and `F` are quadratic; alignment enters as `cos 2theta`), but
   documented so nobody "fixes" it. Also flagged the legacy
   `utils/physical_calculations.buoyancy_of_field` (g in km/s^2, rho_ref=1025) as
   do-not-use.
3. **The "native basis, no rotation" rule was unimplementable.** Both
   `calculate_native_gradient_tracer` and `calculate_jacobian` rotate to geographic via
   `CS`/`SN`. Corrected the rule to what actually matters: `grad b` and the Jacobian must be
   in the *same* basis (they are), `G` and `F` are rotational invariants, and only the
   **departure points** stay in native index space. Planning §5.2 rewritten.

**Three traps recorded in §2** for the execution prompts: `process_llc4320_grid` calls
`reset_coords()` and can drop comodo attrs (must re-run `_ensure_comodo_attrs` before
`set_xgcm_grid`); the confirmed `halo_mask.py:75` bug returns a single all-True 2-D face
from inside the loop when a face is entirely land, aborting the rest and inverting the
convention; and **two live timestamp formats** — `dbof` uses `'%Y-%m-%d %H:%M:%S'` while
`front_tracking.parse_time` requires `'%Y-%m-%dT%H_%M_%S'`. Also: `halo_km` is in
kilometres, not cells (our 7-cell requirement is ~13 km); `calculate_native_strain_vorticity`
returns a **dict** with shear strain and vorticity on **corners**; and
`colocate_fronts_with_properties` defaults to `nan_policy='propagate'`, which is wrong for
our NaN-masked fields.

**Still blocking:** Q11 (branch strategy). M0 cannot start until it is settled, since it
determines which branches we install and work from.

### 2026-09-19 — Planning prompt 6 (Claude Opus 5) — PAUSED, deck built but not uploaded

**State: the deck is built and QA'd; the upload to Drive did not happen.** Paused at JXP's
request to resume on his workstation.

**Done.**

- Destination confirmed on Drive: `data/HIINet/Frontogenesis`
  (folder id `15rA8LYT3yGrQaKwvy9LRHRWf92nkJ7PL`, parent `HIINet`
  `1zrn-v1yVvljkdCpgAkYy8SodT3dpsgXS`, grandparent `data`
  `1xb7TQPwXFwbfknSvuvVWsuOi-4t-CGE0`, on a shared drive). The `Frontogenesis` folder was
  created 2026-09-19, i.e. just before the request.
- 13-slide deck built with `python-pptx` in an "Ocean Gradient" palette (midnight / deep
  blue / teal, amber reserved for risk content). Slides: title; the question; the budget;
  why surface-only works (and where the discretisation takes it back); what exists vs what
  is new; the data; method (five commitments); **the residual is not purely diabatic**;
  four validation gates; figures; milestones; open decision + risks; null-result criteria.
- Saved to the repo so it survives the session:
  `dev/frontogenesis/deck/build_deck.py` (generator) and
  `dev/frontogenesis/deck/Frontogenesis_Planning.pptx`.

**QA.** Geometry and text-overflow checks pass programmatically (no off-slide shapes, no
estimated overflow, margins >= 0.5 in); slide-by-slide content dump verified. **Visual QA
was not possible** — there is no LibreOffice on this machine (`soffice` absent), so the
usual render-to-image inspection could not run. Worth a human eye on first open.

**Why the upload stalled.** `mcp__claude_ai_Google_Drive__create_file` can create a native
Google Slides file but only from uploaded content, which must be passed inline as
base64. The deck is 54 KB (39 KB after stripping the 10 unused default slide layouts) =
~52 KB of base64, which the shell persists to a file rather than returning inline. Splitting
it into three chunks and reassembling by hand risks a single-character corruption that would
silently produce an unopenable file — not worth it.

**To finish (next session).** Either (a) re-run the base64 upload on a machine/session where
the full string can be passed in one piece, or (b) simpler and safer: open
`dev/frontogenesis/deck/Frontogenesis_Planning.pptx` and drag it into the Drive folder above,
letting Drive convert it to Google Slides (File > Save as Google Slides), or (c) install
LibreOffice first so the deck can also get a proper visual QA pass before upload.

**Dependency note.** `python-pptx` is not in any conda env here; it was installed into a
throwaway venv in the session scratchpad. `build_deck.py` needs `pip install python-pptx`
to re-run.

### 2026-09-26 — Planning prompt 7 (Claude Opus 5, branch/transfer survey by Fable)

**Input:** `dev/frontogenesis/lauren_comments_on_planning.txt` (6 comments, on S4, S5.3,
S5.7, S6 and S10/Q11). Adopted essentially in full. Response written to Q&A as round 4,
with Q13-Q15 posed.

**Method.** Answered Lauren's three direct questions from current repo state (I re-fetched
both repos — her comments assume a state that is two weeks old and has moved). One Fable
subagent surveyed the chunk-transfer machinery and current branch topology in the llc repo;
I checked the fronts repo myself.

**The data offer turns out to be worth more than it looked.** I had written in planning §4
that heat fluxes are in neither OSN store, so the diabatic term could only be inferred as a
residual. True of OSN, **false of the CHUNKS store**, whose variable list already carries
`oceQnet`. With hourly chunks we would get `k=1,2` Theta/Salt (hence `b_z`), subsurface `W`,
and `oceQnet` at every step — turning the finite-top-cell vertical term from a *bound* into a
*measured* term and computing most of `grad b . grad B` **directly**. That directly attacks
the adversarial review's worst objection (numerical diffusion and air-sea forcing being
inseparable in one residual). Asked Lauren to also add `oceQsw` — a large fraction of
shortwave is absorbed inside the ~1 m top cell where our `b` lives, and net flux alone blurs
exactly the noon-peaking term Figure 6 is about.

Incidental confirmation: the source has **51 levels to ~968 m, `drF[0] = 1.0 m`,
`Z[0] = -0.5 m`** — which pins the "~1 m top cell" assumption in planning §2.2 that had been
marked *to confirm*. Cost: ~539 MB/timestep, 61 new stores ≈ **33 GB**. A `k=0..4` subset
(~4.5 GB) is **not** currently expressible — level count comes from the source
(`transfer/zarr_io.py:326`) and nothing in the config or CLI selects `k`; it would need ~10
lines. Flagged a footgun: `config._only_known()` silently drops unknown YAML keys, so a
hand-added `k_max:` would be ignored rather than rejected.

**Lauren's tracking comment is the most valuable of the six, and stronger than she framed it.**
Her integer-label worry dissolves — we advect the **boolean mask**, never labels — and the
prediction enters non-invasively because `score_candidate` already accepts a `weights` dict.
But she wrote it "seems relevant" to M4; in fact **M4's reconciliation is meaningless without
it**: if `follow()` links a front at `t` to a different physical front at `t+dt`, the
per-front tendency is not a material derivative at all. And I had missed a further point her
comment exposed: even a perfect flow-following track is insufficient, because front-mean `G`
is a mean over a **changing pixel set**, so the reconciliation must be done on the
**advected pixel set** (Lagrangian-matched pixels). That is a genuine change to the M4
acceptance criterion. Two free by-products: a principled split/merge detector, and a
Phase-3-wide quality metric (distribution of `follow()`-chosen minus flow-predicted
displacement).

**Validation figures: adopted and promoted** from optional to M1 acceptance deliverables —
one PNG per validation test; the half-cell-shift interpolation demo exactly as she described
it (truth vs bilinear-`G` vs cubic-`b`, negative bias at the maximum annotated); the
discrete-null slope **drawn** on Figure 2 as a baseline line rather than quoted; and a new
main figure 3b, rows `{b, G, 2F, tau}` x columns `{L=0,2,4,8}`, so §5.4 stops being abstract.

**Q11 is effectively resolved, and two of Lauren's premises were out of date in our favour.**

- `frontogenesis` was **not** branched off `viz_tools` — it is `origin/build_v5` minus
  `fcf9cb1` plus 5 commits that touch **only** `dev/frontogenesis/`. So its rebase is
  trivial, and the add/add conflicts I worried about in round 3 dissolve: they get resolved
  once, by Lauren, when `viz_tools` meets a `main` that already contains build_v5.
- `tiles-surface-only` **no longer needs a rebase** — it is 0 behind / 4 ahead of
  `origin/main`, a clean fast-forward. `COMODO_COORD_META` on main does carry `-0.5`.
- The actual blocker is **PR #24 ("Build v5"), still OPEN** since 2026-09-01, 10 ahead and 0
  behind main. Everything queues behind it.
- Still not on `origin/main`: `run_series` and the OSN pipeline (only on
  `tiles-surface-only`).

**Flagged for later:** the coding doc §2 API table is pinned to branch line numbers that are
about to move; it must be re-verified after the merges, before M0 runs against it.

### 2026-09-26 — Planning prompt 8 (Claude Opus 5, consistency audit by Fable)

**Deliverables.** Both planning docs updated for Q13-Q15, and **six execution prompt docs**
written: `frontogenesis_prompt_1.md` .. `_6.md`, one per milestone M0-M5, indexed in a new table
at the top of `frontogenesis_coding.md`.

| Prompt | Milestone | Gate |
|---|---|---|
| 1 | M0 access and reconnaissance | |
| 2 | M1 operators and validation | **HARD** |
| 3 | M2 data pull | |
| 4 | M3 field-level budget | **HARD** |
| 5 | M4 fronts and flow-informed tracking | |
| 6 | M5 figures and report | |

**Doc updates from Q13-Q15.** Planning §2.2 — the finite-top-cell vertical term becomes
*measured* rather than bounded; `drF[0] = 1.0 m`, `Z[0] = -0.5 m` now stated as fact. §2.3 — the
residual reduces to **numerical diffusion + interior KPP** rather than a catch-all, since the
vertical and surface-flux terms are computed. §4 — the hourly full-depth chunk transfer, and the
correction that heat fluxes are absent from *OSN* but present in the *chunk* store. §5.7 — the
flow-informed tracking requirement in full, including the advected-pixel-set reconciliation.
§6-§7 — validation figures promoted to deliverables; Figures 2b and 3b added. §10 — Q11 closed
with the four-step branch sequence. Coding doc gained `vertical.py`, flow-informed `tracking.py`
signatures, the chunk data contract, and strengthened M3/M4 acceptance criteria.

**Then a Fable subagent audited all eight documents for mutual consistency, and found 18 real
defects.** All 18 are fixed. The ones that would actually have broken execution:

1. **M2 secretly depended on M1.** Both the coding doc and prompt 3 asserted "M2 does not depend
   on M1", yet M2 was asked to write `tile330_masks.nc` — which needs `masking.py`, an M1 module.
   Resolved by moving mask creation to **M1** and the static grid to **M0**, leaving M2 pulling
   raw fields only. The stated parallelism is now true.
2. **The M3 gate equation differed between planning and coding/prompt** — one subtracted
   `numerical` as if it were known, the other put it on the right-hand side. Unified.
3. **M1's tests needed M2's products.** V2 and V6 require the real tile grid; gate 3's
   real-velocity variant needs two consecutive hours. Fixed by making `tile330_grid.zarr` and a
   **two-hour** pull M0 deliverables, and marking those two tests as not pure-offline.
4. **Prompt 4 forbade front finding while its own acceptance needed front pixels.** Resolved by
   defining M3's front-pixel selection as a percentile of `G` at the midpoint time inside
   `mask_analysis` — no labelling, no `tile_find` — which also keeps the budget from being
   entangled with thresholding choices. Bootstrap units now differ by milestone: spatial blocks
   in M3 (no objects exist yet), frontal features in M4.
5. **`drF` had no source.** `vertical.py` needs it; OSN's grid is 2-D and does not carry it. It
   now comes from the chunk store's 3-D grid in M2, and was removed from M0's question list
   (five questions, not six).
6. **V4/V4b/V5 numbering was incoherent** across three docs, with "four gates" heading a list of
   six functions. Now: **four gates V1-V4, two supporting figures V5-V6, six PNGs.**
7. **Stale pre-Q13 text** still said the vertical term would be "bounded" and that heat fluxes
   were "in neither store" — directly contradicting the same documents' own updated sections.
8. **Figures were scheduled in M5 but required in M3/M4.** They are now produced where their data
   lands (M1: V1-V6; M3: 1-7, 10; M4: 8-9), with M5 consolidating captions and regeneration.
9. Plus: module ownership of `coarsegrain.py` and `vertical.py`; `halo_mask` taking km in one
   place and cells in another (now cells everywhere, converted from measured `dxC`, no hard-coded
   13 km); `fig10_validation` vs `fig10_term_budget`; an 85th/90th percentile mismatch (now
   "read `finding_config_D.yaml`" rather than asserting either); `tile_find`, `score_candidate`,
   config `D` and `flow_weighted_score` called but never specified; the trap count ("three" for
   four marked traps); a Python-version disagreement; a duplicated paragraph in planning §5.2;
   the decisions table still listing Q11 as open with Q13-Q15 missing; and three
   inconsistently-used names for the measured tendency (now deliberately three-tiered and
   documented as such).

**Worth noting for its own sake:** delegating the audit caught things a re-read would not have.
Several defects were *between* documents I had written at different times — exactly the class of
error that reading each one in isolation cannot surface.

**Still outstanding (not blocking):** PR #24 remains open, so the Q15 merge sequence has not run;
once it does, the coding doc's §2 API table must be re-verified before M0 executes against it.

### 2026-09-27 — Execution prompt 1, task 1: environment (Fable)

**Scope.** Task 1 of `frontogenesis_prompt_1.md` only (env). Tasks 2-5 not started.

**Env `frontogenesis`, Python 3.13.15.** 3.14 was checked and rejected: conda-forge does carry
`scikit-fmm 2025.6.23` builds for `python 3.14.* *_cp314` and `xgcm 0.9.0` is noarch
(`Requires-Python >=3.9`), so the prompt's literal test *passes* — but PyPI has **no** scikit-fmm
wheel for 3.14 (`pip download --only-binary=:all:` under the py3.14 `ocean14` env: "from versions:
none"), `xgcm 0.9.0` depends on the unmaintained `future` (1.0.0, Feb 2024, predates 3.13/3.14),
and `fronts` drags `timm==0.3.2`, PyQt6, pyvista/trame and healpy, none of which are worth
debugging on 3.14 for this project. The prompt says "prefer 3.13"; 3.13 it is.

**Install commands (in order).**

```bash
# dbof worktree (main checkout stays on full_globals_run_for_cutouts_v2_2)
cd ~/Oceanography/python/llc4320-native-grid-preprocessing && git fetch origin
git worktree add ~/Oceanography/python/llc4320-tiles-surface-only -b tiles-surface-only origin/tiles-surface-only

mamba create -y -n frontogenesis -c conda-forge python=3.13 numpy scipy xarray dask zarr pandas \
  matplotlib scikit-image scikit-learn h5py h5netcdf netcdf4 pyyaml cartopy cmocean seaborn bokeh \
  tqdm astropy astropy-healpix healpy gsw "xgcm<0.10" xmitgcm scikit-fmm s3fs fsspec ujson \
  pytorch torchvision pyarrow boto3 cftime emcee corner ipython pyvista skan smart_open \
  umap-learn llvmlite pyqtgraph pytest importlib-metadata pip setuptools future

PY=~/miniforge3/envs/frontogenesis/bin/python
(cd ~/Oceanography/python/llc4320-tiles-surface-only && $PY -m pip install -e . --no-deps)
(cd ~/Oceanography/python/fronts && $PY -m pip install -e .)   # with deps: a --dry-run showed it
      # adds only PyQt6, timm 0.3.2, trame*, pyobjc — torch untouched
```

`kerchunk` not installed, as specified. `pip check` reports two metadata-only complaints from
`dbof`'s pins (`xarray==2025.10.1`, `xgcm>=0.10`); harmless for the first, see below for the second.

**Resolved versions.**

| package | version | | package | version |
|---|---|---|---|---|
| python | 3.13.15 | | s3fs | 2026.9.0 |
| numpy | 2.5.3 | | fsspec | 2026.9.0 |
| xarray | 2026.7.0 | | ujson | 6.0.0 |
| dask | 2026.8.0 | | h5netcdf | 1.8.1 |
| zarr | 3.4.0 | | matplotlib | 3.11.2 |
| xgcm | **0.9.0** | | torch | **2.13.0** (conda-forge cpu; not the 2.8.0 pin) |
| xmitgcm | 0.5.2 | | scipy / pandas | 1.18.1 / 3.0.6 |
| scikit-fmm | 2025.6.23 | | torchvision | 0.28.0 |

- `dbof` — editable from worktree `~/Oceanography/python/llc4320-tiles-surface-only`, branch
  `tiles-surface-only` @ **`938bce1`** (2026-09-16 "nb check"), tracking origin. Untouched.
- `fronts` — editable from `~/Oceanography/python/fronts`, branch **`frontogenesis`** @ `5b30711`
  (2026-09-27, merge of origin/main). Not switched.
- Exports: `dev/frontogenesis/env/frontogenesis_env.yml` (`conda env export --no-builds`) and
  `dev/frontogenesis/env/frontogenesis_pip_freeze.txt`.

**Smoke test (no data pulled).** All of these import from the env: `dbof`; `rect_ij_to_tile`,
`process_llc4320_grid`, `get_remote_gridfile`, `get_remote_llc_data`, `get_remote_llc_wind_data`,
`set_xgcm_grid`, `osn_date_to_iteration`, `_tile_indexer`, plus `native_gradient`,
`calculate_fields`, `static_masks`, `halo_mask` (so `skfmm` links); `fronts` and
`fronts.finding.algorithms.fronts_from_gradb2`; `xgcm` 0.9.0 (< 0.10 asserted); `skfmm`.
`rect_ij_to_tile(13320, 9720)` -> `TileInfo(tile_idx=330, face_idx=10, j_face_slice=0:720,
i_face_slice=2880:3600)` as the prompt states; `osn_date_to_iteration('2012-07-02 00:00:00')`
-> 1022976.

**Surprises / contradictions with the prompt and coding doc — three, one of them blocking.**

1. **BLOCKING: `xgcm<0.10` is incompatible with `dbof @ tiles-surface-only`.** `set_xgcm_grid`
   (`dbof/llc4320_ingestion/grid.py:121`) calls `xgcm.Grid(ds_grid, padding='fill')`, and
   `padding=` exists only in xgcm >= 0.10 (0.9.0's constructor takes `periodic`/`boundary`;
   0.10's takes `padding`). Verified offline on a 4x3 synthetic staggered dataset:
   `set_xgcm_grid(ds, use_connections=False)` -> `TypeError: Grid.__init__() got an unexpected
   keyword argument 'padding'`, whereas `xgcm.Grid(ds, periodic=False, boundary='fill')` works.
   The branch's own `pyproject.toml` pins `xgcm>=0.10` with the comment "earlier releases cannot
   exchange a staggered vector pair across a rotated face connection on dask-backed input", and no
   other `dbof` code passes `periodic=` or `boundary=`. So the coding doc's reason for `<0.10`
   ("0.10 removed `periodic` and renamed `boundary`") is *true of xgcm* but *backwards for dbof*:
   dbof already targets the 0.10 API. Left at 0.9.0 as instructed; the fix is one of
   (a) `mamba install -n frontogenesis -c conda-forge "xgcm>=0.10"` — recommended, it is what
   `calculate_fields`/`native_gradient` were developed against — then re-export the env files, or
   (b) keep 0.9 and build the grid in `osn_tiles.py` with
   `xgcm.Grid(g_tile, periodic=False, boundary='fill')` instead of `set_xgcm_grid`. Decide before
   task 2.
2. **`_ensure_comodo_attrs` does not exist** on `tiles-surface-only` @ 938bce1. Coding doc §2.2
   places it at `tile_utils.py:334` and `_tile_indexer` at L372; in fact `_tile_indexer` is at
   L334 and the comodo helper is the *public* `ensure_comodo_attrs(ds, *, strict=False,
   source=None)` in `dbof/llc4320_ingestion/grid.py` (commit 8af5371 "update comodo",
   2026-09-02, then cd31497 "restructure"). The task-2 snippet must import that instead.
3. **Branch state differs from the prompt.** PR #24 (`build_v5` -> `main`) was **merged today**,
   2026-09-27 12:27 UTC, not "still open". `fronts` is on `frontogenesis`, which is `main` + 18
   commits (it contains PR #24). It is **not** based on `viz_tools`: `origin/viz_tools` has 65
   commits absent from both `main` and `frontogenesis` (merge-base 6807f82), and two files the
   coding doc §2.5 cites exist **only** on `viz_tools`: `fronts/front_tracking.py` and
   `fronts/llc/tiles.py` (also the `run_v5_*_chunks.yaml` configs Lauren mentioned). The other
   §2.5 files (`finding/algorithms.py`, `finding/pyboa.py`, `properties/colocation.py`,
   `properties/algorithms.py`, `runs/prototypes/one_full/build_v5.py`, `viz/curtains.py`) are
   present on `frontogenesis`. M0-M3 do not need `front_tracking.py`; M4 does, so `viz_tools`
   has to be merged into `frontogenesis` (or the file cherry-picked) before then.

Minor: `dbof`'s declared deps also pin `boto3==1.41.5`, `dask==2025.10.0`, `s3fs==2025.9.0`;
all ignored via `--no-deps`, and nothing imported so far cares.

**Addendum (2026-09-28) — xgcm upgraded, overriding the prompt's `xgcm<0.10` pin.** The pin is
wrong for `tiles-surface-only`: `set_xgcm_grid` (`grid.py:133,136`) passes `padding='fill'`,
which exists only in xgcm>=0.10, and the branch's `pyproject.toml` itself requires
`xgcm>=0.10`. Ran `mamba install -n frontogenesis -c conda-forge "xgcm>=0.10"`, which moved
only xgcm, from 0.9.0 to **0.10.1**; torch is still 2.13.0. Checked on a synthetic staggered
dataset: `ensure_comodo_attrs` followed by `set_xgcm_grid(ds, use_connections=False)` builds
X and Y axes, not periodic, with `padding='fill'`. `env/frontogenesis_env.yml` and
`env/frontogenesis_pip_freeze.txt` were re-exported. `pip check` still lists `ecco-v4-py`,
`reader` and `seawater` as missing, but grep finds no import of any of them in
`src/dbof`, so they are unused declared deps and were left out. The remaining pin mismatches
(torch==2.8.0, xarray, dask, s3fs, boto3) are expected with `--no-deps`.
**Contradiction with the planning/coding docs:** any reference there to `xgcm<0.10` (or to
`periodic=`/`boundary=` arguments) should be corrected to `xgcm>=0.10` with `padding=`.

### 2026-09-28 — Execution prompt 1, task 2: first contact with the data (Fable)

**Scope.** Task 2 of `frontogenesis_prompt_1.md` only: `py/osn_tiles.py` far enough to load one
timestamp from both OSN stores. Tasks 3-5 (five questions, `tile330_grid.zarr`, QA plot) not
started; no physics written.

**Written.** `dev/frontogenesis/py/osn_tiles.py` (new `py/` dir, ~170 lines): `TILE_RECT_I/J`,
`OSN_ENDPOINT`, `CORE_VARS`, `WIND_VARS`; `tile_spec()` (wraps `rect_ij_to_tile`),
`tile_indexer()` (wraps `dbof`'s private `_tile_indexer`), `load_grid(endpoint, tile=None)`
(`get_remote_gridfile` -> `process_llc4320_grid` -> `isel(face=[10], tile)` -> `.compute()` ->
`ensure_comodo_attrs`, plus the §3.1 attrs `face_index, j_face_start, i_face_start, rect_i,
rect_j, source`), `build_xgcm()` (`set_xgcm_grid(..., use_connections=False)`),
`load_hour(ts, tile=None, endpoint, keep=CORE_VARS, compute=True)` and
`load_wind_hour(ts, ...)` (`keep=KPPhbl, oceTAUX, oceTAUY`). Both hourly loaders share a tail
that selects `keep`, subsets to the tile with the same indexer, restores `time` as a length-1
dim (`expand_dims`, so hours concatenate later), computes, and **raises `ValueError` if the
store's decoded `time` differs from the requested timestamp** — the iteration conversion is
checked against the store's own clock on every pull. `pull_series` (§4.1) not written (M2).
Nothing outside `dev/frontogenesis/` touched; `tiles-surface-only` untouched.

**Run: `'2012-07-02 00:00:00'` -> OSN iter 1022976, tile 330 (face 10, j 0:720, i 2880:3600).**

- `get_remote_gridfile`: 14 s (13 kerchunk JSONs, lazy). Raw store dims
  `(face 13, j/i/j_g/i_g 4320, time 10312)`; **every variable arrives as a coordinate** (zero
  data_vars), 38 coords including `XG, YG, hFacW, hFacS, rAw, rAs, drC, drF, Z, Zl, Zu, Zp1,
  PHrefC, rhoRef, niter, time`. `hFacC` is `(face, j, i)` — 2-D, no `k`.
- `load_grid()`: 18 s, 24.9 MB in memory, dims `(face 1, j 720, i 720, i_g 720, j_g 720)`, all
  twelve §3.1 vars float32 `(1,720,720)`: `XC YC rA Depth hFacC SN CS` on `(j,i)`; `dxC dyG` on
  `(j,i_g)`; `dyC dxG` on `(j_g,i)`; `rAz` on `(j_g,i_g)`. Box lon -127.990..-113.010, lat
  26.659..**38.267** (planning §4 says 38.20 — minor). `dxC` median **1796 m**, `dyC` median
  **1950 m** (quick read only; the proper answer to question 3 is task 3's).
- `build_xgcm()`: builds in <0.01 s — X (`i` center, `i_g` left) and Y (`j` center, `j_g` left),
  not periodic, `padding='fill'`, xgcm 0.10.1.
- `load_hour`: 7.0 s, 12.5 MB; `Theta Salt W Eta (time,face,j,i)`, `U (time,face,j,i_g)`,
  `V (time,face,j_g,i)`, all float32 `(1,1,720,720)`; scalar coords `k=0, k_l=0, niter=1022976`,
  `time=['2012-07-02T00:00:00']` (encoding `seconds since 2011-09-10`). Value ranges: Theta
  11.3..33.1, Salt 28.4..48.4, U -1.01..1.17, V -1.00..1.75, W -9.2e-5..1.5e-4, Eta -2.8..2.0.
- `load_wind_hour`: 5.6 s, 6.2 MB; `KPPhbl (time,face,j,i)` 0.5..60.3 m, `oceTAUX (…,j,i_g)`,
  `oceTAUY (…,j_g,i)`; same `niter`/`time` as the surf store. Store also offers `PhiBot`, `SIarea`.
- Ocean: **`hFacC>0` fraction 0.6884** (hFacC min 0, max 1; `Depth>0` fraction identical).
  Incidental, *not* task 3's answer: the finite fraction of every centred field (`Theta, Salt, W,
  Eta, KPPhbl, oceTAU*`) is also **0.6884** and `U`/`V` 0.6866/0.6873 — i.e. land looks like NaN
  in the data, not 0. Task 3 should confirm this against `hFacC` cell by cell.
- Wall total ~45 s for grid + two stores; bytes on the wire not measured (the kerchunk reads go
  through s3fs; in-memory sizes above are the float32 tile arrays).

**Trap confirmations.**

1. **Comodo attrs survive `process_llc4320_grid` on real OSN data.** Traced at four points: the
   raw gridfile already carries `axis` (+ `c_grid_axis_shift=-0.5` on `i_g`/`j_g`, plus
   `long_name`, `standard_name`, `swap_dim`) on all four horizontal index coords; identical after
   `process_llc4320_grid` (`reset_coords()` demotes only non-index coords — index coords keep
   their attrs), after `isel`, and after `ensure_comodo_attrs` (a no-op here, since
   `comodo_attrs` defers to an existing `axis`). So the §2.1 trap ("can DROP comodo attrs") does
   **not** fire for the OSN gridfile; `ensure_comodo_attrs` stays in `load_grid` as a cheap
   guard for other stores, and `set_xgcm_grid` builds either way.
2. **Timestamp formats.** (a) Both OSN stores are keyed by the *same* OSN iteration
   (`osn_date_to_iteration`, MIT + 10368): iter 1022976 gives `time == 2012-07-02T00:00:00` and
   `niter == 1022976` from **both** `llc_surf` and `llc_wind`, equal to the requested timestamp.
   There is no second OSN time convention; the "two timestamp formats" of the acceptance criteria
   are dbof's `'%Y-%m-%d %H:%M:%S'` vs `front_tracking`'s `'%Y-%m-%dT%H_%M_%S'` (§2.5 trap).
   (b) That round trip verified: `'2012-07-02 00:00:00'` -> `'2012-07-02T00_00_00'` -> back,
   equal. `load_hour` accepts only the dbof format and rejects the underscore form with
   `ValueError` (strptime against `DATE_FMT`), so a wrong-format string cannot silently pull
   the wrong hour. The tracking-side converter itself belongs in `tracking.py` (M4), not here.
3. `calculate_jacobian` args and `halo_mask.py:75` — not exercised (M1 / task 3).

**Contradictions / corrections to the docs.**

- **`drF` is in the OSN gridfile.** Coding doc (prompt 1 §3, coding §6 M0) says "OSN's grid is
  2-D and carries no `drF`; confirmed in M2 from the chunk store's 3-D grid". The 2-D part is
  right (`hFacC` has no `k`), but the gridfile carries the top-cell vertical scalars:
  **`drF = 1.0`, `Z = -0.5`, `Zl = 0.0`, `Zp1 = 0.0`** (all 0-d float32, k = 0 only), plus
  `drC`, `Zu`, `PHrefC`, `rhoRef`. `process_llc4320_grid` drops them (not in `coords_to_keep`);
  `process_llc4320_3d_grid` would keep them. So `drF[0] = 1.0 m`, `Z[0] = -0.5 m` is confirmable
  from OSN now; M2's chunk-grid check becomes a cross-check, not the only source.
- **§4.1 `load_hour` "(lazy)"**: implemented with `compute=True` by default (the task asked for
  computed tiles); `compute=False` returns the lazy view. The signature `load_hour(ts, tile,
  endpoint=...)` is honoured with `tile=None` defaulting to `tile_spec()`.
- **§3.2 dims `(time, j, i)`**: the loaders keep a length-1 `face` dim (`(time, face, j, i)`)
  because `native_gradient` documents its inputs as `(face, j, i)` and `face_seam_mask` needs
  it; drop it at zarr-write time (task 4) if §3.2 is to be taken literally, or amend §3.2.
- `get_remote_gridfile` output has a `time` dim of 10312 hourly records (2011-09-13T00 ->
  2012-11-15T15) with `niter` — a ready-made iteration<->time table matching planning §4's
  coverage. Harmless; `process_llc4320_grid` drops it.
- Planning §4's box lat upper bound 38.20 is 38.267 on the tile's `YC`.
- Minor: `get_remote_llc_data` prints six progress lines per call; fine for one hour, noisy for 72.

**Nothing blocks task 3.** Network access from this machine works anonymously; all five
questions can be answered from `load_grid()` + `load_hour()` (+ `load_wind_hour()`) as written.

### 2026-09-28 — Execution prompt 1, task 3: five empirical questions (Fable)

**Scope.** Task 3 of `frontogenesis_prompt_1.md` only. Tasks 4-5 (`tile330_grid.zarr`, QA plot)
not started; no physics module written. Everything below is from
`dev/frontogenesis/py/m0_recon.py` (new, throwaway but reproducible, ~330 lines; runs in ~280 s,
network-bound) against tile 330 at `2012-07-01 23:00`, `2012-07-02 00:00` (= t0, all statistics)
and `01:00`, plus the `llc_wind` store at t0, plus `hFacW`/`hFacS` cut from the raw gridfile
(`process_llc4320_grid` drops them). Where a gradient is needed it is the **dbof operator M1
will use** (`calculate_native_gradient_tracer`, `calculate_jacobian`,
`calculate_native_strain_vorticity`, `buoyancy_of_field`, `_frontogenesis_formula`), so the
numbers are for the real stencils. "Interior" = ocean, 3 cells clear of land and of the tile
rim (345,420 of 356,877 ocean cells); "front" = interior with `G` above its interior p90
(34,542 cells). Q5 is documentary (web) plus a kernel calculation that needs no data.

**Q1 — Land is NaN, not 0; cell-for-cell equal to the `hFac` masks.** 518,400 cells; `hFacC==0`
in 161,523 (ocean fraction 0.6884); `hFacC` takes only the values 0 and 1 at k=0 (no partial
cells). `hFacW==0` in 162,445 and `hFacS==0` in 162,088, and `hFacW[j,i] ==
min(hFacC[j,i-1], hFacC[j,i])`, `hFacS[j,i] == min(hFacC[j-1,i], hFacC[j,i])` at 100% of
cells. Cell-by-cell: `Theta, Salt, W, Eta` (core) and `KPPhbl, PhiBot, SIarea` (wind store)
are NaN in exactly the 161,523 `hFacC==0` cells — 0 finite-on-land, 0 NaN-on-ocean. `U` is NaN
in exactly the 162,445 `hFacW==0` cells and `V` in exactly the 162,088 `hFacS==0` cells (so `U`
is NaN in 922 and `V` in 565 cells whose *centre* is ocean: the coast-facing velocity faces).
Exact zeros in the ocean: `Theta/Salt/Eta` none; `W` 4 cells (`hFac=1`, float32 zeros, noise);
`SIarea` all ocean cells (no ice, as expected). The NaN pattern is identical at t-1h and t+1h.
One oddity: **`oceTAUX`/`oceTAUY` are masked with the centred `hFacC` mask, not `hFacW`/`hFacS`**
— 922 / 565 finite values sit on faces the model treats as land. Harmless for context plots,
but any stress-divergence must re-mask with `hFacW`/`hFacS`.
*Verdict:* **NaN.** `b(0,0)` never happens; there is no coastal gradient ribbon to see. Since
the dbof stencils propagate NaN, the 3-cell stencil part of the 7-cell halo is automatic; the
explicit halo is still needed for the 4-cell filter support and for `skfmm` distances.

**Q2 — `W[k_l=0]` is not ~0: it is `dEta/dt`.** OSN carries only the `k_l=0` interface (dims
`(time, face, j, i)`, scalar coord `k_l=0`); there is no second interface to compare against.
Over the ocean `|W|`: median **5.5e-5**, p90 8.2e-5, p99 9.6e-5, max 1.5e-4 m/s; signed median
+5.4e-5 (the whole tile is rising at t0: tile-mean `Eta` 1.058 -> 1.304 -> 1.427 m over t-1h,
t0, t+1h, spatial std of the hourly change 0.084 m — a tide, on a ~1.3 m mean offset).
Against the free-surface rate: centred `(Eta(t+1h)-Eta(t-1h))/2h` vs `W(t0)`: **corr 0.9936,
regression slope W/(dEta/dt) = 1.037, rms residual 3.6e-6 m/s = 6% of rms W**; forward
`(Eta(t+1h)-Eta(t0))/1h`: corr 0.81, slope 1.35 (it is centred half an hour late). A centred
±1 h difference under-recovers the rate of an M2 harmonic by `sin(x)/x = 0.958`, i.e. the
expected slope is 1.044 — the measured 1.037 is that. So `W(k_l=0) = dEta/dt(t0)` to within
the sampling error. This is exactly what the model does: LLC4320 runs a **linear** implicit
free surface with `exactConserv=.TRUE.` (Q5; `nonlinFreeSurf`/`select_rStar` unset), and
`integrate_for_w.F` integrates continuity from the bottom, so `wVel(k=1) = dEtaHdt`.
Horizontal gradient at the surface: `|grad_h W|` interior median **9.5e-10**, p90 2.0e-9, p99
3.4e-9, max 1.3e-8 s^-1 — 10-30x the `~1e-10 s^-1` quoted in planning §2.2. For scale, on the
same stencils: `|grad b|` median 4.1e-8, p99 4.3e-7 s^-2; `|F|` median 9.7e-21, p99 3.1e-18
s^-5; `|delta|` median 1.1e-5, p99 5.3e-5 s^-1; `drF*|grad delta|` median 4.8e-9, p99 2.2e-8
s^-1. Note **`|W(0)|` (5.5e-5) is 5x `drF*|delta|` (1.1e-5)**: the vertical velocity at the base
of the 1 m top cell is dominated by the free-surface motion, not by the convergence.
Tilting term `T = -b_z (w_x b_x + w_y b_y)` vs `F`, with `b_z` bracketed because surface-only
data cannot give it (1e-5 well-mixed, 1e-4 moderate, 4e-4 diurnal warm layer, s^-2; planning
§2.2), reported as `rms(T)/rms(F)` over the interior and, in brackets, the median pointwise
`|T|/|F|` at fronts:
- using the surface `W` gradient (what OSN gives): **0.03% [0.15%], 0.3% [1.5%], 1.2% [6%]**;
- using the cell-base convergence part `drF*grad(delta)`: **0.4% [1.6%], 3.7% [16%], 14% [63%]**
  (pointwise p90 at fronts reaches 4.2 at `b_z=4e-4`, where `F` itself is near zero).
*Physics.* In a linear-free-surface MITgcm the top cell is a fixed 1 m box that the free
surface moves through. Checked in the source (`pkg/generic_advdiff/gad_advection.F`, the
default non-`GAD_MULTIDIM_COMPRESSIBLE` branch; `model/src/calc_adv_flow.F`): the surface
transport is set to zero, the vertical sweep subtracts `T*(rTrans_base - 0)`, so the top-cell
vertical tendency is the advective form `-w_base (T_base - T)/drF` with `w_base = W(0) +
drF*delta` (z up), and the global non-conservation `w(0)*T` is left alone (`linFSConserveTr`
unset). There is no spurious `T*dEta/dt/drF` term, but the cell **is** ventilated through its
base at `~dEta/dt` (tidal, reversing). Consequences for §2.2: (i) the continuum argument's
premise "w vanishes at z=0" is false as stated — what vanishes is the velocity *relative to the
free surface*, and the along-surface budget `D_h b/Dt = B` follows from the kinematic condition
`w = D eta/Dt`, not from `w = 0`; the conclusion survives, the sentence does not; (ii) the
discrete top-cell vertical term is real and must be built from the **model's `W(k_l=1)`** (M2
chunk store) — `w(-dz) = -dz*delta` has the wrong sign (continuity gives `+dz*delta`) and
omits the dominant `dEta/dt` part; (iii) the `b_z`-uniform form `-b_z grad w . grad b` drops
`-w grad(b_z) . grad b`, which with `w ~ 5e-5` and front-scale changes in stratification is
not obviously smaller — `vertical.py` should compute `-grad_h b . grad_h[ w_base (b_base -
b)/drF ]` directly rather than the factorised form.
*Verdict:* **"W[k_l=0] ~ 0" is contradicted**; `W(0) = dEta/dt`. The tilting term's *surface*
part is <~1% of `F` in rms for any plausible `b_z`, so surface-only data remain sufficient for
the kinematic side; the *cell-base* part is 0.4-14% rms (order-one pointwise) across the `b_z`
bracket — same order as planning's "~30% by day" — and stays a measured M2/M3 term.

**Q3 — Spacing is 1.7-2.1 km, not 1.8-2.3; and face 10 is rotated 90 degrees.** `YC` spans
26.659-38.267N; the 36.5-37.5N band is 46,800 cells. In that band: `dxC` **1.698-1.719 km
(median 1.708)**, `dyC` **1.838-1.862 km (median 1.850)**; full tile `dxC` 1.681-1.903 (median
1.796), `dyC` 1.819-2.070 (median 1.950); `dxG == dxC` and `dyG == dyC` to three figures.
`dyC/dxC` is a constant **1.086** (1.0816-1.0878). Orientation: `XC` is constant along `i` and
changes along `j` by exactly `1/48 deg` per cell (`XC[0,0] = -127.990`, `XC[-1,0] = -113.010`);
`YC` changes along `i` (`YC[0,0] = 38.267`, `YC[0,-1] = 26.659`). With `CS = 0`, `SN = -1`
(Q4) the dbof rotation is `u_east = V`, `v_north = -U`: **on this face `j`/`V`/`dyC` are
zonal (eastward) and `i`/`U`/`dxC` are meridional (`i` increasing southward)**; `dyC` is the
`(1/48) deg cos(lat)` zonal spacing (1.852 km at 37N — matches), and the meridional spacing is
8% smaller. Expected "1.8-2.3 km" is high; the 7-cell halo is 12.0 km meridional x 13.0 km
zonal at 37N ("~13 km" stands). Speed at centres (`interp_pair_to_center` of raw `U,V`), ocean:
median **0.193**, p90 0.381, p99 0.643, max 1.74 m/s (band: median 0.25, p99 0.72). Hourly
displacement in index space, `di = |u| dt/dxC`, `dj = |v| dt/dyC`: ocean `|d|` median **0.37
cells**, p90 0.75, p99 **1.28**, max 3.5; 30% of ocean cells exceed 0.5 cell, 3.4% exceed 1,
0.4% exceed 1.5; at fronts (`G > p90`) median **0.54**, p90 1.10, p99 **1.64**, max 2.5; in the
37N band median 0.52, p99 1.51.
*Verdict:* planning §5.3's "typical 0.2-0.4 cells" (0.37) and "exceeds 1.5 cells at the strong-
front tail" (p99 1.64 at fronts) are both confirmed; the km spacing and the axis orientation
need correcting (coding §2.5's `follow(km_per_px=2.3)` default is 25% too large here: ~1.8).

**Q4 — Rotation terms are identically zero on this tile; the metric term is ~0.1%.** `SN` is
exactly `-1.0` in every cell (one distinct float32 value); `CS` is zero to rounding (12,087
distinct values, all with `|CS| <= 1.24e-12`); the grid angle `atan2(SN, CS)` is `-90.000 deg`
everywhere, range 0.000 deg. Hence `|grad CS|` median 8e-17 m^-1 (dbof operator and
`np.gradient` agree), `|grad SN| = 0`, and `u|grad CS|` median 1.5e-17 s^-1, max 3e-16 —
**0.0000%** of 1e-5 and of the measured strain. Measured strain `|sigma| =
sqrt(sigma_n^2 + sigma_s^2)` from `calculate_native_strain_vorticity` (shear squared then
`interp_corner_squared`), interior: median **1.9e-5**, p90 4.1e-5, p99 7.9e-5, max 3.9e-4
s^-1; at fronts median 2.9e-5, p99 1.2e-4 (the interpolated Jacobian gives median 1.7e-5).
Spherical metric term `u tan(phi)/a`: median **1.8e-8**, p90 3.9e-8, p99 6.8e-8, max 1.3e-7
s^-1; vs the nominal 1e-5: median 0.18%, p99 0.68%, max 1.26%; vs the local `|sigma|`
pointwise: median **0.09%**, p99 0.54% (fronts 0.09%, 0.60%); ratio of medians 0.096%.
*Verdict:* confirmed, more strongly than claimed — there is no rotation error at all on face
10 (rotate-then-differentiate and differentiate-then-rotate are the same exact axis swap), and
the only neglected term is the metric one at 0.1% median, 0.5-0.7% in the low-strain tail.

**Q5 — `tempAdvScheme = saltAdvScheme = 7` (OS7MP), no explicit horizontal diffusion, linear
free surface, staggered time step.** No MITgcm namelist exists anywhere under `~/Oceanography`
(grep for `tempAdvScheme|select_rStar|nonlinFreeSurf|staggerTimeStep` and `find -name data`
both empty). Source: **`MITgcm_contrib/llc_hires/llc_4320/input/data`** (GitHub mirror
https://github.com/MITgcm-contrib/llc_hires; raw
https://raw.githubusercontent.com/MITgcm-contrib/llc_hires/master/llc_4320/input/data;
production-era commit `4627a7a8`, 2014-06-27, differs from master only in `viscC4Leith`,
`chkptFreq` and two I/O flags — both fetched and diffed here). `&PARM01` verbatim:
`viscAr=5.6614e-04, no_slip_sides=.TRUE., no_slip_bottom=.TRUE., diffKrT=5.44e-7,
diffKrS=5.44e-7, rhonil=1027.5, eosType='JMD95Z', hFacMin=0.3, implicitDiffusion=.TRUE.,
implicitViscosity=.TRUE., viscC4Leith=2.0 (2014) / 2.15 (master), viscC4Leithd=same,
viscA4GridMax=0.8, useAreaViscLength=.TRUE., highOrderVorticity=.TRUE.,
bottomDragQuadratic=0.0021, tempAdvScheme=7, saltAdvScheme=7, StaggerTimeStep=.TRUE.,
multiDimAdvection=.TRUE., vectorInvariantMomentum=.TRUE., implicitFreeSurface=.TRUE.,
exactConserv=.TRUE., convertFW2Salt=-1., useRealFreshWaterFlux=.TRUE., implicSurfPress=0.6,
implicDiv2DFlow=0.6`; `&PARM03`: `deltaT=25., abEps=0.1, dumpfreq=3600.`; `&PARM04`:
`usingCurvilinearGrid=.TRUE., delR = 1.00, 1.14, 1.30, ...` (**90 values**; `drF[0]=1.0 m`
agrees with the OSN gridfile). **Absent, hence MITgcm defaults: `diffKhT=diffK4T=0`,
`viscAh=viscA4=0`, `nonlinFreeSurf=0`, `select_rStar=0`, `linFSConserveTr=.FALSE.`** — so no
z*, a linear implicit free surface, Crank-Nicolson-ish barotropic stepping, KPP
(`data.pkg: useKPP=.TRUE.`; `data.kpp: Ricr=0.3559, Riinfty=0.6998`), tidal potential via
`data.exf apressurefile='EOG_pres_tide'`. Code: `checkpoint65v` with only CPP/SIZE headers in
`code/` at production time (`readme.txt`); the 2023 "Skitka modified Leith" files on master are
for later reruns. Leith timeline from the file's git history: 2.0 until the crash/restart at
step 870912 (~2012-05-19; commit `8b35601d`), 2.1 after, later 2.15 (commit `aa9d6571`, step
undocumented) — **our July 2012 window is after the restart, so 2.1 or 2.15**. Published
confirmation: NASA S-MODE model description
(https://data.nas.nasa.gov/smode/smodedata/data/Info/model_description.pdf): "same as LLC4320:
... a flux-limited, seventh-order, monotonicity-preserving advection scheme (Daru and Tenaud,
2004) and the modified Leith scheme of Fox-Kemper and Menemenlis (2008) ... KPP ... Cd =
0.0021"; Rocha et al. 2016 JPO App. D (dt = 25 s, points to `MITgcm_contrib/llc_hires`). Su
2018 / Torres 2018 / Arbic 2018 full texts not retrievable (paywall); not quoted.
*`kappa_num`.* The unlimited OS7MP kernel is the 7th-order upwind-biased face interpolation
`(-3, 25, -101, 319, 214, -38, 4)/420` on `i-3..i+3`; its semi-discrete Fourier symbol gives
the exact scale-dependent implicit diffusivity `kappa/(|u| dx)` = **0.093 at 2dx, 0.066 at 3dx,
0.023 at 4dx, 4.7e-3 at 5.6dx (10 km), 6.8e-4 at 8dx, 1.1e-4 at 11dx (20 km)**; the
modified-equation form `|u| dx^7 k^6 / 280` agrees for `lambda >= 4dx` and overshoots at 2dx.
Courant `u dt/dx` at dt=25 s is 0.003 (median) to 0.009 (p99), so the one-step time
correction is negligible. With `dx = 1.8 km` and the tile's speeds (`|u| dx` = 342 / 684 /
1152 m^2/s at median / p90 / p99), the damping rate of `G` is `2 kappa k^2`:
- **grid scale 2dx (3.6 km):** kappa 32 / 63 / 107 m^2/s, `G` e-folds in **1.4 / 0.7 / 0.4 h**;
- **4dx (7.2 km):** kappa 7.9 / 16 / 27 m^2/s, `2 kappa k^2` = 1.2 / 2.4 / 4.1e-5 s^-1, `G`
  e-folds in **23 / 11.5 / 6.8 h**, 4% / 0.2% / 0% of `G` survives 72 h;
- **front scale 10 km:** kappa **1.7 / 3.3 / 5.6 m^2/s**, `2 kappa k^2` = **1.3 / 2.6 /
  4.4e-6 s^-1**, `G` e-folds in **212 / 106 / 63 h** (8.8 / 4.4 / 2.6 d), 71% / 51% / 32% of `G`
  survives the 72 h window;
- 20 km: kappa 0.04-0.12 m^2/s, e-folding ~1-4 x 10^4 h — nothing.
Where the MP limiter engages (1-2-cell fronts, local extrema) the local diffusivity rises
toward the first-order-upwind bound `|u| dx / 2` = 171-576 m^2/s (`G` e-folding at 10 km of
2.1-0.6 h), so at the ~1.5-cell fronts of planning §5.3 the numerics are the whole story.
Relative to the kinematic rate `2F/G ~ 2|sigma| ~ 4e-5 s^-1`: numerics are **3-10% at 10 km,
30-100% at 4dx, >100% at 2dx**. Planning §2.3's "kappa ~ (0.01-0.1) u dx ~ 6-60 m^2/s,
(0.7-7)e-5 s^-1 at 4dx, 0.1-1 f" is the 4dx number and holds there (0.023 u dx; 0.14-0.5 f
at f = 8.9e-5); it does not transfer to 10 km, where the same scheme is an order of magnitude
gentler. Uncertainties, honestly: the relevant `|u|` is the grid-relative speed along each
split direction, not the flow relative to the front; the limiter's engagement is
front-by-front, so the true value at a given front lies anywhere between the unlimited kernel
and the first-order bound; explicit vertical `diffKrT = 5.44e-7` and KPP are separate,
diabatic, and in the residual by design.

**Contradictions with the planning doc (and what should change — docs not edited).**

1. **§5.5 "MITgcm stores land as 0"** — OSN stores **NaN**, cell-for-cell equal to `hFacC`
   (centred), `hFacW` (`U`), `hFacS` (`V`). No coastal ribbon exists; task 5's QA plot will not
   show one (a 3-cell NaN rim from the stencils will appear instead). The 7-cell halo's
   justification becomes filter support + `skfmm`, not `b(0,0)`.
2. **§2.2 "At z=0, w vanishes identically along the surface"** — `W(k_l=0)` is `dEta/dt`,
   median 5.5e-5 m/s, corr 0.994 / slope 1.04 against the centred `Eta` difference. The
   surface budget's conclusion stands via the kinematic BC; the stated reason is wrong. Also in
   §2.2: "`w(eta)` ... horizontal gradient `~1e-10 s^-1`" is 9.5e-10 median, 3.4e-9 p99; "`z*`
   dilation `O(eta/H)`" does not apply (linear free surface, no `z*`); "`w(-dz) = -dz delta`"
   has the wrong sign and omits the dominant `dEta/dt` part (`|W(0)|` is 5x `drF|delta|`) —
   `vertical.py` must use the chunk `W(k_l=1)` directly, and should not assume `b_z` uniform.
3. **§4 / prompt 1 "~1.8-2.3 km"** — 1.68-2.07 km; 1.71 (meridional) x 1.85 (zonal) km at 37N.
   And **face 10 is rotated**: `j`/`V`/`dyC` are zonal, `i`/`U`/`dxC` meridional, `i` increasing
   southward (`CS=0, SN=-1`). Coding §2.5 `follow(km_per_px=2.3)` should be ~1.8 for this tile;
   §3.1's grid attrs should record the orientation.
4. **§5.2 "grid angle changes by a few degrees across 720 cells ... ~0.1%"** — the angle is
   exactly -90 deg everywhere; the rotation terms are identically zero; only the metric term
   remains, at 0.1% median / 0.5-0.7% p99. The §5.2 caveat can be reduced to the metric term.
5. **§2.3 "confirm the exact scheme"** — confirmed: OS7MP (`tempAdvScheme=7`), `diffKhT=0`,
   biharmonic Leith 2.1-2.15 on momentum, linear free surface, `StaggerTimeStep`, JMD95Z EOS
   (consistent with §5.1's JMD95 choice). The `kappa` bracket is right at 4dx and an order of
   magnitude too pessimistic at 10 km; §2.3 should state the scale dependence.
6. Confirmed, not contradicted: §5.3's displacement numbers (0.37 median, 1.64 p99 at fronts);
   §2.2's "~30% of F by day" is inside the measured 4-14% rms / order-one pointwise range.
7. Data notes for M2: `oceTAUX/Y` are masked with the centred mask (re-mask with
   `hFacW/hFacS`); `Eta` carries a ~1.3 m mean offset plus a tide of ~0.1 m/h; `hFacC` is
   binary at k=0.

**Before tasks 4-5.** (a) The QA plot's brief ("see the coastal gradient ribbon if question 1
answers 0") is void — show the stencil NaN rim and the high-edge rim instead. (b)
`tile330_grid.zarr` should also carry `hFacW`, `hFacS` (from the raw gridfile; they are the
`U`/`V` masks) and the 0-d `drF, Z, Zl` scalars, and record `CS=0, SN=-1` / axis orientation
in attrs. (c) No change to the operators is implied by Q4; the rotation is an exact axis swap.

Files: created `dev/frontogenesis/py/m0_recon.py`; modified this log only.

**Addendum (2026-09-28, same session) — corrections applied to the docs, on JXP's approval.**
Each spot edited in place, marked "(corrected 2026-09-28, M0 task 3)" or similar; nothing
unrelated rewritten.

- `frontogenesis_planning.md` header ("Reviewed"): appended the five M0 corrections to the
  list of overturned claims.
- planning §2.2: premise rewritten (kinematic BC `w = D eta/Dt`, not `w = 0`; data numbers);
  linear free surface / no `z*`; `|grad_h W| ~ 1e-9`; `w(-dz) = w(0) + dz delta` with the
  `dEta/dt` part; surface-only bracket of the tilting term; the term is built from the chunk
  `W(k_l=1)` and the unfactorised form (drops nothing); `drF`/`Z`/`Zl` are in the OSN gridfile.
- planning §2.3 item 1: OS7MP `tempAdvScheme = 7`, namelist settings and source URL,
  scale-dependent `kappa_num` (2dx / 4dx / 10 km / 20 km) and the limiter bound.
- planning §4 "Region": 1.68-2.07 km, lat to 38.27, face-10 orientation (`CS=0, SN=-1`, `i`
  southward, `j` zonal); "Fields pulled": `hFacW, hFacS, drF, Z, Zl` added to the grid;
  "Second OSN store": `oceTAUX/Y` centred-mask note.
- planning §5.2 "Known approximation": rotation terms identically zero on face 10; metric term
  0.09% median / 0.5-0.7% p99.
- planning §5.3 item 1: displacement numbers confirmed (annotation only).
- planning §5.5: land is NaN (cell-by-cell evidence); halo justification restated; the
  "unverified" paragraph removed.
- planning §6 Phase-0 table: four rows marked done with the results; V6 description; §7 V6
  line; §10 risk table: land-contamination row retired, `W(0)=dEta/dt` row added.
- `frontogenesis_coding.md` §2.5: `follow(km_per_px=2.3)` annotated (~1.8 for tile 330).
- coding §3.1: `hFacW, hFacS, drF, Z, Zl` and orientation / spacing / `land_fill` attrs.
- coding §3.2: land-NaN and `oceTAUX/Y` mask notes. §3.3: `W` on `k_l = 0..2`, `drF(k)`.
- coding §4.2: 12-14 km at 1.7-2.1 km; `dxC` meridional; `ocean_mask == isfinite(Theta)` assert.
- coding §4.6: `vertical_term` signature now takes `b, b_x, b_y, b_k1, W_k1, drF` and the
  rationale (chunk `W(k_l=1)`, unfactorised form).
- coding §6 M0: task 3 answers recorded; task 4 grid contents; task 6 QA-plot brief and
  acceptance (no ribbon expected). §6 M2: `drF` wording, `W(k_l=1)` requirement.
- coding §8 pitfalls: four new items (`W(k_l=1)`, face orientation, `oceTAU` re-mask,
  scale-dependent `kappa_num`).
- `frontogenesis_prompt_1.md`: `drF` note, status line for tasks 1-3, task 4 grid contents,
  task 5 QA-plot brief, acceptance bullet.
- `frontogenesis_prompt_2.md`: halo km (12-14) and the `ocean_mask` assertion.
- `frontogenesis_prompt_3.md`: grid list / `oceTAU` note; `drF` for `k = 0..2` and `W` on
  `k_l = 0..2`.
- `frontogenesis_prompt_4.md`: `vertical_term` bullet; the numerical-diffusion "do not".
- `frontogenesis_prompt_6.md`: the overturned-claims list extended with the five M0 items.

Not changed, deliberately: `frontogenesis_prompt_5.md` (no restated claim; `km_per_px` is
never hard-coded there and the NaN-land remark at its line 33 was already correct); planning
§2.3 items 2-4, §5.1, §5.4, §5.6-§5.7, §11-§12 (no claim touched by task 3); the dbof
`native_gradient` docstrings that call model-x "zonal" (outside `dev/frontogenesis/`); and
the task-3 log entry above, which stays as the record of what was found.

### 2026-09-28 — Execution prompt 1, task 4: tile330_grid.zarr and two hours (Fable)

**Scope.** Task 4 of `frontogenesis_prompt_1.md` only: the static grid store (§3.1) and a
two-hour raw product in the §3.2 layout. Task 5 (QA plot) not started; no 72-hour pull; no
physics module (`operators`/`semilag`/`masking`) written; `pull_series` (M2, resumable) not
written. Nothing outside `dev/frontogenesis/` touched; `tiles-surface-only` untouched
(`dbof` at `938bce1`).

**Written.**
- `dev/frontogenesis/data/tile330_grid.zarr` — **1.8 MB** on disk (29.1 MB in memory).
- `dev/frontogenesis/data/tile330_raw_20120702T00_2h.zarr` — **21 MB** on disk (41.5 MB in
  memory); hours `2012-07-02 00:00:00` and `01:00:00` from both OSN stores.
- `dev/frontogenesis/data/.gitignore` (`*`, `!.gitignore`): the fronts `.gitignore` ignores
  `*.nc/*.npy/*.parquet/*.csv/*.png` but **not `*.zarr`**, so without this the stores would
  show up as untracked; `git check-ignore` confirms the zarrs are ignored.
- `dev/frontogenesis/py/osn_tiles.py` extended (~170 -> 358 lines; §1.3's ~400-line cap
  respected): `CORE_GRID_VARS`, `GRID_EXTRA_VARS = hFacW, hFacS, drF, Z, Zl`, `ORIENTATION`,
  `DATA_DIR = dev/frontogenesis/data`; `load_grid()` now opens the gridfile **once**, runs
  `process_llc4320_grid` on it and cuts the extras from the same raw dataset with the same
  tile indexer (`raw.reset_coords()[GRID_EXTRA_VARS]`, as `m0_recon.py` did), merges, computes,
  `ensure_comodo_attrs`, and adds the §3.1 attrs (`orientation`, `dx_km_37N`/`dy_km_37N`
  measured from the data as band medians, `land_fill='NaN'`); new `write_grid(grid_ds, out,
  clobber)`, `open_grid(path, with_face=True)`, `load_hours(timestamps, ..., include_wind,
  grid_ds)` (concat along `time` from both stores, in memory — the building block `pull_series`
  will loop over), `write_raw(ds, out, clobber)`; private helpers `_git_commit`, `_provenance`
  (`git_commit`, `dbof_commit`, `created`), `_drop_face`, `_clean_encoding`, `_out_path`.
  `_finish_hour` (both hourly loaders) gained two lines: index coords cast to int64 and
  `niter` made a `time` coord (see "found" below).
- `dev/frontogenesis/py/m0_write.py` (new, 190 lines): pulls, writes, re-opens and verifies
  both stores; every check raises on failure; prints sizes and wall times. Reproducible with
  `/Users/xavier/miniforge3/envs/frontogenesis/bin/python m0_write.py`.

**Decisions.**
1. **Face dim dropped in the stored products, per §1.2/§3.1/§3.2** (task 2 flagged the
   loaders keep it). On disk: grid `(j, i)` + `i_g, j_g`; raw `(time, j, i)` + `i_g, j_g`.
   `face` survives as a **scalar coord (= 10)**, so `expand_dims('face')` restores the
   `(face, j, i)` layout the dbof operators expect — `open_grid(with_face=True)` (the default)
   does exactly that; `with_face=False` gives the stored layout. In-memory `load_grid`/
   `load_hour`/`load_wind_hour` still keep the length-1 face dim, unchanged for M1's operator
   work. `face_index=10` is also an attr on both stores.
2. **Output location `dev/frontogenesis/data/`**, as coding §3 ("All under
   `dev/frontogenesis/data/`") and planning Q4 specify; `.gitignore` added there (above).
3. **`hFacW`, `hFacS`, `drF`, `Z`, `Zl` are data variables** (the raw gridfile has them as
   coords), matching the "vars" line of §3.1; `drF/Z/Zl` are 0-d float32 with their source
   attrs (`units='m'`, `standard_name`, `positive='down'`).
4. **Zarr format 3** (zarr-python 3.4.0 default; xarray 2026.7.0), Zstd codec, one
   `(720, 720)` chunk per variable in the grid and **one `(1, 720, 720)` chunk per hour per
   variable** in the raw product (the append unit for M2's `pull_series`). The kerchunk
   source encoding (chunks still carrying the face dim) is stripped before writing, otherwise
   `to_zarr` rejects the squeezed arrays. `time` encoded as `int64` `seconds since 2011-09-10`,
   the stores' own convention. xarray warns that consolidated metadata is not part of the v3
   spec; harmless, both stores re-open with plain `xr.open_zarr`.
5. `write_grid` **asserts `SN == -1` and `|CS| < 1e-6`** before writing the `orientation`
   attr, so the string cannot outlive a change of tile; `dx_km_37N`/`dy_km_37N` are computed
   (median `dxC`/`dyC` over `36.5 <= YC <= 37.5`, rounded to 0.01 km) and the verification
   asserts they equal the task-3 values 1.71/1.85.
6. `git_commit` is recorded as `09b5643+dirty` (the `+dirty` because `osn_tiles.py` itself
   is uncommitted at write time); `dbof_commit = 938bce1` (clean). Re-run `m0_write.py` after
   committing if a clean hash is wanted — the pull is ~5-8 min, network-bound.

**Verification — `tile330_grid.zarr` re-opened from disk (all pass).**
- dims `{j: 720, i: 720, i_g: 720, j_g: 720}`, no `face` dim; coords `face, i, i_g, j, j_g`.
- all **17** vars present: `CS, Depth, SN, XC, YC, Z, Zl, drF, dxC, dxG, dyC, dyG, hFacC,
  hFacS, hFacW, rA, rAz`; dims `hFacW(j, i_g)`, `hFacS(j_g, i)`, `hFacC(j, i)`, `dxC(j, i_g)`,
  `dyC(j_g, i)`, `rAz(j_g, i_g)`, `drF/Z/Zl ()`; **all float32**; `drF=1.0, Z=-0.5, Zl=0.0`.
- comodo attrs survive the round-trip: `j: axis=Y`, `i: axis=X`, `j_g: axis=Y, shift=-0.5`,
  `i_g: axis=X, shift=-0.5`; index coords int64 `0..719` / `2880..3599`.
- `build_xgcm` (= `set_xgcm_grid(use_connections=False)`) builds from the re-opened store in
  both layouts: X (`i` center, `i_g` left), Y (`j` center, `j_g` left), not periodic,
  `padding='fill'`.
- attrs: `face_index=10, j_face_start=0, i_face_start=2880, rect_i=13320, rect_j=9720,
  source='OSN', endpoint, git_commit='09b5643+dirty', dbof_commit='938bce1',
  created='2026-09-28T18:14:34+00:00', orientation='CS=0, SN=-1: j/V/dyC zonal (eastward),
  i/U/dxC meridional (i increasing southward); u_east=V, v_north=-U', dx_km_37N=1.71,
  dy_km_37N=1.85, land_fill='NaN'`.
- every variable equals the in-memory pull **bit-for-bit** (NaN-aware).
- against a real hour (t0): `isnan(U) == (hFacW == 0)` at **162,445** land faces, 0
  mismatches; `isnan(V) == (hFacS == 0)` at **162,088**, 0 mismatches; `isnan(Theta) ==
  (hFacC == 0)` at **161,523**, 0 mismatches (task 3's counts reproduced from disk).

**Verification — `tile330_raw_20120702T00_2h.zarr` re-opened from disk (all pass).**
- dims `{time: 2, j: 720, i: 720, i_g: 720, j_g: 720}`; vars exactly `Theta, Salt, U, V, W,
  Eta, KPPhbl, oceTAUX, oceTAUY`, each `(2, 720, 720)` float32 on the §3.2 dims (`U`,
  `oceTAUX` on `i_g`; `V`, `oceTAUY` on `j_g`); coords `time, niter(time), XC(j, i),
  YC(j, i), i, i_g, j, j_g, face=10, k=0, k_l=0`; comodo attrs on all four horizontal dims,
  index values equal to the grid store's.
- `time = ['2012-07-02T00:00:00', '2012-07-02T01:00:00']`; attrs `iterations = [1022976,
  1023120]` (difference **144** = one hour), `niter(time)` coord equal to it, `timestamps`,
  `endpoint='https://mghp.osn.xsede.org'`, `stores=['llc_surf', 'llc_wind']`, tile attrs,
  `land_fill='NaN'`, `git_commit`, `dbof_commit`, `created`.
- chunks `(1, 720, 720)`; `time` encoding `seconds since 2011-09-10`.
- NaN pattern at **both** hours: `Theta, Salt, W, Eta, KPPhbl` NaN exactly where `hFacC == 0`,
  `U` where `hFacW == 0`, `V` where `hFacS == 0` — 0 mismatches in all 14 checks; the Theta
  mask is identical at t0 and t1. `oceTAUX`/`oceTAUY` are NaN exactly where `hFacC == 0`
  (centred mask; 922 / 565 cells differ from `hFacW`/`hFacS`), as task 3 found and §3.2 now
  states — stored as they come.
- concat is correct: the two hours differ (max `|t1 - t0|`: Theta 1.87 C, U 0.68 m/s, Eta
  0.53 m, KPPhbl 24 m); `Theta[t1]` and `V[t1]` equal a **fresh `load_hour('2012-07-02
  01:00:00')` bit-for-bit**, and `Theta[t0]` a fresh `load_hour(t0)`; `XC/YC` equal the grid
  store's.

**Wall times** (network-bound; OSN was slow during the final run — the same calls took
45-56 s / 7 s / 6 s in earlier runs today): `load_grid` 192 s (52 s and 56 s in the two
earlier runs), `write_grid` 0.2 s, `load_hours` (2 x core + 2 x wind) 186 s (76 s earlier),
`write_raw` 0.2 s, one `load_hour` 21-47 s. Writing is negligible; the pull dominates, so M2's
72 hours will be ~30-90 min at these rates and `pull_series` must be resumable.

**Found on the way (data / code facts, not in the docs).**
1. **Both hourly stores decode every index coord as float64** (`i, i_g, j, j_g, k, k_l,
   niter` — kerchunk's fill-value promotion), whereas the gridfile gives int64. Merging the two
   stores with `combine_attrs='drop'` then stripped the comodo attrs from `i_g`/`j_g` (xarray's
   `combine_attrs` applies to *variable* attrs too) — caught by the verification, fixed:
   `_finish_hour` now casts the index coords to int64 (attrs kept) and `load_hours` merges
   with `compat='override', combine_attrs='override'`, so the raw product's index coords are
   int64 and identical to the grid's. `xr.merge([hour, grid])` in M1/M3 now aligns on equal
   int indexes rather than float-vs-int.
2. A first version computed `dx_km_37N` with `dxC.where(band)`; `dxC` sits on `(j, i_g)` and
   the band on `(j, i)`, so xarray broadcast to 4-D and the "band" median was the tile-wide
   1.80 km. Caught by the verification against task 3's 1.71; fixed with positional numpy
   masking (as `m0_recon.py` does). Worth remembering for M1: **any `where`/arithmetic
   between a centred mask and a staggered field silently broadcasts** — go through xgcm
   `interp` or numpy.
3. `to_zarr` refuses the loaded hours unless the kerchunk `encoding` (chunks `(1, 1, 720,
   720)` with the face dim) is cleared first; `_clean_encoding` does that.

**Contradictions with the docs — none new; two points for the record.**
- §3.1/§3.2 say `(j, i)`; task 2's loaders keep `face`. Resolved by decision 1 (stored
  without the dim, restored on open); the docs need no change, but §3.1/§3.2 could add "`face`
  kept as a scalar coord; `open_grid(with_face=True)` restores the dim".
- §3.2 lists `attrs: iterations, endpoint, stores, git_commit`; the product also carries
  `timestamps`, `dbof_commit`, `created`, `land_fill` and the tile attrs, and the coords
  `niter(time)`, `k`, `k_l`, `face` beyond `time, XC, YC`. Supersets, not conflicts.
- Prompt 1 task 4's name for the two-hour product is unspecified; used
  `tile330_raw_20120702T00_2h.zarr` by analogy with §3.2's `_72h`.
- **§3.1 vs §3.2 disagree on `XC`/`YC`**: §3.1 lists them as *vars* (and
  `process_llc4320_grid`'s `reset_coords()` makes them data variables in the grid store), §3.2
  as *coords* on the raw product. Both stores follow their own section, so a naive
  `xr.merge([hour, grid])` raises `MergeError: unable to determine if these variables should
  be coordinates or not ... {'YC', 'XC'}` (found in an offline re-open test after the log
  above was written). Either `hour.drop_vars(['XC', 'YC'])` or `grid.set_coords(['XC', 'YC'])`
  before the merge works (both verified: 26 / 24 data vars). **The docs should pick one** —
  making `XC`/`YC` coords in §3.1 too (`set_coords` in `load_grid`) is the smaller change and
  matches the raw gridfile; not done here since it alters the stored §3.1 contract.

**For task 5 (QA plot).** `open_grid()` gives the grid with `face` restored and
`xr.open_zarr(DATA_DIR / 'tile330_raw_20120702T00_2h.zarr')` the two hours (`expand_dims('face')`
before handing a snapshot to the dbof operators, or merge with `open_grid(with_face=False)`);
`hFacC > 0` is the land mask; land is NaN in every field so no gradient ribbon; the stencil NaN
rim and the high-edge rim are what to show. `m0_recon.py` still runs but its own raw
`hFacW/hFacS` pull is now redundant (`load_grid()` carries them).

Files: modified `dev/frontogenesis/py/osn_tiles.py`; created `dev/frontogenesis/py/m0_write.py`,
`dev/frontogenesis/data/.gitignore`, the two zarr stores; this log.

### 2026-09-28 — Execution prompt 1, task 5: QA plot and M0 acceptance (Fable)

**Scope.** Task 5 of `frontogenesis_prompt_1.md` (QA plot, no halo) plus the M0 acceptance
audit, including the two §2 traps left unconfirmed by tasks 2-4 (`calculate_jacobian` args;
`halo_mask.py:75`). Offline, from the two task-4 stores; no network. No physics module
(`operators`/`semilag`/`masking`) written, no 72 h pull, no halo adopted. Nothing outside
`dev/frontogenesis/` touched; `dbof` read-only at `938bce1`.

**Written.**
- `dev/frontogenesis/figs/m0_qa_tile330_20120702T00.png` (200 dpi, 3800 x 2300, 1.8 MB).
- `dev/frontogenesis/py/m0_qa_plot.py` (~300 lines): loads `open_grid(with_face=True)` and
  `tile330_raw_20120702T00_2h.zarr`, merges t0 (after `drop_vars(['XC','YC'])` on the hour —
  the §3.1/§3.2 disagreement from task 4 — and `expand_dims('face')`), casts to float64,
  computes `b` via `calculate_fields.buoyancy_of_field` (JMD95, p=0), `G` via
  `calculate_grad_squared_tracer` (the repo's `grad_b2` stencil) and via
  `calculate_native_gradient_tracer` (`b_x^2 + b_y^2`), the Jacobian via `calculate_jacobian`,
  prints every diagnostic, draws the figure. Run: `<env python> m0_qa_plot.py` (~30 s).
- `dev/frontogenesis/py/m0_qa_checks.py` (~200 lines): the trap checks and the tile-edge
  crop test, read-only, importable by M1's tests. A positional-numpy replica of the ECCO
  interp-rotate-diff-interp-rotate stencil lives here (`jacobian_numpy`) — a check, not an
  operator.

**Figure — what it shows** (maps via `pcolormesh(XC, YC, ...)`, so north-up / east-right
regardless of the rotated face; the inset likewise, with cell edges drawn).
- (a) `Theta` at `2012-07-02 00:00`, 11.3-33 C (colour p1-p99), land `hFacC = 0` grey.
- (b) `log10 G`, `calculate_grad_squared_tracer`, colour clipped to the interior p1-p99.5
  (`-16.6 .. -12.4`); the north (`i = 2880`) and west (`j = 0`) tile edges saturate — see (f).
- (c) validity map of the whole tile with a rectangle marking the inset; counts in the box.
- (d) cell-level inset, Monterey Bay (`j 263..310, i 2952..2999`): the G NaN rim is exactly
  one cell wide along the coast (red), the Jacobian rim two cells (orange).
- (e) ribbon test: median / p90 `G` vs taxicab distance to land, tile edges excluded.
- (f) tile-edge crop test: fraction of cells whose value changes when the tile edge moves.

**Numbers.**
1. **`isfinite(Theta) == (hFacC > 0)` cell for cell:** 518,400 cells, 356,877 ocean, 356,877
   finite `Theta`, **0 mismatches** (task 3/4 reproduced from the on-disk stores).
2. **Stencil NaN rim.** `G` finite in 354,703 cells (both stencils); ocean cells with NaN `G`:
   **2,174**, and they are *exactly* the ocean cells at taxicab distance 1 from land
   (`array_equal` True; chessboard-distance histogram also all at 1, i.e. diagonal-only
   neighbours of land keep a finite `G`). **Width 1 cell, not "~3"** — the tracer stencil is
   diff (1 cell) + interp back (1 cell), which reaches one centre from a NaN. The Jacobian's
   NaN rim is **4,204** ocean cells = exactly taxicab `d <= 2` (interp U/V to centres, diff,
   interp: two centres from a NaN). So `F = -(J : grad b grad b)` is undefined within 2 cells
   of land; task 3's "3 cells clear" interior and §4.2's "3 for the Jacobian+interp stencil"
   are one cell conservative — safe, no change needed.
3. **No coastal gradient ribbon.** Median `G` (interior, `d >= 20`, tile edges excluded)
   **1.86e-15 s^-4**, p90 2.29e-14. By taxicab distance from land, median/interior =
   **75.8x (d=2), 44.4x (3), 28.1x (4), 20.7x (5), 17.4x (6), 15.9x (7), 13.0x (8), 11.3x (9),
   9.8x (10)** (`b_x^2+b_y^2` stencil: 64.6x, 39.8x, 27.1x, ... 10.6x). A smooth 10-cell decay
   is the coastal upwelling front, not a stencil artefact: a `b(0,0)`-type ribbon would be
   ~1e-8 s^-4 (**~1e7x** the interior) and confined to `d = 2` — and the tile's own `j = 0`
   edge, where xgcm *does* difference against a 0 fill, shows exactly that: median `G`
   7.3e-9 (dotted line in (e)). Nothing upstream has filled NaN with 0. (M3 note: the coastal
   band dominates the upper `G` percentiles; the 100 km offshore mask removes it.)
4. **Tile-edge rim — both edges are invalid, and the values are finite, not NaN.** xgcm
   `padding='fill'` with `set_xgcm_grid`'s `fill_value=None` pads with **0** (checked: max
   `|diff_X(b)[i_g=2880] - b[i=2880]| = 0.0`). Crop test (recompute on the tile cropped by 32
   cells on every side, diff against the full-tile values at the same cells; a cell that changes
   is one the edge contaminates): **`G`: 1 cell on every edge** (offset 0 changes in 100% of
   cells, offset 1 in 0%, on all four edges, both stencils); **Jacobian: 1 cell on the low
   edges, 2 cells on the high edges** (offsets 0 and 1 at 100% on `j = 719` and `i = 3599`).
   Why: the low staggered point of each cell is in the tile, the high one is not, so at a low
   edge the *diff* sees the 0 fill (bogus, huge) and at a high edge the *interp* sees it
   (halved); the Jacobian's extra interp-before-diff carries the high-edge error one cell
   further. Magnitudes (median vs offsets 1-3): `G` low edges **1.7e6x / 8.3e5x**, high edges
   **0.50x / 0.58x**; `|J|` low edges 3.0x / 4.8x, high edges 0.8x / 1.0x (the high-edge
   Jacobian error is a mix of halving and sign, invisible in a median of magnitudes — the
   crop test is the authority). Ocean cells in the union rim: 2,634.
5. **Two `G` stencils differ by ~9%**: interior median of `(b_x^2 + b_y^2) /
   calculate_grad_squared_tracer` = **0.911** (interp-then-square attenuates). See
   "contradictions" (4) — M1 must pick one.

**Trap confirmations (the two outstanding ones; all four now done).**
- **(i) `calculate_jacobian(u_x, v_y, ...)` takes the raw staggered `U`, `V`.** Source
  (`native_gradient.py` L130-200 at `938bce1`): L164 passes `u_x, v_y` to
  `rotate_vector_to_geographic`, whose `interp_pair_to_center` interpolates `u_x` along X
  (needs `i_g`) and `v_y` along Y (needs `j_g`); `compute_velocity_jacobian`
  (`calculate_fields.py` L162) calls it with `ds_merge.U, ds_merge.V`. Empirically: a
  positional-numpy replica of the full stencil from `U`/`V` (`m0_qa_checks.jacobian_numpy`)
  reproduces all four components **bit-for-bit** (max abs diff 0.0 on 352,673 cells, NaN
  patterns equal); on this face (`CS=0, SN=-1`) `du_east/dx_east` equals `d(V_c)/d(model j)`
  to 2.4e-16. The swapped call `calculate_jacobian(V, U, ...)` raises `KeyError` on
  dask-backed inputs — but on numpy-backed inputs it does **not** raise: xgcm interps `V`
  along X (it has `i`), the `CS`/`SN` multiply broadcasts `(j_g, i_g) x (j, i)` to 4-D and the
  process was OOM-killed (exit 137). Independent physics check: `du_dx + dv_dy` vs the
  flux-form `divergence_center` from `calculate_native_strain_vorticity` (no rotation, no
  interpolation): **corr 0.969, regression slope 0.80** (n 347,333) — the right quantities in
  the right slots, with the interpolated Jacobian **20% attenuated** relative to the flux
  form, inside planning §6's 0.7-1.4 bracket. V3 will see this.
- **(ii) `halo_mask.py:75` is not reached.** L74-75 read `elif (phi==-1).any(): return mask_f`
  — the branch for a face with masked cells and **no** unmasked cells, i.e. a face that is
  entirely land (`hFacC == 0` everywhere). Tile: `phi == -1` in 161,523 cells, `phi == 1` in
  356,877, so the L59 condition `(phi==-1).any() and (phi==1).any()` is True and the skfmm
  branch runs. Run read-only on the tile with `halo_km = 12.0` (`hFacC == 0` as an xarray
  `(face, j, i)` bool; the function indexes `mask[face].values`): returns `ndarray` bool
  **(1, 720, 720)**, 342,682 retained of 356,877 ocean (96.0%; 14,195 excluded). The bug
  reproduced on an all-land face: returns the **2-D** `(720, 720)` input, all True. M1's
  `masking.halo_mask` should assert `out.ndim == 3` exactly as §2.4 says.
- (iii) comodo attrs and (iv) timestamp formats: confirmed in task 2 (attrs survive
  `process_llc4320_grid` on the OSN gridfile, `ensure_comodo_attrs` a no-op guard; the
  dbof-vs-`front_tracking` round trip verified, `load_hour` rejects the underscore form).

**dbof line numbers vs coding §2 (at `938bce1`).** `native_gradient.py` is **+77 lines**
throughout: `rotate_vector_to_geographic` L91 (doc L13), `calculate_jacobian` L130 (L53),
`calculate_native_gradient_tracer` L203 (L126), `calculate_grad_squared_tracer` L301 (L224),
`calculate_native_strain_vorticity` L378 (L301). `grid.py` `COMODO_COORD_META` L11 (doc L9);
`tile_mapping.py` `class TileInfo` L48 (doc L47). All others match (`date_iterations` L31/38/64,
`get_raw_data` L21/151/241, `preproc_llc_core_data` L37, `rect_ij_to_tile` L112,
`ensure_comodo_attrs` L46, `set_xgcm_grid` L121, `_tile_indexer` L334, `calculate_fields`
L66/96/122/134/145/168/254/627/654, `physical_constants` G L12 / RHO0 L19, `static_masks` L5,
`halo_mask` L5; the L74-75 bug is where the doc says).

**M0 acceptance audit (prompt 1).**

| Criterion | Verdict | Evidence |
|---|---|---|
| Two consecutive hours load end to end, both OSN stores, reproducible env | **PASS** | task 1 env (py3.13, resolved versions logged); task 4: `tile330_raw_20120702T00_2h.zarr`, `time = [00:00, 01:00]`, iterations 1022976/1023120 (diff 144), `llc_surf` + `llc_wind` vars, `m0_write.py` reproduces it |
| `tile330_grid.zarr` written (§3.1) | **PASS** | task 4: 17 vars incl. `hFacW/hFacS/drF/Z/Zl`, orientation/spacing/`land_fill` attrs, comodo attrs survive re-open; used from disk by this task (`open_grid`) |
| All five questions answered in the log, with numbers | **PASS** | task 3 entry (NaN land; `W(0) = dEta/dt`; 1.71 x 1.85 km, rotated face; rotation terms 0, metric 0.1%; OS7MP + scale-dependent `kappa_num`) |
| Four §2 traps confirmed on real data | **PASS** | comodo (task 2), timestamps (task 2), `calculate_jacobian` args (here, bit-identical replica), `halo_mask.py:75` (here, branch not reached; bug reproduced on an all-land face) |
| QA plot written; coastline inspected; NaN rim expected, gradient ribbon not | **PASS** | this figure: rim = taxicab `d = 1` exactly (2,174 cells); coastal `G` decays smoothly 76x -> 10x over `d = 2..10`; a ribbon would be ~1e7x at `d = 2` and is absent |

M0 closes. "Do not" list respected: no physics module, no 72 h pull, nothing outside
`dev/frontogenesis/`.

**Contradictions with the docs / things that should change before M1 (docs not edited).**
1. **"Invalid rim on the high edges"** (prompt 1 task 5; coding §2.2 `_tile_indexer` note, §6
   M0 task 6; `osn_tiles.tile_indexer` docstring; dbof's own `_tile_indexer` docstring
   "derivative-based properties NaN an edge rim"): **all four edges are invalid, and the
   values are finite, not NaN.** Low edges are the worse ones (`G` ~1e6x, from differencing
   against xgcm's 0 fill); high edges are halved (`G`) and two cells deep (Jacobian). The
   land halo will not remove them (a tile edge is not land; `skfmm` distance is measured
   from `hFacC == 0`), nor will the 100 km offshore mask on the open-ocean west and north
   edges. **`masking.analysis_mask` (§3.5/§4.2) needs an explicit tile-edge margin**: at
   least 2 cells for the raw operators, and the same 7 cells as the land halo once the filter
   support (half-width 4) and the Jacobian reach are counted. Wording to change in §2.2, §6
   and prompt 1.
2. **"`G` undefined within ~3 cells of land"** (prompt 1 task 5, coding §6 M0 task 6): it is
   exactly **1 cell** for `G` and **2 cells** for the Jacobian / `F`. The 7-cell halo budget
   ("3 for the Jacobian+interp stencil") is one cell conservative; keep it, fix the prose.
3. **§2's `native_gradient.py` line numbers are stale** (+77; list above), plus
   `COMODO_COORD_META` L11 and `TileInfo` L48. The §8 checklist item "re-verify if the Q15
   merges have landed" should just say "re-verified 2026-09-28 at `938bce1`" with these values.
4. **Which `G` stencil?** §1.1 says "`G = |grad_h b|^2`; the repo's `gradb2`" (=
   `calculate_grad_squared_tracer`, squares on the staggered points), §4.3 has
   `gradb2(b, grid_ds, grid)` unspecified, and `F` is built from the *components*
   (`calculate_native_gradient_tracer`). The two differ by **0.91x** in the interior median.
   The discrete identity `F = (1/2) DG/Dt` only holds when `G` is formed from the same
   `b_x, b_y` that enter `F`; using the repo's `gradb2` for `G` and the components for `F`
   biases V3's null slope by ~0.9 before any physics. **M1 must fix `operators.gradb2 =
   b_x^2 + b_y^2` from `grad_b`** (or derive both from the staggered squares) and say so in
   §1.1/§4.3; the repo's `gradb2` stays for front *finding* (M4), where only the pattern matters.
5. **Interpolated Jacobian is 20% attenuated** against the flux-form divergence (slope 0.80,
   corr 0.97). Not a contradiction — planning §6 predicted 0.7-1.4 — but it is the size of
   the V3 correction to expect, and worth recording next to the "quote every slope against
   the M1 null baseline" rule.
6. **The swapped-argument Jacobian call fails silently on numpy-backed data** (4-D broadcast,
   OOM), loudly only on dask-backed data. Task 4's "centred mask x staggered field broadcasts
   to 4-D" pitfall is general: add "assert `out.dims == ('face', 'j', 'i')` after every dbof
   operator call" to §8, and keep M1's operators on dask or assert dims.
7. **`figs/*.png` are git-ignored** by `fronts/.gitignore` (`*.png`, line 7), so the QA plot
   and M1's six acceptance PNGs will not be tracked unless `dev/frontogenesis/figs/` gets a
   negating `.gitignore` (as `data/` got one for the opposite reason) or the figures are
   committed with `git add -f`. Decide before M1.
8. Confirmed, not contradicted: land is NaN (0 mismatches), `hFacC` binary, the task-3/4
   counts, the §2.4 bug at L74-75, and `set_xgcm_grid`'s `padding='fill'` (xgcm 0.10.1).

Files: created `dev/frontogenesis/py/m0_qa_plot.py`, `dev/frontogenesis/py/m0_qa_checks.py`,
`dev/frontogenesis/figs/m0_qa_tile330_20120702T00.png`; this log.

**Addendum (2026-09-28, same session) — corrections applied to the docs and code, on JXP's
approval.** Each spot edited in place and marked "(corrected 2026-09-28, M0 task 5)" or
"(added 2026-09-28, M0 task 5)"; nothing unrelated rewritten.

- `frontogenesis_planning.md` header: a second "Updated 2026-09-28 (M0 task 5)" paragraph
  listing the four new corrections (tile-edge rim, stencil rim widths, `G` stencil, Jacobian
  attenuation). §2.1: `G = b_x^2 + b_y^2` from the same `b_x, b_y` as `F`; the repo's `gradb2`
  (`calculate_grad_squared_tracer`, 0.911x) is for front finding only. §5.5: measured rim
  widths (1 / 2 cells, counts) and the coastal decay; "3 for the Jacobian+interp stencil"
  annotated as one cell conservative; new paragraph on the finite four-edge tile rim and the
  `edge_cells` margin. §6 test 3: measured trace-vs-divergence slope 0.80 / corr 0.97 as the
  expected correction; V6 description: all four edges, finite, plus the margin. §10 risk
  table: the operator-bias row annotated with 0.80 and the `G`-stencil rule; new tile-edge row.
- `frontogenesis_coding.md` §1.1: `G` row rewritten (component stencil, not the repo's
  `gradb2`). §2.2: `TileInfo` L48, `COMODO_COORD_META` L11, `_tile_indexer` note rewritten
  (all four edges, finite, fill 0, crop-test widths). §2.3: `native_gradient.py` line numbers
  +77 (L91 / L130 / L203 / L301 / L378) with the re-verification note; `calculate_jacobian`
  trap extended (numpy-backed swapped call broadcasts to 4-D and is OOM-killed, dask raises
  `KeyError`); `calculate_grad_squared_tracer` annotated "front finding only". §3.1: `XC`, `YC`
  moved from `vars` to `coords`, `face` scalar coord and `open_grid(with_face=True)` noted.
  §3.2: coords line completed (`niter`, `face`, `k`, `k_l`) and the same `face`/merge note.
  §3.5: `mask_edge` added. §4.2: new `edge_mask(grid_ds, edge_cells=7)`, `analysis_mask`
  gains `edge_cells=7`, with the rationale; the 7-cell budget annotated with the measured
  reach. §4.3: `gradb2` comment (component form, not `calculate_grad_squared_tracer`). §4.9:
  V3 comment (expect ~0.8), V6 comment (edge rim + margin), the `figs/.gitignore` note. §5
  test table: `test_masking.py` guards the `ocean_mask` identity and the edge margin against
  `m0_qa_checks.check_edge_rim`. §6 M0 task 6: rim widths and four-edge wording fixed, "Done"
  line with the file names; "M0 closed 2026-09-28" status; M1 tasks: `analysis_mask` includes
  the edge margin; M1 acceptance 3: the 0.80 measurement and the `G`-stencil rule. §8: PNG
  item extended (`figs/.gitignore`); line-number item reworded (re-verified at `938bce1`);
  three new items (assert dims after every dbof call; tile-edge margin; `G` from the same
  `b_x, b_y` as `F`).
- `claude_prompts/frontogenesis_prompt_1.md` task 5: "~3 cells" and "high edges" corrected as
  a record, with a "Done 2026-09-28" line.
- `frontogenesis_prompt_2.md`: masking — measured rim reach; new bullet for `edge_mask` /
  `edge_cells` with the crop-test evidence and a `test_masking.py` requirement; operators —
  new bullets for `gradb2 = b_x^2 + b_y^2` (0.911x) and for asserting output dims (with the
  `calculate_jacobian` confirmation); V3 — the 0.80 / 0.91 numbers; acceptance 5 — the
  `figs/.gitignore` note and V6 must show the edge rim + margin.
- `frontogenesis_prompt_3.md`: grid list — `XC, YC` as coords; `face` scalar coord note.
- `frontogenesis_prompt_6.md`: the overturned-claims list extended with the four M0 task-5
  items.
- Not changed, deliberately: `frontogenesis_prompt_4.md` and `_5.md` (no restated claim:
  `mask_analysis` is used by name only, and prompt 5's `gradb2` is the front-finding use, which
  stays on the repo's stencil); planning §4 "Fields pulled" (lists names only); the task-5 log
  entry above, which stays as the record; dbof (read-only).

**Code and data (same session).**
- `py/osn_tiles.py`: `load_grid` now `set_coords(['XC', 'YC'])` (with the reason), so the
  in-memory grid and the store both carry them as coords; `write_grid`'s presence check uses
  `variables` rather than `data_vars`; `tile_indexer` and `write_grid` docstrings updated
  (four-edge finite rim; coords).
- `py/m0_write.py`: presence check on `variables`; new checks "XC/YC are coords, not data vars"
  and "`xr.merge([hour, grid])` works without dropping XC/YC" (21 data vars).
- `py/m0_qa_plot.py`: the `drop_vars(['XC', 'YC'])` workaround removed (plain merge); `XC`/`YC`
  read with `.squeeze()` since `expand_dims('face')` leaves coords on `(j, i)`.
- `figs/.gitignore` (`!*.png`, with a comment): `git check-ignore -v` now reports the negating
  rule (`figs/.gitignore:3:!*.png`) and the PNG shows in `git status`.
- **Re-run `m0_write.py`** (OSN fast this time: `load_grid` 10.6 s, `load_hours` 15.0 s):
  both stores rewritten, **90 checks ok, 0 FAIL**, including the two new ones; grid 1.9 MB /
  raw 21 MB on disk as before; `git_commit = 49b1867+dirty`, `dbof_commit = 938bce1`.
- **Re-run `m0_qa_plot.py`** on the rewritten stores: every number identical (Jacobian replica
  max diff 0.0; `halo_mask` (1, 720, 720), 342,682 retained; 0 mismatches; rim 2,174 / 4,204;
  ribbon profile 75.8x .. 9.8x; `G_comp/G_sq` 0.9106; edge crop test unchanged); figure
  regenerated and inspected, unchanged.

### 2026-09-28 — Execution prompt 2, task 1: masking, tile330_masks.nc, V6 (Fable)

**Scope.** Task 1 of `frontogenesis_prompt_2.md` only: `py/masking.py`, `data/tile330_masks.nc`,
`py/tests/test_masking.py`, `py/validate.py` with `qa_land_halo` (V6). Tasks 2-7 not started: no
`operators.py`, `semilag.py`, `coarsegrain.py`, no other V-function, no data pulled. Offline, from
the two M0 stores. Nothing outside `dev/frontogenesis/` touched; `dbof` read-only at `938bce1`.

**Written.**
- `py/masking.py` (~270 lines): `ocean_mask`, `halo_mask(grid_ds, halo_cells=7)`,
  `coast_distance_km`, `offshore_mask(min_km=100)`, `edge_mask(edge_cells=7)`,
  `analysis_mask(halo_cells=7, min_km=100, edge_cells=7)` exactly as coding §4.2, plus
  `spacing_km`, `halo_width_km` (the cell→km rule), `build_masks` (the §3.5 dataset with attrs),
  `write_masks`, `open_masks`, `MASK_VARS`. All outputs plain bool numpy on `(j, i)`, **True =
  retained, land = False**; `coast_distance_km` float, NaN on land. A length-1 `face` is
  squeezed and a `k` dim collapsed to `k=0` before anything else (`_positional`).
- `py/m1_write_masks.py` (~180 lines, `m0_write.py` style): builds, writes, re-opens, verifies
  (41 checks, all ok) and runs the Gulf of California check (`gulf_of_california_check`, reused
  by the test and by V6).
- `data/tile330_masks.nc` — 6.0 MB, netCDF4/zlib, dims `(j: 720, i: 720)`, coords `j, i, XC,
  YC` (+ scalar `face = 10`), vars exactly the §3.5 six in order; bools round-trip as bool. Attrs:
  `convention`, `halo_cells=7`, `halo_km=12.573`, `halo_km_rule`, `edge_cells=7`,
  `offshore_km=100`, `distance_method`, `dxC_median_km=1.7962`, `dyC_median_km=1.9504`,
  `dxC_mean_km=1.7948`, `dyC_mean_km=1.9483`, `dxC_min/max_km`, `n_ocean … n_analysis`, the tile
  attrs and `orientation` from the grid, `source_grid_commit`, `git_commit`, `dbof_commit`,
  `created`. Ignored by `data/.gitignore` like the zarrs (regenerate with `m1_write_masks.py`,
  <1 s).
- `py/validate.py` (~240 lines): `qa_land_halo(grid_ds, png=True) -> dict` only, plus the private
  `_snapshot_fields` (grid + t0 merged, JMD95 `b`, `G = b_x^2 + b_y^2` via the component stencil,
  dims asserted) and `_classify`. **V6 →
  `figs/V6_land_halo_tile330.png`** (200 dpi, 3800 x 2300, 0.85 MB): (a) mask stages on the whole
  tile, (b) Monterey Bay cell-level inset, coastline before/after the halo, (c) the distance field
  with the 100 km and 12.6 km contours, (d) Gulf of California zoom with the survivors (none),
  (e) the finite tile-edge rim (median |G| vs offset, all four edges, log scale) with the
  `edge_cells` margin shaded and the crop-test offsets marked, (f) halo width histogram in
  taxicab cells. `git check-ignore -v` → `figs/.gitignore:3:!*.png`; the PNG shows in `git status`.
- `py/tests/test_masking.py` (17 tests), `py/tests/conftest.py` (sys.path, session fixtures
  `grid_ds`/`raw_ds` that **skip** when the M0 stores are absent), `py/tests/pytest.ini`
  (registers `needs_grid` and `network`, `--strict-markers`, `-m "not network"` by default).

**Design choice: wrap the helper for the halo, own skfmm for the distance.** `halo_mask` wraps
`llc_native_grid_halo_mask` (it is the mask the dbof cutout pipeline would produce, and §4.2 says
"wraps"), with the guards: (1) a grid with no ocean returns all-False 2-D *without* calling it —
the helper's L74-75 path returns its 2-D input, all True; (2) `k` is collapsed first — on a
`(face, k, j, i)` mask the helper does not go silent, `skfmm` raises `ValueError: dx must be of
length len(phi.shape)`; (3) after the call the shape is asserted `(1, nj, ni)` bool (an
unconditional `raise AssertionError`, not the `assert` statement, so `-O` cannot strip it) and
no land cell may be retained. `coast_distance_km` has to call `skfmm` directly because the
helper thresholds and does not expose the distance; it uses the helper's exact recipe (`phi = +1`
ocean / `-1` land, so the coastline sits half a cell from the last land centre; **uniform
per-axis mean spacing** `dx=(mean dyC, mean dxC)` in km), and `test_masking.py` pins the two to
each other: on the real grid `halo_mask == raw helper[0] == ocean & (coast_distance_km >=
halo_km)`, cell for cell (`np.array_equal`). A variable-spacing distance would have been ~6%
more accurate locally (`dxC` 1.68-1.90 km) but would break that identity; immaterial for a
nominal 100 km cut.

**Halo: `halo_km = 7 x median(dxC) = 7 x 1.7962 = 12.573 km`** (§4.2's "roughly 12-14 km"
holds), measured from `tile330_grid.zarr`, never hard-coded; the fast-marching itself runs on
the mean spacing 1.7948 x 1.9483 km (the helper's choice; the median/mean difference is 0.08%).
Because `dxC` is the meridional (`i`) spacing and the smaller one, 12.573 km is 7.0 cells along
`i` and 6.45 cells along `j`; with the half-cell interface offset the first retained cell is at
index distance 8 (`i`) / 7 (`j`) — **min taxicab distance of a retained cell 7, min chessboard
5** (diagonal reach of a Euclidean halo), max taxicab of a removed ocean cell 10 (chessboard 7);
the split in km is exact (min retained 12.574, max removed 12.572); min centre-to-centre
Euclidean distance of a retained cell 13.15 km = 7.33 cells. The helper's result at this
`halo_km` is 341,960 retained (M0 task 5 reported 342,682 at the round 12.0 km).

**Retained counts at each stage (518,400 cells).** ocean **356,877** (68.8%) → halo
**341,960** (removes 14,917) → offshore ≥ 100 km **273,431** (removes a further 67,154; the
halo is a strict subset of the offshore cut) → tile-edge margin, `edge_cells = 7`: geometry
498,436 cells, 344,378 of them ocean; it removes 10,506 cells that the two other cuts keep (the
open-ocean west and north edges, exactly the case §4.2 describes) → **`mask_analysis` 262,925**
(73.7% of the ocean, 50.7% of the tile). The analysis mask is a **single connected component**.

**Gulf of California — the ≥ 100 km cut removes it, no polygon needed.** Inside the tile
(lat ≥ 26.66N) the Baja peninsula separates the Gulf from the Pacific, so the Gulf is its own
connected component of `mask_ocean` (`scipy.ndimage.label`; the component holding the cell
nearest 30.5N, 114.0W, disjoint from the one holding 35N, 125W): **9,891 cells**, lon
−114.885 … −113.010 (it touches the east edge `j = 719` on 125 cells, lat 28.48-30.54), lat
28.475-31.709. Land east of the edge is invisible to `skfmm`, so the distances inside the Gulf
are *over*-estimates, which makes the check conservative — and still **max coast distance
73.9 km** (at 30.41N, 113.82W; 63.5 km on the east edge itself), **0 cells ≥ 100 km, 0 in
`mask_analysis`**. For the record the other stages alone would not do it: the halo removes
2,988 Gulf cells, the edge margin 934, and 6,368 survive both. Also found: the model treats
several inland basins as ocean — the tile has **16 ocean components**; besides the Pacific
(345,368) and the Gulf, the Salton Sea (1,128 cells, 32.7-33.7N), Death Valley (252 cells,
35.9-36.7N, 117.2-116.7W, below sea level), the Sacramento-San Joaquin Delta (161) and San
Pablo Bay (33) plus ten specks; max coast distance ≤ 18 km, so **all 11,509 non-Pacific ocean
cells are removed by the offshore cut**. Asserted in `m1_write_masks.py` and in
`test_offshore_and_analysis_counts` (one component survives).

**Tile-edge rim vs the margin** (crop test, `m0_qa_checks.check_edge_rim`, re-run here on t0
with `G = b_x^2 + b_y^2`): G changed at offset `[0]` on all four edges, Jacobian at `[0]` on the
low edges and `[0, 1]` on the high — max contaminated offset **1**, so `edge_cells = 2` is the
minimum and 7 covers it with the filter half-width to spare. The rim is finite: median |G| at
offset 0 relative to offsets 1-3 is 1.2e6x (`j = 0`) and 5.6e5x (`i = 2880`) on the low edges,
0.49x / 0.44x on the high edges; offset 2 is within 0.7-1.4x on every edge (both asserted).

**Tests — `pytest dev/frontogenesis/py/tests`: 17 passed in 1.2 s** (11 offline synthetic-grid
tests, 6 `needs_grid`; `-m "not needs_grid"` → 11 passed, 6 deselected, 0.06 s). Offline: True =
retained / land False for every mask and NaN in the distance; `analysis_mask` is the
intersection; `edge_mask` geometry; `halo_width_km` uses the **median** `dxC` (a skewed tail
in `dxC` leaves it unchanged); halo width in cells on a uniform grid, `halo_cells = 3` and `7`,
both directions (first retained at index distance `halo_cells + 1`, last removed at
`halo_cells`, and km split against `coast_distance_km`); anisotropic spacing (8 along `i`, 7
along `j` for `dyC/dxC = 1.086`); **all-land face** (the raw helper's 2-D all-True on record,
the wrapper all-False 2-D); **`k`-carrying `hFacC`** (raw helper `ValueError`; ours identical
to the 2-D grid for all six functions); no-land / no-face-dim grid; a two-face grid is
rejected. `needs_grid`: `ocean_mask == isfinite(Theta)` cell for cell at **both** hours
(356,877, 0 mismatches); wrapper == raw helper == distance threshold; halo width on the real
grid (min taxicab 7, ≤ 8, max removed ≤ 10); offshore/analysis nesting and the single
component; the Gulf check; the edge margin covers the crop-test rim and the rim magnitudes.

**Contradictions / things to flag (docs not edited except where noted).**
1. **The 7-cell halo is Euclidean, and its chessboard width is 5.** §4.2/prompt 2 budget the 7
   cells as "3 stencil + 4 filter half-width", i.e. in stencil (chessboard) terms, but the halo
   is a fast-marching *distance*, so a cell 5 diagonal steps from land (7.07 cells away) is
   retained. This is harmless as long as the filter propagates NaN (land is NaN, every dbof
   stencil propagates it, and NaN-aware reductions drop the cell) — the halo's job is then only
   the distance field and consistency, as task 3 concluded. **If task 2's `lowpass` renormalises
   over valid cells instead** (finite values from partial stencils near land), a square kernel
   of half-width 4 would touch land from chessboard 4, and the halo should be judged in
   chessboard cells (≈ 9-10 Euclidean). Decide in task 2; nothing to change here.
2. §2.4/§4.2 "a `k`-carrying `hFacC` makes the mask 4-D and breaks `skfmm`" — true, but it
   is *loud*: `ValueError` from `skfmm.distance`, not a silent wrong answer. Only the all-land
   path is silent. Wording only.
3. The helper measures distance on the per-axis **mean** spacing while the contract converts
   cells with the **median** `dxC`; both are recorded in the nc attrs. 0.08% apart here.
4. `edge_mask` is pure geometry (True on land inside the interior rectangle), per the §4.2
   signature "False within edge_cells of ANY tile edge"; every other mask has land False.
   `mask_analysis` carries the land. Noted in the var's `long_name`.
5. `tile330_masks.nc` is `.gitignore`d (`data/.gitignore: *`), like the zarrs; §3.5 does not
   say whether it should be tracked. It is 6 MB and regenerates in < 1 s from the grid store.
6. Not a contradiction, for the record: the OSN gridfile's ocean includes the Salton Sea,
   Death Valley and the Delta (item above); none survives the offshore cut, but anything that
   ever uses `mask_halo` alone (e.g. M4 front finding on the full tile) will see them.

**Status line updated** in `frontogenesis_prompt_2.md` (task 1 done). Coding §6 M1 untouched
until the milestone closes (task 7).

Files: created `py/masking.py`, `py/m1_write_masks.py`, `py/validate.py`, `py/tests/conftest.py`,
`py/tests/pytest.ini`, `py/tests/test_masking.py`, `data/tile330_masks.nc` (ignored),
`figs/V6_land_halo_tile330.png`; modified `claude_prompts/frontogenesis_prompt_2.md` (status
line) and this log.

### 2026-09-29 — Execution prompt 2, task 2: operators.py and the regression oracle (Fable)

**Scope.** Task 2 of `frontogenesis_prompt_2.md` only: `py/operators.py`, `py/tests/test_operators.py`,
the criterion-7 regression against `calculate_fields.frontogenesis_tendency`, the 0.911x ratio, and
the `lowpass` NaN-policy decision left open by task 1 (flag 1). Tasks 3-7 not started: no
`semilag.py`, `coarsegrain.py`, no V1-V5, no data pulled. Offline, from the two M0 stores and
`tile330_masks.nc`. Nothing outside `dev/frontogenesis/` touched; `dbof` read-only at `938bce1`;
`masking.py` and `tile330_masks.nc` unchanged.

**Written.**
- `py/operators.py` (389 lines, functions only): the eight contract functions of coding §4.3 —
  `buoyancy(ds)`, `lowpass(field, L_cells)`, `grad_b`, `gradb2`, `jacobian`, `frontogenesis`,
  `strain_divergence`, `strain_alignment` — plus `strain_from_jacobian(u_x, u_y, v_x, v_y)` (the
  Jacobian-consistent `(delta, sigma_n, sigma_s, sigma_mag)`, which decomposes `F` exactly) and
  the dims guards `require_centred` / `require_u_point` / `require_v_point` / `assert_dims`
  (unconditional `raise`, not `assert`). Every dbof call is preceded by a staggering check on its
  inputs and followed by a dims assertion on its output; everything is computed eagerly in
  float64 with names/units attrs. No function filters or masks internally.
  - `buoyancy`: `calculate_fields.buoyancy_of_field` on `Theta, Salt` cast to float64; JMD95 at
    `p = 0`; sign left as is (`+g sigma0/rho0`; on t0 all 356,877 ocean values are positive,
    median 0.25 m s^-2). Bit-identical to the repo call.
  - `grad_b` / `gradb2`: `calculate_native_gradient_tracer`, and `G = b_x^2 + b_y^2` from that
    same pair (never `calculate_grad_squared_tracer`).
  - `jacobian`: raw staggered `U (j, i_g)`, `V (j_g, i)` into `calculate_jacobian`; the
    pre-check is what prevents the 4-D broadcast (below).
  - `frontogenesis`: `-(u_x b_x^2 + (u_y + v_x) b_x b_y + v_y b_y^2)` from the two above; attrs
    carry the convention `F = (1/2) DG/Dt`, compare `2F`.
  - `strain_divergence`: `calculate_native_strain_vorticity` (dict, as §2.3 says);
    `strain_shear_corner` is averaged to centres with two `grid.interp(..., padding='fill')`;
    then — **not in the contract, but required** — the strain pair is rotated from the model
    basis to geographic by `2 alpha` (`cos 2a = CS^2 - SN^2`, `sin 2a = 2 CS SN`; on face 10 a
    sign flip of both), so it shares `grad_b`'s basis. `delta` and `|sigma|` are invariant.
  - `strain_alignment`: `theta` from the **compressional** axis, folded to `[0, pi/2]`:
    `cos 2theta = -(sigma_n (b_x^2 - b_y^2) + 2 sigma_s b_x b_y) / (|sigma| G)`, so that
    `F = -(1/2) delta G + (1/2) |sigma| G cos 2theta` (see contradiction 1).
- `py/tests/test_operators.py` (21 tests; 16 offline on a synthetic C-grid in both the `CS = 1`
  and the face-10 `CS = 0, SN = -1` orientation, with `dx != dy` so a swapped metric shows; 5
  `needs_grid`). `py/tests/conftest.py`: a `masks_ds` fixture (skips when the nc is absent).

**`lowpass`: the kernel and the NaN policy (decision).** The docs fix the scale set
`{0, 2, 4, 8}`, "the same filter on `b`, `U`, `V`", and "half-width 4 for the widest filter",
but neither the kernel shape nor the NaN handling. Chosen: a separable **top-hat of half-width
`L/2`** (support `L + 1` cells, weights `1/(L+1)`; `L` must be even; `L = 0` returns the input),
applied in index space along whichever of `j`/`j_g`, `i`/`i_g` the field carries, so `U` on
`i_g` and `V` on `j_g` get the identical kernel; numpy in, numpy out (last two axes); float64.
**NaN propagates and is never renormalised**: the convolution is direct (`ndimage.convolve1d`,
not a running sum), the tile edge is padded with NaN, and a cell whose `(L+1)^2` footprint
touches a NaN is NaN. Reasons: (i) renormalising over the valid part of the footprint is a
different, cell-dependent kernel at every coastal cell, which breaks the shift-invariance that
makes the filter commute with the diff/interp stencils — the property the coarse-grained budget
and the Germano identity (planning §5.4, task 4) rest on; (ii) it would make `lowpass` the only
operator in the pipeline that turns a land neighbour into a finite number, whereas every dbof
stencil propagates NaN; (iii) it keeps the halo reasoning honest — nothing finite is ever
contaminated, so validity is `isfinite(field)` and the halo only has to size the filter
support. Verified: constants pass unchanged, the mean is preserved, a wave of wavelength `L + 1`
cells is annihilated exactly, a 48-cell wave passes at the analytic box response (0.996 / 0.983 /
0.944 for `L = 2 / 4 / 8`), `grad_b(lowpass b) == lowpass(grad_b b)` to 1e-12 wherever finite,
and on the real tile `isfinite(lowpass(b, L))` equals *exactly* `chessboard distance to a NaN >
L/2` inside the `L/2` edge band, for `L = 2, 4, 8`.

**Consequence for `halo_cells` — keep 7; no change to `masking.py` or the nc.** Reach of `F`
from filtered inputs, measured on t0 as the minimum chessboard distance from land of a finite
`F`: **2 / 3 / 4 / 6** at `L = 0 / 2 / 4 / 8` (`= L/2 + 2`: filter half-width plus the
Jacobian's reach; `G` one less). The Euclidean halo retains cells at chessboard 5 (task 1), so
at `L = 8` those are NaN in `F`: **249 of 341,960 `mask_halo` cells** (0.07%; the count inside
`mask_halo & mask_edge`, i.e. coast-related), all inside the 100 km cut. **`F` is finite on all
262,925 `mask_analysis` cells at every `L`**, and `G` likewise (0 NaN). So the flag-1 concern
("judge the halo in chessboard cells, ~9-10 Euclidean") applies only to the renormalising
policy that was not adopted; under propagation a wider halo would only relabel cells that are
NaN anyway. Recommendation: leave `halo_cells = 7`; treat `mask_halo` as a *selection* mask for
statistics (`& isfinite`), never as a multiplier on `b, U, V` before differencing — multiplying
it in would push the NaN reach of `F` at `L = 8` to ~13 cells from land for no gain (the values
on the cells where both are finite are identical either way). The same holds for the tile edge:
at `L >= 2` the NaN padding replaces the finite-but-wrong rim (the 0-fill is never reached
because NaN is hit first), `F` is NaN 6 cells deep at `L = 8`, and `edge_cells = 7` covers it.
Synthetic check (`test_lowpass_halo_consequence_diagonal_coast`): chessboard-5 halo cells exist
only with the tile's anisotropic spacing (`dxC = 1796`, `dyC = 1950` m; the halo is 7.0 cells
along `i` but 6.45 along `j`); on a square grid the minimum is 6.

**Regression check (criterion 7) — bit-for-bit.** Hour 0 merged with the grid on
`(face, j, i)` in float64, `b = operators.buoyancy(ds)`, `F = operators.frontogenesis(b, U, V,
ds, grid)` vs `calculate_fields.frontogenesis_tendency(ds, grid)`: NaN patterns identical
(352,673 finite cells); over the 262,925 `mask_analysis` cells (all finite in both) **max |dF| =
0.0, max relative difference 0.0, 262,925 / 262,925 exactly equal**; `max |F| = 2.50e-17 s^-5`
on that set. Same on hour 1 (max |dF| 0.0 on all 352,673 finite cells; max |F| 3.19e-17). This
is expected — both sides call the same `calculate_native_gradient_tracer` / `calculate_jacobian`
and the same three-term formula — so the oracle checks the *wiring* (arguments, staggering,
basis, order of operations), not the numerics; the independent numerical checks are the analytic
synthetic tests below and M0 task 5's numpy replica. Oracle only, as the prompt says.

**0.911x confirmed.** `operators.gradb2` / `calculate_fields.grad_b2` interior median on M0 task
5's cell set (`ocean & taxicab >= 3`): **0.9106** on t0 (M0 reported 0.9106), 0.9108 on t1;
0.9115 / 0.9119 with the 3-cell tile-edge band also excluded; **0.9098 / 0.9099 on
`mask_analysis`**. Asserted in `test_gradb2_ratio_to_repo_grad_b2_is_0911` (`|ratio - 0.911| <
0.002`).

**Strain: rotation sign and the 0.80x, re-measured on the analysis mask.** Regressing the
Jacobian-derived quantities on the flux-form ones over `mask_analysis` (n 262,925): `delta`
slope **0.853**, corr 0.986; `sigma_n` **0.854**, corr 0.983; `sigma_s` **1.000**, corr 1.000;
`|sigma|` 0.925, corr 0.980. Slopes positive, so the `2 alpha` rotation of the model-basis pair
is the right sign (unrotated, `sigma_n` and `sigma_s` would regress at -0.85 / -1.00 on this
face); `sigma_s` from the averaged corners equals the Jacobian's `u_y + v_x` to three figures
(interp-then-diff and diff-then-interp commute on the near-uniform metric). M0 task 5's 0.80
(corr 0.97) was measured on the whole ocean interior including the coastal band; **on the
analysis mask the interpolated Jacobian's attenuation is ~0.85** — the number V3 should expect
there. With `strain_from_jacobian`, `F = -(1/2) delta G + (1/2) |sigma| G cos 2theta` closes to
3e-33 (max |F| 2.5e-17) on the real hour and to 1e-12 relative on synthetic fields; on
`mask_analysis`, median `theta` is 0.72 rad and 54% of cells have `theta < pi/4`.

**Tests — `pytest dev/frontogenesis/py/tests`: 38 passed in 2.1 s** (`test_masking.py` 17 +
`test_operators.py` 21; `-m "not needs_grid"` → 27 passed, 11 deselected, 0.25 s). Offline
(each in both orientations where marked ×2): gradient of `sin(kx x) cos(ky y)` to < 1% and
within the `(k dx)^2/6` truncation (×2); `gradb2` equals `b_x^2 + b_y^2` and is the smaller of
the two stencils at a front (×2); the Jacobian of `u = -a x, v = a y` is `(-a, 0, 0, a)` to
1e-12 (×2); **the factor of two** (×2): on the exact solution `b = b0 tanh(x e^{at}/ell)`,
`G ~ exp(2at)` along a parcel so `DG/Dt = 2aG`, and on the grid `F = aG` to 1e-10, hence
`2F = DG/Dt` and `F` alone is off by 2x; flux-form strain of the deformation
(`sigma_n = -2a`) and of a pure shear (`sigma_s = c`) in geographic components on both
orientations, with the raw helper's model-basis pair shown to be the negative on face 10 (×2);
the alignment angle (0 across the compressional axis, `pi/2` along it) and the exact
decomposition with the plus sign (the minus-sign form misses by > 10%); **the dims guard** on an
8 x 8 grid: `b * U` is 4-D and `assert_dims` catches it, and `jacobian(V, U)`, `jacobian(U, U)`,
`grad_b(U)`, `gradb2(V)`, `frontogenesis(U, U, V)`, `frontogenesis(b, V, U)`,
`strain_divergence(V, U)`, `buoyancy` with `Theta` on `i_g`, all raise `ValueError('expected
dims ...')` before any dbof call, and a numpy array raises `TypeError`; `lowpass` identity /
odd-`L` rejection / constants / mean / annihilated wavelength / analytic long-wave response /
numpy and staggered inputs; NaN propagation (an `(L+1)^2` box around one NaN, chessboard `L/2`
from a diagonal coast, every other value identical to the NaN-free result); the halo consequence
above; commutation with `grad_b`. `needs_grid`: `buoyancy` bit-identical to the repo call, sign
and count; the regression oracle (prints the numbers); the 0.911 ratio; the strain rotation sign
and the decomposition on the real face; `lowpass` NaN footprint at the real coast, `F` finite on
all of `mask_analysis` at `L = 2, 4, 8`, the 249 halo cells, and the `L/2 + 2` reach.

**dbof behaviour vs coding §2.3.**
1. `calculate_jacobian` with swapped arguments on **numpy-backed** inputs *does* raise —
   `KeyError: "DataArray cannot have more than 1 axis dimension, but found {'i_g', 'i'}"` from
   xgcm — but only *after* the `CS`/`SN` multiply has built a 5-D `(face, j_g, i_g, j, i)`
   intermediate (8^5 on the test grid). On the tile that intermediate is 720^5 doubles, which is
   the OOM kill M0 task 5 saw, so "raises only on dask" is the practical truth but not the
   mechanism: the raise comes too late. Hence the pre-call staggering check.
2. `calculate_native_strain_vorticity`'s strain pair is model-basis (§2.3 says "magnitude
   calculations only"); the contract's `strain_alignment(b_x, b_y, sigma_n, sigma_s)` needs it
   rotated, which §4.3 did not say. Added (coding §4.3, "added 2026-09-29, M1 task 2").
3. `frontogenesis_tendency` is bit-identical to ours (same helpers, same formula); nothing
   else in §2.3 differed. Line numbers unchanged at `938bce1`.

**Contradictions / things to flag.**
1. **Planning §2.4's decomposition had the wrong sign on the strain term.** It wrote
   `F = -(1/2) delta G - (1/2) |sigma| G cos 2theta` with `theta` "the angle between `grad b`
   and the compressional axis" — but for `u = -a x, v = a y` and a front `b(x)`, `theta = 0` and
   `F = +a b_x^2 > 0`, so the sign must be **plus** for the compressional-axis angle (minus is
   the extensional-axis convention). Corrected in planning §2.4 with a marked note; coding §4.3
   comment added. `strain_alignment` implements the compressional-axis definition, which is the
   one the physics statement ("F peaks where `grad b` aligns with the compressional axis")
   needs.
2. The filter kernel and its NaN policy were unspecified in coding §1.2 / §4.3 and planning
   §5.4. Fixed by decision, recorded in coding §1.2 ("added 2026-09-29, M1 task 2"). If a
   Gaussian is preferred later, `lowpass` is the single place to change it and every test that
   pins the box response (`test_lowpass_identity_constants_and_scale`) will say so.
3. Task 1's flag 1 resolved: the chessboard-5 halo reach is harmless under NaN propagation
   (above). Its origin is the anisotropy (`7 x dxC = 6.45 dyC`), not the diagonal alone.
4. §8 "Land halo applied **before** any differencing, not after": with land NaN and NaN
   propagation the order changes no finite value, only how many cells are NaN; recommended usage
   is the halo as a selection mask after the operators (wording, no edit made).
5. M0 task 5's 0.80x Jacobian attenuation is **0.85x on `mask_analysis`** (0.80 included the
   coastal band). Not a contradiction; the V3 expectation on the analysis mask should be quoted
   as ~0.85 (coding §4.9 / §6 M1 say "~0.8"; left as is, this entry is the record).
6. `frontogenesis` returns `F`, per the contract; `two_F = 2 * F` is formed at the call site
   (M3), and `F.attrs['convention']` says so. Coding §1.1's "name the predicted variable
   `two_F`" applies to the budget dataset, not to this operator.
7. The oracle being bit-for-bit means criterion 7 cannot fail independently of the wiring; the
   `< 1%` gates V1/V2 (task 5) and the discrete null V3 (task 6) are where the numerics are
   judged. Worth keeping in mind when reading "regression to round-off" in the acceptance list.

**Status line updated** in `frontogenesis_prompt_2.md` (tasks 1-2 done). Coding §6 M1 untouched
until the milestone closes (task 7).

Files: created `py/operators.py`, `py/tests/test_operators.py`; modified `py/tests/conftest.py`
(`masks_ds` fixture), `frontogenesis_planning.md` (§2.4 sign, marked), `frontogenesis_coding.md`
(§1.2 kernel/NaN policy; §4.3 strain rotation and alignment comments, marked),
`claude_prompts/frontogenesis_prompt_2.md` (status line) and this log.

### 2026-09-29 — Execution prompt 1, task 6: M0 acceptance deck

**Scope.** Task 6 of `frontogenesis_prompt_1.md` only. No physics, no network, no data store
touched; nothing outside `dev/frontogenesis/deck/` written except the task-6 status line in
`frontogenesis_prompt_1.md`. **Full log is in `dev/frontogenesis/deck/README.md`**, as task 6
directs; this entry is the pointer.

**Built.** `deck/Frontogenesis_M0_Acceptance.pptx` — 12 slides, 1.3 MB. Title; Contents; one
slide per task (1-6); an M0 acceptance slide; and **three finding slides** beyond the
one-per-task brief: the two overturned planning claims, implicit numerical diffusion by scale,
and hourly displacement. The extra three are justified by the prompt's own framing — a
contradicted planning claim is "the most valuable output of this milestone", and burying two of
them inside a five-row task-3 slide would have understated them.

Scripts kept beside the deck as instructed: `deck/make_m0_figs.py` (three figures plus a LANCZOS
downscale of the real QA plot) and `deck/build_m0_deck.py`. Figures in `deck/figs_m0/`.

**Provenance discipline.** Every number on the slides is quoted from the task 1-5 log entries
above; `make_m0_figs.py` opens no store and makes no network call. The deck therefore cannot
drift from the log — but it also inherits any error in the log, so it is a presentation
artefact, not an independent check. Said plainly in the README.

**QA.** Geometry and overflow checked programmatically across all 12 slides: no off-slide
shapes, no estimated text overflow, margins >= 0.5 in; content dumped and verified against the
logs; 4 images placed. **Visual QA was again not possible** — no LibreOffice on this machine
(`soffice` absent), the same limitation recorded for the planning deck on 2026-09-19. The
geometry check is a proxy, not a substitute.

**Dependency.** `python-pptx 1.0.2` installed into the `frontogenesis` env. Deliberately **not**
added to `env/frontogenesis_env.yml` — nothing in the analysis path imports it, and the env
export should stay a record of the *scientific* stack.

**Flagged while writing it.** The 2026-09-19 planning deck still asserts the two claims M0
overturned (land stored as 0; `w` vanishing at the surface). It should be rebuilt from the
corrected planning doc before it is shown to anyone; noted in `deck/README.md`. It also remains
un-uploaded to the AIOcean Drive.

### 2026-09-29 — Execution prompt 2, task 3: semilag.py and V5 (Fable)

**Scope.** Task 3 of `frontogenesis_prompt_2.md` only: `py/semilag.py`, `py/tests/test_semilag.py`,
`validate.demo_interp_half_cell` (V5). Tasks 4-7 not started: no `coarsegrain.py`, no V1-V4 gate,
no data pulled. Offline, from the two M0 stores and `tile330_masks.nc`. Nothing outside
`dev/frontogenesis/` touched; `dbof` read-only at `938bce1`; `masking.py`, `operators.py` and
`tile330_masks.nc` unchanged. Nothing committed. The user approved the planning §2.4 sign
correction (M1 task 2) on 2026-09-29, provisionally ("for now").

**Written.**
- `py/semilag.py` (423 lines, functions only; §1.3's ~400 exceeded by the module docstring, which
  carries the interpolation/NaN rationale — flagged below): the five contract functions of coding
  §4.4 — `centre_velocities(U, V, grid_ds, grid)`, `departure_index(u_c, v_c, grid_ds, dt=3600,
  n_iter=3, vel_order=1)`, `interp_to_departure(field, di, dj, order=3, *, allow_low_order=False)`,
  `measured_DGDt(b_t, b_tp1, u_mid, v_mid, grid_ds, grid, dt=3600, order=3, n_iter=3, *,
  allow_low_order=False)`, `eulerian_DGDt(G_t, G_tp1, u_mid, v_mid, grid_ds, grid, dt=3600)` — plus
  `midpoint_time(f_t, f_tp1) = 0.5 (f_t + f_tp1)` (the helper for `F` and front selection),
  `grad_b_at_departure` and `gradb2_at_departure(b, di, dj, grid_ds, order)` (the piece V3/V4/V5
  reuse). Reuses `operators.grad_b`/`gradb2`, the dims guards, `masking._positional` and dbof's
  `interp_pair_to_center`; every output has dims asserted; float64 throughout. `u_mid`/`v_mid` may
  be the raw staggered midpoint pair (centred internally) or an already centred pair.
- `py/tests/test_semilag.py` (15 tests: 14 offline on `test_operators.synthetic_cgrid`, both
  orientations where the axis pairing matters; 1 `needs_grid`).
- `py/validate.py` (+~200 lines): `demo_interp_half_cell(png=True) -> dict` → **V5 →
  `figs/V5_interp_half_cell.png`** (200 dpi, 3800 x 1240, 0.45 MB; shows in `git status`,
  `git check-ignore -v` → `figs/.gitignore:3:!*.png`), plus `synthetic_uniform_grid` for the
  offline gates. `qa_land_halo` untouched.

**Design: what `G(x_d, t)` is — and what it is not.** `measured_DGDt = [G(x, t+dt) − G(x_d, t)]/dt`
with `G(x, t+dt) = operators.gradb2(b_tp1)` and `G(x_d, t)` from `b_t` interpolated onto the
**five-point tracer stencil centred at the departure point** — `x_d`, `x_d ± e_i`, `x_d ± e_j`,
the displacement held fixed across the stencil — followed by the *same* diff / `dxC` / interp /
`CS,SN`-rotate stencil as `calculate_native_gradient_tracer`, replicated in numpy in the same
operation order (at zero displacement the result is **bit-for-bit** `operators.gradb2`, max
difference 0.0; `dxC`, `dyC`, `CS`, `SN` taken at the arrival cell, a 0.02%/cell effect). It never
interpolates `G`. The tempting shortcut — interpolate `b_t` at `x_d(x)` for every `x` and hand the
shifted field to `operators.gradb2` — is **wrong**, and this is the one thing in the contract that
reads as if it asked for it: `grad[b_t(x_d(x))] = (I − grad d)^T grad b(x_d)` carries the strain
of the departure map, and in the adiabatic limit `b_t(x_d(x))` *is* `b_{t+dt}(x)`, so
`[G_{t+dt} − G(shifted b)]/dt` is the residual, not `DG/Dt`. Measured on the pure deformation
`u = −a x, v = a y` with the exact solution `b = b0 tanh(x e^{at}/ell)` (`a = 1e-5`, `dt = 3600`;
`test_deformation_measures_DGDt_not_the_residual`): the stencil construction gives measured/`2F_mid`
**0.966 [0.960, 0.967]** at `ell = 4 dx` and **0.979 [0.956, 0.989]** at `8 dx` (0.951 at 3 dx,
0.976 at 6 dx; face-10 orientation 0.963 / 0.979), the shift-then-differentiate construction gives
**+0.0003 / +0.0001** (max |·| < 0.025). The residual 2-4% is not interpolation (order 5 changes
it by < 0.5%): it is the centred-difference truncation not being conserved as the front sharpens
(measured/exact-along-parcel 0.982 at 4 dx, 0.996 at 8 dx, 0.965 at 3 dx — for a 3-cell tanh the
stencil returns 0.930 of the true `G`, and that factor changes under strain). That is exactly the
chain-rule violation V3 (task 6) is designed to measure; nothing was tuned here.

**Design: interpolation and NaN.** `interp_to_departure` is tensor-product **Lagrange**
interpolation of odd degree `order` on the `order + 1` nearest nodes per axis (2 x 2 bilinear,
4 x 4 cubic, 6 x 6 quintic), evaluated at `(j − dj, i − di)`; `order` must be odd (even raises);
`order < 3` raises unless `allow_low_order=True` (test-only, used by the order-1 bias test and V5).
Not `scipy.ndimage.map_coordinates`: a B-spline of order ≥ 2 needs the recursive prefilter, whose
response decays as `0.268^k` for the cubic, so a NaN must be filled first and the fill then leaks
into the coefficients **46% / 12% / 3.3% / 0.9% / 0.24% / 0.06%** of the jump at 1 / 2 / 3 / 4 /
5 / 6 nodes away (measured on a Gaussian with one node zeroed) — "the stencil touches NaN → NaN" could not be
made exact without masking ~6 cells from every NaN; `prefilter=False` would turn the spline into a
smoothing approximation that attenuates fronts. The Lagrange kernel is local: the support is
exactly the `(order+1)^2` nodes, so the NaN rule is exact — **NaN wherever the support contains a
NaN or leaves the tile, or `di`/`dj` is NaN; nothing is filled** (nodes whose weight happens to be 0
at an integer offset count too, deliberately, so the NaN pattern does not jump as `d → 0`). Integer
displacements reproduce node values bit-for-bit (weights exactly 1 and 0), which is what makes the
whole-cell translation exact; polynomials of degree `order` are reproduced to 1e-13. Because the
five stencil points share one fractional offset, "interpolate then difference" equals "difference
then interpolate" away from NaN. `centre_velocities` uses `interp_pair_to_center` (the Jacobian's
first step) and sets the **last centre along each interpolated axis to NaN** — the tile has no high
staggered face there and xgcm's `padding='fill'` would average with 0 (M0 task 5); the coast-facing
`U`/`V` faces are NaN in the stores, so `u_c` is NaN one cell into the ocean. `departure_index`:
`di = u_c dt/dxC`, `dj = v_c dt/dyC` with the spacing at the centre (mean of the two faces, NaN at
the last cell), refined by `n_iter = 3` midpoint iterations `d ← dt u(x − d/2)` with the velocity
interpolated **bilinearly** (`vel_order = 1`): the order rule protects the sharp front in `b`, the
velocity is smooth at the grid scale, and on the real hour `vel_order = 3` moves the departure by
at most **0.032 cell** (median 0.0014). The iteration itself matters at the 0.04-0.17-cell level
(below). On a linear flow it converges to `d = dt u/(1 − a dt/2)`, within 1e-3 cell of the exact
`x(e^{a dt} − 1)` (the first guess is 0.03 cell off). NaN reach at zero velocity from a NaN in `b`:
chessboard 1 / 2 / 3 at order 1 / 3 / 5 (the `G(t+dt)` rim plus the support); with flow it grows
with the displacement. `eulerian_DGDt = (G_tp1 − G_t)/dt + u·grad[0.5 (G_t + G_tp1)]` with `grad`
from `operators.grad_b` (geographic) and the centred model-basis velocity rotated with `CS`/`SN`.

**Velocity-spacing pairing, verified on `tile330_grid.zarr`.** `dxC` is on `(j, i_g)`, `dyC` on
`(j_g, i)`. `dxC[j, i_g = i]` equals the haversine distance between the centres `(j, i−1)` and
`(j, i)` to within 0.9995-1.0002, and `dyC[j_g = j, i]` the distance between `(j−1, i)` and `(j, i)`
to 0.9997-1.0001; the cross pairing (`dyC` against the along-`i` distance) is 1.081-1.088 off.
`dxG/dxC` 1.00007-1.00010, `dyG/dyC` 0.99990-0.99993; both spacings vary along `i` only (latitude;
`std` along `j` 0.02 m), by at most 0.02% per cell. So `U` (on `i_g`, the component along `i`)
pairs with `dxC` and index `i`; `V` (on `j_g`) with `dyC` and `j` — in native index space with no
rotation, exactly as planning §5.2/§5.3 say. That `i` is meridional (southward) and `j` zonal on
face 10 is irrelevant to the departure: `U` is the component along `i` whatever `i` points at.
Pinned in `test_departure_pairing_on_face10_no_rotation` (an eastward flow is model `V` and moves
along `j` by `c dt/dyC`, not `c dt/dxC`; a northward flow is `−U` along `−i`).

**Bias numbers (V5 and `test_interpolation_order_bias_at_a_front_maximum`).** Front
`b = b0 erf((x − x0)/(√2 σ_b))`, `σ_b = √2 σ_G`, so `G = b_x²` is a Gaussian of `σ_G = 1.5` cells
(planning §5.3's "front ~1.5 cells wide"; its prediction `dx² G_xx/8G = −1/(8 σ_G²) = −5.56%`
at the maximum; the analytic 1-D bilinear value is −5.40%). Shift by half a cell; truth = the same
discrete stencil on the exactly shifted `b` (so the stencil's own attenuation — the discrete
maximum is 0.95 of the continuum — cancels and only the interpolation error remains). Relative
error at the maximum: **bilinear `G` −4.94%**, **bilinear `b` then the stencil −4.99%** (the same
leading-order bias: `b_x` errs by `dx² b_xxx/8`, squared), **cubic `b` −0.54%** (9x smaller),
**quintic `b` −0.10%** (5x smaller again); all negative, i.e. the interpolation flattens the
maximum and fabricates frontogenesis, and positive on the convex flanks (panel b, tracking
`+dx² G_xx/8G`). Against the front width (panel c, half-cell shift, −bias at the maximum):
`σ_G` = 1.0 / 1.25 / 1.5 / 2 / 3 / 4 / 6 cells → bilinear `G` 9.7 / 6.8 / 4.9 / 2.9 / 1.35 / 0.77 /
0.35%; cubic `b` 2.0 / 1.0 / 0.54 / 0.19 / 0.041 / 0.013 / 0.003%; quintic `b` 0.67 / 0.25 / 0.10 /
0.022 / 0.002 / 0.0004 / 0.00003%. The peak-vs-peak ratio is 0.951 (bilinear `G`) vs 0.995
(cubic `b`). V5 annotates the four biases, the prediction and the ~5.5%.

**Real-hour smoke (`test_two_hours_smoke`, M0's two hours, `L = 0`).** Displacement over the
hour (ocean): median **0.364**, p90 0.732, p99 **1.252**, max 2.09 cells (M0 task 3 from the raw
`|u| dt/dx`: 0.37 / 0.75 / 1.28 / 3.5 — the max is lower because the iteration averages the
midpoint velocity and the 0-padded edge centres are now NaN); `di`/`dj` NaN in 5,235 ocean cells
(coast and edge); midpoint iteration vs first guess p99 0.042, max 0.167 cell on the analysis
mask. `measured_DGDt` and `eulerian_DGDt`: dims `('face', 'j', 'i')`, `(1, 720, 720)`, float64,
**finite on all 262,925 `mask_analysis` cells** (finite ocean cells 345,961 / 351,843 of 356,877;
NaN reach chessboard 2 from land), 1.0 s for both. On the analysis mask: rms **1.21e-18 vs
1.23e-18 s^-5** (same order), median |·| 5.96e-20 vs 6.33e-20, **corr 0.740, slope (semilag on
Eulerian) 0.730**. Diagnostic only (V3 is task 6): against `2F` at the midpoint time, all analysis
cells corr 0.49 / 0.39 and slope 0.970 / 0.784 (semilag / Eulerian); on front pixels (`G_mid` >
p90 in the mask, n 26,293) corr 0.505 / 0.424, slope **0.967 / 0.779**; median `2F dt/G` on those
pixels is 0.015, median `|DGDt| dt/G` 0.126. Order sensitivity on front pixels: order 1 − order 3
median `(ΔDGDt) dt/G` = **+0.052** — the fabricated 5% of `G` per hour, on real data, of the same
size as the V5 prediction; order 5 − order 3: rms difference 0.108 of rms, slope vs `2F` 0.911
(order 3: 0.967). So the choice between cubic and quintic is a ~5% effect on the real front pixels
(they include 1-2-cell features); **V4 (task 5) should sweep `order = 3, 5`** and the order used
should be quoted with the bias.

**Tests — `pytest dev/frontogenesis/py/tests`: 53 passed in 4.6 s** (`test_masking.py` 17 +
`test_operators.py` 21 + `test_semilag.py` 15; `-m "not needs_grid"` → 41 passed, 12 deselected,
0.6 s). `test_semilag.py`: the kernel reproduces tensor polynomials of its degree to 1e-13 and
integer shifts bit-for-bit, DataArray/numpy in and out; the order rule (raises below 3 without the
override, even orders always, in `interp_to_departure`, `gradb2_at_departure` and
`measured_DGDt`); **zero velocity** (×2 orientations): `G(x_d)` bit-for-bit `operators.gradb2`,
`measured_DGDt` and `eulerian_DGDt` exactly 0; **whole-cell translation** (×2 orientations ×2
axes, `dxC = 1800`, `dyC = 2250` so both `dx/dt` are exact binary ratios): `di`/`dj` exactly 1,
`DGDt` exactly 0, `G(x_d)` bit-for-bit the rolled `G`; the face-10 pairing; the midpoint iteration
on a linear flow; the **order bias** (numbers above, with bounds); **NaN at a synthetic coast**
(diagonal coast plus an island, the tile's anisotropic spacing, `U`/`V` NaN on coast-facing faces):
every finite `DGDt` is *identical* to the land-free result and the NaN set equals an independent
loop-based reconstruction of "support touches NaN or leaves the tile" for the velocity iteration
and the five stencil points, plus `G(t+dt)`'s rim; the **deformation** test (×2) with the naive
construction ≈ 0 and the Eulerian estimate within 0.90-1.05 of `2F` and correlated > 0.99 with the
semi-Lagrangian one; the real-hour smoke (bounds on the displacement statistics, finiteness, corr
> 0.5, rms ratio in [0.5, 2], the order-1 fabrication > 2%/h).

**Contradictions / things to flag.**
1. **The contract's wording invites the wrong construction.** Prompt 2 ("interpolates `b` to the
   departure points and then differentiates with the same `operators.grad_b` stencil") and coding
   §4.4 ("interpolates `b`, then differentiates") read naturally as "shift the field, then call
   `gradb2`", which measures `DG/Dt − 2F` (≈ 0 under pure strain, above) and would have sent V3 to
   slope 0. Planning §5.3's "onto the departure-point stencil" is the correct reading. Clarified
   in coding §4.4 and planning §5.3 item 2 ("clarified/measured 2026-09-29, M1 task 3"); pinned by
   `test_deformation_measures_DGDt_not_the_residual`.
2. **`operators.grad_b` cannot literally be called at the departure points** (it differences in
   the arrival frame), so `grad_b_at_departure` replicates the dbof stencil in numpy — the same
   kind of replica as `m0_qa_checks.jacobian_numpy`, and pinned bit-for-bit to `operators.gradb2`
   at zero displacement and to the rolled `G` at a whole-cell shift. Not duplication of
   `operators.py` (which is a wrapper), but a second copy of the `calculate_native_gradient_tracer`
   arithmetic; if that dbof stencil ever changes, the zero-velocity test will say so.
3. **Coding §4.4's "cubic+ REQUIRED" and "cubic splines" (task prompt) → local Lagrange, not
   splines**, for the NaN reason above. `map_coordinates` is not used anywhere.
4. **Order 3 vs 5 is a ~5% effect on the real front pixels** (slope vs `2F` 0.967 vs 0.911; rms
   difference 11%), larger than the synthetic 1.5-cell-front numbers suggest because real fronts
   include 1-2-cell features (order 3 bias at `σ_G = 1`: −2.0%). The default stays `order = 3`
   per the contract; V4 must report the order and sweep it.
5. **The chain-rule violation at 3-4 dx is 3-7%, not "order unity".** Prompt 2 criterion 3 says
   centred differences violate the chain rule "at `O((k dx)²)` — ~2.5 at a `4 dx` feature, order
   unity". Measured on the pure deformation: measured/`2F` 0.951 at 3 dx, 0.966 at 4 dx, 0.979 at
   8 dx (the `(k dx)²` is the prefactor; the coefficient is small). The 0.80-0.85x Jacobian
   attenuation (M0 task 5 / M1 task 2) is the larger of V3's two known effects. Wording only; the
   gate's ±0.05 stands.
6. `measured_DGDt` is NaN where the departure support touches the coast, so its reach is
   displacement-dependent (chessboard 2 minimum on the real hour, up to ~5 where a parcel comes
   from near land); `mask_analysis` (≥ 100 km) is untouched, but anything evaluated inside the halo
   alone must use `isfinite`, as M1 task 2 already recommended.
7. **Planning §5.3's displacement max (3.5 cells) was the raw `|u| dt/dx` at a centre**; the
   semi-Lagrangian departure (midpoint velocity, edge centres NaN) has max 2.09 on this hour.
   p99 1.25 agrees. Not a contradiction; the tail claim ("exceeds 1.5 cells") still holds at the
   front p99 (1.64 in M0).
8. `semilag.py` is 423 lines against §1.3's "~400", the excess being the module docstring's
   interpolation/NaN rationale; `validate.py` is 437 and will grow with V1-V4 — it should be split
   at task 5 (e.g. `validate_gates.py` / `validate_figs.py`) rather than nested.
9. Not a contradiction, for the record: the Eulerian and semi-Lagrangian estimates agree only to
   corr 0.74 / slope 0.73 on the real hour at `L = 0`, and neither correlates with `2F` above 0.5
   pointwise. No interpretation here (M1 produces no science); M3's closure and the filter sweep
   are where this belongs.

**Status line updated** in `frontogenesis_prompt_2.md` (tasks 1-3 done). Coding §6 M1 untouched
until the milestone closes (task 7).

Files: created `py/semilag.py`, `py/tests/test_semilag.py`, `figs/V5_interp_half_cell.png`;
modified `py/validate.py` (`demo_interp_half_cell`, `synthetic_uniform_grid`, docstring),
`frontogenesis_coding.md` (§4.4 clarification, marked), `frontogenesis_planning.md` (§5.3 item 2
measured note, marked), `claude_prompts/frontogenesis_prompt_2.md` (status paragraph) and this log.

### 2026-09-29 — Execution prompt 2, task 4: coarsegrain.py (Fable)

**Scope.** Task 4 of `frontogenesis_prompt_2.md` only: `py/coarsegrain.py` and
`py/tests/test_coarsegrain.py`. Tasks 5-7 not started: no V1-V4 gate, no data pulled, no budget on
real data beyond the one-hour smoke test the task asks for. Offline, from the two M0 stores and
`tile330_masks.nc`. Nothing outside `dev/frontogenesis/` touched; `dbof` read-only at `938bce1`;
`masking.py`, `operators.py`, `semilag.py`, `validate.py` and the nc unchanged. Nothing committed
(the user committed tasks 1-3 mid-session; the only uncommitted files are this task's).

**Written.**
- `py/coarsegrain.py` (273 lines, functions only): the two contract functions of coding §4.5 —
  `subfilter_flux(b, U, V, L_cells, grid_ds, grid) -> (tau_x, tau_y)` and
  `subfilter_term(b_bar, tau_x, tau_y, grid_ds, grid, tau_delta=None) -> term` — plus
  `subfilter_bdelta(b, U, V, L_cells, grid_ds, grid) -> tau_delta` (the dilatation part, below),
  `subfilter_advection(tau_x, tau_y, grid_ds, grid, tau_delta=None) -> sigma`,
  `flux_divergence(fx, fy, grid_ds, grid)` (the model's flux-form divergence at the centres,
  bit-for-bit `calculate_native_strain_vorticity`'s `divergence_center` away from the last
  row/column, which are NaN here instead of xgcm's finite 0-padded value),
  `b_at_velocity_points(b, grid)` and `filt(field, L_cells)` (`operators.lowpass`, or a sequence
  of scales applied in turn — the composite filter the Germano identity needs). Uses
  `operators.lowpass`, `grad_b` and the dims guards; every xgcm call is preceded by a staggering
  check and followed by a dims assertion; float64; NaN propagates as in `lowpass`, nothing filled.
- `py/tests/test_coarsegrain.py` (435 lines, 10 tests: 9 offline on `test_operators.synthetic_cgrid`,
  1 `needs_grid`).

**Factor and sign (derivation, in the module docstring).** Overbar = `lowpass` (linear, normalised,
shift-invariant; commutes with the discrete gradient, M1 task 2). Filter `d_t b + u.grad b = B` and
split the advection into resolved + subfilter: `d_t bbar + ubar.grad bbar = Bbar − sigma`,
`sigma := mean(u.grad b) − ubar.grad bbar`. Take `grad`, dot with `grad bbar`, and use
`−grad bbar . grad(ubar.grad bbar) = F(ubar, bbar) − ubar.grad(Gbar/2)` (the identity behind
`F = ½ DG/Dt`): **`Dbar/Dt (Gbar/2) = F(ubar, bbar) + grad bbar.grad Bbar − grad bbar.grad sigma`**.
So the term in **F units** (s^-5) is `T = −grad bbar . grad sigma`, which is what `subfilter_term`
returns — the same units as planning §5.4's equation, which is written for `Dbar/Dt (Gbar/2)`.
In the `DG/Dt` budget the study compares (`two_F` vs `DGDt`, coding §1.1):
`Dbar Gbar/Dt = 2 Fbar + 2 T + 2 grad bbar.grad Bbar`, so **M3's `subfilter` field must be
`2 * subfilter_term`**. The sign: `sigma` is a sink of `bbar`, hence the minus; `T > 0` where the
subfilter advection sharpens the resolved gradient. Attrs carry `convention` and `form`.
The flux form: `u.grad b = div(u b) − b delta` ⇒ `sigma = div tau − tau_delta` with
`tau = mean(u b) − ubar bbar` and `tau_delta = mean(b delta) − bbar deltabar`. For a non-divergent
flow `tau_delta = 0` and `T = −grad bbar.grad(div tau)` (coding §4.5, planning §5.4). **The surface
flow is divergent** (`delta` 0.1-0.5 f at fronts, planning §2.2) and `tau_delta ~ b' delta'` is the
same order as `div tau ~ u' b'/L`, so `subfilter_bdelta` supplies it and `subfilter_term` takes it
as an optional argument — omitted, the contract's non-divergent form is returned.

**Flux placement (decision): staggered velocity points, flux form, model basis.** `b` is averaged
to the U point `(j, i_g)` and the V point `(j_g, i)` (two-point mean; the first face along each axis
NaN — no low neighbour in the tile, xgcm would average with 0), `U b_u` and `V b_v` are filtered
*there* with the kernel that filters `U` and `V` (`lowpass` acts on `i_g`/`j_g` as on `i`/`j`), so
`tau_x = mean(U b_u) − Ubar mean(b_u)` on `(j, i_g)`, `tau_y` on `(j_g, i)`, with
`mean(b_u) = interp(bbar)` exactly by commutation. `div tau` is the model's flux-form divergence
`(Δ_X(tau_x dyG) + Δ_Y(tau_y dxG))/rA` at the centres (last centre NaN), `delta` for `tau_delta` the
same operator on `(U, V)`; `grad sigma` and `grad bbar` are both `operators.grad_b` (geographic; the
dot product is invariant, so `tau` stays model-basis). Why: (i) it is how the model advects — `U`
times the face value is its tracer flux (OS7MP reconstructs the face value at higher order; the
*resolved* flux of the filtered budget is the second-order one); (ii) "the same filter on `b`, `U`
and `V`" (coding §1.2) is then literal, the flux living on the velocity's own points; (iii) the
divergence reaches the centres in one difference, with no interpolation of the flux and no extra
half-cell attenuation of the third derivative the term is. The alternative — everything at the
centres from `interp_pair_to_center(U, V)` — differs by the `O(dx²)` product-rule mismatch between
the flux and advective forms; the closure test measures that error together with the chain-rule
violation V3 is about, and finds the `bbar` budget closing ∝ dx² (below), so the placement is
consistent with `semilag`'s centred departure velocity to the order of the scheme.

**Germano: holds to round-off (max 4e-16 relative), for the composite filter.** With `tau` at `L1`,
`T` at the composite level (`L1` then `L2`, `filt` with a sequence) and the Leonard flux
`Leo = subfilter_flux(lowpass(b, L1), lowpass(U, L1), lowpass(V, L1), L2)`: `T − lowpass(tau, L2) =
Leo` on every finite cell (identical NaN sets), for `(L1, L2) = (2, 4), (4, 2), (2, 2)`, both
components. It is an algebraic identity for any linear filter, so "exactly" up to floating point —
**provided the combined filter is the composition**. It is *not* one of our `lowpass` scales: two
top-hats compose to a trapezoid, and using a single top-hat of scale `L1 + L2` as the "combined"
filter misses by 31-43% rms. Hence `L_cells` accepts a sequence.

**`tau → 0` as `L → 0`.** At `L = 0` `tau` is exactly 0.0 on every finite cell (`lowpass` is the
identity, so `mean(ub) − ubar bbar` is the same product twice). For smooth fields (50-70 km scales)
`tau_x / [M2 (U_x b_x + U_y b_y)]`, `M2 = L(L+2) dx²/12` the kernel's second moment (the Clark /
gradient model): median **0.968 / 0.910 / 0.731** at `L = 2 / 4 / 8`, rms `tau` growing x2.8 and
x2.7 — i.e. `tau = O(L²)`.

**Closure (the main test; `test_closure_of_the_coarse_grained_budget[shear|divergent]`).** Exact
solutions of `d_t b + u.grad b = 0` (verified to 1e-3 by fine differences): `b0` = a 30°-tilted
tanh front of width 7.2 km plus a 21.6 x 18 km sinusoid; flows (a) **shear** `u = 0.3 sin(2πy/28.8
km)`, `v = 0` (non-divergent, `ubar ≠ u`), `b = b0(x − u(y) t, y)`; (b) **divergent** `u = 0.2
sin(2πx/36 km)`, `v = 0.3 sin(2πy/28.8 km)` (separable; the 1-D sine back-trajectory
`tan(θ0/2) = tan(θ/2) e^{−akt}` is analytic, `delta` varies at 20-29 km). Physical set-up fixed, grid
refined: `dx = 3.6 / 1.8 / 0.9 km` with `L = 2 / 4 / 8` (filter 10.8 / 9 / 8.1 km; 64 x 32 → 256 x
128 cells), `dt = 3600 s`. Measured = `semilag.measured_DGDt(bbar_t, bbar_tp1, Ubar, Vbar)`,
predicted = `2 F(bbar_mid, Ubar, Vbar) + 2 T(mid)`, on interior front pixels (`Gbar > 0.2 max`,
n = 70-1738), rms residual over rms measured:

| flow | dx, L | 2T / meas | residual without T | **with T** | flux form only |
|---|---|---|---|---|---|
| shear | 3600, 2 | 0.38 | 0.55 | **0.20** | 0.20 |
| shear | 1800, 4 | 0.48 | 0.55 | **0.085** | 0.085 |
| shear | 900, 8 | 0.44 | 0.47 | **0.041** | 0.041 |
| shear | 900, 8, dt 900 | 0.46 | 0.47 | **0.022** | 0.022 |
| divergent | 3600, 2 | 0.59 | 0.85 | **0.33** | 0.44 |
| divergent | 1800, 4 | 0.60 | 0.68 | **0.12** | 0.42 |
| divergent | 900, 8 | 0.55 | 0.58 | **0.090** | 0.39 |
| divergent | 900, 8, dt 900 | 0.54 | 0.57 | **0.032** | 0.38 |

The `bbar` budget at the midpoint, `mean(d_t b) + ubar_c.grad bbar + sigma` (exact `d_t b` by a 60 s
centred difference; `ubar_c` from `centre_velocities` rotated to geographic), rms over rms of the
resolved advection: shear **0.121 / 0.036 / 0.0093** (∝ dx², x3.3 and x3.9 per halving) with
`sigma` 35-40% of the advection and 0.36-0.49 without it; divergent **0.22 / 0.051 / 0.014** with,
0.42-0.75 without, **0.70 / 0.38 / 0.28 with the flux form alone**. So: the term is O(1) (`2T` is
38-60% of the measured tendency), the budget does not close without it, it closes to **< 10% at
900 m** and the residual shrinks with resolution; the 4-9% floor at `dt = 3600` is the
midpoint-field time discretisation (`0.5 (b_t + b_tp1)` vs `b(t_mid)`, `(dt u k)²/8` of the
sinusoid), since `dt = 900 s` at the same `dx` takes it to 2-3% while the `bbar` budget (evaluated
at `t_mid` exactly) is unaffected. Stated tolerance in the test: `res < 0.10` and `b_res < 0.04` at
900 m, `res < 0.5 res_no` at every level, monotone in resolution, `b_res(900) < 0.4 b_res(3600)`,
`res(dt 900) < 0.6 res(dt 3600)`; on the divergent flow the flux form leaves > 3x the closed
residual at the two finer levels (1.2x at the coarsest, where discretisation dominates); on the shear
flow `tau_delta` is exactly 0. Face-10 orientation (`test_closure_on_the_rotated_face`, 1.8 km,
`L = 4`): 0.076 / 0.116 (shear / divergent) vs 0.085 / 0.119 unrotated — `tau` in the model basis and
the invariant dot product are handled.

**NaN at a synthetic coast** (diagonal coast + island, the tile's anisotropic spacing, `U`/`V` NaN on
the coast-facing faces, `L = 2, 4, 8`): every finite `tau`, `tau_delta` and term is *identical* to
the land-free result; the term is NaN within chessboard **`L/2 + 2`** of land (measured min reach)
and finite beyond `L/2 + 3`; the tile-edge rim is NaN `L/2 + 2` deep. Same reach as `F` (M1 task 2),
so `halo_cells = 7` / `edge_cells = 7` still cover it.

**Real hour (`test_first_hour_smoke`, hour 0, `L = 2, 4, 8`).** `tau`, `tau_delta`, `Fbar` and the
term finite on all 262,925 `mask_analysis` cells; dims `('face', 'j', 'i_g')` / `('face', 'j_g',
'i')` / `('face', 'j', 'i')`. rms(term)/rms(`Fbar`), median |term/`Fbar`|, corr(term, `Fbar`):

| L | analysis mask | front pixels (`Gbar` > p90, n 26,293) | rms `Fbar` (s^-5) |
|---|---|---|---|
| 2 | **0.31**, 0.35, −0.66 | 0.30, 0.25, −0.67 | 1.8e-19 |
| 4 | **0.50**, 0.71, −0.60 | 0.48, 0.46, −0.62 | 9.7e-20 |
| 8 | **0.70**, 1.15, −0.54 | 0.68, 0.71, −0.55 | 4.0e-20 |

So the term is O(1) in the sense the planning means — not small at any `L` — but it **grows with
`L`** (0.3 → 0.7 of `Fbar` in rms while `Fbar` itself falls 4.5x from `L = 2` to 8) rather than
staying constant, and it is anti-correlated with `Fbar`. **The flux form alone overstates it 2.2x
in rms at every `L`** (flux-only / full = 2.17 / 2.30 / 2.15): the `grad delta . grad b` part of
`div tau` is cancelled by `tau_delta` (in the Clark limit `div tau − tau_delta ≈ M2 ∂_i u_j ∂_i∂_j b`,
the `M2 grad delta . grad b` piece dropping out), and on this divergent surface flow that piece is
larger than what remains. M3 must pass `tau_delta`. No interpretation beyond that (M1 produces no
science).

**`L → dx` behaviour (characterised, not toleranced).** The explicit term is exactly 0 at `L = 0`
and `O(L²)` at small `L` (Clark limit above; on the real hour 0.31 of `Fbar` at `L = 2`, the
smallest scale in the set). It therefore does not "become" the model's numerical-diffusion term:
the OS7MP implicit dissipation acts on the model's `b` before we ever see it and is never in an
explicit `tau` built from the model fields — it sits in the residual at every `L`, filtered along
with everything else (and, being grid-scale-selective, its filtered magnitude decreases with `L`).
Planning §5.4's sentence holds for the *total* subfilter flux (explicit + implicit); noted there.

**Tests — `pytest dev/frontogenesis/py/tests`: 63 passed in 6.0 s** (`test_masking.py` 17 +
`test_operators.py` 21 + `test_semilag.py` 15 + `test_coarsegrain.py` 10; `-m "not needs_grid"` →
50 passed, 13 deselected, 1.5 s). `test_coarsegrain.py`: the exact solutions satisfy the advection
equation (x2 flows); `tau` at `L = 0` and the Clark asymptotics; Germano (three scale pairs, both
components, and the single-top-hat mismatch); `flux_divergence` vs the repo's `divergence_center`
(bit-for-bit, rotated grid) and the dims guards (`flux_divergence(V, U)`, `subfilter_flux(U, U, V)`
raise before xgcm); closure (x2 flows, four levels each, both budgets); the rotated face; NaN at the
coast; the real hour.

**Contradictions / things to flag.**
1. **Coding §4.5 / planning §5.4's `−grad(bbar).grad(div tau)` is the non-divergent form and does not
   close the surface budget.** The surface flow is divergent; the exact term needs
   `sigma = div tau − tau_delta`. On the synthetic divergent flow the flux form leaves 38-44% of the
   `Gbar` tendency unclosed (vs 9-12% with `tau_delta`); on the real hour it overstates the term
   2.2x. Added `subfilter_bdelta` and the optional `tau_delta` argument (the contract signatures
   are unchanged); coding §4.5 and planning §5.4 corrected, marked.
2. **Units of the budget field `subfilter` were unspecified** (coding §3.4, §4.7, planning §6 Phase
   2 write `measured − 2F − subfilter − …`). `subfilter_term` returns F units, matching planning
   §5.4's `Dbar/Dt (Gbar/2)` equation; the budget field must be `2 * subfilter_term`. Recorded in
   coding §4.5; M3's `compute_budget` should name it accordingly (or `two_` it, per §1.1).
3. **"As `L → dx` it becomes the numerical-diffusion term"** (planning §5.4, prompt 2 contract and
   task 4 text) is true of the total subfilter flux, not of the explicit `tau`, which is exactly 0 at
   `L = 0` and `O(L²)` after (above). Noted in planning §5.4; the sweep's `L = 0` column is `2F` vs
   measured with the whole numerical term in the residual, not a limit of the `tau` term.
4. **"`O(1)` at every `L`"**: measured 0.30 / 0.50 / 0.70 of `Fbar` at `L = 2 / 4 / 8` on hour 0 —
   O(1) but growing with `L`, not constant. Recorded in planning §5.4.
5. **The Germano identity is exact only for the composite filter.** Coding §5's "Germano
   consistency" cannot be checked between two members of `{2, 4, 8}` directly (two top-hats do not
   compose to a top-hat; 31-43% mismatch). `filt` / `L_cells` as a sequence is the accommodation.
6. The closure tolerance has a `dt` floor: at `dt = 3600` the `Gbar` budget on the synthetic
   fields closes to 4% (shear) / 9% (divergent) at 900 m, dominated by the midpoint-field time
   discretisation of the *test* (`0.5 (b_t + b_tp1)` vs `b(t_mid)`), not by the operators. On real
   hours the same midpoint construction is used by design (coding §1.2), so a few-% term of this
   kind is part of M3's closure budget; V3 (task 6) will see it too.
7. `filt`'s composite `L_cells` is a small extension of the contract's `L_cells: int` (coding §1.2
   "integer `L_cells` in {0, 2, 4, 8}"); the integer path is unchanged.
8. Not a contradiction, for the record: on the real hour the term is *anti*-correlated with `Fbar`
   at every `L` (−0.54 to −0.67). No interpretation here.

**Status paragraph updated** in `frontogenesis_prompt_2.md` (tasks 1-4 done). Coding §6 M1 untouched
until the milestone closes (task 7).

Files: created `py/coarsegrain.py`, `py/tests/test_coarsegrain.py`; modified `frontogenesis_coding.md`
(§4.5: `subfilter_bdelta`, `tau_delta`, units, placement — marked), `frontogenesis_planning.md`
(§5.4: divergence and the `L → dx` precision — marked), `claude_prompts/frontogenesis_prompt_2.md`
(status paragraph) and this log.

### 2026-09-29 — Execution prompt 2, task 5: gates V1, V2, V4 (Fable, restarted)

**Scope.** Task 5 of `frontogenesis_prompt_2.md` only: `validate.test_cartesian_deformation` (V1),
`test_native_metric` (V2), `test_interpolation_bias` (V4), their PNGs, `py/tests/test_validate.py`, and
the split of `validate.py` a previous (stalled) session had half-done. Tasks 6-7 not started: no V3,
no `test_nan_finding.py`, no data pulled, nothing committed. `masking.py`, `operators.py`, `semilag.py`,
`coarsegrain.py` and the data stores untouched. The user reaffirmed approval of the planning §2.4 sign
correction (for now) on 2026-09-29.

**Written / the split.**
- `py/validate.py` (512 lines): the numbers only. `test_cartesian_deformation(alpha=1e-5, png=True, ...)`,
  `test_native_metric(grid_ds, png=True, ...)`, `test_interpolation_bias(png=True, ...)` — the §4.9
  signatures, extra keyword arguments (grid size, widths, sweeps) defaulted — each returning a dict
  with a `gate` entry, plus the unchanged `demo_interp_half_cell` (V5) and `qa_land_halo` (V6) and
  their helpers (`_snapshot_fields` stays, `test_masking.py` imports it). `__test__ = False` on the
  three `test_*` names; pytest collects nothing from it (checked with `--collect-only`).
- `py/synthetic.py` (294 lines, inherited and reviewed): `synthetic_cgrid`, `model_components`, `da`,
  `deformation_fields` (the grid helpers `test_operators.py` used to define — it now re-exports them
  from here, so `test_semilag` / `test_coarsegrain` are unchanged), `synthetic_uniform_grid`,
  `inner`, `erf_front`, `deformation_case` / `deformation_step` / `deformation_series` (V1),
  `uniform_shift_bias` (V4/V5); added `wave`, `sphere_radius` (V2) and `real_hour_fractions`,
  `REAL_HOUR_CELLS` (V4). All of the inherited builders were read and exercised; they are sound:
  the deformation departure is the converged midpoint rule (`x (1 + e/2)/(1 - e/2)` vs `e^e`,
  `e = a dt`, 4e-6 apart), the backward chain interpolates the displacement bilinearly, which is exact
  for the linear flow, and `uniform_shift_bias` builds `b_tp1` as the exactly translated `b_t` (integer
  shifts give `rel = 0.0` to the bit, tilted front included).
- `py/validate_figs.py` (473 lines, inherited): `fig_V1`, `fig_V2`, `fig_V4`, `fig_V5`, `fig_V6`, one
  per PNG, taking the dicts/arrays `validate.py` computed. Fixed: `fig_V2` took eight positional arrays
  and computed the latitude bins itself (now `(res, ctx)`, the bins come from `validate.py` with the
  local truncation prediction per component); `fig_V1`'s profile panel assumed four widths; `fig_V4`
  panel (b) put an exact 0 on a log axis (symlog) and panel (c)'s title hard-coded "sigma_G^-4" (now
  the fitted slopes). V5/V6 re-rendered **byte-identical** (`git status` shows them unmodified).
- `py/tests/test_validate.py` (6 tests: V1, V2 `needs_grid`, V3 slot skipped, V4, V5, V6 `needs_grid`;
  `png=False`). `python validate.py` writes all six PNGs.

**V1 — Cartesian deformation: PASS.** `u = -a x, v = a y`, `a = 1e-5 s^-1`, `dt = 3600`, front
`b = b0 tanh(x/ell)` on a 128^2 grid of 1.8 km, both orientations (`CS = 1` and face 10's `CS = 0,
SN = -1`: **bit-identical**, max difference 4e-15). What is compared: `G` from `operators.gradb2` at
the arrival cells and from `semilag.gradb2_at_departure` (order 3) at the departure points of
`semilag.departure_index`, on the exact solution `b(x, t) = b0 tanh(x e^{at}/ell)`;
`semilag.measured_DGDt` reproduces `(G_1 - G_d)/dt` **bit-for-bit** (max rel diff 0.0).
- *The gate:* parcels arriving at every front pixel (`G >= 0.2 max`, n = 1568) followed backwards
  through **8 chained semi-Lagrangian hours** at the reference width `ell = 8 dx` (tanh; `sigma_G`
  ~ 3.6 cells): `max |G(t_n)/G(t_0) / exp(2 a t_n) - 1|` over parcels and steps = **0.776%**
  (rms at 8 h 0.53%; `exp(2at)` reaches 1.78) — **< 1%, PASS**.
- *Vs front width (one-step growth rate `ln[G(x, t+dt)/G(x_d, t)]/(2 a dt)`, front pixels):* median
  0.9949 [0.9896, 1.0030] at 8 dx, 0.9928 at 6 dx, 0.9825 at 4 dx, 0.9663 at 3 dx, 0.9235 at 2 dx;
  rms error **0.65 / 1.11 / 2.42 / 4.19 / 9.03%** at 8 / 6 / 4 / 3 / 2 dx, i.e. **order 1.90 in
  `dx/ell`** (4-8 dx). Over the 8 h chain the max error is 0.78 / 1.57 / 3.09 / 4.43 / 7.86%.
  The deficit sits on the flanks (panel d), where the discrete gradient of a sharpening tanh is
  attenuated most: it is the **centred stencil's truncation**, not the scheme.
- *The semi-Lagrangian step alone* (`G_d` against the same stencil applied to the analytic `b` at the
  exact departure point, so the stencil's truncation cancels): max **0.354 / 0.174 / 0.085 / 0.026 /
  0.010%** at 2 / 3 / 4 / 6 / 8 dx (order 3); 0.148 / 0.028 / 0.009 / 0.001 / 0.0006% (order 5).
  **< 1% at every width — PASS** (this is planning §6's "scheme in isolation").

**V2 — native-grid metric: PASS.** `f = sin(2 pi (lon - lon0)/2 deg) cos(2 pi (lat - lat0)/2 deg)`
(96 x 124 cells; `lon0 = -123.52`, `lat0 = 31.54` = the analysis-mask means) through
`operators.grad_b` on `tile330_grid.zarr` (face 10, `CS = -2e-17`, `SN = -1`), against the exact
gradient on a sphere of **R = 6370 km** (MITgcm `rSphere`; the grid's own `dxC`/`dyC` over the
haversine centre distances give **6370.0 / 6369.2 km**). Errors normalised by `max |grad f|` on
`mask_analysis` (262,925 cells; all finite): **`b_x` max 0.077%, rms 0.033%, p99 0.071%; `b_y` max
0.041%, rms 0.018%, p99 0.037%** — **< 1%, PASS** by 13x; pointwise relative error where
`|grad f| > 0.5 max`: 0.080%. Where it is worst: `b_x` at 37.54N 124.49W (j 168, i 48; the north end,
where a lon-wave crest coincides with the largest phase advance per cell), `b_y` at 32.03N 127.01W.
The error is the stencil's truncation, predicted per cell as `-(theta^2/6) f'` with `theta` the
phase advance per cell (measured/predicted max 0.077/0.071%, 0.041/0.038%; the rms-vs-latitude curves
lie on the prediction, panel d; halving/doubling the wavelength scales the max error by 3.8x / 3.2x —
order 1.8). *The metric alone* (linear `lon - lon0`, `lat - lat0`: no truncation): **max 0.0122% /
0.0114%**, scale medians 0.99994 (`dyC` is 0.012% under `R cos(lat) dlambda` at 6370 km) / 1.0000.
Components swapped (a wrong `CS`/`SN`) would give **87%**. `dxC / dyC`: 1.69 / 1.83 km at the north
end, 1.90 / 2.06 km at the south.

**V4 — interpolation bias: recorded.** `semilag.measured_DGDt` under a uniform flow that translates
an `erf` front (`G` Gaussian of `sigma_G` cells) by a prescribed displacement per hour, `b_tp1` the
exactly shifted `b_t`, so the true `DG/Dt = 0`. Reported as `rel = DGDt dt / G(t+dt)`, the fabricated
tendency per hour as a fraction of `G` (positive at the maximum = fabricated frontogenesis), rms over
front pixels (`G >= 0.2 max`) and signed at the maximum. 64 x 96 grid, 8-cell margin.
- **Headline error bar (definition):** the rms over front pixels of `DGDt dt/G` for the
  **`sigma_G = 1.5`-cell front at order 3**, averaged in quadrature over the sub-cell cross-front
  displacement implied by the task-3 real-hour distribution (`|d|` lognormal with median 0.364 and
  p99 1.252 cells, capped at 2.09; direction isotropic; only the fractional part of the cross-front
  component matters, integer shifts being exact): **0.28% of `G` per hour** (all-cross-front upper
  bound 0.33%; signed at the front maximum +0.30%). Against the 7-20% per-hour signal `2F dt/G`:
  **1.4-4.0%** of the signal. Order 1: 2.30% (11-33% of the signal); order 5: 0.060% (0.3-0.9%).
- *Sub-cell fraction* (`sigma_G = 1.5`, rms / signed at the max): order 3 **0.36% / +0.54%** at a
  half cell, 0.31% at 0.25, 0.26% at 0.75, 0.14% at 0.1, **0 at 0 and 1** (rms 0.0); order 1
  3.46% / +4.99%; order 5 0.076% / +0.10%. Asymmetric about 0.5 because the Lagrange nodes are
  floor-based (`-1..2`), a property of the kernel.
- *Direction* (0.5 cells at 0 / 30 / 45 / 60 / 90 deg from `i`): front along `j`, order 3: 0.36 /
  0.43 / 0.39 / 0.31 / **0.000%** (along-front is exact); front tilted 30 deg: 0.25 / 0.26 / 0.25 /
  0.21 / 0.03% — both kernel axes engaged, no larger than the 1-D case.
- *Front width* (rms at a half cell → at the real-hour distribution), `sigma_G` = 1.0 / 1.5 / 2 / 3 /
  4 / 6 cells: order 3 **1.61 / 0.36 / 0.135 / 0.031 / 0.010 / 0.002% → 1.02 / 0.28 / 0.099 / 0.021 /
  0.007 / 0.001%**; order 5 0.62 / 0.076 / 0.018 / 0.002 / 0.0004 / 0.00003% → 0.38 / 0.060 / 0.013 /
  0.001 / 0 / 0%; order 1 6.8 / 3.5 / 2.0 / 0.90 / 0.51 / 0.23% → 4.6 / 2.3 / 1.35 / 0.62 / 0.35 /
  0.16%. Fitted slopes for `sigma_G >= 1.5`: **`sigma_G^-3.8` (order 3), `^-5.6` (order 5),
  `^-2.0` (order 1)**.
- *Order sweep, summary:* at the recorded operating point (order 3, 1.5 cells) the bias is 8x below
  order 1 and 4.6x above order 5; at 1 cell it is 1.0% of `G`/h at order 3 and 0.38% at order 5.

**Tests — `pytest dev/frontogenesis/py/tests`: 68 passed, 1 skipped (the V3 slot) in 15.6 s**
(`test_masking` 17 + `test_operators` 21 + `test_semilag` 15 + `test_coarsegrain` 10 +
`test_validate` 5 + 1 skip; `-m "not needs_grid"` → 53 passed, 1 skipped, 15 deselected, 8.3 s).
`test_validate.py` asserts: V1 series < 1% (both orientations), semilag-only < 1% at every width,
`measured_DGDt` bit-for-bit, convergence order in [1.5, 2.5], orientations identical; V2 max < 1%,
metric alone < 0.1%, swapped > 50%, wavelength order in [1.7, 2.3], measured < 2x predicted,
R within 5 km of 6370; V4 integer shifts exact, order hierarchy (1 > 5x order 3 > 3x order 5), order-1
half-cell > 2%, order-3 < 0.6%, positive at the maximum, width slope < -3, and a **regression bound**
(headline < 1% of `G`/h, i.e. < 15% of the smallest signal) — not a criterion, the criteria set none;
V5 reproduces the task-3 biases (-4.94 / -4.99 / -0.54 / -0.10%); V6 reproduces the task-1 counts.
PNGs (200 dpi, all in `git status`, `git check-ignore -v` → `figs/.gitignore:3:!*.png`):
`figs/V1_cartesian_deformation.png`, `figs/V2_native_metric.png`, `figs/V4_interpolation_bias.png`
(new); `figs/V5_interp_half_cell.png`, `figs/V6_land_halo_tile330.png` (re-rendered, identical).

**Contradictions / things to flag.**
1. **Criterion 1's "< 1%" is front-width dependent.** The literal comparison (`G` along parcels vs
   `exp(2at)`) passes at `ell = 8 dx` (0.78% over 8 h) and fails it at `<= 6 dx` (1.6% at 6 dx, 3.1%
   at 4 dx, 7.9% at 2 dx): the centred stencil's truncation, second order in `dx/ell`, which both
   sides of the budget share. The semi-Lagrangian step itself (planning §6's "scheme in isolation")
   is < 0.36% at every width. Not tuned — the gate is stated at its reference width and the
   dependence reported; the chain-rule/truncation mismatch is exactly what V3 (task 6) measures on
   the `2F` side (task 3 measured 0.966 at 4 dx). Coding §6 M1 criterion 1 left as is; the width is
   recorded here.
2. **The synthetic error bar understates the real-hour order sensitivity.** Order 3 vs 5 differ by
   0.22% of `G`/h at `sigma_G = 1.5`, but task 3 found ~5% of `G`/h between them on the real front
   pixels (which include <= 1-cell features: at `sigma_G = 1.0` the order-3 bias is 1.0% of `G`/h,
   order 5 0.38%). **Quote the error bar with its width: 0.28% (1.5-cell) to 1.0% (1-cell) of `G`
   per hour at order 3**, i.e. up to 14% of a 7% signal on the sharpest real fronts — and M3 should
   report the slope at order 5 alongside order 3 (a 5%-of-signal-class check, per task 3).
3. **The direction of the real displacement relative to the front is unknown**, so the headline
   assumes an isotropic direction; the all-cross-front value (0.33%) is the bound. V3 with `llc`
   velocities (task 6) is where the real distribution enters.
4. `validate.py` is 512 lines against §1.3's ~400 even after the split (the docstrings state what
   each gate compares, which is the point of the module); `validate_figs.py` 473. Flagged, not
   trimmed further — a third split (`validate_gates.py`) would move the contract names off
   `validate.py`.
5. Not a contradiction: the grid's zonal metric `dyC` implies 6369.2 km against `dxC`'s 6370.0
   (0.012%), the whole of the "metric alone" error; a 1-cell V2 wavelength sweep would be needed to
   see anything metric-like above the truncation, and there is nothing (panel d).
6. Coding §4.9 says V2 "needs the real tile grid" only; it also reads `data/tile330_masks.nc` for
   `mask_analysis` (falls back to `masking.build_masks`), like V6.

**Status paragraph updated** in `frontogenesis_prompt_2.md` (tasks 1-5 done). Coding §6 M1 untouched
until the milestone closes (task 7).

Files: created `py/synthetic.py`, `py/validate_figs.py`, `py/tests/test_validate.py`,
`figs/V1_cartesian_deformation.png`, `figs/V2_native_metric.png`, `figs/V4_interpolation_bias.png`;
modified `py/validate.py` (split + V1/V2/V4), `py/tests/test_operators.py` (grid helpers imported
from `synthetic.py`), `claude_prompts/frontogenesis_prompt_2.md` (status paragraph) and this log.

### 2026-09-29 — Execution prompt 2, task 6: gate V3, the discrete null (Fable)

**Scope.** Task 6 of `frontogenesis_prompt_2.md` only: `validate.test_discrete_null` (V3, the hard
gate), the operator change it forced, the re-run of the whole suite, `figs/V3_discrete_null.png`,
the V3 slots in `test_validate.py`. Task 7 not started; no data pulled; nothing committed; nothing
outside `dev/frontogenesis/` touched (`deck/`, prompt 1 untouched); `dbof` read-only at `938bce1`.
Every python/pytest command ran under `timeout 300`; the longest was 33 s.

**Construction.** A tracer `b_t` is advected one hour by **our own** semi-Lagrangian step:
`u_c, v_c = semilag.centre_velocities`, `(di, dj) = semilag.departure_index` (midpoint iteration,
`n_iter = 3`), `b_tp1 = interp_to_departure(b_t, di, dj, order 3)` — so the truth satisfies our
discrete advection exactly. Measured side: `semilag.measured_DGDt(b_t, b_tp1, U, V)` (the same
departure). Predicted side: `2F` from `operators.frontogenesis` at the trajectory midpoint
`b_mid = midpoint_time(b_t, b_tp1)` with the (steady / time-midpoint) `U, V`.
*`'strain'`*: `synthetic.null_strain_case` — `b = b0 tanh(s/ell)` with the front normal at
0 / 30 / 60 deg from the compressional axis (`F = +aG / +aG/2 / −aG/2` for the deformation alone)
and `ell = 2, 3, 4, 6, 8 dx` (`sigma_G = 1-4` cells; equal jump `b0`), 15 cases on 128² grids of
1.8 km; velocity = deformation `a = 1e-5` plus sinusoidal shear (`0.25 m/s`, 40 dx: vorticity +
shear strain), along-x divergence (`0.15 m/s`, 36 dx) and along-y divergence (`0.20 m/s`, 48 dx),
so the strain varies along every front; front pixels pooled under one threshold. `ell = 1, 1.5`
run out of the pool as diagnostics. *`'llc'`*: the real tile grid, hour-0 JMD95 `b` as the tracer
(the real front-width distribution), `0.5 (U_t + U_tp1)` of M0's two hours, `mask_analysis`.
Displacement on front pixels: median 0.41, p99 1.61, max 2.07 cells.

**Declared before any number was seen** (module constants in `validate.py`): front pixels =
`G_mid = gradb2(b_mid) >= p90` over the valid set (`mask_analysis` & finite; the interior with an
8-cell margin, pooled, for strain) — planning §11's rule for M3, independent of both endpoints,
and what tasks 3-4 already used (`G_mid > p90`, n 26,293); *not* config D (85th percentile in
64-px windows with `sharpen`/`despur`), which is M4's front *finding*. Gate estimator = **OLS of
measured on `2F` with intercept** (`2F` is the smooth side in a null whose only error is
discretisation); also reported: through-origin OLS, inverse OLS, geometric mean (RMA), orthogonal
(TLS; both axes share units), ratio `sum y/sum x`, corr. Bootstrap over **32 x 32-cell spatial
blocks** (1000 draws, 2.5-97.5%), never pixels. The strain pool widths and the out-of-pool
diagnostics were fixed at the same time. Nothing was changed after the first numbers.

**First attempt — FAIL, both variants** (the operators as they were: chain-rule `F`, bilinear
velocity in the departure iteration):
- *strain*: OLS **0.9496 [0.9432, 0.9616]** (n 18,816, 120 blocks; orthogonal 0.967, inverse OLS
  0.986, GM 0.968, origin 0.949, ratio 0.827, corr 0.981). Per width: **0.937 / 0.969 / 0.981 /
  0.990 / 0.993** at 2 / 3 / 4 / 6 / 8 dx (width share of the front pixels 14 / 17 / 20 / 24 / 25%);
  out of the pool 1 dx **0.817**, 1.5 dx 0.897. Per case the 0-deg fronts are lowest (0.814 at 2 dx).
- *llc*: OLS **0.7579 [0.7333, 0.7839]** (n 26,293, 249 blocks; orthogonal 0.790, inverse 0.844,
  GM 0.800, ratio 0.693, corr 0.947). Median `2F dt/G` on front pixels 0.010; median `|DGDt| dt/G`
  0.045 (the null's measured side is dominated by the ~4%/h of strain-driven change).

**Diagnosis (the finding about the discretisation).**
1. **The Jacobian attenuation is invisible to a semi-Lagrangian null — by construction.** The
   Jacobian's `u_x` is `interp(diff(interp(U)))`, which on the C-grid is *exactly* the wide centred
   difference `[u_c(i+1) − u_c(i−1)]/(2dx)` of the centred velocity `u_c = (U_i + U_{i+1})/2`, i.e.
   the `(1,2,1)/4` mean of the flux-form divergence (measured: Jacobian trace vs numpy `D_h u_c`
   slope 1.0000, corr 1.0000). The departure map is built from the same `u_c`, and the stencil
   applied to `b_t(x − d(x))` sees `D_h d`. So both sides see the same 0.85x: on front pixels the
   departure strain regresses on the Jacobian trace at 0.965 (bilinear velocity) / **0.992** (cubic),
   the Jacobian on the flux-form `delta` at 0.856, the departure strain on `delta` at 0.847. The
   0.80 / 0.85 the docs told V3 to expect cannot appear here; see contradictions.
2. **What the null measures is the chain-rule violation.** With `L` the (linear) gradient stencil
   and `b_tp1 = b_t(x − d)`, `d/dt (L b_tp1) = −L(u·grad b)` exactly, so the semi-Lagrangian
   difference tends to `−2 (L b)·[L, u·grad] b` — the commutator of the stencil with advection.
   In the continuum `[grad, u·grad] b = (grad u) grad b` and the textbook `F` follows; on the grid
   `[L_k, u·grad] b = [(u(x+e_k) − u(x))·grad b(x+e_k) + (u(x) − u(x−e_k))·grad b(x−e_k)]/(2h_k)`:
   the one-sided velocity differences times the *true* gradient at the two stencil neighbours,
   whose average is `b' + h² b'''/2` where `L b = b' + h² b'''/6`. At the centre of `tanh(x/ell)`
   the product form is therefore `1 + (2/3)(dx/ell)²` too large: **0.83x at 2 dx, 0.93 at 3, 0.96 at
   4, 0.99 at 8** — the per-width slopes above, softened by the flanks (where `b'''` changes sign).
   Task 3's 0.966 at 4 dx and V1's width dependence were the same effect.
3. **A third, separate defect: the bilinear velocity in the departure iteration.** Low-passing the
   *velocity* (`L = 8`, `b` raw) took the consistent form from 0.938 to 1.001 on the real hour;
   low-passing `b` (`L = 8`, velocity raw) left it at 0.951 / 0.981 (bilinear / cubic). Bilinear
   interpolation of `u_c` at the sub-cell midpoint `x − d/2` smooths its grid-scale structure by
   `f(1−f) dx² grad² u / 2`; the displacement barely notices (≤ 0.03 cell, task 3) but its
   *gradient* — the strain the front responds to — does (0.965 of the Jacobian's, above).

**Every change tried (OLS on the pre-declared front pixels).**

| change | strain | llc |
|---|---|---|
| chain-rule `F`, bilinear departure velocity (**first attempt**) | 0.9496 | 0.7579 |
| chain-rule `F`, cubic departure velocity | 0.9512 | 0.7914 |
| consistent `F`, 2nd-order neighbour gradient (`D_2h`), bilinear velocity | — | 0.9824 |
| consistent `F`, 2nd-order neighbour gradient, cubic velocity | 1.0223 (per width 1.029 → 1.002) | 1.0267 |
| consistent `F`, **4th-order neighbour gradient**, bilinear velocity | — | 0.9382 |
| consistent `F`, 4th-order neighbour gradient, **cubic velocity** (**adopted**) | **1.0044** | **0.9806** |
| … with `vel_order = 5` / `order = 5` for `b` / `n_iter = 0` | — | 0.987 / 0.986 / 0.962 |
| … with `2F` interpolated to the spatial trajectory midpoint `x − d/2` | — | 1.024 (over-corrects; kept at the arrival cell per coding §1.2) |
| … velocity low-passed `L = 2 / 4 / 8` (`b` raw; bilinear vel.) | — | 0.966 / 0.987 / 1.001 (cubic, `L = 8`: 1.006) |
| … `b` low-passed `L = 8` (velocity raw) | — | 0.981 (bilinear 0.951) |
| 4th-order gradient stencil on **both** sides, chain rule (the prompt's other option; angle 0) | 0.942 / 0.977 / 0.985 / 0.984 / 0.979 at 2 / 3 / 4 / 6 / 8 dx | — |
| face-10 orientation of the strain case (2 / 4 dx, angle 0) | 1.0003 / 0.9959, bit-identical to CS = 1 | — |

The 2nd-order neighbour gradient (`Ā_h L b = D_2h b`, error `h² b'''/2` vs the needed `h² b'''/3`)
over-corrects by half the deficit, +2.9% at 2 dx, and "passes" the real hour with the bilinear
velocity only because the two errors cancel — rejected. Raising the gradient order everywhere
halves the deficit but the product rule still fails (0.94 at 2 dx, and it changes `G`, the
measured side, the rims and the oracle) — rejected. The 4th-order neighbour gradient
`(4 D_h − D_2h)/3` is also the node derivative of the cubic Lagrange interpolant the step uses,
averaged over its two one-sided limits, which is the natural closure of the argument.

**Final — PASS, both variants** (`form='discrete'`, `vel_order = 3`):
- *strain*: OLS **1.0044 [0.9950, 1.0171]** (se 0.006; n 18,816); origin 1.004, inverse 1.039, GM
  1.021, orthogonal 1.022, corr 0.983; ratio 0.851 — ill-conditioned for a pool with both signs of
  `2F` (the 60-deg fronts are frontolytic), reported and not used. **Per width 1.006 / 1.003 /
  1.001 / 1.000 / 1.000**; out of the pool 1 dx 1.004, 1.5 dx 1.007 — the consistent form is right to
  `O((dx/ell)^4)` all the way to the resolution limit on the synthetic fronts.
- *llc*: OLS **0.9806 [0.9698, 0.9942]** (se 0.006; n 26,293; 249 blocks); origin 0.980, inverse
  1.017, GM 0.999, orthogonal 0.999, ratio 0.950, corr 0.982. Inside the ±0.05 gate, but the CI
  excludes 1: the remaining −2% is the real velocity's grid-scale structure (it vanishes when the
  velocity is low-passed and does not move when `b` is), part of it the cubic interpolation of
  the velocity at the midpoint (`vel_order = 5`: 0.987) and part the evaluation of `F` at the
  arrival cell rather than along the trajectory (the spatial-midpoint variant over-corrects to
  1.024). **This 0.981 is the baseline Figure 2 draws** (`res['slope']`), with its CI.

**Operator changes (`operators.py`, `semilag.py`).**
- `operators.frontogenesis(b, U, V, grid_ds, grid, form='discrete')`: the discretely consistent
  `F = −sum_k (L_k b)[L_k, u·grad] b` is the **default**; `form='chain'` is the unchanged
  repo-equivalent path (**still bit-for-bit `frontogenesis_tendency`**; the criterion-7 oracle test
  now calls `form='chain'` explicitly — that is the "keep the oracle against the old path" option);
  `form='discrete_o2'` is the rejected 2nd-order variant, kept for the record. New helpers
  `centred_model_velocity(U, V, grid)` (one implementation of the centred velocity, now shared by
  `semilag.centre_velocities` and the consistent `F`, so the departure map and `F` cannot see
  different velocities) and `model_basis_gradient(b_x, b_y, grid_ds)` (the exact inverse rotation;
  `F` is an invariant and is formed along the stencil axes). `F.attrs['form']` records which.
  On the real hour the consistent form regresses on the chain form at **0.790** over the top-decile
  cells (logged by the oracle test) — the size of the correction M3 would otherwise have carried
  as a baseline.
- `semilag.departure_index(..., vel_order=3)` (was 1); `measured_DGDt(..., vel_order=3)` passes it
  through. Displacement statistics unchanged (median 0.364, p99 1.248, max 2.10 on the ocean;
  bilinear vs cubic max 0.032 cell).
- NaN reach of the default `F` (measured on the diagonal-coast geometry and the real tile, both
  forms, `L = 0/2/4/8`): the **chessboard** minimum of a finite `F` is unchanged (`L/2 + 2`), the
  **taxicab** minimum is one more (`L/2 + 4` vs `L/2 + 3`), and `F` is finite only beyond
  chessboard `L/2 + 3` (chain `L/2 + 2`). Consequences: NaN in `mask_halo & mask_edge` at `L = 8`
  **1,627** cells (0.48%; chain 249), 0 at `L ≤ 4` on the tile; **finite on all 262,925
  `mask_analysis` cells at every `L`, both forms**; the tile-edge reach at `L = 8` is **exactly 7
  cells = `edge_cells`** (no slack left; `edge_cells` must not shrink).
- Coarse-grained closure (task 4's test, unchanged tolerances) **improved** with the consistent
  `F`: `Gbar` residual/measured with the term, shear `dx = 3600/1800/900`: 0.117 / 0.052 / 0.037
  (was 0.20 / 0.085 / 0.041; `dt = 900`: 0.012, was 0.022); divergent 0.194 / 0.089 / 0.088 (was
  0.33 / 0.12 / 0.090; `dt = 900`: 0.023, was 0.032) — the chain-rule mismatch task 4 said the
  closure "measures together with the discretisation" is now gone from it at the coarse levels.

**Tests — `pytest dev/frontogenesis/py/tests`: 70 passed, 0 skipped in 33 s** (was 68 + 1 skip):
`test_masking` 17, `test_operators` 21, `test_semilag` 15, `test_coarsegrain` 10, `test_validate` 7
(V1, V2, **V3 strain**, **V3 llc** `needs_grid`, V4, V5, V6). Updated with a stated reason, no
tolerance loosened: `test_factor_of_two_pure_deformation` (×2: `F = aG` is the chain form's
identity → `form='chain'`, plus new assertions that the discrete form is 0.945-0.970 of it at the
4-dx centre and within 0.4% at 16 dx); `test_strain_alignment_definition_and_decomposition` and
`test_strain_rotation_sign_on_the_real_face` (the Jacobian decomposition is of the chain form →
`form='chain'`); `test_regression_vs_repo_frontogenesis_tendency` (oracle on `form='chain'`, still
max |dF| = 0 on 262,925 cells; plus the default form finite on the mask and its 0.79 ratio);
`test_lowpass_halo_consequence_diagonal_coast` and `test_lowpass_nan_at_the_coast_real_tile`
(both forms, the reach numbers above, taxicab added); `test_nan_propagates_at_a_synthetic_coast`
(the velocity support is cubic now); `test_two_hours_smoke` (compares `vel_order = 1` to the new
default). `test_validate.py` V3: strain asserts the gate, the CI inside ±0.05, every pooled width
within 2% and the out-of-pool widths within 3% for the discrete form, the chain form monotone in
width and < 0.95 at 2 dx / < 0.85 at 1 dx, the 2nd-order variant > 1.02 at 2 dx, estimators within
3% of each other; llc asserts the gate and CI, n 26,293 / 262,925, the first attempt < 0.85, cubic >
bilinear, the low-passed-velocity run within 2% of 1, departure-vs-Jacobian > 0.98 and
Jacobian-vs-flux-form in 0.8-0.9 (the blindness, pinned).

**V3 → `figs/V3_discrete_null.png`** (200 dpi, 4000 x 2500, 0.99 MB; `git status` shows it,
`git check-ignore -v` → `figs/.gitignore:3:!*.png`): (a, b) strain variant, first attempt and the
gate, 2-D histograms of measured vs `2F` on front pixels with the 1:1 line, the OLS fit and CI, the
orthogonal / inverse / GM slopes; (c, d) the same for the LLC variant; (e) the slope vs front width
per form with the `1 − (2/3)(dx/ell)²` prediction, the out-of-pool widths, the 4th-order-everywhere
alternative and the LLC slopes; (f) every change tried as bars against the gate band, with the
departure / Jacobian / flux-form strain ratios in the title. `test_discrete_null(png=True)` computes
the other variant too so the figure always carries both (the LLC half is skipped without the M0
stores).

**Contradictions / things to flag (explicitly).**
1. **The expected 0.80-0.85x Jacobian attenuation does not and cannot appear in this null**
   (prompt 2 task 6 and criterion 3, coding §4.9 / §6 M1, planning §6 test 3 all say to expect it).
   The Jacobian trace *is* `D_h u_c`, the `(1,2,1)/4` mean of the flux-form divergence, and the
   semi-Lagrangian departure is built from `u_c` — a semi-Lagrangian truth can only ever confirm
   the centred description against itself. Whether the model's flux-form advection sharpens fronts
   with the h-scale strain the `2h` tracer stencil cannot see is **untested and open for M3**; the
   test that would settle it is a finite-volume (flux-form) advection step as the truth. Noted in
   planning §6 (marked) and coding §4.9 (marked). If that strain does act, M3's slope would be biased
   *high*, not low.
2. **Criterion 3's "order unity at a 4 dx feature"**: the chain-rule violation is `(2/3)(dx/ell)²` —
   4% at 4 dx, 17% at 2 dx, order unity only at 1 dx (0.82). Task 3 flagged the same; wording only.
3. **Criterion 7 as worded ("unfiltered `operators.frontogenesis` agrees with `frontogenesis_tendency`
   to round-off") is now true of `form='chain'`, not of the default.** The oracle test says so; the
   science product is the consistent form. Coding §4.3 corrected and marked.
4. **Coding §4.4 / task 3's "bilinear velocity is fine"** was true of the displacement, false of its
   gradient. `vel_order = 3` is now the default (coding §4.4 marked).
5. **Planning §11's ratio estimator** `sum y/sum x` is ill-conditioned whenever the front pool has
   both signs of `2F` (0.85 on the synthetic pool against 1.004 for every other estimator); M3
   should quote it separately for `X > 0` and `X < 0` as §11 already asks for the binned means.
6. **The llc CI excludes 1** (0.970-0.994): the gate (±0.05) passes but the baseline is
   significantly 2% below 1, and that 2% is the real velocity's grid-scale structure, not the fronts.
   Figure 2 should draw 0.981 with its band, not "1".
7. **V4's error bar does not enter this null**: with our own step as truth the interpolation
   bias cancels identically (integer and fractional shifts alike), so V3 cannot confirm V4's 0.28-1.0%
   of `G`/h; it applies to the real comparison only. Not a contradiction; a limit of the design.
8. Coding §4.9's signature `test_discrete_null(velocities='strain', png=True)` grew keyword
   arguments (`order, forms, widths, angles, n, a, margin, n_boot, diag_widths, grid_ds, changes`),
   all defaulted; the contract call works as written. The docs' "`G_mid` top decile" front rule was
   only implicit (planning §11 "a stated percentile"); p90 is now stated in `validate.py`.
9. `validate.py` is 863 lines (§1.3's ~400 exceeded further; V3 alone is ~350 with the
   estimators and the changes-tried bookkeeping); `validate_figs.py` 600, `operators.py` 552.
   Flagged, not split.
10. M2 (prompt 3) and M3 (prompt 4) must call `operators.frontogenesis` with the default and
    `semilag` with the defaults; anything pinned to "`F` = `frontogenesis_tendency`" in those
    prompts now means `form='chain'`. Not edited here.

Files: created `figs/V3_discrete_null.png`; modified `py/operators.py` (`frontogenesis(form=)`,
`_frontogenesis_discrete`, `centred_model_velocity`, `model_basis_gradient`, `_shift`, docstrings),
`py/semilag.py` (`centre_velocities` delegates; `departure_index(vel_order=3)`;
`measured_DGDt(vel_order=)`; `ng` import dropped), `py/synthetic.py` (`null_strain_case`,
`NULL_MODES`), `py/validate.py` (V3: constants, `slope_estimators`, `block_bootstrap_ols`, `two_F`,
`null_step`, `front_pixels`, `_null_strain`, `_fourth_order_everywhere`, `_discrete_null`,
`test_discrete_null`), `py/validate_figs.py` (`fig_V3`), `py/tests/test_operators.py`,
`py/tests/test_semilag.py`, `py/tests/test_validate.py`, `frontogenesis_coding.md` (§4.3, §4.4, §4.9
marked), `frontogenesis_planning.md` (§6 test 3 marked), `claude_prompts/frontogenesis_prompt_2.md`
(status paragraph) and this log.

### 2026-09-29 — Execution prompt 2, task 7a: test_nan_finding.py (Fable)

**Scope.** The `test_nan_finding.py` piece of task 7 of `frontogenesis_prompt_2.md` only:
`fronts_from_gradb2` under finding config D on NaN land, synthetic and real. The M1 audit, the
Status paragraph, criteria and coding §6 are the audit agent's; `test_validate.py` untouched; V3b is
a concurrent agent. Nothing outside `dev/frontogenesis/` touched — in particular **the `fronts`
package is unmodified**; the three failures below are recorded as `xfail(strict=True)` with the
exact mode, and the fixes are described here, not applied. No data pulled; nothing committed.
Every python/pytest command ran under `timeout 300`; the full suite takes 64 s, the new file 34 s.
Env: `~/miniforge3/envs/frontogenesis/bin/python` (3.13), `fronts` editable from this checkout
(`pip show`: editable project location `~/Oceanography/python/fronts`, branch `frontogenesis`),
skimage 0.26.0, scipy 1.18.1, skan 0.13.1, numpy 2.5.3.

**Config D, read from the YAML** (`fronts/finding/configs/finding_config_D.yaml`, `binary:`):
`window 64, threshold 85, thresh_mode 'pool', thin false, sharpen true, despur true, Lspur 10,
dilate false, min_size 7, connectivity 2`. Five of these differ from `fronts_from_gradb2`'s defaults
(40, 90, 'generic', False, False) — the docs were right to say "read it". Callers in `fronts`
(`finding/run.py:52`, `runs/prototypes/finding/explore_hyper.py:131`) pass `**bparam` with
`bparam['n_workers'] = 10` hard-coded; the config alone is not runnable (below).

**How NaN flows through config D** (traced in the code, then measured on 160x200 fields with a
2-px Gaussian ridge and a 40-column NaN land block):

1. `pyboa.front_thresh` — `scipy.ndimage.vectorized_filter` (or `generic_filter`, or the pool's
   row chunks) with `np.nanpercentile` over the 64x64 window and `cval=nan` outside the array. The
   local threshold is the percentile of the window's *finite* cells; a NaN cell compares
   `nan > q = False`, so no NaN cell is ever flagged; a window entirely on land gives `q = nan`
   (numpy `RuntimeWarning: All-NaN slice` in 'vectorized' mode — 1,440 of them on the 160x200
   field; none from 'generic', which filters NaN explicitly; none visible from 'pool', whose
   children swallow them) and flags nothing. **NaN-safe.** The three modes agree bit for bit on the
   NaN field (`generic == vectorized == pool`). No coastal bias: on a flat noise field with a NaN
   coast the 85th-percentile window flags 0.150 of the 32 coastal columns vs 0.149 of the interior.
2. `sharpen.global_sharpen_pq` — a priority-queue thinning keyed on `gradb2`, padded with 0; it
   only reads `gradb2` on foreground pixels (never NaN) and ends with `morphology.thin`.
   **NaN-safe.** Note it thins, so config D's fronts are 1 px wide even with `thin: false`.
3. `pyboa.cropping` — `spur` (LUT, `padding=1`), `remove_small_objects(min_size=7)` (skimage
   0.26 deprecates `min_size` for `max_size`, FutureWarning every call), then
   **`remove_small_holes()` with its default `area_threshold=64`**: a background region < 64 px
   enclosed by front pixels is filled *without looking at `gradb2`*. A NaN island < 64 px that a
   front wraps around is filled, and the final thin draws the skeleton across it. **Not NaN-safe.**
4. the final `morphology.thin` — binary, NaN-blind.
5. `despur.prune_short_spurs` — `skeletonize`, then `skan.Skeleton(skeleton)`; on an **empty**
   skeleton skan raises `ValueError: index pointer size 0 should be 1`. **Breaks** whenever nothing
   survives cropping (an all-NaN field; an all-land or featureless tile). `despur=False` returns
   the all-False mask correctly, so steps 1-4 are fine.

**What works.** A NaN land block: config D runs (given `n_workers`), returns `(nj, ni)` `bool`,
puts **0 front pixels on NaN**, and the fronts on every column >= 10 cells from the coast are
**pixel-for-pixel equal** to the land-free field's (0 differing pixels; 0 in the 10 coastal columns
too; the tolerance in the test is <= 2). A front running into the coast: 0 on NaN, no spurious
front along the coast, the skeleton stops at the *second* ocean column (`i = 41` for a coast at
40 — a 1-px endpoint retraction, the same as at an open array edge), away from the coast equal to
the land-free mask (0 pixels differ; 1 in `40 <= i < 50`). A NaN island beside (not enclosed by) a
front: 0 on NaN.

**What breaks** (each an `xfail(strict=True)` in the file, so a `fronts` fix shows up as XPASS):
- **`fronts_from_gradb2(G, **config_D)` raises `TypeError`** before touching NaN: `thresh_mode
  'pool'` with the default `n_workers=None` reaches `np.array_split(np.arange(nrows), None)`
  (`pyboa.py:797`). And 'pool' is a `ProcessPoolExecutor` (spawn on macOS): a caller script
  without an `if __name__ == '__main__'` guard dies with `BrokenProcessPool` (my first probe did;
  pytest is fine). The tests pass `n_workers=2`.
- **Front pixels on NaN via `remove_small_holes`**: a 6x5 NaN island on the ridge → **6 front
  pixels on NaN** (the skeleton runs straight through the island).
- **All-NaN field → `ValueError` in despur** (skan on an empty skeleton).

**Workarounds** (caller side, tested): filling NaN before finding is **not safe** — 0-fill and
ocean-median-fill both change the fronts away from land (15 pixels differ at >= 10 cells from the
coast, 16 in the coastal columns; the filled cells enter the percentile window and move the local
threshold). The safe recipe is: pass NaN as-is, pass `n_workers` explicitly, keep `despur` off on a
possibly-empty field, and **`fronts &= isfinite(gradb2)` afterwards** — that removes exactly the
island fill and changes nothing else (the mask is already False on every other NaN cell).

**Real tile** (`needs_grid`; hour 0 of `tile330_raw_20120702T00_2h.zarr`, the repo's `grad_b2` =
`calculate_grad_squared_tracer`, the finding field per the operators contract; NaN on all 161,523
land cells plus 2,174 ocean cells at taxicab 1 from land, finite 354,703): config D with
`n_workers=4` → **11,836 front pixels in 7.1-7.3 s** (the 'vectorized' threshold alone is 13 s
single-process; the whole M2 series at 72 h is ~9 min), **0 on NaN**, **878 (7.4%) inside the
7-cell halo** (`mask_ocean & ~mask_halo`), 7,885 on the analysis mask. Coastal artefact: the front
density is 0.167 at the first finite cell from NaN, 0.093 / 0.058 / 0.046 / 0.034 at 2 / 3 / 4 / 5,
0.034 at 7, 0.032 at >= 8 — i.e. a coastal excess that decays by ~5 cells, entirely inside the
halo. It comes from the *input*: `grad_b2`'s median at the first finite cell is 1.4e-13 vs 2.1e-15
in the interior (~70x); the finder itself shows no coastal bias on a flat field (above). Whether
that 70x is the coastal-upwelling front or the stencil next to the NaN rim is not interpreted here
(M1 produces no science); the halo removes it either way.

**Recommended fixes for `fronts` (described, not applied).**
1. `pyboa.front_thresh` 'pool': default `n_workers` to `os.cpu_count()` (or fall back to
   'vectorized') when `None`; document the `__main__` guard.
2. `pyboa.cropping`: after `remove_small_holes`, re-apply the caller's finite mask — simplest is
   `fronts_from_gradb2` doing `res_frnt_crop &= np.isfinite(gradb2)` after cropping (and after
   dilation), or passing `remove_small_holes` a hole mask that excludes NaN cells.
3. `despur.prune_short_spurs`: `if not skeleton.any(): return skeleton` before `Skeleton()`.
4. `pyboa.cropping`: `remove_small_objects(max_size=min_size - 1)` (skimage 0.26 deprecation; note
   the off-by-one in the new semantics) — cosmetic.
5. 'vectorized' mode: wrap `nanpercentile` in `warnings.catch_warnings` (or use 'generic'), so an
   all-land window does not emit 1,000s of RuntimeWarnings.

**Tests.** `py/tests/test_nan_finding.py`: 14 tests — 11 pass, 3 xfail (strict). Offline: config D
values vs defaults; the `n_workers` TypeError (xfail); land block (shape/dtype/no-NaN; equality away
from land); threshold modes agree + NaN-safe (+ the All-NaN warning); no coastal bias in the
percentile window; front into the coast; enclosed island (xfail); island beside a front; all-NaN
(xfail) and all-NaN with `despur=False`; fill workarounds unsafe; output-masking workaround safe.
`needs_grid`: hour-0 `grad_b2` under config D with the numbers above asserted as ranges. Full
suite: **81 passed, 3 xfailed** in 64 s (was 70 passed).

**Contradictions / notes on the docs.**
1. Coding §2.5 says `build_v5.py` step 1 → `build.tile_find` → `fronts.preproc.gradb2.
   generate_tile_gradb2`. **Neither `generate_tile_gradb2` nor `tile_find` exists in this
   checkout** (`grep` over `fronts/`, all `*.py`); `build_v5.py` step 1 calls
   `generate_for_channels` / `export_channels`, and `fronts/preproc/gradb2.py` has only
   `generate_gradb2`. Planning-prompt survey notes cite `gradb2.py:89-98` for it — stale against
   the current `frontogenesis` branch. The finding entry point for M4 is `fronts_from_gradb2`
   directly (as `finding/run.py` does); the §2.5 line should be corrected by the audit.
2. Planning §10 / the prompt-1 survey call the finding chain "already NaN-safe (`nanpercentile`,
   `cval=np.nan`)". True for the threshold step only; `remove_small_holes` and the empty-skeleton
   despur are not, and config D as written does not run without `n_workers`.
3. Prompt 2 criterion 6 / coding §5 name the test; both satisfied. Coding §5's "all offline
   except one `network` test" holds — the real-tile test is `needs_grid`, on disk.
4. `fronts_from_gradb2`'s docstring: `threshold` is documented as "used in the cropping function
   to determine size threshold"; it is the percentile passed to `front_thresh`. `thin: false` in
   config D does not mean unthinned output — `sharpen` thins.
5. `pyboa.front_thresh`'s `ValueError` message lists 'generic', 'vectorized', 'dask' — 'pool' is
   accepted but unlisted.

Files: created `py/tests/test_nan_finding.py`; this log. Nothing else.

### 2026-09-29 — Execution prompt 2, task 6b: V3b, the finite-volume null (Fable)

**Scope.** Task 6b of `frontogenesis_prompt_2.md` (M1-Q2, option (a): V3b in M1, before any science,
as a **recorded bias, not a gate**): `py/fvadvect.py` (new), `validate.test_fv_null` (V3b),
`validate_figs.fig_V3b`, `figs/V3b_fv_null.png`, the V3b tests in `tests/test_validate.py`, the task
entry `### 6b` in prompt 2. No operator module touched (`operators.py`, `semilag.py`, `masking.py`,
`coarsegrain.py` unchanged); `validate.py` gained an injectable truth (`null_step(advect=)`), the llc
input loader `_llc_inputs` / `_fit_llc` factored out of `_discrete_null` (V3 re-run: 1.0044 / 0.9806,
bit-identical), and `null_step` now also returns `G_tp1`. Nothing pulled, nothing committed, nothing
outside `dev/frontogenesis/`; every command under `timeout 300` (longest 49 s). Grids ≤ 128² synthetic;
the final llc numbers use the full `mask_analysis` (262,925 cells, n front 26,293).

**Why V3b.** V3's truth is our own semi-Lagrangian step, so both sides see the same centred velocity
`u_c` and the null is blind to the 0.80-0.85x attenuation of the interpolated Jacobian relative to the
flux-form strain (M0 task 5; 0.853 on `mask_analysis`, M1 task 2; 0.856 on the V3b front pixels).
LLC4320 advects `b` in flux form with OS7MP. V3b replaces the truth by a flux-form finite-volume step on
the C-grid with V3's midpoint velocity; everything else (measured `semilag.measured_DGDt`, predicted
`2 * operators.frontogenesis` at the midpoint, `form='discrete'` and `'chain'`, the pre-declared
`G_mid >= p90` front pixels, OLS-with-intercept estimator, 32-cell block bootstrap with 1000 draws) is V3's
code path, unchanged.

**The truth (`fvadvect.fv_advect`), MITgcm conventions.** Face transports `uTrans = U dyG hFacW`,
`vTrans = V dxG hFacS` (`drF` cancels in one layer; the OSN NaN velocity faces are `hFacW = 0` cell for
cell, set to zero transport); per directional sweep `db/dt = -[Delta(uTrans b_f) - b Delta(uTrans)] /
(rA hFacC)`; sweeps X then Y on even sub-steps, Y then X on odd (as the model alternates them), forward
in time, the second sweep on the field the first updated; `dt_sub = 100 s` (36 sub-steps, `c <= 0.06` on
the tile; the model's 25 s is an option). Land filled with its nearest ocean value for the reconstruction
only (its faces carry zero transport; NaN again on output); the tile edge padded zero-gradient. Schemes:
- `centred`: `b_f = (b_{i-1} + b_i)/2`, integrated unsplit with SSP-RK3 (forward-in-time centred is
  anti-diffusive at `O(c^2)`). No dissipation: **the pure C-grid stencil effect**, the reference.
- `dst3`: MITgcm's third-order direct-space-time scheme (`tempAdvScheme = 30`, no limiter):
  `b_f = b_{i-1} + d0 (b_i - b_{i-1}) + d1 (b_{i-1} - b_{i-2})`, `d0 = (2-c)(1-c)/6`, `d1 = (1-c^2)/6`.
  The cross-check with a *larger* implicit diffusion.
- `os7`: the unlimited seventh-order one-step scheme of Daru & Tenaud (2004): the exact sub-step time
  average of the degree-6 reconstruction through `b_{i-4..i+2}` (four upwind, three downwind), a
  degree-6 polynomial in `c` derived from the primitive function (`os7_weight_matrix`; checked: `c -> 0`
  gives `(-3, 25, -101, 319, 214, -38, 4)/420`, `c = 1` the exact shift, weights sum to 1).
- `os7mp`: `os7` + the Suresh & Huynh (1997) monotonicity-preserving limiter (`alpha = 4`, the `d^M4`
  curvature bounds, `UL/LC/MD`), the construction OS7MP and `gad_os7mp_adv_x.F` follow. **How it differs
  from MITgcm's OS7MP:** the model's limiter bounds carry the Courant number (a second-order-in-`c`
  difference in where the limiter engages, at the model's `c <= 0.02` / our `<= 0.06`), and the model
  reduces the stencil order with `maskW` next to land where we fill. The unlimited scheme is the same.
  So: an honest OS7MP-like scheme, exact in the smooth limit, approximate in the limiter's bounds.
Checks: a uniform tracer stays uniform under the divergent strain-case flow for every scheme (max
`|b - 3| = 0`); translation of a Gaussian by one cell per hour converges as `sigma^-2.8` (centred),
`^-3.6` (dst3), `^-7` (os7/os7mp); the one-step schemes are exact at `c = 1` (1e-16).

**Divergence treatment (decision).** The truth is `d_t b + div(u b) = b div u`, the advective form
built from flux differences — `D b/Dt = 0` for the tracer. Justification: (i) the null is about the
kinematic identity `F = (1/2) DG/Dt`, which is what a surface tracer obeys when only its horizontal
kinematics are modelled; (ii) it is exactly MITgcm's default multi-dimensional branch
(`GAD_MULTIDIM_COMPRESSIBLE` undefined): each 1-D sweep subtracts `tracer * (uTrans(i+1) - uTrans(i))`;
(iii) the model's top-cell budget under the linear free surface (M0 task 3, from the source): the surface
transport is zero and the vertical sweep contributes `-w_base (b_base - b)/drF` with `w_base = W(0) +
drF delta` (`W(0) = dEta/dt`), so the *horizontal* part of the model's top-cell tendency is precisely this
advective form and the vertical part is M3's separately measured top-cell term (`vertical.py`), not part
of this null. The conservative alternative `-div(u b)` alone (`form='conservative'`, kept to show its
size) adds `-b delta` — 4% of `b` per hour at `delta ~ 1e-5` — and gives a nonsensical slope of **13.3**
(discrete) / 9.0 (chain) on the real hour: a spurious `-2 delta G`-type signal far larger than the strain's.

**Slopes (OLS of measured `DG/Dt` on `2F`, front pixels, 32-cell block bootstrap 2.5-97.5%).**

| truth | strain, chain | strain, discrete | llc, chain | llc, discrete |
|---|---|---|---|---|
| semi-Lagrangian (= V3) | 0.951 [0.944, 0.964] | 1.004 [0.995, 1.017] | 0.791 [0.753, 0.815] | 0.981 [0.970, 0.994] |
| FV centred (stencil only) | 0.932 [0.895, 0.996] | 0.983 [0.951, 1.037] | 0.811 [0.768, 0.849] | 0.974 [0.930, 1.026] |
| FV OS7 (unlimited) | 0.931 [0.920, 0.948] | 0.985 [0.977, 0.996] | 0.803 [0.778, 0.823] | 0.983 [0.958, 1.018] |
| **FV OS7MP-like** | 0.931 [0.920, 0.948] | **0.985 [0.977, 0.996]** | 0.790 [0.753, 0.814] | **0.975 [0.954, 1.003]** |
| FV DST3 (3rd order) | 0.909 [0.889, 0.932] | 0.962 [0.947, 0.975] | 0.666 [0.622, 0.725] | 0.845 [0.784, 0.918] |

Correlations (discrete): strain 0.983 / 0.833 / 0.980 / 0.980 / 0.960; llc 0.982 / 0.717 / 0.932 / 0.918 /
0.856 — the centred truth is dispersive on the real field (grid-scale wiggles in `b_tp1`), which widens
its CI without moving its slope; os7 and os7mp coincide on the smooth synthetic fronts (the limiter never
engages on a monotone tanh). n front 18,816 (strain, 15 pooled cases) / 26,293 (llc).

**Slope vs front width (strain, per width 2 / 3 / 4 / 6 / 8 dx; out of the pool 1 / 1.5 dx).**
- discrete: V3 1.006 / 1.003 / 1.001 / 1.000 / 1.000 (1.004 / 1.007); centred 0.982 / 0.983 / 0.986 /
  0.991 / 0.994 (0.998 / 0.988); **os7(mp) 0.979 / 0.992 / 0.995 / 0.997 / 0.998 (0.893 / 0.958)**; dst3
  0.951 / 0.981 / 0.991 / 0.996 / 0.997 (0.807 / 0.906).
- chain: V3 0.938 / 0.970 / 0.982 / 0.991 / 0.995; centred 0.917 / 0.950 / 0.967 / 0.982 / 0.989; os7(mp)
  0.911 / 0.960 / 0.976 / 0.988 / 0.993; dst3 0.884 / 0.949 / 0.972 / 0.987 / 0.992.
So a model-like advection sharpens a front *less* than the discrete `F` says by ~2% at 2 dx, 0.8% at 3,
0.5% at 4, 0.2% at 8 dx, and by 4% at 1.5 dx and 11% at 1 dx: the scheme's own truncation on fronts it
does not resolve (the seventh-order error is not small at `k dx ~ 1`). The chain form adds its
`(2/3)(dx/ell)^2` violation on top. The rule for M3: the numerics-of-advection shortfall is width-dependent
and is what a slope < 1 on the sharpest fronts *must* be corrected for before any diffusion is inferred.

**Does the 0.85 attenuation appear? No.** On the real hour the FV truths give 0.974 (centred), 0.983
(os7), 0.975 (os7mp) against V3's 0.981 — a shift of −0.006 [−0.027, +0.022] for the OS7MP-like truth,
not the +0.17 (`1/0.85`) a fully acting attenuation would give (dotted red line in V3b (d, e)). The Jacobian
trace regresses on the flux-form divergence at **0.856** on these very front pixels, so the strain the
tracer is advected with *is* 15% larger at the h scale than the strain `F` sees — but the front strength
we measure is `|L b|^2` with `L` the same `2 dx` stencil, and `d/dt (L b) = -L(u . grad b)` averages the
h-scale strain over the same `(1, 2, 1)/4` footprint before it reaches `G`. The h-scale strain of the
face velocities sharpens sub-stencil structure that the resolved `G` does not see; the interpolated
Jacobian's attenuation is therefore *not* a bias of the pipeline but the consistent description of the
resolved front strength. (An interpretation supported by the numbers, not a theorem: the chain form
moves the same way, 0.791 → 0.790.)

**Stencil vs implicit diffusion (discrete `F`; llc, then strain).**
- **C-grid stencil** = centred slope − 1: **−0.026** (llc; CI half-width ±0.05), −0.017 (strain).
- **Implicit diffusion** = scheme − centred, on the scheme's own front pixels (exact: one `b_t`, one
  departure, so `measured(s) − measured(centred) = [G(b_tp1^s) − G(b_tp1^centred)]/dt`): os7 **+0.009**,
  os7mp **+0.001**, dst3 **−0.129** (llc); os7(mp) +0.002, dst3 −0.021 (strain). As a `DG/Dt` term its OLS
  slope on `2F` is +0.008 / −0.001 / −0.145 (llc) and its rms is 0.81 / 0.91 / 0.87 of rms `2F` — the large
  rms against the centred reference is the centred scheme's dispersive noise, not diffusion.
- **The MP limiter alone** = os7mp − os7 (same reconstruction): slope shift **−0.008** (discrete) /
  −0.013 (chain) on llc; its `DG/Dt` term regresses on `2F` at −0.009, rms 0.22 of `2F`; `dt/G` median
  **0.0%** of `G` per hour, mean −0.24%, **p10 −1.9%**, negative on 48% of front pixels — the limiter is a
  tail effect on the sharpest fronts, as planning §2.3 says (first-order-upwind-like where it engages).
- The seventh-order dissipation is below the noise of this one-hour null (`+0.009 ± 0.03`); the
  third-order DST3 cross-check shows what a genuinely diffusive scheme would do: −0.13 in the slope,
  `dt/G` median −1.5% per hour, p10 −11%, negative on 68% of front pixels, `2 kappa k^2 ~ 4e-6 s^-1` at
  the front scale — the order of planning §2.3's estimate for `4 dx`.

**Sensitivities (llc, os7mp × discrete unless stated).** `dt_sub` 25 / 100 / 300 s: 0.9753 / 0.9747 /
0.9733 (centred RK3 25 / 300 s: 0.9741 / 0.9741). Land fill nearest vs ocean-mean: **identical** on every
`mask_analysis` cell (max |Δmeasured| = 0). Tile-edge crop test (rerun on the tile cropped 16 cells,
`max |Δb| / std(b)` vs cells from the crop edge): 0.42 / 0.076 / 0.043 / 0.011 / 0.0042 / 0.0018 / 6e-4 /
**1.3e-4 at 7 cells** (= `edge_cells`) / 7e-5 / 8e-6 / 6e-6 / 1e-6 at 11 — the zero-gradient pad's reach
decays 3-5x per cell and is below 1e-4 of `std(b)` at the edge margin.

**Headline recorded bias for M3.** `bias = 0.975 [0.954, 1.003]` (llc, OS7MP-like truth, `form='discrete'`;
strain 0.985 [0.977, 0.996]): the OLS slope our pipeline returns when the tracer is advected as the model
advects it and nothing else happens. Relative to Figure 2's baseline of 0.981 (M1-Q4, the V3 slope) that is
**−0.006 [−0.027, +0.022]**: no correction, and in particular *no upward correction* for the Jacobian
attenuation. Recommended use in M3: (1) keep the baseline at 0.981 with its V3 band, and widen the
*systematic* band to the V3b CI, 0.954-1.003 (report M3's slope against 0.981, quote the V3b interval as
the pipeline's model-advection systematic); (2) report the slope per front width and subtract the
width-dependent advection-numerics shortfall above (−2% at 2 dx, −4% at 1.5 dx, −11% at 1 dx, discrete
form) before attributing anything on the sharpest fronts to diffusion; (3) for `form='chain'` the same
truths give 0.79 [0.75, 0.81] — the chain form is a different baseline, as M1-Q1 already treats it; (4) the
number to quote if one is needed: **0.975 ± 0.025**. Not a gate; `res['bias']` carries it.

**Tests — `pytest dev/frontogenesis/py/tests`: 84 passed, 3 xfailed in 91 s** (was 81 + 3 xfailed with
`test_nan_finding.py`; 70 before it): `test_validate.py` gains `test_V3b_fv_step_basics` (the OS7 weight
limits, uniform-tracer invariance under a divergent flow for every scheme, the conservative form's
`-b delta`, exactness of the one-step schemes at `c = 1`),
`test_V3b_fv_null_strain` (offline: runs, finite, `scheme='semilag'` reproduces V3's 1.0044 to 1e-12,
centred FV at 8 dx within 2% of 1, os7 = os7mp on smooth fronts to 1e-4 (4e-6 measured: the limiter
touches a few flank cells of the sheared cases), os7(mp) per width monotone and within
3% of 1 in the pool, dst3 below os7 at 2 dx, the OS7 weight limits, the uniform-tracer invariance and the
conservative form's spurious `-b delta`) and `test_V3b_fv_null_llc` (`needs_grid`: runs, finite,
`semilag` reproduces V3's 0.9806 / 0.7914, n front 26,293 on 262,925, the OS7MP-like slope finite with a
CI that contains it, the limiter attribution present). **No assertion on the headline number** (a
recorded bias).

**V3b → `figs/V3b_fv_null.png`** (200 dpi, 4000 x 2500, 1.0 MB; `git status` shows it,
`git check-ignore -v` → `figs/.gitignore:3:!*.png`): (a) strain variant with the OS7MP-like truth;
(b, c) the LLC variant with the centred and the OS7MP-like truth (2-D histograms of measured vs `2F` on
front pixels, OLS with CI, the other estimators); (d) slope vs front width per truth, discrete solid /
chain dashed, out-of-pool widths open, the `1/0.85` line; (e) every truth × form × variant with its CI
and the recorded-bias band; (f) the stencil vs implicit-diffusion attribution with the limiter's numbers.

**Contradictions / things to flag.**
1. **The 0.80-0.85x Jacobian attenuation is not a bias of the slope** (prompt 2 criterion 3, coding §4.9,
   planning §6 test 3, task 6's contradiction 1 and M1-Q2 all expected it to bias M3 *high* if it acted).
   V3b measures the shift of the OLS slope under a flux-form truth at −0.006 ± 0.025; the attenuation is
   the resolved `G`'s consistent view of the strain. The "open systematic for M3" of task 6 is closed by
   this entry as a recorded bias; the docs' wording is left for the audit.
2. Planning §2.3's implicit-diffusion estimate for OS7MP (`kappa_num` 8-27 m² s⁻¹ at 4 dx, e-folding of
   `G` in 7-23 h) is **not** seen in one hour on the resolved `G` of the real field with the seventh-order
   scheme (+0.009 ± 0.03 in the slope; limiter tail p10 −1.9%/h); it *is* seen with the third-order DST3
   (−0.13). §2.3's numbers are for the unlimited kernel's Fourier symbol at the grid scale and are not
   contradicted — they act on scales the `2 dx` `G` stencil does not resolve — but as a caveat on the
   *resolved* headline slope they are an over-estimate by an order of magnitude for OS7MP.
3. The OS7MP here is OS7 + the Suresh-Huynh MP limiter, not a transcription of `gad_os7mp_adv_x.F`
   (its cfl-dependent limiter bounds and land masking differ; stated in the module docstring). The
   unlimited seventh-order kernel is the same; the limiter's whole effect is −0.008 in the slope, so the
   approximation cannot move the headline by more than that order.
4. `validate.py` is now ~1,020 lines (§1.3's ~400 exceeded further; V3b adds ~150), `validate_figs.py`
   ~700, `fvadvect.py` 330. Flagged, not split.
5. The centred FV truth's correlation on the real field (0.72) is low enough that its CI (±0.05) is the
   widest of the table; the stencil-vs-diffusion split is quoted with that width.

Files: created `py/fvadvect.py`, `figs/V3b_fv_null.png`; modified `py/validate.py` (`null_step(advect=)`,
`G_tp1`, `_llc_inputs`, `_fit_llc`, `_null_strain(advect=)`, `FV_*` constants, `_fv_pool_strain`,
`_fv_attribution`, `_fv_null`, `test_fv_null`), `py/validate_figs.py` (`_null_panel(verdict=)`,
`fig_V3b`), `py/tests/test_validate.py` (V3b tests), `claude_prompts/frontogenesis_prompt_2.md` (task
entry `### 6b` only) and this log. Not touched: `operators.py`, `semilag.py`, `masking.py`,
`coarsegrain.py`, `tests/test_nan_finding.py`, the Status paragraph, the criteria, coding §6, planning.

### 2026-09-30 — Execution prompt 2, task 7: Q&A decisions and M1 acceptance audit (Fable)

**Scope.** Task 7 of `frontogenesis_prompt_2.md`, the audit half (task 7a delivered
`test_nan_finding.py`; task 6b delivered V3b): apply JXP's answers to M1-Q1..Q8, fix the layout of
`figs/V3b_fv_null.png`, run the M1 acceptance audit in the style of M0's task-5 audit, and close M1 if
every criterion passes. Task 8 (slides) not started. No operator module touched (`operators.py`,
`semilag.py`, `masking.py`, `coarsegrain.py`, `fvadvect.py`, `validate.py` unchanged); the only code
edit is the layout of `validate_figs.fig_V3b`. No data pulled; nothing committed; nothing outside
`dev/frontogenesis/`; `frontogenesis_prompt_1.md` and `deck/` untouched. Every python/pytest command
under `timeout 300`; the full suite is the longest at 98 s.

**Step 1 — the Q&A decisions applied** (each spot marked "(decided 2026-09-30, M1-Qn)" or
"(corrected 2026-09-30, M1 task 7)"; nothing rewritten around them).
- **M1-Q1 (both forms; discrete primary, chain alongside, the difference a stated systematic).**
  `frontogenesis_prompt_4.md` (M3): new "Both forms of `F`" bullet under *Runs*; acceptance 4
  extended. `frontogenesis_coding.md` §4.3 `frontogenesis` comment (two lines); §6 M3 new "Carried
  from M1" paragraph.
- **M1-Q2 (a) — V3b as the recorded bias, per the 6b log.** `frontogenesis_prompt_4.md`: new
  paragraph "The baseline and its bands" after the stats paragraph (baseline 0.981 [0.970, 0.994];
  systematic band 0.954-1.003 = 0.975 ± 0.025; no upward correction; slope per width with the −2% /
  −4% / −11% shortfall at 2 / 1.5 / 1 dx subtracted first; the ratio estimator split by sign).
  `frontogenesis_coding.md` §4.9 (`test_fv_null` added to the signature block with its numbers;
  "seven PNGs"); §6 M3 "Carried from M1". The docs that predicted the 0.80-0.85 attenuation would
  bias M3 high, each with the short marked note *V3b measured −0.006 ± 0.025, so the attenuation
  does not bias the slope*: prompt 2 criterion 3; coding §4.9 V3 comment; coding §6 M1 acceptance
  3; planning §6 test 3 (a second "Done" note after task 6's); the M1-Q2 text itself is left as the
  record.
- **M1-Q3 (sign final).** `frontogenesis_planning.md` §2.4 note now reads "Corrected 2026-09-29,
  M1 task 2; **final**, decided 2026-09-30, M1-Q3"; `frontogenesis_coding.md` §4.3
  `strain_alignment` comment "sign FINAL". "Provisionally" / "for now" occur only in the log
  entries (the record), not in the docs — nothing to remove.
- **M1-Q4 (Figure 2 baseline at 0.981 with its band).** `frontogenesis_planning.md` §7 Figure 2
  and the V3 bullet; `frontogenesis_coding.md` §4.9 (the "return the fitted slope" sentence) and
  §4.10 (`fig02`); `frontogenesis_prompt_4.md` Figure 2 bullet; `frontogenesis_prompt_6.md` (M5)
  Figure 2 bullet. Each also names V3b's systematic band 0.954-1.003 as the thing drawn beside it.
- **M1-Q5 (8 dx reference width, with the effect stated).** Prompt 2 criterion 1 restated (< 1% at
  `ell = 8 dx`; 0.78 / 1.57 / 3.09 / 4.43 / 7.86% at 8 / 6 / 4 / 3 / 2 dx over 8 h from the
  centred-stencil truncation `G` and `F` share; the semi-Lagrangian step alone < 0.36% at every
  width); `frontogenesis_coding.md` §6 M1 acceptance 1 likewise; `frontogenesis_planning.md` §6
  test 1 a "Passed" note with the same numbers.
- **M1-Q6 (bar 0.28-1.0% of `G`/h at order 3; order 3 default; order 5 as an M3 sensitivity).**
  Prompt 2 criterion 4 note; `frontogenesis_prompt_4.md` "Interpolation order" bullet under *Runs*
  and acceptance 4; `frontogenesis_coding.md` §4.4 (after the departure paragraph), §4.9 V4
  comment, §6 M1 acceptance 4, §6 M3.
- **M1-Q7 (criterion 7 names `form='chain'`).** Prompt 2 criterion 7 reworded;
  `frontogenesis_coding.md` §6 M1 gains criteria 6-7 (the test files; the chain-form oracle), so
  the coding doc's M1 list now matches prompt 2's seven.
- **M1-Q8 (leave all three).** No edit: `data/tile330_masks.nc` stays git-ignored (regenerates in
  < 1 s from the grid store), `validate.py` is not split (1,020 lines against §1.3's ~400, flagged
  in tasks 3-6b and left), the subfilter-term trend with `L` (0.30 / 0.50 / 0.70 of `Fbar` at
  `L = 2 / 4 / 8`, anti-correlated −0.6) is left to M3's filter sweep.
- Also, from task 7a's request to the audit: `frontogenesis_coding.md` §2.5 gains a marked note
  that `build.tile_find` / `generate_tile_gradb2` do not exist at this checkout and that M4's entry
  point is `fronts_from_gradb2` directly. Prompt 5 (M4) still names the old path; not edited (not an
  M3/M5 prompt) — listed under open issues below. Prompt 2's Q&A header records that all eight were
  answered and applied.

**Step 2 — V3b figure layout (`validate_figs.fig_V3b`, layout only).** The 6b agent reported that
panels (e) and (f) had long titles clipped at the right edge and that the legend in (e) covered the
last row's label (`LLC: dst3, discrete`). Fixed: the (e) title and x-label are each wrapped onto two
lines, the (f) title's limiter line onto three, and (e) gets `set_ylim(-0.7, n_rows + 1.3)` with the
legend at `upper left`, so it sits in the empty band above the first row. No number, estimator, colour
or panel changed. Regenerated with `validate.test_fv_null('llc', png=True, schemes=('semilag',
'centred', 'os7', 'os7mp', 'dst3'))` (25 s; the `'semilag'` scheme is not in `FV_SCHEMES`, so it must be
named explicitly to get the V3 reference rows the 6b figure carried — a first regeneration without it
dropped those rows and was redone). Numbers in the regenerated figure vs the 6b log table (llc, chain /
discrete): semilag 0.791 [0.753, 0.815] / 0.981 [0.970, 0.994]; centred 0.811 [0.768, 0.849] / 0.974
[0.930, 1.026]; os7 0.803 [0.778, 0.823] / 0.983 [0.958, 1.018]; os7mp 0.790 [0.753, 0.814] / **0.975
[0.954, 1.003]**; dst3 0.666 [0.622, 0.725] / 0.845 [0.784, 0.918]; stencil effect −0.026; implicit
diffusion +0.009 / +0.001 / −0.129 (os7 / os7mp / dst3); limiter −0.0081, median +0.00% of `G`/h, p10
−1.9%; `bias = 0.975 [0.954, 1.003]` — **all identical to the 6b table**. Inspected: nothing clipped,
nothing overlapped. `figs/V3b_fv_null.png` 1.0 MB, in `git status` (`??`).

**Step 3 — M1 acceptance audit (prompt 2).**

*Tests* (`timeout 300 ~/miniforge3/envs/frontogenesis/bin/python -m pytest dev/frontogenesis/py/tests
-q`): **84 passed, 3 xfailed in 98 s**; `-m "not needs_grid"`: **66 passed, 18 deselected, 3 xfailed in
50 s**. Per file (collected): `test_masking.py` 17, `test_operators.py` 21, `test_semilag.py` 15,
`test_coarsegrain.py` 10, `test_validate.py` 10 (V1, V2 `needs_grid`, V3 strain, V3 llc `needs_grid`,
V3b basics, V3b strain, V3b llc `needs_grid`, V4, V5, V6 `needs_grid`), `test_nan_finding.py` 14 (11
pass + **3 `xfail(strict=True)`**, each documenting a `fronts` bug, task 7a: config D with the default
`n_workers=None` raises `TypeError` in `pyboa.front_thresh` 'pool'; `remove_small_holes` puts front
pixels on an enclosed NaN island; an all-NaN field raises `ValueError` in `despur` via skan on an empty
skeleton). Strict, so a `fronts` fix shows up as XPASS. No skips.

*PNGs.* `git status`: `V1_cartesian_deformation.png`, `V2_native_metric.png`,
`V4_interpolation_bias.png`, `V5_interp_half_cell.png`, `V6_land_halo_tile330.png` are **tracked and
unmodified** (committed by the user in `9be37dc`, so they do not appear as changes — `git ls-files figs/`
lists them); `V3_discrete_null.png` and `V3b_fv_null.png` are **untracked and shown** (`??`).
`git check-ignore -v` on the two untracked ones → `figs/.gitignore:3:!*.png` (the negating rule); none of
the seven is ignored.

| Criterion | Threshold | Verdict | Evidence |
|---|---|---|---|
| 1. V1 Cartesian deformation | `G ∝ exp(2at)` to < 1% at the 8 dx reference width (M1-Q5) | **PASS** | task 5: 0.776% max over 8 chained hours, n 1568 parcels, both orientations bit-identical; the scheme alone < 0.36% at every width; `test_validate.py::test_V1_cartesian_deformation`; `figs/V1_cartesian_deformation.png`. Width effect recorded: 1.57 / 3.09 / 4.43 / 7.86% at 6 / 4 / 3 / 2 dx |
| 2. V2 native-grid metric | analytic gradients to < 1% | **PASS** | task 5: `b_x` max 0.077%, `b_y` max 0.041% on `mask_analysis` (262,925 cells); metric alone 0.012%; swapped components 87%; R = 6370.0 km; `test_V2_native_metric` (`needs_grid`); `figs/V2_native_metric.png` |
| 3. V3 discrete null | slope = 1 ± 0.05 on front pixels, both variants; return the slope | **PASS** | task 6: strain **1.0044 [0.9950, 1.0171]** (n 18,816), llc **0.9806 [0.9698, 0.9942]** (n 26,293), with `form='discrete'` + `vel_order=3` (first attempt 0.950 / 0.758 — the change is the discretisation finding); `test_V3_discrete_null_strain`, `test_V3_discrete_null_llc`; `figs/V3_discrete_null.png`. **Recorded bias V3b** (task 6b): flux-form OS7MP-like truth **0.975 [0.954, 1.003]** llc, 0.985 [0.977, 0.996] strain — −0.006 ± 0.025 from the baseline, so the 0.80-0.85 attenuation does not bias the slope; `figs/V3b_fv_null.png`; three V3b tests, no assertion on the headline |
| 4. V4 interpolation bias | record it | **PASS (recorded)** | task 5: **0.28% of `G`/h** (1.5-cell front, order 3, real-hour displacements) to **1.0%** (1-cell); order 1 2.3%, order 5 0.06%; falls as `sigma_G^-3.8`; `test_V4_interpolation_bias`; `figs/V4_interpolation_bias.png`. Quoted per M1-Q6 as 0.28-1.0% (order 3), order 5 an M3 sensitivity |
| 5. PNGs V1-V6 in `figs/`, in `git status`; V6 shows the edge rim + margin; V5 as Lauren asked | all six (seven with V3b) | **PASS** | listed above; V5 (task 3) annotates −4.94 / −4.99 / −0.54 / −0.10% at the maximum; V6 (task 1) panel (e) is the four-edge finite rim with the `edge_cells = 7` margin |
| 6. Tests pass, incl. `test_nan_finding.py` | six files | **PASS** | 84 passed, 3 strict xfailed (counts per file above); `test_nan_finding.py` exercises `fronts_from_gradb2` under config D on a NaN land block, a front into the coast, an island, an all-NaN field and the real hour-0 tile (11,836 front pixels, 0 on NaN) |
| 7. Regression oracle | `form='chain'` bit-for-bit `frontogenesis_tendency` (M1-Q7) | **PASS** | task 2 / task 6: max \|dF\| = 0.0 on all 352,673 finite cells, both hours; `test_operators.py::test_regression_vs_repo_frontogenesis_tendency` (also pins the default form's 0.79 ratio to the chain form) |

*Discharges vs criteria.* Task 1 (5: V6; 6: `test_masking.py`) — V6 written, 17 tests; ok. Task 2
(6: `test_operators.py`; 7) — 21 tests, oracle bit-for-bit; ok, with criterion 7 reworded to
`form='chain'` after task 6 changed the default. Task 3 (5: V5; 6: `test_semilag.py`) — ok. Task 4 (6:
`test_coarsegrain.py`) — ok. Task 5 (1, 2, 4; 5: V1, V2, V4) — ok, criterion 1 at the reference width
now stated. Task 6 (3; 5: V3) — ok. Task 6b (nothing; recorded under 3) — ok. Task 7a / 7 (6: the
remaining tests; the audit) — ok. Every "Discharges" line is honoured and nothing is claimed twice.

*Open issues carried forward (to M2 / M3 / M4).*
1. **`fronts` bugs (task 7a; for Lauren / JXP, not applied here):** (i) `pyboa.front_thresh` 'pool'
   with `n_workers=None` → `TypeError` at `np.array_split(rows, None)` (`pyboa.py:797`); default it to
   `os.cpu_count()` or fall back to 'vectorized', and document the `__main__` guard the
   `ProcessPoolExecutor` needs; (ii) `pyboa.cropping`'s `remove_small_holes(area_threshold=64)` fills an
   enclosed NaN island and the final thin draws the skeleton across it — re-apply
   `&= np.isfinite(gradb2)` after cropping (and after dilation), or pass a hole mask that excludes NaN;
   (iii) `despur.prune_short_spurs` → skan `ValueError` on an empty skeleton — `if not skeleton.any():
   return skeleton`; (iv) `remove_small_objects(min_size=)` deprecated in skimage 0.26 (use `max_size =
   min_size - 1`); (v) 'vectorized' mode emits thousands of All-NaN `RuntimeWarning`s on land windows.
   The three strict xfails in `test_nan_finding.py` flip to XPASS when (i)-(iii) are fixed.
2. **Coding §2.5 / prompt 5 (M4) `build.tile_find` → `generate_tile_gradb2` path does not exist** at
   this checkout; the entry point is `fronts_from_gradb2` directly, as `finding/run.py` does. §2.5 now
   carries a marked note; **prompt 5 task 1 still names the old path and must be corrected when M4
   is prepared.**
3. **Caller-side safe recipe for NaN finding (M4):** pass NaN as-is (0-fill or median-fill changes
   fronts away from land — 15-16 pixels on the test field), pass `n_workers` explicitly, keep `despur`
   off on a possibly-empty field, and **`fronts &= isfinite(gradb2)`** afterwards (removes exactly the
   island fill, nothing else). Config D values: `window 64, threshold 85, thresh_mode 'pool', sharpen,
   despur, Lspur 10, min_size 7, connectivity 2` — five differ from the function defaults.
4. **M3 requirements from the decisions:** both `form='discrete'` (primary) and `form='chain'`, the
   difference a stated systematic (M1-Q1); `order = 3` default with **order 5 as a sensitivity**
   (M1-Q6); V4's bar 0.28-1.0% of `G`/h quoted with the front width; `tau_delta` passed to
   `subfilter_term` (task 4); the ratio estimator split by the sign of `2F` (task 6).
5. **The V3b systematic band (M1-Q2):** baseline 0.981 with its V3 band [0.970, 0.994]; model-advection
   systematic **0.954-1.003 (0.975 ± 0.025)**; **no upward correction** for the Jacobian attenuation;
   slope per front width with the advection-numerics shortfall (−2% at 2 dx, −4% at 1.5 dx, −11% at
   1 dx) subtracted before any diffusion is inferred on the sharpest fronts. Also from 6b: planning
   §2.3's OS7MP implicit-diffusion estimate is an order of magnitude too large as a caveat on the
   *resolved* slope (+0.009 ± 0.03 in one hour); the third-order DST3 cross-check (−0.13) shows what a
   diffusive scheme does. §2.3 itself is not edited (its numbers describe the grid scale).
6. Smaller, for the record: `tile330_masks.nc` stays ignored, `validate.py` stays at ~1,020 lines,
   the subfilter-term growth with `L` is M3's (M1-Q8, all three left as they are); the tile-edge reach
   of the default `F` at `L = 8` is exactly 7 cells, so `edge_cells` must not shrink (task 6); M2
   (prompt 3) and M3 (prompt 4) must call `operators.frontogenesis` and `semilag` with their defaults
   — anything pinned to "`F = frontogenesis_tendency`" means `form='chain'` (task 6, flag 10); the M0
   planning deck still asserts the two claims M0 overturned (task 6 of prompt 1).

**Step 4 — closure. M1 closed 2026-09-30.** Every criterion passes. Marked in the Status paragraph of
`frontogenesis_prompt_2.md` (with the per-task summary) and in `frontogenesis_coding.md` §6 M1 ("M1
closed 2026-09-30", after the criteria list). "Do not" list respected: no data pulled, nothing
physical interpreted, no budget on real data. Task 8 (slides) is the next session's.

Files: modified `claude_prompts/frontogenesis_prompt_2.md` (criteria 1, 3, 4, 5, 7; Q&A header;
Status paragraph), `claude_prompts/frontogenesis_prompt_4.md` (stats paragraph, Runs, Figure 2,
acceptance 4), `claude_prompts/frontogenesis_prompt_6.md` (Figure 2 bullet), `frontogenesis_coding.md`
(§2.5, §4.3, §4.4, §4.9, §4.10, §6 M1, §6 M3), `frontogenesis_planning.md` (§2.4, §6 tests 1 and 3,
§7 Figure 2 and V3), `py/validate_figs.py` (`fig_V3b` layout only), `figs/V3b_fv_null.png`
(regenerated, same numbers) and this log. Not touched: every other module, the tests, the data
stores, `frontogenesis_prompt_1.md`, `_3.md`, `_5.md`, `deck/`.

### 2026-10-02 — Planning prompt 9: planning deck v2 (Claude Opus 5.5)

**Deliverable.** `deck/Frontogenesis_Planning.pptx`, now **v2** — 17 slides (was 13), rebuilt
by `deck/build_deck.py`, which was rewritten in place (v1 stays in git history). Work log in
`deck/README.md` ("Planning deck v2").

**The four requests.**

1. **No font below 20 pt.** Every run is clamped at `MIN_PT = 20` (the `build_m1_deck.py`
   helpers). v1 ran 10-17 pt in bodies, so this was a reflow, not a resize: shorter wording,
   three-column cards turned into full-width rows, footers and tags at 20 pt.
   `check_m1_deck.py Frontogenesis_Planning.pptx`: **minimum 20.0 pt, no offender.**
2. **Glossary** — slides 15-16: 9 physics terms (front/frontogenesis, b, G, F, DG/Dt,
   residual, tilting term, diabatic B, strain/θ) and 10 method/data terms (semi-Lagrangian,
   front pixels, filter scale L, subfilter flux τ, discrete null, OS7MP, KPP/KPPhbl,
   follow()/IoU, advected pixel set, tile 330/OSN/chunks).
3. **Tracking slide** — slide 11, from planning §5.7 / Q14. It has a cartoon: front A at t,
   its mask advected by u, v (dashed), the real continuation inside that prediction, and a
   nearer stationary neighbour B that position alone could pick. Beside it are four steps:
   label both hours independently; advect the *boolean* mask with `semilag` and threshold at
   0.5; score candidates with `follow()`'s position/overlap/length/area/orientation terms plus
   IoU with the predicted mask; link if the best score ≤ 2.5, otherwise record a gap. A closing
   band covers the advected pixel set and split/merge flags. I checked the score terms and
   `MAX_SCORE` against `fronts/front_tracking.py` on `origin/viz_tools`.
4. **Version number** — a v2 badge on the title slide, "Frontogenesis planning · v2" plus the
   slide number in every footer, and a version-history slide (17).

**Beyond the letter of the prompt (flag if unwanted).** v1 predated M0 and Q13-Q15, and
`deck/README.md` already said it had to be corrected before anyone saw it. So v2 also brings
six statements into line with the current `frontogenesis_planning.md`:
- surface `w = Dη/Dt`, not 0;
- the top-cell vertical term is measured from hourly chunks, not bounded;
- OSN land is NaN, not 0;
- surface fluxes come from the chunk store;
- numerical damping is ~0.1-0.5 f at 4Δx, not 0.1-1 f;
- the branch decision is resolved (Q15).

The "OSN path never run" risk was retired by M0. "Tracking links to the wrong front" replaced
it. Slide 17 lists all of this. No M1 results were added, so the deck stays a planning summary.

**QA.** Rendered with LibreOffice and all 17 pages inspected. Three layout fixes were made and
the render was repeated.

**Not done:** the Drive upload (still pending from prompt 6, and not asked for here).

**Upload (2026-10-02, follow-up to Planning prompt 9).** v2 pushed to the AIOcean shared drive,
`data/HIINet/Frontogenesis/`, alongside the M0/M1 decks, using the local **rclone** remote
`AIOcean:` rather than the Drive connector, which is what stalled the v1 upload. Two files:
`Frontogenesis_Planning.pptx` (62,950 bytes, id `1JRbjZeLCYZAJz9R-KU24MBKUXSuvPHhM`) and
the native Google Slides version `Frontogenesis_Planning`
(https://docs.google.com/presentation/d/1GG-CD82oxCbrmxAelGQ7mRha9IeEggHlBE4RSPi3UpA/edit,
17 slides, confirmed through the connector's metadata). The Slides version was converted with
`rclone copy --drive-import-formats pptx`, and the .pptx was uploaded with
`--drive-skip-gdocs`. This closes the upload left open since Planning prompt 6.

### 2026-10-03 — Execution prompt 3, task 1: pull_series and verify_series (Fable)

**Scope.** Task 1 of `frontogenesis_prompt_3.md` only: `osn_tiles.pull_series` (resumable,
atomic per hour), `verify_series`, and `tests/test_pull_series.py` offline plus one network
smoke test. Task 2 (the 72-hour pull) not run; tasks 4-5 (chunk store) not touched. *(entry
started early; extended below as the work proceeds)*

Nothing outside `dev/frontogenesis/` touched; `deck/`, the M1 modules and M1 tests untouched;
nothing committed; `tile330_raw_20120702T00_2h.zarr` kept (M2-Q4). Every python/pytest call under
`timeout 300`, no background jobs; the full suite is the longest at 139 s.

**Written.**
- `py/zarr_series.py` (new, 247 lines) — the generic resumable, per-hour-atomic append
  machinery, written as its own module so task 5's `vertical.load_chunk_levels` reuses it rather
  than copies it: `append_hour(path, ds_hour, encoding, attrs)`, `repair_trailing(path, log)
  -> int`, `present_times(path)`, `with_retries(fn, attempts, backoff, sleep, log, what)`,
  `consolidate(path)`, `RETRY_BACKOFF_S = (5, 20, 60)`. Knows nothing about OSN or §3.2.
- `py/osn_tiles.py` (370 -> 556 lines): `pull_series(timestamps, out_zarr, tile=None,
  endpoint=OSN_ENDPOINT, include_wind=True, clobber=False, *, grid_ds=None, attempts=3,
  backoff=RETRY_BACKOFF_S, sleep=time.sleep, log=None, report=None) -> str` — the §4.1
  signature positionally, keyword-only extras after it (see contradictions). `load_hours` was
  refactored onto three shared helpers (`_load_merged_hour`, `_series_attrs`,
  `_finish_series`) with identical behaviour, so M0's `m0_write.py` path is unchanged; `_sync_attrs`
  re-syncs the root attrs on resume.
- `py/series_verify.py` (new, 208 lines): `verify_series(out_zarr, timestamps, grid_ds=None)
  -> dict` and `summarize(res) -> str`.
- `py/tests/test_pull_series.py` (new, 461 lines): 20 offline tests + 1 `network` smoke test.
  `pytest.ini` needed no change: M1 had already registered `network` and `addopts` deselects it
  (`-m "not network"`); a CLI `-m network` overrides that (verified: "1 passed, 20 deselected").

**The product.** The store is M0's `write_raw` layout written incrementally: 9 vars float32,
one `(1, 720, 720)` chunk per hour per variable (Zstd level 0, little-endian bytes codec, NaN
fill, `_FillValue` attr `AAAAAAAA+H8=`), `time` int64 `seconds since 2011-09-10`
(`proleptic_gregorian`), `niter(time)` int64, scalar `face=10`, `k`, `k_l`, `XC`/`YC(j, i)` coords,
`U` on `i_g`, `V` on `j_g`, comodo attrs on all four horizontal dims, index coords int64, root
attrs `iterations, timestamps, endpoint, stores, face_index, j_face_start, i_face_start, rect_i,
rect_j, land_fill, git_commit, dbof_commit, created`. **Encoding parity with
`tile330_raw_20120702T00_2h.zarr`** is tested (`test_encoding_parity_with_m0_store`,
`needs_grid`): for every data var and for `time, niter, face, XC`, the zarr v3 `data_type`,
`codecs`, `fill_value`, `chunk_key_encoding`, `dimension_names`, `_FillValue` and the set of
`coordinates` are equal to M0's, and the `time` attributes (`units`, `calendar`) are identical. The
one deliberate difference: the series store chunks `time` and `niter` **one hour per chunk**
`(1,)` (the append unit), whereas M0's two-hour store, written in one call, has a single `(2,)`
chunk; `XC`/`YC` are auto-chunked by zarr at the first write in both. Both stores open with the same
`xr.open_zarr` and the same decoded coords, which is what "readable identically" needs; task 2's
spot check of hours 0-1 is on values and will not see this.

**Atomicity design — measured first, then built** (scratch experiments on xarray 2026.7.0 /
zarr 3.4.0, recorded in the `zarr_series` module docstring):
1. `to_zarr(append_dim='time')` is not atomic. Killing it after the third array write
   (`zarr.Array.__setitem__` patched to raise) left ``time`` at length 2 with `Theta`/`W` resized
   (one of them with its chunk written) and `Salt`/`Eta`/`niter` at length 1 — in that experiment
   `time` was written **first**, exactly the half-hour the prompt warns about. In the test's
   crash (`test_resume_after_crash_inside_append`) the order differs (xarray writes in dataset
   variable order); the repair does not rely on it.
2. `zarr.Array.resize` to a smaller shape does **not** delete the chunks beyond it (`c/2` stayed
   after resizing 3 -> 2).
3. xarray **drops the consolidated metadata** from the root `zarr.json` while an append is in flight
   and rewrites it at the end, so an interrupted store may have no consolidated view, or a stale one.
4. On every append xarray **rewrites every non-time variable's chunks** (`XC`, `face`, `i`, `j`,
   `i_g` — measured by mtime) and **replaces the root attrs** with the appended dataset's.

So: (a) *stage in memory, write once* — core and wind are loaded (with retries) into one merged
in-memory hour before any write, so a network failure never touches the store; (b) *append only the
time-dimensioned variables* — `XC`/`YC`, index coords and scalars are written at creation and
dropped from later appends (measured: only the new hour's chunk files, the resized arrays'
`zarr.json` and the root `zarr.json` change); (c) *repair on resume, before trusting `time`* —
`repair_trailing` opens the array metadata directly (`use_consolidated=False`), truncates every
time-dimensioned array to the shortest, then walks back from the trailing hour while any array's
trailing slab is **unwritten** (chunk key missing), **unreadable** (chunk fails to decode — a chunk
truncated by the kill) or **all fill value** (never true of a real hour: land is 31% of the tile;
`time`/`niter` are never the int fill 0), deleting the orphan chunks `resize` leaves and
re-consolidating; (d) *"present" is the store's own `time` coord* after the repair, never a side
file; (e) the root attrs carry the full `iterations`/`timestamps` list on every append and
`_sync_attrs` rewrites them from the store's `time` if a crash left them one hour ahead. The
common case costs nothing: lengths agree, the trailing slabs read (9 x 2 MB), the loop exits at
once. `repair_trailing` is idempotent and returns the number of hours removed (`report['repaired']`).

**Gap policy — stop at the first hour that still fails, and justify it against criterion 1.**
Retries: `attempts=3`, backoff `(5, 20, 60)` s, any `Exception` treated as transient
(`BaseException` — Ctrl-C — is not caught), `sleep` injectable. If the hour still fails,
`pull_series` records it (`report['failed']`, `report['not_attempted']` = every later hour, a
`FAILED` line through `logging` and the `log` callback) and **returns**. Why not "allow gaps and
fill later": acceptance criterion 1 is "72 steps, **no gaps**", so a store with a hole is never the
product; an append-along-`time` store cannot have a middle hour inserted later without a rewrite
(filling would need a preallocated 72-hour axis written by region, which breaks "present = the
store's own `time`"); and the invariant *store == contiguous prefix of the requested series* is
what makes "present" a one-line `isin` and the resume trivially correct. The prompt's "record it and
move on, so one bad hour cannot stall 72" is honoured in the sense that matters — the run does not
stall, it finishes immediately with a report, and the next run (task 2's script re-launched; "an
interrupted pull is restarted, not debugged") retries the failed hour first. A *persistently* bad
hour therefore blocks the hours after it, deliberately: that is a finding about the source to
report, not a gap to paper over. Tested in `test_persistent_failure_stops_at_gap`: hours 0-1 on
disk, hour 2 failed, 3-5 not attempted, `verify_series` reports the three missing, the re-run with the
source back completes the series byte-compatibly.

**Other rules.** Duplicated or non-increasing `timestamps` raise `ValueError` before any pull; a
requested hour earlier than the store's last hour and not already in it raises too (the store can
only grow forwards; clobber or use a new store); a requested sub-window already present is a plain
no-op. `clobber=True` `rmtree`s the store first. `grid_ds` (keyword-only) is the `XC`/`YC` source,
needed only when the store is created; default M0's `tile330_grid.zarr` if on disk, else one
`load_grid`. Progress: logger `frontogenesis.osn_tiles` / `frontogenesis.zarr_series` plus the
`log` callback (one line per event: hours present, each hour's wall time and count, retries,
failures, repairs), which is what task 2's detached script writes to `data/m2_pull.log`.

**`verify_series(out_zarr, timestamps, grid_ds=None) -> dict`** (`series_verify.py`), lazy, one
hour at a time (`ds[vars].isel(time=k).load()` = 9 chunk reads per hour): `time` (missing, extra,
duplicates, order vs `timestamps`), `schema` (vars exactly the §3.2 nine; dims per var; float32;
chunks `(1, nj, ni)`; coords `time, XC, YC, niter, face, k, k_l`; `face == 10` scalar; `time`
encoding `seconds since 2011-09-10` int64; comodo attrs and int64 on `j, i, j_g, i_g`; attrs
`iterations, endpoint, stores, git_commit`; `stores == [llc_surf, llc_wind]`), `land_nan` (every
hour, every var: `isnan == (hFac == 0)` with `U`->`hFacW`, `V`->`hFacS`, everything else including
`oceTAUX`/`oceTAUY` -> `hFacC`, per M0 task 3; reports the worst mismatch count per var and the
first bad hour), `niter` (steps all 144; equal to `osn_date_to_iteration(ts)` for each stored hour;
equal to the `iterations` attr), `KPPhbl` (present; finite on every `hFacC > 0` cell in every
hour). Top-level `ok`; `summarize()` prints one line per check. On the real hour below it ran in
< 1 s; 72 hours is ~650 chunk reads, seconds.

**Tests.** `test_pull_series.py`, 20 offline + 1 network, 37 s offline. Synthetic 12 x 12 hours
with the loaders' real shape (`(time, face, j, i)`, `U`/`oceTAUX` on `i_g`, `V`/`oceTAUY` on `j_g`,
scalar `k`/`k_l`, `niter(time)` from `osn_date_to_iteration`, comodo attrs, land NaN from a
synthetic `hFacC`/`hFacW`/`hFacS` that differ by a row/column), values a deterministic function of
(hour, variable) so the store is compared with the truth; only `load_hour`/`load_wind_hour` are
monkeypatched (plus `zarr.Array.__setitem__` for the mid-append crash). Covered: the fresh pull
(schema, dtype, chunking, time encoding, values, one chunk file per hour); encoding parity with the
M0 store (`needs_grid`); the no-op re-run (**sha256 of every file identical**, zero loader calls);
resume after a crash between hours (hours 0-2 not rewritten); resume after a crash **inside**
`to_zarr` (store genuinely half-written: arrays at lengths 3 and 4; `repaired == 1`, the hour is
re-pulled, values and attrs right); five directly built trailing defects (`time` extended but vars
not; vars extended but `time` not; chunk missing; chunk truncated to 7 bytes; slab all-NaN) each
repaired to the 3 good hours with the good chunk files untouched and the repair idempotent;
clobber; duplicate / out-of-order / earlier-than-store errors; retry then success (sleeps exactly
`backoff[0], backoff[1]`); persistent failure -> stop at gap -> next run completes; `verify_series`
on a good store, and failing on a gap, an extra hour, a duplicate/out-of-order `time`, float64 +
wrong chunks + missing `KPPhbl` + wrong time units, finite-on-land and NaN-on-ocean cells, a
`niter` step of 145, and a missing store.

**Full suite** (`timeout 300 ~/miniforge3/envs/frontogenesis/bin/python -m pytest
dev/frontogenesis/py/tests -q`): **104 passed, 3 xfailed, 1 deselected (network) in 139 s** =
M1's 84 + 3 strict xfails plus the 20 new.

**Network smoke** (`pytest py/tests/test_pull_series.py -m network -s`, run once): pulled
`2012-07-02 02:00:00` from both stores into a `tmp_path` store, `verify_series` **ok on real data**
(schema, land-NaN vs the M0 grid for all nine vars, `niter = 1023264`, `KPPhbl` finite on ocean),
dims `(time 1, j 720, i 720, i_g 720, j_g 720)`; **22 s wall** for the hour (load + append; OSN
was fast today — M0 saw 21-90 s per hour). At 22-90 s/hour the 72 hours are **~30-110 min**;
task 2 must run detached and will finish in one or two restarts.

**Contradictions / deviations from the docs — flagged.**
1. **Coding §4.1 signature vs the prompt's extra requirements.** `pull_series`'s return type is
   `str` by contract, but task 1 wants failures "recorded (returned and logged)", an injectable
   backoff and a progress log. Resolved with **keyword-only** extras after the contract's positional
   signature (`grid_ds, attempts, backoff, sleep, log, report`); `report` is a dict filled in place
   (`pulled, skipped, failed, not_attempted, repaired, wall_s`). Every call written to §4.1 is
   valid; §4.1 was not rewritten (a marked one-line note added, see files).
2. **Coding §1.3's ~400-line cap: `osn_tiles.py` is now 556 lines** (370 before). The split that
   respects the cap — moving the grid functions (`load_grid`, `write_grid`, `open_grid`,
   `build_xgcm`) to their own module — would ripple into `m0_write.py`, `conftest.py` and the M1
   tests' imports, which this task may not touch. Left over the cap and flagged, like
   `validate.py` (M1-Q8); the new machinery itself went into separate modules (`zarr_series`,
   `series_verify`) precisely to keep the overflow small.
3. **`verify_series` has no module in either doc** (prompt 3 task 1 names it bare; §4.1 does not
   list it). It lives in `py/series_verify.py`, with the execution prompt's `grid_ds=None`.
4. **Prompt 3 task 1 "record it and move on" vs acceptance criterion 1 "no gaps"** — resolved as
   stop-at-gap, above. If "move on" was meant literally (keep pulling later hours), the design would
   have to change to a preallocated time axis with region writes, and "present from the store's own
   `time` coord" would no longer hold.
5. **`pytest.ini` already had `network`** registered and deselected by default (M1), so the prompt's
   "register it if missing" was a no-op; the way to select it is `-m network` on the command line.
6. For the record, the §3.2 attrs list (`iterations, endpoint, stores, git_commit`) is a subset of
   what the store carries (as M0 task 4 noted), and `verify_series` checks the §3.2 four plus
   `stores`' value.

**For task 2 (`m2_pull.py`).** `pull_series(TS72, DATA_DIR / 'tile330_raw_20120702T00_72h.zarr',
report=rep, log=progress_file.write)`; re-run in a loop until `rep['failed']` is empty (or the same
hour fails twice, which is a source finding); then `series_verify.verify_series(path, TS72)` and
`summarize`. The no-op proof is `rep['pulled'] == []` plus a sha256 snapshot of the chunk files
(`test_rerun_is_noop_and_byte_identical` has the recipe). `osn_date_to_iteration` is a pure offline
function, so `iterations`/`niter` can be checked without the network. `get_remote_llc_data` prints
six progress lines per hour (M0 task 2) — 432 lines in the detached log; harmless.

Files: modified `py/osn_tiles.py` (imports, module docstring, `load_hours` refactor,
`pull_series`, `_sync_attrs`, helpers), `frontogenesis_coding.md` (§4.1, one marked note),
`claude_prompts/frontogenesis_prompt_3.md` (Status paragraph), this log; created `py/zarr_series.py`,
`py/series_verify.py`, `py/tests/test_pull_series.py`. Not touched: `pytest.ini`, `conftest.py`,
every M1 module and test, `deck/`, the data stores (the smoke test wrote only under `tmp_path`).

### 2026-10-03 — Execution prompt 3, task 2: the 72-hour OSN pull (Fable)

**Complete** (phase 1 09:15 PDT, phase 2 09:44 PDT; results below).

**Scope.** Task 2 of `frontogenesis_prompt_3.md`: the detached 72-hour pull into
`data/tile330_raw_20120702T00_72h.zarr`. Run in two phases: phase 1 (this part) wrote the
driver script, launched the pull detached and confirmed it is progressing; phase 2 (a later
session, after the main session sees `data/m2_pull_done.json`) runs `verify_series`, the no-op
re-run with checksums, the hours 0-1 spot check against the M0 two-hour store, and the
wall-time / volume report. Nothing outside `dev/frontogenesis/` touched; nothing committed;
`tile330_raw_20120702T00_2h.zarr` kept (M2-Q4). Every interactive command under `timeout`; the pull
itself runs under `nohup`, never in the foreground (M2 rules).

**Written: `py/m2_pull.py`** (new, ~170 lines; `m0_write.py` is the style precedent). A thin
driver on task 1's `pull_series`:
- builds the 72 timestamps `2012-07-02 00:00:00` … `2012-07-04 23:00:00` from `T_START` with
  `timedelta(hours=h)` in `DATE_FMT`, and asserts the count and both endpoints;
- opens M0's grid (`open_grid(GRID_PATH, with_face=False)`, the stored layout) and passes it as
  `grid_ds` — the `XC`/`YC` source for the store's creation;
- `pull_series(ts, OUT_ZARR, grid_ds=grid, log=say, report=rep)`; paths come from
  `osn_tiles.DATA_DIR` (resolved from `__file__`), and the script puts its own directory on
  `sys.path`, so it can be launched from anywhere;
- `Progress`: the `log` callback writes one timestamped line per event (hours present, each hour's
  wall time with a running mean and ETA computed from the live `report['wall_s']`, retries,
  failures, repairs, totals, the final report) to **`data/m2_pull.log`** (append, flushed per
  line) and to stdout (so `data/m2_pull.nohup` also has them, interleaved with dbof's six
  kerchunk progress lines per hour);
- **`data/m2_pull_done.json`**, written atomically (tmp + `replace`) when the run ends, with
  `status` (`ok` / `failed` / `error` / `interrupted`), `started`, `finished`, `wall_s`,
  `n_present` (from the store's own `time`), `n_requested`, `pid`, `host`, the full `report`
  (`pulled, skipped, failed, not_attempted, repaired, wall_s`) and any traceback. Written on every
  path — the stop-at-gap return (`status='failed'`), an exception (`except BaseException`, so a
  `KeyboardInterrupt` writes it too), and the complete or no-op run (`status='ok'`, `pulled=[]`
  on a no-op). A stale done-file is deleted at start. Exit code 0 iff `ok`;
- `--dry-run` lists the 72 timestamps with present/missing from `zarr_series.present_times` (no
  repair, no network, no writes) plus the store/log/done/grid paths, and exits;
- `if __name__ == '__main__': sys.exit(main())`.

**Dry run** (`timeout 120 python m2_pull.py --dry-run`, 09:15 PDT): store absent, 0 present, 72
to pull, grid present.

**Launch** (09:15:55 PDT, 2026-10-03):
```
cd dev/frontogenesis/py && nohup ~/miniforge3/envs/frontogenesis/bin/python m2_pull.py > ../data/m2_pull.nohup 2>&1 &
```
**PID 19049** (host MacBook-Pro-4.local). Log start line `09:15:56`; `0 hours present, 72
requested` at `09:15:57` (store created on the first append).

**First per-hour wall times** (load core + wind with retries, then one atomic append), no retries,
no repairs, no failures so far:

| hour | wall |
|---|---|
| 2012-07-02 00:00 | 21.7 s |
| 2012-07-02 01:00 | 22.4 s |
| 2012-07-02 02:00 | 22.9 s |
| 2012-07-02 03:00 | 20.7 s |
| 2012-07-02 04:00 | 22.1 s |

~22 s/hour, the same rate as task 1's network smoke test (22 s) and the fast end of M0's 21-90 s;
the store is 42 MB after 4 hours (~10.5 MB/hour, so ~0.75 GB for 72, as M2-Q2 estimated). ETA
from the script's own running mean: **~25 min, i.e. ~09:42 PDT** if OSN stays this fast; up to
~1.8 h if it slows to M0's worst 90 s/hour. The main session watches for
`data/m2_pull_done.json`.

**Phase 2 — results.** The pull **completed on the first launch, no restart needed**:
`m2_pull_done.json` (run 1, preserved as `data/m2_pull_done_run1.json`) has `status=ok`,
`n_present=72`, `pulled` = all 72, `skipped=[]`, `failed=[]`, `not_attempted=[]`, `repaired=0`,
`error=None`; start `09:15:56`, end `09:43:13` PDT (16:15:56-16:43:13 UTC). The log has no
retry, `FAILED`, repair or exception line (the only grep hits are the "0 failed / 0 repaired"
totals), so nothing exercised the retry/repair paths on real data; they remain covered by the
offline tests only.

- **Wall time.** Total **1636.9 s = 27.3 min**; the 72 per-hour walls (load core + wind, one
  atomic append) sum to 1636.2 s, so the script's own overhead (grid open, repair check, totals)
  was 0.7 s. Per hour: **median 22.4 s, mean 22.7 s, range 20.7-33.1 s**, p10/p90 21.0/24.7 s;
  the slowest hour was 2012-07-03 10:00 (33.1 s), then 07-04 11:00 (27.4), 07-03 14:00 (26.2).
  OSN was uniformly fast: 72 hours in a 21-33 s band, with none of the 90-192 s stalls M0 saw
  on 2026-09-28. Network-bound (writing an hour is ~0.2 s, M0 task 4).
- **Volume.** `du -sh` **765 MiB** (783,048 KiB allocated = 802 MB; 799.7 MB of file bytes),
  **11.1 MB/hour**, 834 files: 9 vars x 72 chunks (648) + 72 `time` + 72 `niter` chunks + the
  static `XC`/`YC`/index-coord chunks and the `zarr.json` metadata. Per variable 71 MB (`Salt`)
  to 91 MB (`V`); data chunks 1.03-1.33 MB each (Zstd on a 2.07 MB float32 `720 x 720` slab
  with 31% NaN land).
- **Extrapolation** at 22.7 s and 11.1 MB per hour: 1 week (168 h) ~64 min, 1.9 GB; the full
  504-hour OSN series ~3.2 h, 5.6 GB; 30 days (720 h) ~4.5 h, 8.0 GB. At M0's worst 90 s/hour
  those become 4.2 h / 12.6 h / 18 h, which is why the pull stays detached and resumable. Per
  tile the cost is linear in hours; a second tile is a second series of the same size.
- **`verify_series(out, TS72)` — OK in 0.9 s**, every check `ok`: `time` 72 in store / 72
  expected, no missing, extra or duplicate hours, ordered; `schema` no problems (the nine §3.2
  vars, dims, float32, `(1, 720, 720)` chunks, coords, `face=10`, time encoding `seconds since
  2011-09-10` int64, comodo attrs, §3.2 attrs, `stores`); `niter` steps `{144}`, equal to
  `osn_date_to_iteration` for every hour and to the `iterations` attr; `land_nan` **72 hours
  checked, 0 mismatch cells for all nine variables** (`U` vs `hFacW`, `V` vs `hFacS`, the rest
  incl. `oceTAUX`/`oceTAUY` vs `hFacC`), land fraction 0.3116, so M1's masks hold for every hour;
  `KPPhbl` present and finite on every ocean cell in every hour (criterion 3).
- **No-op re-run — proven byte-identical.** sha256 of all 834 files and `stat` (mtime, size)
  of each taken before; `timeout 300 python m2_pull.py` re-run at 09:44:54 (pid 73461): `72 hours
  present, 72 requested` -> `0 pulled, 72 skipped, 0 failed, 0 not attempted, 0 repaired`,
  wall **0.9 s**, done-file `status=ok` with `pulled=[]`, `wall_s={}`; sha256 and stat taken
  again: **0 differing lines in either**, newest mtime in the store still 09:43:13 (the end of
  run 1). Criterion 2 holds on real data. Snapshots kept in the session scratchpad only.
- **Hours 0-1 vs `tile330_raw_20120702T00_2h.zarr` — identical.** Same variable set (9 vars,
  11 coords); for all 20 variables/coords the dims, shape, dtype, variable attrs and values are
  equal (NaN-aware: 323,046 NaN in the centred fields = 2 x 161,523, 324,890 in `U`, 324,176 in
  `V`); `time` values and encoding equal, `niter` `[1022976, 1023120]`. Root attr **keys** equal;
  values differ only in `git_commit` (`e59b675+dirty` vs `49b1867+dirty`), `created`, and the
  longer `iterations`/`timestamps` lists (72 vs 2 entries) — provenance, as expected. The
  2-hour store is kept (M2-Q4); M1's tests are untouched.

**Criteria discharged.** 1 (72 steps, no gaps, §3.2 schema), 2 (no-op re-run, byte-identical),
3 (`KPPhbl`), the land-NaN replacement for the struck masks item (all 72 hours), and the OSN
half of 7 (765 MB, 27.3 min, 22.4 s/hour median). The chunk half (criteria 5-6, tasks 4-5)
is untouched.

**Contradictions / deviations — none from the docs; three notes.**
1. The done-file describes the *latest* run and is overwritten by every re-run, by design;
   a watcher tells a no-op from a pull by `report.pulled`, not by `status` (both are `ok`).
   Run 1's copy is `data/m2_pull_done_run1.json`.
2. `--dry-run` reads `present_times` **without** `repair_trailing` (it must not write), so on a
   store left half-written by a crash it could list the trailing hour as present; the real run
   repairs first, so the pull itself is unaffected.
3. The `status` paragraph in prompt 3 was updated to "tasks 1-2 done"; coding §6 M2 was not
   touched (that is the task-6 audit's job).

**For task 3 (`m2_qa.py`).** `xr.open_zarr(DATA_DIR / 'tile330_raw_20120702T00_72h.zarr')`
is 72 lazy hours, one chunk per hour per variable, so per-hour statistics stream at 9 chunk
reads per hour (the whole store loads in memory at ~1.3 GB float32 if wanted);
`expand_dims('face')` before any dbof operator; `open_grid()` for the masks; 71 hour pairs for
`semilag.departure_index`; the M2-Q3 extra (6 pairs through `validate.test_discrete_null`) is
~1 min per pair.

**Full suite** after the task (`timeout 300 python -m pytest dev/frontogenesis/py/tests -q`):
**104 passed, 3 xfailed, 1 deselected (network) in 143 s**, identical to task 1's count; the M1
baseline (84 + 3 strict xfails) still passes. `git status` shows only this log and the new
script changed; the data products are git-ignored.

Files: created `py/m2_pull.py`, `data/tile330_raw_20120702T00_72h.zarr` (765 MB),
`data/m2_pull.log`, `data/m2_pull.nohup`, `data/m2_pull_done.json` (latest run) and
`data/m2_pull_done_run1.json` (the 72-hour run); modified
`claude_prompts/frontogenesis_prompt_3.md` (Status paragraph), this log. Not touched: every
module and test, `pytest.ini`, `deck/`, `frontogenesis_coding.md`, `tile330_grid.zarr`,
`tile330_masks.nc`, `tile330_raw_20120702T00_2h.zarr`. Nothing committed.


### 2026-10-03 — Execution prompt 3, task 3: series QA and baseline stability (Fable)

**Scope.** Task 3 of `frontogenesis_prompt_3.md` (the time-series QA of the 72-hour store, no
`F`/`G` budgets or slopes) plus the extra step JXP approved in M2-Q3 (the V3 real-velocity null
re-run on 6 hour pairs across the window). Tasks 4-5 (chunk store) not touched. Offline, from
`data/tile330_raw_20120702T00_72h.zarr`, `tile330_grid.zarr`, `tile330_masks.nc`. Nothing
outside `dev/frontogenesis/` touched; `deck/` untouched; nothing committed; every python command
under `timeout 300`, no background jobs. *(entry started early; extended below as the work
proceeds)*

**Written.**
- `py/m2_qa.py` (638 lines, functions only) → **`figs/m2_qa_series.png`** (200 dpi, 3600 x 3400,
  1.2 MB, 8 panels in `validate_figs` style). Streams the store one hour at a time (72 hours in
  6 s) and one pair at a time (71 pairs in 42 s); per-hour / per-pair numbers cached as JSON in
  `data/m2_qa_hours.json`, `data/m2_qa_pairs.json`, summary `data/m2_qa_summary.json`
  (git-ignored; resumable, `--pairs a:b`, `--figure-only`, `--measured k,...`).
- `py/m2_baseline_stability.py` (422 lines) → **`figs/m2_v3_stability.png`** (3400 x 2300,
  0.6 MB); cache `data/m2_v3_stability.json` (resumable; `--pairs`, `--all a:b`, `--changes`,
  `--check-default`, `--roughness`, `--robust a:b`).
- **M1 code touched (flagged):** `py/validate.py` only — two **backward-compatible keywords**
  `store=None, t0=0` on `_llc_inputs`, `_discrete_null` and `test_discrete_null` (the real store
  path and the pair's first hour; `_llc_inputs` now loads `isel(time=[t0, t0+1])` instead of the
  whole store, and returns `hours`, `store`, `t0`; the `res` strings `tracer`/`velocity` are
  rewritten only when a non-default is passed). Defaults reproduce M1 **bit for bit**: the
  unchanged default call gives slope 0.980590 [0.969822, 0.994233], identical in every
  estimator, CI and threshold to the 72-hour-store pair 0 (`--check-default`). `test_fv_null`
  untouched (calls the defaults). No definition changed.

**QA numbers (tile-mean / ocean percentiles, 72 hours).**
- **Eta — the tide is there.** Tile-mean range **2.009 m** (−0.512 to +1.497 m; over
  `mask_analysis` 2.027 m, so it is tile-wide, not the Gulf of California); the p5-p95 band is
  ±0.2-0.3 m around the mean. FFT of the detrended tile mean: peak bin 12.0 h (72-h record,
  1/72 h⁻¹ resolution), parabolic refinement **12.41 h**; a free-period sinusoid fit gives
  **12.38 h, amplitude 0.612 m** (r² 0.61 alone, because of the diurnal inequality); M2 + K1 fit:
  **M2 0.619 m, K1 0.494 m, r² 0.989**, rms residual 0.07 m; S2 is not separable from M2 in 72 h.
  Highs at hours 1-2, 14-15, 26, 39, 51, 64; lows at 9, 20, 33-34, 44-45, 58, 69.
- **KPPhbl — the diurnal cycle is there.** 24-h fit **amplitude 6.4 m**, mean 22.1 m, trend
  −0.042 m/h, r² 0.85; **maximum at 8.85 UTC = 0.8 h local solar time (lon −120.5: UTC − 8.0 h;
  01:51 PDT)**, minimum at ~21 UTC = 13 h solar. Daily tile-mean min / max: 13.1 / 25.1 m (07-02,
  max at 09 UTC, min at 21), 11.8 / 26.8 (07-03, 09 / 21), **7.9 / 26.9** (07-04, 10 / 21) — the
  afternoon minimum deepens day by day (7.9 m at hour 69, p50 4.2 m) as the tile-mean |tau|
  falls from 0.10-0.12 to 0.06 N m⁻² (panel b, right axis). Figure 6 has its signal.
- **Theta** tile mean 16.955-17.313 °C (p5 ~14.0, p95 ~19.3), a slow cooling with a weak
  diurnal bump; **Salt** 33.640-33.648; **W** tile mean −1.2e-4 to +1.0e-4 m/s (= dEta/dt, M0).
- **|u|** at centres (`semilag.centre_velocities`; speed is rotation-invariant): p50
  **0.144-0.230 m/s**, p95 0.373-0.508, max 1.07-3.14 m/s, all three with a clear semidiurnal
  modulation; tile-mean geographic components with the face-10 orientation `u_east = V`,
  `v_north = −U`: u_east −0.08 to +0.02, **v_north −0.19 to −0.03 m/s** (southward, the
  California Current), both tidally modulated.
- **Land-NaN fraction — constant:** centred fields **0.31158** (161,523 cells), `U` 0.31336
  (162,445, `hFacW`), `V` 0.31267 (162,088, `hFacS`) in all 72 hours; **0 mismatch cells** vs
  `hFacC/W/S` for all nine variables in every hour (confirms task 2's `verify_series`).
- **oceTAUX / oceTAUY re-masking:** finite on `hFacW`/`hFacS` land **922 / 565 per hour** before
  (66,384 / 40,680 over the window — M0 task 3's numbers, constant), **0 / 0 after** re-masking,
  every hour.
- **Anomaly flags — none.** No frozen field (no identical consecutive hours for any variable;
  the largest identical-cell fraction is 0.6 % for `KPPhbl`, 0.26 % for `oceTAUX`; min rms hourly
  change of `Theta` 0.028 °C); no NaN-pattern change (pattern equal to hour 0's for every
  variable and hour); no outlier jumps in the tile means (max |z| of the hourly difference vs
  the MAD: Eta 1.6, KPPhbl 2.7, Theta 2.9, |u| 1.6, Salt 1.7, W 1.2; none > 5); the tide is not
  missing. The only flag raised is the L = 8 edge-support one below.

**Displacement envelope (71 pairs, `departure_index` defaults, midpoint velocity).**
- Pair 0-1 reproduces M1 exactly: ocean median 0.364, p99 1.248, max 2.10 cells.
- **Ocean:** median **0.268-0.441** (mean 0.351), p99 **1.05-1.38** (mean 1.22), per-pair max
  1.80-4.05; **window max 4.05 cells at pair 59 (07-04 11:00-12:00)** — at lon −113.57, lat
  29.25, 6.8 km from the coast in the **Gulf of California midriff** (the Ballenas Channel tidal
  jet), where every pair's max > 2.5 sits; the max series peaks every ~12.4 h (3.3-4.05 at pairs
  9, 22, 34, 47, 59, 70). None of those cells is in `mask_analysis`.
- **`mask_analysis`:** median **0.252-0.439**, p99 **0.93-1.34**, per-pair max 1.69-2.27;
  **window max 2.27 cells (pair 45, 07-03 21:00)**; 0 NaN departures on the analysis mask in any
  pair. Fraction of ocean cells moving > 1 cell: up to ~3.5 %; > 1.5 cells: ~0.5 %. So M1's
  assumptions (median 0.36, p99 1.25, max 2.1) hold over the window on the analysis domain to
  within the tidal modulation; the only exceedance is the Gulf of California (excluded).
- Front pixels of the V3 runs (next section): median 0.33-0.56, p99 1.30-1.79, max 1.66-2.27.

**Edge support — verdict: `edge_cells = 7` holds at `L ≤ 4` exactly; at `L = 8` it is short by
0-5 cells for ≤ 34 analysis cells per pair.** Geometric test per pair: the order-3 departure
support of `measured_DGDt` (the five-point stencil at the departure point on the 4-node
Lagrange kernel: nodes `floor(p) − 2 .. floor(p) + 3` per axis) against the finite part of a
field low-passed at `L` (the tile at `L = 0`; `[L/2, n − 1 − L/2]` otherwise, because `lowpass`
pads the edge with NaN):
- **`L = 0, 2, 4`: 0 `mask_analysis` cells in all 71 pairs** (a parcel would need > 5 / 4 / 3
  cells inward at the first analysis row; the window max there is 2.27).
- **`L = 8`: 1,450 cells over 71 pairs, 7-34 per pair (mean 20; 0.003-0.013 % of 262,925), worst
  pair 44 (07-03 20:00)**, all at edge distance exactly 7 (the first analysis row: a parcel
  arriving there from > 1 cell towards the edge needs node 4 − 1 = 3, which is in the NaN rim).
  **Cross-checked empirically**: `semilag.measured_DGDt` on JMD95 `b` (raw and `op.lowpass(b, 8)`)
  gives **0 / 21, 0 / 34, 0 / 28, 0 / 21 NaN analysis cells (L = 0 / 8) on pairs 0, 44, 46, 59**
  — identical to the geometric count, all within reach of the tile edge, min edge distance 7.
  So M1 task 6's "edge reach at `L = 8` is exactly 7, no slack" becomes, with the real
  displacement, a loss of ≤ 34 cells per pair at the analysis rim when `b` is low-passed at
  `L = 8` *before* the semi-Lagrangian step. Recommendation for M3: **keep `edge_cells = 7`**
  (making it exact at `L = 8` would need 9-10 cells for a 0.03 % gain), and reduce over
  `isfinite(DGDt) & mask_analysis` at `L = 8` — the rule M1 tasks 2-3 already state. Note also
  that `mask_edge`'s outermost analysis rows carry high leverage on day 3 (below), which is a
  stronger reason to report an edge-band sensitivity in M3 than the NaN count.

**Extra step (M2-Q3) — stability of the V3 baseline. The construction is M1 task 6's,
unchanged**: hour `t0` JMD95 `b`, `0.5 (U_t + U_tp1)`, our semi-Lagrangian step (order 3,
`vel_order` 3, `n_iter` 3), `measured_DGDt` vs `2F` at the midpoint, front pixels `G_mid ≥ p90`
over `mask_analysis` & finite (n = **26,293 of 262,925 in every pair**), OLS with intercept as
the gate, 32-cell block bootstrap with 1000 draws, forms `chain`, `discrete_o2`, `discrete` (the
gate). Because `changes=False` costs only **~2 s per pair** (not the minute M2-Q3 assumed), the
six planned pairs were run **and then all 71**; the planned pairs (chosen from the Eta / KPPhbl
phases above, at least one per day, plus a 7th at the window's shallowest mixed layer) are the
marked points in the figure. `data/m2_v3_stability.json` has every pair's estimators.

| pair (UTC) | phase | discrete (gate) | 95 % CI | n | gate | chain | corr |
|---|---|---|---|---|---|---|---|
| 0-1, 07-02 00 | high tide, ML deepening, 16 h solar — M1's pair | **0.9806** | [0.9698, 0.9942] | 26,293 | PASS | 0.7914 | 0.982 |
| 9-10, 07-02 09 | low tide, ML max, 01 h solar | 0.9872 | [0.9800, 0.9939] | 26,293 | PASS | 0.8053 | 0.982 |
| 21-22, 07-02 21 | rising, ML min of day 1, 13 h solar | 0.9851 | [0.9762, 0.9941] | 26,293 | PASS | 0.8052 | 0.965 |
| 33-34, 07-03 09 | low tide, ML max of the window | 0.9802 | [0.9745, 0.9908] | 26,293 | PASS | 0.7559 | 0.985 |
| 45-46, 07-03 21 | low tide, ML min of day 2, largest analysis displacement | 0.9775 | [0.9563, 0.9913] | 26,293 | PASS | 0.7971 | 0.958 |
| 62-63, 07-04 14 | high tide, smallest displacement, 06 h solar | **0.9462** | [0.9086, 0.9953] | 26,293 | **FAIL** | 0.6778 | 0.963 |
| 69-70, 07-04 21 (extra) | ML min of the window, 13 h solar | 0.9544 | [0.9163, 0.9849] | 26,293 | PASS | 0.7086 | 0.951 |

- **Reproduction:** pair 0-1 gives **0.98059** = M1's 0.9806, CI [0.9698, 0.9942] = M1's
  [0.970, 0.994]; chain 0.7914; `changes=True` reproduces every M1 variant (first attempt 0.7579,
  bilinear discrete 0.9382, velocity low-passed 1.0057, `b` low-passed 0.9806, strain seen
  0.992 / 0.856 / 0.847).
- **The 7 marked pairs:** slopes 0.946-0.987, mean 0.973 (std 0.016), weighted mean 0.984,
  χ²/dof 1.4 against their bootstrap CIs; 5/7 inside the baseline band, all 7 CIs overlap it,
  6/7 inside the V3b band; all 7 CIs exclude 1 from above (upper < 1).
- **All 71 pairs:** slopes **0.902-0.997**, mean **0.972**, std **0.020**, weighted mean
  **0.9828**; **64/71 pass the gate**, 66/71 CIs have upper < 1, **71/71 CIs overlap the
  baseline band**, 46/71 slopes inside it, 62/71 inside the V3b band 0.954-1.003; χ²/dof **1.8**
  (excess scatter beyond the bootstrap 0.013). Day means **0.984 / 0.978 / 0.953**. Failures:
  pair 36-37 (0.947) and pairs 62-67 (0.946, **0.903, 0.902**, 0.924, 0.914, 0.923), all with
  CIs 2-4x wider than hour 0's (0.82-0.99) and corr 0.92-0.95. Chain form 0.616-0.824 (std
  0.048), moving with the gate form; `discrete_o2` 0.94-1.03.
- **What the dips are (diagnosed, not tuned; nothing in the gate was changed):**
  1. *Leverage, not a drift of the operators.* Dropping the top 1 % of |2F| front pixels (263 of
     26,293) gives **0.971-1.002, mean 0.9869, std 0.0053** over all 71 pairs (pair 64: 0.902 →
     0.986; pair 36: 0.947 → 0.978; hour 0: 0.981 → 0.987); dropping 5 %: 0.98-1.00. The
     kurtosis of 2F on the front pixels rises from 112 (hour 0) to 200-340 (pairs 63-67). The
     largest leave-one-block-out shift is ≤ 0.02 for 62 pairs and **+0.058 to +0.080 for pairs
     63-67** (one 32-cell block carries 29-44 % of the 2F variance there).
  2. *Where:* on day 3 that block is at lon −124.3, lat 38.0 — the **block containing the
     northern tile edge** (`i = 0-30`). At hour 64 the median `G` in it is **~1e-13 on rows
     3-11 from the edge (10-100x the interior; absent at hour 0)**: a very sharp front strip
     entering the analysis domain at the tile boundary. The OLS on front pixels **7-9 rows from
     the northern edge is 0.71, 10-12 rows 0.83, ≥ 13 rows 1.02 / 0.98**; at hour 0 the same
     bands give 0.98 / 1.19 / 0.99 / 1.00. The strip's `G/threshold` median is 4.6 (interior
     2.7); `b` low-passed at `L = 8` moves pair 64 to 0.978 and the velocity low-passed to 0.985
     (hour 0: 0.981 / 1.006), so both a sub-resolved front (V3b's −4 to −11 % at 1-1.5 dx) and
     grid-scale velocity structure contribute; whether the first analysis rows also see the
     boundary itself (zero-gradient / `padding='fill'` leakage beyond `edge_cells`) cannot be
     excluded from this data alone and is **an open item for M3**.
  3. *Grid-scale velocity content* (`rms(u_c − lowpass(u_c, 2))/rms(u_c)` on the analysis
     mask, 0.028-0.042 across the pairs) explains only part: corr −0.45 with the slope.
- **Interpretation.** 0.981 is **a property of the operators on this tile, reproduced within the
  bootstrap over most of the window** (weighted mean 0.983; day-1 and day-2 means 0.984 / 0.978;
  every CI overlaps the baseline band; the systematic −2 % below 1 is present in 66/71 pairs),
  **with a leverage-driven tail**: on 7 of 71 pairs a single extreme front region (day 3: a
  sharp front at the northern boundary) pulls the OLS gate down by 0.03-0.08, and the OLS
  estimator with a `p90` pool is sensitive to it (the orthogonal / GM estimators on pair 64 give
  0.976 / 0.978 against OLS 0.902). The hour-to-hour spread (std 0.020 raw, **0.005 trimmed**)
  is therefore larger than the one-hour CI (±0.012) but is not a drift of the kinematics.
- **Recommendation (no doc or figure changed here):** keep Figure 2's baseline at **0.981
  [0.970, 0.994]** (M1-Q4) and the V3b band 0.954-1.003; add to M3's reporting (planning §11)
  (i) the window spread as the baseline's **temporal systematic — 0.972 ± 0.020 over 71 pairs,
  0.987 ± 0.005 with the top 1 % |2F| trimmed** — quoted beside the bootstrap CI; (ii) the
  trimmed and orthogonal estimators alongside the OLS gate for every pair (the gate's definition
  stays OLS); (iii) an edge-band sensitivity (slope with `edge_cells` = 7 vs 13); (iv) for the
  M3 headline, hours with a front strip at the tile boundary (07-04 14-20 UTC) should be shown
  separately, not dropped.

**Suite** (`timeout 300 python -m pytest dev/frontogenesis/py/tests -q`): **104 passed, 3
xfailed, 1 deselected (network) in 151 s** — unchanged from tasks 1-2; all M1 tests pass with
the `validate.py` keywords. `git status` shows `figs/m2_qa_series.png` and
`figs/m2_v3_stability.png` as new (`figs/.gitignore` `!*.png`), `py/m2_qa.py`,
`py/m2_baseline_stability.py` new, `py/validate.py` modified (+26/−11), plus task 2's uncommitted
`py/m2_pull.py` and the two prompt files. Nothing committed.

**Contradictions / deviations — flagged.**
1. **M2-Q3's cost estimate ("about a minute per pair") is wrong by 30x**: a V3 llc run with
   `changes=False` is ~2 s (the minute is `changes=True`'s variants plus the figure). That is
   why all 71 pairs were run; the 6-pair design asked for would have shown 5/6 pass and one
   failure without the context to interpret it.
2. **The M1 gate does not pass on every hour of the window**: 7/71 pairs fail `1 ± 0.05`
   (0.902-0.947), against M1 task 6 / coding §4.9's "PASS" recorded on hour 0-1. The gate as
   defined (OLS on the `p90` pool) is leverage-sensitive; the pass is robust once the top 1 %
   |2F| pixels are excluded (0.971-1.002). Not a change to any definition; a finding for the
   M3 prompt and the audit.
3. **M1 task 6's "edge reach at `L = 8` is exactly 7 = `edge_cells`, no slack"** was for the
   stencils at zero displacement; with the real displacement the `L = 8` reach exceeds 7 for
   7-34 analysis cells per pair (0.013 % max). `edge_cells = 7` is kept; `isfinite` on the mask
   is required at `L = 8`, as already recommended.
4. **Planning §5.3 / M1 task 3's displacement max (2.09-2.1 cells)** is an hour-0 and
   analysis-domain number; over the window the ocean max is **4.05 cells** (Gulf of California
   tidal jet, every ~12.4 h), 2.27 on `mask_analysis`. The median and p99 (0.36 / 1.25) hold to
   within ±25 % / ±10 % tidally.
5. `KPPhbl`'s diurnal cycle is 180° out of phase with the SST diurnal cycle as a "daytime"
   quantity: its **maximum is at ~01 h local solar (night-time convective deepening) and its
   minimum at ~13 h** — Figure 6's "diurnal residual" axis should be phased on the mixed-layer
   *minimum* (afternoon), and its day-to-day deepening (13 → 12 → 8 m) is a wind trend
   (|tau| 0.11 → 0.06 N m⁻²), not noise.
6. `m2_qa.py` is 638 lines (coding §1.3's ~400 exceeded, as `validate.py` is); the per-pair
   cache and figure code are the bulk. Flagged, not split.
7. The prompt's "6 hour pairs" was exceeded (7 marked + all 71); the construction is unchanged
   and the 6-pair result is a strict subset of what is reported.

Files: created `py/m2_qa.py`, `py/m2_baseline_stability.py`, `figs/m2_qa_series.png`,
`figs/m2_v3_stability.png`, and (git-ignored) `data/m2_qa_hours.json`, `data/m2_qa_pairs.json`,
`data/m2_qa_summary.json`, `data/m2_v3_stability.json`; modified `py/validate.py` (the two
keywords, flagged above), `claude_prompts/frontogenesis_prompt_3.md` (Status paragraph), this
log. Not touched: every other M1 module and test, `pytest.ini`, `conftest.py`, `deck/`,
`frontogenesis_coding.md`, `frontogenesis_planning.md`, the data stores and masks.

### 2026-10-03 — Execution prompt 3, task 4: chunk-store reconnaissance (Opus; Fable limit reached)

**Scope.** Task 4 of `frontogenesis_prompt_3.md` only: reconnaissance of source B. No bulk load,
no loader (`load_chunk_levels` is task 5). Everything below comes from `py/m2_chunk_recon.py`
(new, read-only), which merges its results into `data/m2_chunk_recon.json` (git-ignored). It
runs in sections (`--sections inventory,grid | hour --hour-vars ... | eta,edges | fluxes`), because
this machine reads Nautilus at only 0.55 MB/s and a single 3-D object takes ~2 min (see "Bytes"
below). The compressed objects fetched for the checks (~430 MB) are cached in the session
scratchpad, not in `data/`.

**Answer to M2-Q1: the transfer is complete.** All 72 hours are there, with `oceQsw` and
`oceFWflx`, all 51 levels, and they cover exactly tile 330.

**1. Location and access.**
- **Where:** `s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/` on NRP Nautilus, endpoint
  `https://s3-west.nrp-nautilus.io`, with path-style addressing.
- **Layout:** one **zarr v3** store per hour, `{YYYYMMDDTHH}.zarr` (no consolidated metadata),
  plus a static `grid.zarr`. It is not kerchunk.
- **Transfer config:** `configs/transfer/run_chunks_monterey_72h.yaml` on llc-repo branch
  `origin/transfer-monterey-72h`, by Lauren, 2026-10-01: commits `ffbe400 monterey72`,
  `e3fbcfd FWflux and Qsw` and `f779ed0 flx`. The last fixes a typo, `oceFWflux` →
  `oceFWflx`. That branch's yaml lists exactly our 72 dates. The `tiles-surface-only` worktree
  still carries the old 17-date `run_chunks_monterey_bay.yaml`, and its
  `docs/Data_Organization.md` still says "17 stores" and lists no `oceQsw`. Both are stale; they
  were not touched.
- **Access:** the bucket needs credentials. This machine already has a default AWS profile
  (`~/.aws/credentials`), which s3fs picks up, so the reads worked with no setup. No secret was
  read or printed. **Anyone running task 5 elsewhere needs Nautilus `dbof` credentials.**
- **How to open it:** `xr.open_zarr(fs.get_mapper(f'{PREFIX}/{hour}.zarr'), consolidated=False)`
  works. The script reads the objects directly instead: it takes the bytes, decodes them with
  zstd and reshapes them.
- **Robustness:** dbof's own reader (`get_raw_data._llc_depth_storage_options`) warns that
  Nautilus "intermittently serves corrupt bytes". No corrupt read occurred here, in 14 large
  and ~100 small objects. Task 5 should still retry on a zstd decode error.

**2. Hours: 72/72 present, 0 missing.**
- Every store from `20120702T00` to `20120704T23` exists.
- For every store, the stored `time` equals the name, and the group attrs agree:
  `selected_iteration` equals the MIT iteration (OSN `niter` − 10368; for example 07-03 T00 is
  MIT 1016064 against OSN 1026432), and `resolved_face=10, j_start=0, i_start=2880,
  tile_size=720`.
- The 11 stores that predated the 72-hour transfer were **rewritten** and now carry the new
  variables too: the layout is uniform across all 72.
- **Outside the window** there are 7 daily 12:00 stores: 06-29, 06-30, 07-01, 07-05, 07-06,
  07-07 and **07-09**. There is no 07-08. These are the old variable set, with **no**
  `oceQsw`/`oceFWflx`.
- The prefix also holds other chunk regions, which we did not open: `amundsen`,
  `bellingshausen`, `gulf_stream`, `ross`, `southern_ocean_scotia_sea` and `weddell`.

**3. Variables (identical in all 72 stores).**
- 3-D: `Theta, Salt` `(k, face, j, i)`; `U` `(k, face, j, i_g)`; `V` `(k, face, j_g, i)`;
  `W` `(k_p1, face, j, i)`.
- 2-D: `Eta, oceQnet, oceQsw, oceFWflx, SIarea` `(face, j, i)`; `oceTAUX` `(face, j, i_g)`;
  `oceTAUY` `(face, j_g, i)`.
- Coordinates: `time` and the index coords `face, j, i, j_g, i_g, k, k_p1`.
- **`oceQsw` and `oceFWflx` are present (Q13 done).**
- `SIarea` is all zero on the tile.

**4. Levels.** `k` has 51 levels, as expected. **`W` sits on `k_p1`, with 52 interfaces
(0..51), not on `k_l`.** Continuity says `k_p1 = k` is the *top* face of cell `k`, the same as
`k_l = k`:
- At 07-03 T00, `rA·(W[k] − W[k+1]) + Σ_out(U·dyG·drF·hFacW, V·dxG·drF·hFacS)` closes to an rms
  of **7e-12 m s⁻¹** (max 1e-10) for cells k = 0, 1, 2, against rms `W` ≈ 1e-4. At k = 49 and
  50 it closes to 1.2e-10, against rms `W` ≈ 3e-3.
- So **`W(k_l = 0..2)` = `W.isel(k_p1 = slice(0, 3))`**, and `W(k_l = 1)`, the cell-base
  velocity coding §4.6 needs, is `k_p1 = 1`.
- `W[k_p1 = 51]`, at the base of level 50 (968.6 m), is finite and non-zero: the model has 90
  levels, and the transfer keeps 51.
- The grid store has `k, k_l, k_u` (51) and `k_p1` (52), plus a 3-D `hFacC/W/S` and
  `mask_c/w/s` on `k`.

**5. Layout, coverage and bytes.**
- **Coverage — no blocker:** `grid.zarr` and every hourly store hold face 10, with `j` and
  `j_g` running 0..719 and `i` and `i_g` running 2880..3599. That is
  **exactly tile 330 (face 10, j 0:720, i 2880:3600)**. It is not a cutout of the tile: it is
  the same 720×720 native block, because the transfer floors 36.8 N, −121.9 E to the enclosing
  native chunk.
- **Grid against `tile330_grid.zarr`:** `XC, YC, dxC, dyC, dxG, dyG, rA, rAz, CS, SN, Depth` and
  `hFacC/hFacW/hFacS` at k = 0 are all **bit-identical**. So the tile indexing, orientation and
  staggering are the same, with no flip or transpose.
- **Chunking — this is the contradiction:** every variable is **a single zarr object per hour**.
  3-D chunks are `(51, 1, 720, 720)`, `W`'s are `(52, 1, 720, 720)`, and 2-D chunks are
  `(1, 720, 720)`. The codecs are bytes (little-endian) + zstd at level 0, no sharding. A zstd
  stream cannot be partially decoded, so **a `k = 0..2` read must fetch the full 51-level
  object.** Level-selective reads are not possible.
- **Bytes per hour, compressed on the store:**

  | Object | Size |
  |---|---|
  | `Theta` | 57.2 MB |
  | `Salt` | 47.5 MB |
  | `U` | 63.8 MB |
  | `V` | 64.2 MB |
  | `W` | 65.4 MB |
  | each 2-D field | ~1.2 MB |
  | **whole store** | **305.6 MB** (305.1-306.1) |
  | **72 h** | **22.0 GB** |

- **What a task-5 read of `Theta, Salt, W, oceQnet, oceQsw, oceFWflx` fetches:** **173.8 MB per
  hour, 12.5 GB for the window**, to keep ~25 MB per hour (3 levels × 4 fields + 3 2-D,
  float32).
- **Throughput measured from this machine: 0.55 MB/s**, flat. It is the same with 16 parallel
  byte-range GETs of one object (0.55 MB/s), so the limit is the link, not the request pattern.
- **So task 5 costs ~316 s per hour and ~6.3 h for 72 hours**, against 22 s per hour for OSN.
  It has to be detached and resumable, as designed. Running it on a machine near Nautilus would
  be much faster.
- Peak memory per hour: about 106 MB per decoded 3-D field. Decode one field at a time.

**6. 3-D grid (`grid.zarr`; float32 values as stored).**

| | k=0 | k=1 | k=2 | dim |
|---|---|---|---|---|
| `drF` | **1.0** | 1.14 | 1.30 | `k` (51) |
| `Z` | **−0.5** | −1.57 | −2.79 | `k` |
| `Zl` | 0.0 | −1.0 | −2.14 | `k_l` (51) |
| `Zu` | −1.0 | −2.14 | −3.44 | `k_u` (51) |
| `Zp1` | 0.0 | −1.0 | −2.14 | `k_p1` (52; last −968.62) |
| `drC` | 0.5 | 1.07 | 1.22 | |

- **Confirmed: `drF[0] = 1.0 m` and `Z[0] = −0.5 m`**, equal to OSN's 0-d scalars in
  `tile330_grid.zarr`.
- `Σ drF = 968.62 m`, and `Zp1[51] = −968.62 m`.
- The bottom levels have `Z` = −900.1 and −945.6 m, and `drF` = 44.87 and 46.05 m.

**7. Consistency with OSN — the two sources are the same model output, bit for bit.**
- **Hour 07-03 T00:**
  - chunk `k = 0` `Theta`, `Salt`, `U` and `V` against `data/tile330_raw_20120702T00_72h.zarr`:
    all **bit-identical**, with max |d| = 0 and identical NaN patterns. The finite counts are
    356 877, 356 877, 355 955 and 356 312.
  - **chunk `W(k_p1 = 0)` against OSN `W`: bit-identical**, which includes the same 4 exact
    zeros.
  - `Eta`, `oceTAUX` and `oceTAUY`: bit-identical. `oceTAU*` carries the same 922 and 565 zeros
    on `hFacW`/`hFacS` land.
- **Whole window:** **`Eta` is bit-identical in all 72 hours**, so there is no time offset of
  even one iteration anywhere. `Theta k = 0` is bit-identical at both ends of the window as
  well, 07-02 T00 and 07-04 T23.
- `corr(W[k_p1 = k], centred dEta/dt)` is **0.998, 0.903 and 0.690** for k = 0, 1 and 2. That
  reproduces M0's `W(0) = dEta/dt` and shows the cell-base `W(k_l = 1)` adding a real
  convergence part.
- **Land is NaN in the chunk store**, as in OSN: `fill_value = NaN`, and the NaN pattern equals
  `hFac == 0` exactly for `Theta`, `Salt` and `W` (`hFacC`), `U` (`hFacW`) and `V` (`hFacS`) at
  k = 0, 1, 2. The 2-D fields follow `hFacC`.
- The ocean fraction is 0.6884 at k = 0, 1, 2 alike. The first 3 levels have the same wet set.

**8. The surface fluxes: a sign-convention trap, and a forcing-resolution caveat.** Tile means
of the three flux fields were computed for all 24 hours of 07-03 (`--sections fluxes`, 2-D only).
- **The `+=down` attrs are wrong for these values.** The stored attrs say `oceQsw`/`oceQnet` are
  "+=down, >0 increases theta" and `oceFWflx` is "+=down, >0 decreases salinity". The data
  instead follow MITgcm's **upward-positive** forcing convention:
  - `oceQsw` is **≤ 0 everywhere at every hour**; no positive value exists. Its tile mean runs
    from −0.1 W m⁻² at 09 UTC (01 local solar time; the tile is ~UTC−8) to **−589 W m⁻² at
    21 UTC (13 LST)**.
  - `oceQnet` is about +115 W m⁻² at night (cooling) and −454 W m⁻² at 21 UTC (heating).
  - `oceFWflx` is about +2-3e-5 kg m⁻² s⁻¹, net evaporation in a summer subtropical ocean,
    i.e. positive upward.
  - **So `surface_flux_term` must flip the sign of all three**, or task 5 should store them
    sign-flipped with an attr saying so. Do not trust the long_name.
- **The shortwave is not a resolved diurnal cycle.** The hourly series is **piecewise linear,
  with kinks every 6 h at 03, 09, 15 and 21 UTC**. It looks like linear interpolation, by the
  model's forcing package, of 6-hourly atmospheric fields. Shortwave is non-zero until 09 UTC
  (01 LST), and its night-time "zero" lasts one instant.
  - Planning §4 and §2.3 argue that `oceQsw` resolves the noon-peaking term Figure 6 is about.
    That is still true at the 6-hourly scale. The diurnal *shape*, however, is a triangle with
    its peak at 13 LST and its minimum at 01 LST, not insolation. M3 and Figure 6 should say
    so.

**Blockers for task 5: none.** Everything task 5 needs exists, for all 72 hours, on exactly our
tile. Practical constraints:
- **(a) Cost:** ~6.3 h of download at 0.55 MB/s (12.5 GB fetched, ~1.8 GB kept). Run it detached
  and resumable, as task 5 already plans.
- **(b) Rename `W`'s dim:** take `k_p1` 0..2 and rename it to `k_l`, to match §3.3.
- **(c) Flux sign:** handle the convention, and record in the store's attrs which one it uses.
- **(d) Corrupt reads:** retry on a corrupt or failed decode.
- **(e) Credentials:** Nautilus credentials are needed, and are present on this machine.
- **(f) `drF`:** take `drF(k = 0..2)` from `grid.zarr`, not from the hourly stores. The hourly
  stores carry no grid.

**Contradictions with the planning, coding and prompt docs — flagged.** Nothing in those docs
was edited.
1. **Prompt 3 (task 4 and source B) and coding §3.3** assume the store is laid out "so a
   `k = 0..2` read touches only those levels", and that "nothing obliges us to read" the other
   levels. **False:** there is one 51-level object per variable per hour, so the other levels
   *must* be downloaded, though not stored. That is 174 MB per hour, 12.5 GB in all.
2. **Coding §3.3 and §4.6 and prompt 3 say `W` is on `k_l`;** the store has **`k_p1` (52)**.
   They are equivalent for 0..2, by continuity to 1e-11; it is a rename.
3. **"~539 MB/timestep" and "~33 GB"** (planning §4, Planning-7, prompt 3) are uncompressed
   figures. On the store an hour is **306 MB**, and 72 hours are **22.0 GB**.
4. **"11 of the 72 stores exist; 61 new"** (prompt 3 "The window", planning §4, coding §6 M2)
   is now stale. All 72 exist, and the 11 old ones were rewritten with the new variables. The
   dbof `Data_Organization.md` (17 stores) and the old `run_chunks_monterey_bay.yaml` are
   stale too.
5. **Flux sign:** coding §4.6 `surface_flux_term` and planning §2.3 implicitly take the
   documented `+=down` convention. The data are **upward-positive** (item 8).
6. **Planning §4 and §2.3 on `oceQsw` "the noon-peaking term":** the forcing is 6-hourly and
   linearly interpolated (item 8), so the diurnal shape is not resolved. This does not
   invalidate Q13. It limits what Figure 6 can claim.
7. Planning §4 calls OSN and the chunk store "different readers of the same physics", a
   genuine cross-check. They are **bit-identical** at k = 0, so the cross-check is trivially
   satisfied. It confirms the iteration mapping, but it cannot catch a model-side issue.
8. Minor: prompt 3 says `process_llc4320_3d_grid` provides `drF`. It is a column filter on a
   grid Dataset. Opening `grid.zarr` directly gives `drF/Z/Zl/Zu/Zp1`, and `Zu`/`Zp1` live on
   `k_u`/`k_p1`.

**Suite:** `timeout 300 python -m pytest dev/frontogenesis/py/tests -q` → **104 passed, 3
xfailed, 1 deselected (network) in 144 s**, unchanged.

**Files:**
- Created: `py/m2_chunk_recon.py`; `data/m2_chunk_recon.json` (git-ignored); the scratchpad
  object cache (outside the repo).
- Modified: this log, and `claude_prompts/frontogenesis_prompt_3.md` (Status, and a note under
  M2-Q1).
- Not touched: every M1 and M2 module and test, `deck/`, the planning and coding docs, the dbof
  worktree, every remote store (read-only), and `data/`, except the JSON.
- Nothing committed.

### 2026-10-03 — M2-Q7: tile-edge margin test (Opus; Fable limit reached)

**Scope.** JXP's answer (a) to M2-Q7 in `frontogenesis_prompt_3.md`: re-run the V3 real-velocity
discrete null on all 71 hour pairs with `edge_cells` = 7 / 10 / 13 / 16, and discriminate tile-edge
contamination from a real front at the northern edge (crop, motion and support tests). Offline,
from the stored data; every python command under `timeout 300`; nothing committed.
*(entry started early; extended below as the work proceeds)*

**Written.** `py/m2_q7_edge_margin.py` (new; `--pairs a:b`, `--crop t0,...`, no flag = analysis +
figure) → **`figs/m2_q7_edge_margin.png`**. Per-pair caches and `analysis.json` are in the
session scratchpad (`.../scratchpad/m2q7/`), not in `data/`. **No M1 code changed:**
`validate._fit_llc` already takes the mask through `inp['ana']`, so no `mask=` keyword was
needed. The null step (`validate.null_step`, forms `chain`/`discrete_o2`/`discrete`, order 3,
`vel_order` 3) does not depend on the mask, so it runs once per pair. `_fit_llc`'s fit is then
applied line for line on each mask. Each mask is `masking.analysis_mask(g, edge_cells=E)`, built
in memory. `analysis_mask(edge_cells=7)` equals `tile330_masks.nc`'s `mask_analysis` (asserted).
The front-pixel rule `G_mid >= p90` is **recomputed on each mask** (as V3 does), with the same
OLS gate and the same 32-cell block bootstrap (1000 draws, seed 0). Runtime is ~1.8 s per pair
for all four masks.

**Reproduction.** With `edge_cells = 7`, all 71 pairs reproduce `data/m2_v3_stability.json`
**exactly** (max |diff| 0.0 over slope, CI, n_front, threshold, chain slope and the trimmed
slope). The result is 64/71 pass, failures 36 and 62-67, hour 0 gives 0.98059
[0.96982, 0.99423].

**Table (71 pairs, gate = OLS slope on `discrete`, 1 ± 0.05).**

| `edge_cells` | n mask | n front | pass | failing | mean ± std | range | weighted mean | trimmed (top 1 % \|2F\| dropped) mean ± std [range] | hour 0-1 |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 262,925 | 26,293 | **64/71** | 36, 62-67 | 0.9720 ± 0.0202 | 0.902-0.997 | 0.9828 | 0.9869 ± 0.0053 [0.971, 1.002] | 0.9806 [0.9698, 0.9942] |
| 10 | 258,491 | 25,850 | 69/71 | 36, 63 | 0.9786 ± 0.0125 | 0.927-1.002 | 0.9843 | 0.9878 ± 0.0051 [0.971, 1.003] | 0.9807 [0.9709, 0.9937] |
| 13 | 254,102 | 25,411 | 69/71 | **36, 70** | 0.9799 ± 0.0118 | 0.940-1.002 | 0.9842 | 0.9870 ± 0.0051 [0.972, 1.005] | 0.9793 [0.9700, 0.9921] |
| 16 | 249,761 | 24,977 | 68/71 | **36, 69, 70** | 0.9778 ± 0.0126 | 0.932-1.001 | 0.9833 | 0.9864 ± 0.0054 [0.971, 1.003] | 0.9792 [0.9689, 0.9916] |

Day-3 pairs (36 and 62-67), E = 7 / 10 / 13 / 16:
- 36: 0.947 / 0.949 / 0.946 / 0.944;
- 62: 0.946 / 0.957 / 0.975 / 0.974;
- 63: 0.903 / 0.927 / 0.980 / 0.980;
- 64: 0.902 / 0.957 / 0.982 / 0.981;
- 65: 0.924 / 0.971 / 0.982 / 0.982;
- 66: 0.914 / 0.968 / 0.987 / 0.985;
- 67: 0.923 / 0.984 / 0.995 / 0.990.

So a wider margin removes the 62-67 failures, but it does so **by excluding a band, not by
cleaning one**. The trimmed statistics do not move (0.987 ± 0.005 for every E). Pair 36 fails
at every E. At E = 13 and 16 **new failures appear (69, 70)**. Leave-one-block-out shows that
pair 36 (E = 7 and 13), pair 70 (E = 7, 13, 16) and pair 69 (E = 16) are driven by **interior**
blocks (i = 64-95, i.e. ≥ 64 cells from any edge, lon −124.3 / −125.0, lat 37.1). Each is a
single sharp front with a LOO shift of +0.02 to +0.03. The same leverage mechanism acts with no
edge nearby. Once the 7-12 band is excluded at E = 13, the p90 pool shifts and these interior
blocks gain weight.

**(i) Crop test** (pairs 63, 64 and 0 as a control; tile cropped by N = 4, 8, 16 cells at the
northern edge, i.e. low `i` on face 10; the full null step is recomputed on the cropped
tile/grid and compared at the same cells, j 16-703):
- The new edge reaches **3 cells into `G_mid` and `measured`, 5 cells into `2F` (discrete)**.
  The affected cells become **NaN** (222 of 222 per column at 0-1, ~10-15 % at 3 / 5). No finite
  values are wrong there: our operators NaN the edge, so xgcm's 0 pad never reaches a finite
  output.
- Beyond 4 (G, DG/Dt) and 6 (2F) cells, the largest change in any column is **≤ 4e-12 of the
  column max**, which is round-off. Some cells fail `rtol=1e-9` (≤ 2.6 % of a column, at 4-33
  cells), but only in `measured`, which is the difference of nearly equal G values. The absolute
  change stays at the 1e-12 level.
- Band slopes on the northern block's front pixels (full → cropped):
  - pair 64: N = 4 gives 10-12: 0.8452 → 0.8452, 13-15: 1.0307 → 1.0307, 16-19: 1.0701 → 1.0701,
    20-31: 0.9472 → 0.9472. N = 8 gives 13-15 and 20-31 unchanged. N = 16 gives 20-31
    0.9472 → 0.9469, because 9 pixels at 4-5 cells from the new edge became NaN; on the same
    surviving pixels the full tile gives 0.9469.
  - The 7-9 band at N = 4 (now 3-5 cells from the new edge) moves 0.637 → 0.969. **This is
    pixel loss, not value change**: 48 of 72 pixels go NaN, and on the 24 survivors the full tile
    also gives 0.969.
  - Pair 63 and pair 0 behave the same way.
- So the low slopes at rows 7-12 are **crop-invariant to round-off** when the edge moves 4 to 16
  cells closer. Contamination would change them.

**(ii) Motion test** (hours 55-70, region j 140-230, i < 48):
- The top-1 % `G_mid` strip (i ≥ 7) stays at **i ≈ 10 (p10-p90 8-13)** throughout. Its j-centre
  drifts 180 → 177, and the top-1 % |measured − 2F| residual drifts j 174 → 170, at i 8-9.
- The block-mean displacement is di +0.2 to +0.56 cells/h (southward, away from the edge) and
  dj −0.22 to −0.37 cells/h (westward).
- The j drift (−3 to −4 cells in 15 h) matches dj. The i position does not follow di: it is
  stationary in i.
- On its own this is **ambiguous**: a stationary front at a confluence (cold inflow from the north
  meeting a warm tongue) is common.
- The raw model `Theta` settles the point (figure panel c). At j = 176 the temperature jumps
  **11.84 → 14.02 °C between i = 7 and i = 11** at hour 64 (2.2 °C in 4 cells). The same front is
  at i ≈ 6-10 at hour 52 and at i ≈ 7-11 at hour 70. At hour 0 the profile is flat (14.7-15.0 °C).
- The front is in the model output itself. The tile edge is inside face 10 (`i_face_start` 2880),
  not a face seam, so nothing in the source data is special at i = 0.

**(iii) Support test** (task-3 displacements, order-3 departure stencil: the lowest node touched
is `floor(i − di) − 2`):
- On `mask_analysis` (E = 7), the lowest node over all 71 pairs is **i = 2** (window max
  displacement towards the edge on the first analysis rows: 2.17 cells).
- On the northern block it is **i = 3** (pairs 63, 64, 67; max displacement towards the edge
  1.16 cells).
- The departure support therefore stays 2-3 cells inside the tile and never reaches the pad. It
  reads raw `b`, which is finite and correct at i = 0-2. The xgcm-padded outputs (G, Jacobian at
  i = 0-1) are NaN in our operators and not used.

**Verdict: not tile-edge contamination. It is a real, near-grid-scale model front that happens
to sit 7-13 cells from the northern edge on day 3.**
- A wider margin "fixes" pairs 62-67 only by excluding the front.
- The same leverage failure occurs at interior fronts (pairs 36, 69, 70), and two of those get
  *worse* with a wider margin.
- Per JXP's option (a) rule, **nothing in masking was changed**: `edge_cells = 7` stays in
  `masking.py`, `tile330_masks.nc`, `test_masking.py`, the coding doc and the V6 caption.
- **Recommendation: (c) with (b)'s sensitivity row.** Keep `edge_cells = 7`, and report the 7
  failures as the temporal systematic (0.972 ± 0.020; trimmed 0.987 ± 0.005). Add to M3 the
  `edge_cells = 13` sensitivity (mean 0.980 ± 0.012, 69/71) and the trimmed / orthogonal
  estimators per pair.
- The failing hours reflect the OLS-on-p90 gate's sensitivity to a single sub-resolved front
  (V3b: −4 to −11 % at 1-1.5 dx), wherever the front is. Task 3's open item ("whether the first
  analysis rows also see the boundary itself cannot be excluded") is **closed: they do not**.
- Since the mask is unchanged, the V3 hour 0-1 baseline is unchanged (0.9806). For information,
  it would be 0.9793 [0.9700, 0.9921] at E = 13.

**Suite** (`timeout 300 python -m pytest dev/frontogenesis/py/tests -q`): **127 passed,
3 xfailed, 2 deselected in 174 s.** The count is up from 104 / 1 deselected because the
concurrent agent added `tests/test_load_chunk_levels.py` (and `vertical.py`, plus a
`series_verify.py` modification); none of their tests fail. My work changed no tested code.

**Contradictions / flags.**
1. The M2-Q7 premise ("a slope recovering to 1 with distance from the boundary is the signature
   of contamination") does not hold here. The recovery follows the front's position, which
   happens to be fixed relative to the edge. Crop invariance shows the boundary plays no part.
2. Task 3's band numbers are reproduced: 7-9 rows 0.637 (task 3 quoted 0.71, presumably over a
   different hour set or pixel subset), 10-12 rows 0.845 (0.83), ≥ 13 rows 1.03 / 1.07. Over
   hours 62-70 the 13-19 bands go *above* 1 (1.03-1.29), so "recovers to 1 beyond 13" holds only
   on average.
3. Widening the margin is not monotone in pass count (64 → 69 → 69 → 68). Raising `edge_cells`
   would have traded one leverage failure for others, so it could not be justified "by the
   numbers".

Files: created `py/m2_q7_edge_margin.py` and `figs/m2_q7_edge_margin.png`; modified this log and
`frontogenesis_prompt_3.md` (a note under JXP's M2-Q7 answer). Not touched: `masking.py`,
`validate.py`, `tile330_masks.nc`, every test, the coding/planning docs, the V3/V6 PNGs,
`vertical.py`, the chunk files, `m2_chunk_pull.py` and `deck/`. Nothing committed.

### 2026-10-03 — Execution prompt 3, task 5: load_chunk_levels and the chunk pull (Opus; Fable limit reached)

**(in progress; phase 2 appends the results.)** Phase 1, this part: the loader, its tests and
the pull script are written, and the 72-hour pull is launched detached and confirmed to be
progressing. Phase 2, a later resume after the main session sees `data/m2_chunk_pull_done.json`,
does the rest: `verify_chunk_series`, the no-op re-run with checksums, the k=0 bit-identity spot
check against OSN for 3 hours, volume and wall time, the missing hours, and the Status update.

**Scope and rules.** Nothing outside `dev/frontogenesis/` was touched, and nothing was
committed. `masking.py`, `tile330_masks.nc`, `validate*.py` and `deck/` were left alone, since
M2-Q7 runs in parallel. The remote store is read-only. The note to Lauren is drafted and not
sent. Every interactive command ran under `timeout`, and the pull runs under `nohup`.

**Written.**
- **`py/vertical.py`** (new, ~525 lines; see deviation 1). It holds only
  `load_chunk_levels(window, k_max=2, out_zarr=None, *, clobber, endpoint, prefix, fs,
  osn_store, local_grid, attempts, backoff, sleep, log, report)` and its private helpers. The
  physics functions of §4.6 are M3's. It returns the path, or an in-memory Dataset when
  `out_zarr=None` (the §4.6 `str | xr.Dataset`).
- **`series_verify.verify_chunk_series(out_zarr, timestamps, levels=None, k_max=2,
  sw_tol=1.0)`** (+154 lines in `py/series_verify.py`, now 362). It reuses `_check_time` and
  `_check_niter`.
- **`py/tests/test_load_chunk_levels.py`** (new): 23 offline tests and 1 `network` smoke test.
- **`py/m2_chunk_pull.py`** (new): the driver. It imports `Progress` and `timestamps_72` from
  `m2_pull.py` rather than copying them.
- **`claude_prompts/note_to_lauren_flux_signs.md`**: the M2-Q6 (c) draft, for JXP to
  forward.

**Design.**
- **Reading.** Each hourly object is one bytes+zstd chunk (task 4). The loader reads
  `zarr.json` and then the single chunk through `vertical._cat(fs, path)`, the one network
  primitive (s3fs on Nautilus, path-style, default AWS profile, no cache). It zstd-decodes the
  bytes, checks that they are exactly `prod(shape) * itemsize`, and reshapes. It does not go
  through `xr.open_zarr`, so every object can be validated and re-fetched on its own. A corrupt
  `Theta` re-fetches 57 MB, not the whole 174 MB hour. One field is decoded at a time, and the
  `k = 0..2` slab is copied so that the 51-level buffer is freed.
- **Validated before anything is written.**
  - The layout: one chunk, bytes (little-endian) + zstd, the default key encoding.
  - The shapes: `(51, 1, nj, ni)`; `W` `(52, …)`; the 2-D fields `(1, nj, ni)`.
  - Land: NaN exactly where the chunk `grid.zarr` `hFacC[k] == 0`. This is checked for
    `Theta`/`Salt` at k = 0..2, for `W` at `k_p1 = 0..2`, and for the fluxes and `Eta` at
    `hFacC[0]`.
  - Values: finite values inside generous plausibility bounds, which catch garbage that still
    decodes.
  - Time alignment, for every hour:
    - the store's own `time` equals the requested timestamp;
    - the group attrs `selected_date_utc` and `selected_iteration` (= OSN `niter` − 10368)
      match, and so do `resolved_face`, `j_start`, `i_start` and `tile_size`;
    - the chunk `Eta` (1.2 MB) is **bit-identical to the OSN 72-hour store's `Eta`** for that
      hour (`osn_store=`; task 4 found this true in all 72 hours);
    - after the append, the store's `time` read back ends at the requested hour.
  - Once per run, the chunk `grid.zarr` is checked: `drF[0] == 1.0`, `Z[0] == −0.5`, it
    starts at `j = 0` and `i = 2880`, and `hFacC[0] != 0`, `XC` and `YC` equal M0's
    `tile330_grid.zarr` (`local_grid=`).
- **Retries and gaps.** `zarr_series.with_retries` wraps each object and each group-attrs
  read: 3 attempts, backoff 5/20/60 s. A failed check counts as a failed read. An hour that
  still fails **stops the run at the gap** (`report['failed']`, `report['not_attempted']`),
  exactly as in `pull_series`; the next run resumes from that hour.
- **Resumability: `zarr_series`, reused.** The hour is staged in memory, then appended once
  (`append_hour`, time-dimensioned variables only). On resume, `repair_trailing` runs before
  `present_times` is trusted. "Present" is the store's own `time`. The root attrs are re-synced
  through `osn_tiles._sync_attrs`. Duplicate or out-of-order timestamps, or an hour earlier than
  the store's end, raise. `clobber` is supported.
- **Schema (§3.3).**
  - Variables, all `float32`, one chunk per hour per variable:
    - `Theta`, `Salt` `(time, k=0..2, j, i)`;
    - `W` `(time, k_l=0..2, j, i)`;
    - `oceQnet`, `oceQsw`, `oceFWflx` `(time, j, i)`;
    - `drF(k)`, static and written once.
  - Coords:
    - `time`, encoded as `seconds since 2011-09-10` int64, as in §3.2;
    - **`niter(time)` = the OSN iteration**, for consistency across the two stores;
    - `mit_iteration(time)` = `niter − 10368`, the source's `selected_iteration`;
    - scalar `face = 10`;
    - `k`, `k_l`;
    - `j` (0..719) and `i` (2880..3599), int64 with comodo attrs, the same values as the OSN
      store;
    - `XC`, `YC`;
    - `Z(k)` and `Zl(k_l)`, which were cheap.
  - Root attrs:
    - the §3.3 set: `source='CHUNKS/monterey_bay'`, `levels='k=0..2, k_l=0..2'` and
      `git_commit`, plus `dbof_commit` and `created`;
    - `source_path`, `endpoint` and `addressing`;
    - `iterations` (OSN), `mit_iterations` and `timestamps`;
    - the tile attrs and `land_fill`;
    - `flux_sign_convention`, `flux_sign_conversion`, `w_interfaces`, `forcing_note` and
      `provenance` (a list that records the negation).
- **`W` rename.** `W(k_l = n)` = source `W(k_p1 = n)`, for n = 0..2. `k_p1 = n` is the *top*
  face of cell n. Task 4 verified this by continuity (rms 7e-12 m s⁻¹ for cells 0..2). The
  mapping is in a code comment, in `W.attrs['interface_mapping']` and `source_dim='k_p1'`, and
  in the root attr `w_interfaces`.
- **Sign conversion (M2-Q6 (a)).** `oceQnet`, `oceQsw` and `oceFWflx` are **negated at write
  time**, so the store is downward-positive, as coding §3.3/§4.6 assume. Each of the three
  carries:
  - a correct `long_name`;
  - `sign_convention='positive downward (into the ocean); …'`;
  - `source_sign_convention`, the original "+=down" long_name;
  - `sign_conversion`, which says the variable was negated and why;
  - `forcing_note`, on the 6-hourly interpolated forcing.

  The root `provenance` says "NEGATED". **Guard:** an hour whose *raw* `oceQsw` exceeds
  +1 W m⁻² is refused. If the source were ever corrected by negating the data, the pull fails
  instead of flipping the sign twice.

**`verify_chunk_series`** checks:
- `time`: gaps, duplicates and order;
- `schema`: the §3.3 variables, dims, float32, chunks `(1, 3, nj, ni)` / `(1, nj, ni)` /
  `(3,)`, `k`/`k_l` = 0..2, coords, scalar `face`, time encoding, root attrs, the per-flux sign
  attrs, and `W.source_dim`;
- `niter`: steps of 144, equal to `osn_date_to_iteration` and to `iterations`, with
  `mit_iteration == niter − 10368`;
- `drF`: equal to the grid's, `drF[0] == 1.0`, `Z[0] == −0.5`;
- `land_nan`: every hour and every level k = 0..2, NaN exactly where `hFacC[k] == 0`;
- `sign`: stored `oceQsw ≥ −sw_tol` everywhere, and > 0 somewhere.

The reference `levels` (land and `drF`) is read by default from the chunk `grid.zarr`, about
0.5 MB of network, which also cross-checks against `tile330_grid.zarr`.

**Tests** (`test_load_chunk_levels.py`, 23 offline + 1 network, 13 s offline).
- **The source mimic.** Synthetic zarr-v3 stores are written under `tmp_path` with the real
  layout:
  - one `YYYYMMDDTHH.zarr` per hour, every variable a single zstd object;
  - 51 levels on `k`, and `W` on `k_p1` (52);
  - a `numpy.datetime64` `time`;
  - the group attrs, with the MIT `selected_iteration`;
  - flux attrs that say "+=down" over upward-positive data;
  - a `grid.zarr` whose land block grows with depth, so k = 0, 1, 2 each have their own NaN
    pattern.

  They are read through a local fsspec filesystem, so the real fetch, decode and validate path
  runs. Only `vertical._cat` is monkeypatched (plus `zarr.Array.__setitem__` for the
  mid-append crash).
- **Covered:**
  - the level subset and the `k_p1 → k_l` rename (values equal to source levels 0..2);
  - schema, dtype and chunking (one chunk file per hour), `niter`/`mit_iteration`, `Z`/`Zl`,
    comodo attrs;
  - `drF = [1.0, 1.14, 1.30]`, written once;
  - the sign conversion (values = −source) and every attr;
  - the in-memory return, and `k_max = 1`;
  - **the no-op re-run: sha256 of every file identical, and zero source reads**;
  - resume after a crash between hours (mid-fetch), with hours 0-2 untouched;
  - **resume after a crash inside `to_zarr`**: the store is half-written, `repaired == 1`,
    the hour is re-pulled, the good chunks are byte-identical, and verify passes;
  - clobber and the order errors;
  - **retry on a corrupt read**, in 4 variants: a truncated zstd stream, zero bytes, a valid
    stream of the wrong size, and a valid stream with a finite value on land. Each costs
    exactly one backoff sleep, and the store is right;
  - **stop-at-gap** on a persistently corrupt `Salt` at hour 2: 2 on disk, 1 failed,
    3 not attempted, and the next run completes;
  - an hour refused for a wrong `time`, a wrong `selected_iteration` (OSN instead of MIT), or a
    source that is already downward-positive;
  - the `Eta` alignment check: it passes against an aligned OSN store and fails against one
    shifted by an hour;
  - `verify_chunk_series` passing, and failing on a gap, an un-negated hour, finite-on-land at
    k = 2 together with NaN-on-ocean in `W(k_l = 1)` (reported per variable with the first bad
    hour), a wrong `drF`, a float64 / missing-variable / wrong-sign-attr schema, and a missing
    store.
- The `network` smoke test, `test_network_one_real_hour`, is marked SLOW (~5 min, 174 MB). It
  was **not run** in phase 1; the detached pull exercises the same path.

**Full offline suite** (`timeout 300 python -m pytest dev/frontogenesis/py/tests -q`):
**127 passed, 3 xfailed, 2 deselected (the two network tests) in 178 s**. That is the previous
104 + 3 xfailed + 1 deselected, plus the 23 new tests and the new network test.

**Dry run** (`python m2_chunk_pull.py --dry-run`): the store is absent, with 0 of 72 present;
the grid and OSN stores are present.

**Live pre-flight (~3 MB, before the launch).**
- `_load_levels` against Nautilus with `local_grid`: 51 levels, 720×720,
  `drF[0..2] = [1.0, 1.14, 1.30]`, `Z = [−0.5, −1.57, −2.79]`, `Zl = [0, −1.0, −2.14]`,
  land fraction 0.3116 at k = 0, 1, 2. `hFacC[0]`, `XC` and `YC` equal `tile330_grid.zarr`.
- At 07-02 T00: `time` and `selected_date_utc` are correct. Raw `oceQsw` spans −454 to
  −252 W m⁻², upward-positive as expected. `Eta` is bit-identical to OSN.

**First launch: stopped at hour 0 by a plausibility bound; fixed and relaunched.** The first
launch was at 16:49:32 PDT, PID 73437.
- `Theta` arrived at 0.52 MB/s.
- `Salt` was then refused 3 times: `values [28.44, 48.42] outside plausible (0, 45)`. The run
  stopped at the gap as designed, `status=failed`, and wrote its done-file (kept as
  `data/m2_chunk_pull_done_run0_salt_bound.json`).
- The value is real, not corrupt. The OSN 72-hour `Salt` maximum is **48.64 psu**, at
  −115.74 E, 32.54 N (the northern Gulf of California, hypersaline), with 32 cells above 45 at
  hour 0.
- The bounds were widened, to `Salt (0, 70)` and `Theta (−3, 45)`, with a comment. The
  bounds are meant to catch garbage, not physics. That cost 7 min. It also shows the
  retry → stop-at-gap → done-file path working on real data.

**Launch** (16:57:12 PDT, 2026-10-03):
```
cd dev/frontogenesis/py && nohup ~/miniforge3/envs/frontogenesis/bin/python m2_chunk_pull.py > ../data/m2_chunk_pull.nohup 2>&1 &
```
**PID 86425** (host MacBook-Pro-4.local). The log is `data/m2_chunk_pull.log`, appended, and
also holds the first launch's lines above `pid 86425`. The done-file is
`data/m2_chunk_pull_done.json`.

**First hour** (no retries, no repairs, no failures in this launch):

| object / hour | size | wall | rate |
|---|---|---|---|
| `20120702T00/Theta` | 57.2 MB | 106 s | 0.54 MB/s |
| `20120702T00/Salt` | 47.5 MB | 86 s | 0.55 MB/s |
| `20120702T00/W` | 65.4 MB | 127 s | 0.51 MB/s |
| **hour 2012-07-02 00:00** (+ fluxes, Eta, time, attrs; append) | **175 MB fetched** | **333.0 s** | |

That is the rate task 4 measured, 0.55 MB/s, and the link is the limit. The read-only sanity
check of the stored hour 0 found:
- dims `k = k_l = 3`, 720×720;
- `niter` 1022976, `mit_iteration` 1012608;
- `drF` [1.0, 1.14, 1.30], `Z` [−0.5, −1.57, −2.79];
- **`Theta`/`Salt` k=0 and `W(k_l=0)` bit-identical to the OSN store**;
- stored `oceQsw` 252 to 454 W m⁻², positive downward at 17 LST;
- tile means: `oceQnet` +240 W m⁻² (ocean warming), `oceFWflx` −2.2e-5 (net evaporation, now
  negative downward).

The store is **14 MB per hour on disk**, so about **1.0 GB** for 72 hours.
**ETA:** 72 × ~333 s ≈ 6.7 h from 16:57, i.e. **~23:40 PDT 2026-10-03**. The script's own
estimate after hour 0 is 394 min. The main session watches for `data/m2_chunk_pull_done.json`.

**Deviations and notes.**
1. **`vertical.py` is about 525 lines, over coding §1.3's ~400-line cap.** About half of it is
   docstrings and the validation helpers. It will grow further when M3 adds the physics
   functions. A split (for example `chunk_store.py` for the reader) is better decided in M3,
   when `vertical.py` gets its physics. Flagged and not done now, because the task names
   `vertical.py`.
2. **§3.3 lists "+ staggered" dims.** There are none: the store has no `U`/`V`. It has
   `j`, `i`, `k`, `k_l` and `time`.
3. **Additions beyond §3.3**, all coords or attrs:
   - `mit_iteration(time)`;
   - `Z(k)` and `Zl(k_l)`;
   - `j`/`i` index coords, matching the OSN store;
   - the sign and provenance attrs.
4. **The `Eta` check needs the OSN store.** It is passed by the script, and off by default in
   the function, because the tests use synthetic sources. If OSN lacks an hour, that hour's
   `Eta` check is skipped, not failed.
5. Nautilus served no corrupt bytes during the pre-flight or the first launch. The four
   corrupt-read variants are covered by the offline tests only.

Files so far: created `py/vertical.py`, `py/m2_chunk_pull.py`,
`py/tests/test_load_chunk_levels.py` and `claude_prompts/note_to_lauren_flux_signs.md`;
modified `py/series_verify.py` (appended `verify_chunk_series`) and this log. Data, which is
git-ignored: `data/tile330_chunk_20120702T00_72h.zarr` (growing), `data/m2_chunk_pull.log`,
`data/m2_chunk_pull.nohup` and `data/m2_chunk_pull_done_run0_salt_bound.json`. Not touched:
`osn_tiles.py`, `zarr_series.py`, `m2_pull.py`, every M1 module and test, `pytest.ini`, the
coding and planning docs, and `deck/`. Prompt 3's Status is left for phase 2. Nothing committed.
