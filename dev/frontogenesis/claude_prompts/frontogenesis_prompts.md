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
