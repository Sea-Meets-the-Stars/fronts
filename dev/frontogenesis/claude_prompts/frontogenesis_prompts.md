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
