# Frontogenesis slides

## Goals

Generate a set of slides to describe the frontogenesis analysis thus far.
We will begin by generating a set of figures and GIFs.

## Context

- See the design docs, `frontogenesis_coding.md` and `frontogenesis_planning.md`.
- See the reports in report/
- See the code in py/
- See example tracked front in `tracked/matched_parcels_track_81.md`

## Code

- Python in the `ocean14` conda environment
- **JXP runs all git, in both repos.**
- When possible reuse existing code 
- Place import commands at the top of the file.
- Generate Python scripts in the py/ directory.
- Have the figures generated appear in `slides/`


## Figures

### Analysis scene

1. Read the Context. Then create a figure of the analysis scene, showing the grid
(label lat and lon) and land.  Indicate the regions avoided (i.e. too close to land).
Show the `G` field, i.e. magnitude of the gradient of buoyancy squared.
I will want one example with the fronts indicated, and one without.
Show the field at the mid-point of the analysis window in time.
If you have any questions, put them in the Q&A/Scene section below.
Name the script `py/slides_scene.py` and the figures `slides/scene_fronts.png` 
and `slides/scene_no_fronts.png`.
Use Opus 5.5. Log your work.

2. See my answers to your Q1 and Q2 questions and modify the script accordingly.
Use Opus 5.5. Log your work.

### Tracking

1. We are going to make a figure and then GIFs to illustrate how well we track
a front.  For the figure, I wish 3 panels covering 3 hours centered at the mid-point
of the analysis window.  In each panel, show the `G` field, and a dotted cyan
curve that indicates all of the parcels of the tracked front. 
For panel 2, indicate show the front curve from panel 1 in dotted red.
For panel 3, indicate show the front curve from panel 2 in dotted red, and the front curve 
from panel 1 in dotted yellow.
The panels should be zoomed in on the front of interest which is `label 81` in `tracked/matched_parcels_track_81.md`.
Name the script `py/slides_tracking.py` and the figure `slides/tracking_81.png`.
If you have any questions, put them in the Q&A/Tracking section below.
Use Opus 5.5. Log your work.

2. Ok, let us make a few modifications after you read my responses to TR-Q1 - Q3:
    - Have panels 1 and 3 show 5 hour offsets from the mid-point of the analysis window, not 1 hour
    - The colored curves should lasso the parcels in the front, as defined in `M4`

## Q&A

### Tracking

**TR-Q1 — "All of the parcels of the tracked front": which pixels?** I drew (a) the pixels of
the label the tracker chose at each pair (the one-pixel-wide skeleton, ~305 px). The
alternative is (b) the matched-parcel band of `matched_parcels_track_81.md` §1.2, i.e. the
label dilated by 1 cell (3 × 3), ~650–900 parcels per pair. That band is the population the
rates are averaged over, but as dots it reads as a ribbon, not a curve.
*Recommendation: (a) for this figure; (b) only if the slide is about the rates.*

>A. I wish a curve that lassoes the parcels in the front, i.e. those in the dilated front

**TR-Q2 — Earlier fronts: as found, or carried by the flow?** As asked, the red and yellow
curves are the earlier fronts **where they were found** (fixed positions). The tracker
actually compares the current label with the previous label *advected* by the flow (the
orange set in `tracked/figs/matched_parcels_track_81_sets_32_39.png`). With a median
displacement of 0.23 cells/h the difference is small over 1–2 h. (a) keep "as found";
(b) advect them to the panel's hour (that needs the `frontogenesis` env, since `tracking.Flow`
imports `fronts`). *Recommendation: (a) for the figure; worth reconsidering for the GIFs,
which span more hours.*

>A. (a) keep "as found"

**TR-Q3 — Colour of `G`.** The scene slides use `cividis`. Here I switched to grey (dark =
strong `G`), because `cividis`'s yellow ridges hid the yellow curve. (a) keep grey here;
(b) use another colour for the 2-h-earlier curve (e.g. magenta) and return to `cividis`
for consistency with the scene. *Recommendation: (a).*

>A. (a) keep grey here

**TR-Q4 — Panel 3 lands just after a split.** At pair 40 the tracker's label (84) has only
111 px, against 281 / 307 at pairs 30 / 35. `matched_parcels_track_81.json` flags a split at
pairs 38 and 40 (and a merge at 39). The tracked piece is the western arm and the northern
loop (27.5–28.3°N, 124.2–126.0°W). The southern north–south arm (~124.1°W, 27.0–27.6°N) is
no longer part of the label, so in panel 3 it carries only the red and yellow lassos.
(a) keep ±5 h: it shows honestly what tracking does through a split; (b) use another
offset, e.g. ±4 h (pair 39: 299 px, but flagged as a merge) or ±6 h (pair 41: 106 px);
(c) keep ±5 h and add a one-line note on the slide. *Recommendation: (c).*

### Scene

**S-Q1 — Which "fronts" go on the scene?** I drew (a) the M4 finder's config-D labels
(`tile330_fronts_D_L0.zarr`, 269 one-pixel-wide fronts at pair 35) — the fronts that are
tracked. Alternatives: (b) the M3 `front` mask (p90 of `G` on `mask_analysis`, 26,293 pixels,
not thinned), which is the population of the M3 budget figures; (c) both, in two colours.
*Recommendation: (a).*  

>A. Yes, we will stick with (a)

**S-Q2 — Highlight the tracked front?** Pair 35 is the anchor of track 81 (the front of
`tracked/matched_parcels_track_81.md`). (a) leave as is; (b) draw label 81 in a second colour
(and/or an inset zoom) on `scene_fronts.png`, to set up later slides on the tracked front;
(c) a third figure for that. *Recommendation: (b) or (c), if the deck will include track 81.*

>A. Ok, edit the script to draw a label 81 in a second colour.  No inset zoom

## Logs
### 2026-11-19 — Figures / Analysis scene, prompt 1 (Opus 5.5)

**Done.** `py/slides_scene.py` (ocean14 env, ~20 s) writes `slides/scene_no_fronts.png` and
`slides/scene_fronts.png`; it prints a JSON summary of every number below (nothing new is
computed — `G` is the stored field).

- **Time.** The analysis window is 72 model hours, 2012-07-02 00 UTC to 07-04 23 UTC, i.e. 71
  hourly pairs with `time_mid` 07-02 00:30 … 07-04 22:30. Its midpoint is **pair 35,
  `time_mid` = 2012-07-03 11:30 UTC** (03:30 LST) — also the anchor pair of tracked front 81.
- **Field.** `G = |∇b|²` from `data/tile330_derived_L0.zarr` (`L = 0`, unfiltered), shown as
  `log10 G` on a fixed scale −17 … −12.5 [s⁻⁴] (`G` p1 / p50 / p99 = 6.0e-18 / 1.3e-15 /
  3.0e-13), `cividis`. Map = `pcolormesh(XC, YC)`, so north-up, east-right; axes in °W / °N.
- **Land.** `~mask_ocean` from `data/tile330_masks.nc`, drawn tan.
- **Avoided regions** (`mask_analysis = mask_ocean & mask_halo & mask_offshore & mask_edge`),
  two classes, each a white/blue veil:
  - within **100 km of the coast** (`~mask_offshore`, hatched): 83,446 ocean cells. It contains
    the 7-cell (12.6 km) land halo, so the halo is not drawn separately. The circle near
    29°N, 118.3°W is the 100-km exclusion around Guadalupe Island.
  - the **7-cell tile-edge rim** (`~mask_edge`, outside the coastal band; light blue, thin):
    10,506 ocean cells.
  - dashed black line = boundary of `mask_analysis` (262,925 of 356,877 ocean cells retained).
- **Fronts** (`scene_fronts.png` only): the M4 finder product, config D, `L = 0`,
  `data/tile330_fronts_D_L0.zarr`, `labels > 0` at pair 35 — **269 fronts, 8,733 pixels**, all
  inside `mask_analysis`. Drawn as one-pixel red squares (no dilation). See Q&A S-Q1.


### 2026-11-19 — Figures / Analysis scene, prompt 2 (Opus 5.5)

**Done.** Applied S-Q1 (a) and S-Q2 (b, no inset) to `py/slides_scene.py` and regenerated both
figures (`scene_no_fronts.png` is unchanged).

- Fronts stay the config-D labels at pair 35 (S-Q1 (a)); no change.
- `scene_fronts.png` now draws the **tracked front, label 81 at pair 35, in cyan** (`TRACK_COL
  = '#00e5ff'`, markers ~2× the other fronts' so it reads on a slide). It is the other 268 fronts
  that are red. Legend: "tracked front (label 81, 307 pixels)"; the total stays 269 fronts /
  8,733 pixels, label 81 included.
- Label 81 at pair 35 is the anchor of M4 track 0, checked against
  `data/matched_parcels_track_81.json` (`anchor = {label 81, step 35}`, `track_index 0`). It
  lies near 27–28.5°N, 124–126°W, inside the analysis region. Its 307 pixels fall within the
  track's 104 / 178 / 413 (min / median / max) range in `tracked/matched_parcels_track_81.md`.
- New module constants `TRACK_LABEL = 81` and `TRACK_COL`, plus a `track_label` argument on
  `draw_scene`, which raises if the label is absent at the pair. The printed summary now
  includes `track_label` and `n_track_pixels`.

### 2026-11-19 — Figures / Tracking, prompt 1 (Opus 5.5)

**Done.** `py/slides_tracking.py` (ocean14 env, ~15 s) writes `slides/tracking_81.png` and
prints a JSON summary of the numbers below. Questions TR-Q1..Q3 are in Q&A/Tracking.

- **Pairs.** Three hours centred on the window midpoint: pairs 34, 35, 36, `time_mid`
  2012-07-03 10:30 / 11:30 / 12:30 UTC.
- **The tracked front** is M4 flow track 0 (`data/m4_tracks.json`, `flow[0]`, anchor label 81
  at pair 35; the script asserts this). The chosen labels are **91 / 81 / 83** with
  **304 / 307 / 307 px**. Labels are renumbered every pair, so the ID changes while the front
  is the same.
- **Panels.** Each panel shows `log10 G` (`L = 0`, grey, dark = strong) with the panel's own
  front as cyan dots on the cell centres. Panel 2 adds panel 1's front in red; panel 3 adds
  panel 2's front in red and panel 1's in yellow. The earlier fronts are drawn where they were
  **found**, not advected (TR-Q2). Older fronts use larger dots and sit underneath, so a pixel
  shared by several fronts shows as nested dots and no curve hides another.
- **Same cell as the previous hour:** 219 of 307 px at pair 35, and 251 of 307 at pair 36.
  These are exact-cell counts, with no tolerance and no advection. The red and yellow that
  show past the cyan are the front's real hour-to-hour changes, mostly at the northern tip
  (~28.5°N, 124.2°W) and a few kinks.
- **Zoom.** A fixed crop holds all three fronts with 12 cells of margin (j 86–215, i 592–715,
  ~26.8–28.9°N, 123.6–126.1°W). The colour limits are p2 / p99.5 of `log10 G` in the crop
  (−16.7 / −12.7), and the top is stretched by +0.6 dex so the ridges stay mid-to-dark grey.
- **Environment note.** `tracking.py` imports `fronts`, which (with `skan`) is not installed in
  ocean14. The script therefore reads the track from the JSON and the labels from the zarr
  directly, and reuses only `figures.hour` and `slides_scene`'s constants.

### 2026-11-19 — Figures / Tracking, prompt 2 (Opus 5.5)

**Done.** I applied TR-Q1 (lasso the dilated front), TR-Q2 (a) (as found) and TR-Q3 (a)
(grey) to `py/slides_tracking.py` and regenerated `slides/tracking_81.png`. One new question,
TR-Q4, is in Q&A/Tracking.

- **Pairs:** the window midpoint and **±5 h**, i.e. pairs **30 / 35 / 40** (`time_mid`
  2012-07-03 06:30 / 11:30 / 16:30 UTC). The module constant is `OFFSET = 5` (it was
  `HALF = 1`). Track-0 labels **90 / 81 / 84**, with 281 / 307 / 111 px.
- **Lasso = the M4 parcels.** Each front's parcel set is the chosen label dilated by
  `BAND = 1` cell with a 3 × 3 (8-connected) structuring element, the construction of
  `tracking.matched_link` and `front_strength_series` (M4-Q3 (a)). The curve is the 0.5
  contour of that band on the cell-centre grid, drawn dotted, so it runs half a cell outside
  the outermost parcels. The parcel counts **1039 / 1119 / 429** match the `n_parcels` of
  steps 30 / 35 / 40 in `data/matched_parcels_track_81.json` exactly.
- **Colours** as in prompt 1: cyan for this panel, red for the previous panel (5 h earlier),
  yellow for the first panel (10 h earlier). All lassos are drawn where they were found.
  Parcels shared between bands (exact cells): pair 35 ∩ pair 30 = 807 of 1119; pair 40 ∩
  pair 35 = 350 and pair 40 ∩ pair 30 = 294 of 429.
- **Panel 3** falls just after a split: the label at pair 40 is the western arm and the
  northern loop only (27.5–28.3°N; flagged split at pairs 38 and 40, merge at 39). The southern
  arm carries only red and yellow lassos (TR-Q4).
- **Zoom** is recomputed on the union of the three bands with a 12-cell margin (j 84–216,
  i 595–716). The `log10 G` limits are unchanged at −16.7 / −12.7, with the +0.6 dex stretch.
