# Frontogenesis execution prompt 5 — M4: Fronts and flow-informed tracking

**Milestone:** M4 (`frontogenesis_coding.md` §6).
**Prerequisite:** M3 passed (budget closure). Do not start otherwise — if the per-pixel budget
does not close, chasing the same question in object space will only obscure why.
**Goal:** the per-front result — how much individual fronts actually sharpen, against what the
strain field predicted — with tracking that genuinely follows the fluid.

**Status 2026-10-26 — task 1 done, on JXP's instruction to start M4 although M3 is NOT closed**
(the prerequisite above is not met: M3 task 8's audit, 2026-10-11, failed criterion 1 at every
gated `L`; JXP chose to proceed, and task 1 is front *finding*, which needs no closure).
`py/tracking.py` (`find_fronts`, `find_fronts_series`, `write_fronts_series` /
`open_fronts_series`, `labels_at_factory`, the two timestamp converters) + `py/m4_find.py`
(runner) + `tests/test_tracking.py` (11 tests; suite **269 passed + 3 xfails**, 2 deselected,
269 s). **The `tile_find` path named in task 1 does not exist on this branch** (M1 task 7a,
coding §2.5): the entry point is `fronts_from_gradb2` with config D **read from the YAML**
(window 64, 85th percentile, 'pool', sharpen, despur `Lspur` 10, `min_size` 7, connectivity
2) plus the three task-7a caller-side rules; the mask is applied as NaN. Output
`data/tile330_fronts_D_L0.zarr` — **71 label arrays** (config D on the derived store's
midpoint `G` at `L = 0`, `mask_analysis`), 248-311 fronts and 8,545-9,115 front pixels per
pair, 0 pixels off the finite masked field, 0.60-0.66 of them on M3's `G >= p90` front
pixels, 10-18 fronts >= 100 px per pair, 551 s at 8 workers. **Open for JXP (M4-Q1):** the
mask — `mask_analysis` (the M3 pool, used) excludes the coastal 100 km, where `mask_halo`
finds a 1,179-px front at pair 0 that is the tile's largest by 3x. **Blocker for task 2:**
`fronts/front_tracking.py` (`follow`, `score_candidate`, `parse_time`) exists only on
`origin/viz_tools` (commit e04d4f2), not on `frontogenesis`. Log entry "Execution prompt 5,
task 1".

---

## Why tracking has to be flow-informed (decision Q14)

`front_tracking.follow()` predicts the next position by extrapolating the **centroid** velocity
from the last two sightings. But a front whose centroid moves because it grew asymmetrically is
not a front that moved with the fluid. And the consequence is not cosmetic:

> **If `follow()` links a front at `t` to a different physical front at `t+dt`, the per-front
> `d(front-mean G)/dt` is not a material derivative, and comparing it to `integral 2F dt`
> compares nothing.**

Since a buoyancy front is advected by the flow and we already have the departure-point
machinery from M1, we can do better.

## Tasks

### 1. Front finding per hour

`tracking.find_fronts_series(...)`, wrapping the `tile_find` path
(`build_v5.py` step 1 -> `build.tile_find` -> `preproc/gradb2.generate_tile_gradb2`) with finding
config `D`, producing a label array per timestep. The finding
chain is already NaN-safe — `pyboa.front_thresh` uses `nanpercentile` with `cval=np.nan`, so NaN
never becomes a front and local thresholds are built from ocean values only. That is exactly why
the halo mask must leave land as **NaN** rather than zero: zeros would drag the local-percentile
window down near the coast and manufacture coastal fronts.

**Read `fronts/finding/configs/finding_config_D.yaml` for the actual window and percentile** —
do not assume them, and note they differ from `fronts_from_gradb2`'s bare defaults.

### 2. Flow-informed tracking — `tracking.py`

```python
def advect_mask(mask_bool, u_c, v_c, grid_ds, dt=3600.0)   -> np.ndarray
def flow_weighted_score(...)                                -> (float, dict)
def track_top_n(labels_by_time, times, n=10, flow=None)     -> list[Track]
def tracking_quality(tracks, flow)                          -> DataFrame
def front_strength_series(track, G_by_time, two_F_by_time,
                          matched_pixels=True)              -> DataFrame
```

- **Advect the boolean mask, never the label field.** Push the mask through `semilag` as a float
  and threshold at 0.5. Labels stay integers and are never interpolated — that is what makes
  this tractable at all.
- Feed `IoU(flow-predicted mask, candidate label)` into `score_candidate` via its **existing
  `weights` dict**. This is additive, not a rewrite.
- **Timestamp trap:** `front_tracking.parse_time` wants `'%Y-%m-%dT%H_%M_%S'` (underscores);
  `dbof` uses `'%Y-%m-%d %H:%M:%S'`. Two formats are live in this project. Convert here.
- `N = 10` largest fronts, looped `follow()`.

### 3. The reconciliation, done correctly

Even perfect flow-following tracking is **not** enough. Front-mean `G` is a mean over a
**changing pixel set** — fronts lengthen, split and merge — so `d/dt` of that mean carries an
extra term from the set's own evolution.

**So reconcile on the advected pixel set (Lagrangian-matched pixels).** Comparing "pixels
labelled front at `t`" against "pixels labelled front at `t+dt`" does *not* satisfy the
acceptance criterion, even though it will look superficially similar.

### 4. Diagnostics that are free, and worth having

- **Tracking quality:** the distribution of (`follow()`-chosen displacement − flow-predicted
  displacement). If these disagree often, the track is not following the fluid and Phase 3 is not
  interpretable. Better to discover that here than in the writeup.
- **Split/merge detection:** the flow-predicted mask overlapping two candidate labels.
  `follow()` has no notion of topology change, and over 72 hours it will certainly happen.

### 5. Front strength

- **Primary:** front-mean `G` via `colocate_fronts_with_properties`. Pass
  **`nan_policy='omit'`** — the default is `'propagate'` and our fields are NaN over land and
  halo. Primary because it is the *same quantity* as the Phase-2 per-pixel measure, which is what
  makes the reconciliation meaningful.
- **Secondary:** cross-front `delta b` and width, from `curtains.perpendicular_path` and
  `path_metrics`. These do not exist yet and are new work.

## Figures

8 (tracked-front case study: 4-6 panel time sequence, observed strength with the `F`-predicted
curve overlaid), 9 (population statistics over the 10 fronts — frontogenetic vs frontolytic
fractions, lifetime vs mean `F`), plus the tracking-quality diagnostic.

---

## Acceptance criteria

1. 10 tracks spanning **>= 12 h each**.
2. **Flow-informed scoring actually in use** — mask advection feeding `score_candidate`'s
   weights, not merely implemented and bypassed.
3. `tracking_quality` reported, with split/merge flags.
4. **Reconciliation on Lagrangian-matched pixels:** per-front measured `d(front-mean G)/dt`
   equals the advected-pixel-set average of the Phase-2 per-pixel `DGDt` to a stated tolerance.
5. **Figures 8 and 9 plus the tracking-quality diagnostic** written. Bootstrap here is over
   **frontal features and hours** (M3 used spatial blocks, since objects did not exist yet).

## Do not

- Do not interpolate the integer label field.
- Do not reconcile on endpoint labels and call criterion 4 satisfied.
- Do not quietly drop fronts that split. Flag them — a split is physics, not a nuisance.

## Log

Number of tracks and their durations; the tracking-quality distribution; how many splits/merges
were flagged; the reconciliation tolerance achieved; and whether flow-informed scoring changed
which fronts were linked compared with plain `follow()` — **that comparison is itself a result**
and is the direct answer to Lauren's question.

## Q&A

### Task 1 → task 2

#### Claude, 2026-10-26 (after task 1)

Two questions before task 2. Every number is from the task-1 log entry ("Execution prompt 5,
task 1") and `data/m4_find_done.json`; nothing was recomputed. Each has a recommendation.

**M4-Q1 — Which mask are fronts found on?** Task 1 ran config D on `mask_analysis` (the M3
pool: ocean minus the 7-cell halo, >= 100 km offshore, inside the 7-cell edge margin; 262,925
cells), applied as NaN. The alternative is `mask_halo` (ocean minus the halo only), which adds
the coastal 100 km. At pair 0 the two give 311 fronts / 9,115 px against 384 / 12,718 px, and
on the analysis cells themselves they agree to 9,079 vs 9,115 px. The difference is the coastal
zone, which holds **one 1,179-px front** — three times the largest front on the analysis mask
(375 px) and the best tracking subject on the tile — but the budget's `valid` pool never
covered it, so `DGDt` there is not among M3's judged pixels.
- (a) `mask_analysis`, as run: criterion 4's reconciliation is then against pixels M3 actually
  evaluated. **Recommended.**
- (b) `mask_halo`: the coastal front is tracked, but criterion 4 reconciles partly on pixels
  outside M3's pool.
- (c) both, (a) primary and (b) a declared sensitivity. Switching is `m4_find.py --mask halo`
  and 9 minutes.

>A. Yes, use (a)

**M4-Q2 — Where does `front_tracking.py` come from?** Task 2 needs `follow`,
`score_candidate`, `parse_time` and `Track` from `fronts/front_tracking.py`, which is **not on
the `frontogenesis` branch**: it exists only on `origin/viz_tools` (commit e04d4f2, Lauren,
2026-08-24: `fronts/front_tracking.py`, 607 lines, with `docs/viz/apps/front_tracking.md`).
That commit also touches `fronts/viz/apps/*` and `fronts/tests/test_viz_apps.py`. A change to
the `fronts` package outside `dev/frontogenesis/` is yours to make, so nothing was done.
- (a) copy the one file: `git show origin/viz_tools:fronts/front_tracking.py >
  fronts/front_tracking.py` (plus its doc page), no other change. **Recommended** — smallest
  footprint, and task 2 only wraps it.
- (b) cherry-pick e04d4f2 onto `frontogenesis` (brings the viz-app changes and their tests).
- (c) merge `origin/viz_tools` into `frontogenesis`.
>A. I have merged in `viz_tools`.  Please use the files you need from it

#### JXP

