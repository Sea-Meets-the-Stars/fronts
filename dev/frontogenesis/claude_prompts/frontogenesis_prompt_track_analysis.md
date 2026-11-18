# Frontogenesis prompt — track analysis: do changes in a front's pixels affect the tracked-front results?

**Milestone:** a follow-up to M4 (`frontogenesis_coding.md` §6), after M5's report. It does not
reopen any M3 or M4 criterion.
**Prerequisites:** M4 complete (prompt 5, tasks 1–5, 2026-10-26 to 2026-10-28); the report
`report/frontogenesis_report.md` (2026-11-18), §3.6 and §9.
**Written 2026-11-18 (Fable 5.2), prompt 6 task 4,** as a starting document for JXP to edit:
it holds the context and the pointers to the code, and **no tasks yet**. Every number below is
from the prompt-5 log entries (`frontogenesis_prompts.md`) or the M4 JSON summaries in `data/`;
no store was opened to write it. Numbers marked **(new)** were tabulated for this document from
`data/m4_strength.json` and `data/m4_quality.json` and appear in no earlier log entry.

**The concern (JXP, prompt 6 task 4).** Some of the results for the tracked fronts may be
affected by changes in the pixels allocated to a given front between timesteps. Such changes
have two possible origins: the flow-informed tracking (which label is chosen at each hour), and
real changes in the front (lengthening, splitting, merging, interaction with other features).
This document sets out what the M4 analysis did about that, what it left open, and where the
code is.

---

## 1. Context: what was done

### 1.1 The setting

- LLC4320, tile 330 (California Current), 72 hourly snapshots from 2012-07-02 00:00 UTC, which
  give **71 pairs** (steps 0–70), each with fields at its midpoint time (`…T00_30_00`, …).
  Everything in M4 is at filter scale **`L = 0`**, on **`mask_analysis`** (at least 100 km
  offshore, land halo and tile-edge margin removed), grid spacing 1.873 km per cell.
- `G = |∇b|²` is the front strength; `2F` is the kinematic frontogenesis prediction for
  `D_hG/Dt`; `DGDt_semilag` is the measured (semi-Lagrangian) tendency stored by M3. The M3
  budget did not close (criterion 1 failed at every `L`), so **no ratio or slope of measured to
  predicted is an efficiency** — that rule carries over to anything done here.

### 1.2 The pipeline, step by step

1. **Finding** (prompt 5 task 1). The `fronts` package's finder in configuration D (local 85th
   percentile in a 64-cell window, sharpened, de-spurred, minimum size 7, 8-connectivity) on
   `G` at each of the 71 midpoints. Fronts are **one-pixel-wide skeletons**, labelled
   independently at every hour: a label number means nothing from one hour to the next. Per
   hour: 248–311 fronts (median 275), 8,545–9,115 front pixels, median front size 16 px.
2. **Tracking** (task 2). The ten largest fronts at the middle step (35, 2012-07-03 11:30 UTC;
   121–307 px) are followed forwards and backwards, one link per hour. Two trackers were run on
   the same anchors:
   - *plain*: `fronts.front_tracking.follow`, which predicts the next position by extrapolating
     the centroid at constant velocity;
   - *flow-informed*: `tracking.follow_flow`, which advects the last-found boolean mask with the
     model's surface flow, builds the position term from that advected mask, and adds
     `1 − IoU(advected mask, candidate)` (on 1-cell dilated masks) to the score with weight 1.0.
   At each hour the tracker picks **one label**. After a split it follows one piece; after a
   merge the chosen label contains pixels that belonged to another front the hour before.
3. **Reconciliation on matched parcels** (task 3). Every pixel of the chosen label at step `t`
   is treated as a parcel and carried by the flow. The **matched rate** is the change of mean
   `G` along those parcels; it is compared with `DGDt_semilag` and with `2F` averaged on the
   same parcels. The construction JXP chose (M4-Q3 (a)) is **band 1 + centred**: the label
   dilated by one cell (3 × 3), parcels seeded at step `t` and sampled at `t − 1` and `t + 1`.
4. **Tracking quality** (task 4). Per link, the chosen label is compared with the previous mask
   advected by the flow; splits and merges are flagged.
5. **Strength series and population** (task 5). Per track and per step: the label mean of `G`
   on the band, the matched rates, four curves integrated from the anchor, the cross-front
   buoyancy step and width; then statistics pooled over the ten fronts (Figures 8 and 9).

### 1.3 Results that bear on the concern

**Tracks.** All ten flow-informed tracks are found at 71 of 71 steps (70 h, no gaps).

**The tracker's choice matters on 3 of 10 tracks.** Of the 649 links both trackers made, 71
differ (tracks 0, 1, 5: 35, 2, 34 links), and the flow-informed tracker made 51 links the plain
one left as gaps (22 and 29 on tracks 1 and 5). At every differing link the flow-informed
tracker chose the larger front (median 159 px against 38 px). Mechanism on track 0: the 295-px
front splits at step 38 into 110-px and 178-px pieces; the plain tracker takes the small piece,
its centroid jumps 25 cells, and from step 39 it sits on 40–61-px fragments 40–50 cells away.
Seven tracks are identical link for link under the two trackers.

**The advection itself adds little at hourly cadence.** Front pixels move a median 0.35 cells
per hour. The advected mask overlaps the next hour's fronts slightly less well than no
advection (dilated IoU 0.759 against 0.774; better on 0 of 70 transitions). The gain is a
prediction that survives a split, not a better hourly overlap.

**Splits and merges are common.** 168 of 700 links are flagged by task 4's test — 93 splits
(the advected previous mask overlaps two or more labels) and 85 merges (the chosen label
advected back overlaps two or more), each needing at least 5 overlapping pixels. On the 532
unflagged links the chosen and flow-predicted centroids differ by 1.63 cells in the median; on
the 168 flagged links by 15.5. Mask to mask, the chosen label lies within one cell of the
prediction on 72 % of links, and is the prediction's majority overlap on 99.0 %.

**The label mean is not a material quantity.** The hour-to-hour change of a front's own mean
`G` (`r_label`) differs from the matched rate by a **set-evolution term 5.2–7.0 times the
matched rate in r.m.s.** (5.44 on band 1 + centred), and correlates with the matched rate at
0.04. On track 0 the label mean jumps by factors of two at splits and merges while the
material curve does not (Figure 8).

**The matched rate reconciles with the stored tendency on the band.** Band 1 + centred:
correlation 0.953, r.m.s. ratio 0.415 over 690 links, against the pre-declared 0.8 and 0.5;
5 of 10 tracks pass individually. The pre-registered construction (1-px skeleton, forward
difference) failed (−0.093, 1.123) — a registration artefact on one-cell ridges, reproduced on
a synthetic ridge. Links flagged as split or merge reconcile no worse (0.917 / 0.483 on 100
flagged links against 0.958 / 0.403 on the rest).

**Population (690 links, band 1 + centred).** Fraction of hours frontogenetic: 0.64
[0.50, 0.77] by the matched rate, 0.81 [0.75, 0.88] by `2F`; sign agreement 0.78 [0.67, 0.86].
`∫2F dt > 0` on 10 of 10 fronts; material `ΔG > 0` on 7. Hourly matched rate against `2F`:
correlation 0.951, amplitude ratio 1.008 — a consistency check, not an efficiency. Per front,
material `ΔG` over `∫2F dt` is 0.35, 0.96, 0.51, −0.84, 0.86, −0.11, 0.72, −0.45, 0.36, 1.10 —
not efficiencies either.

**Per track (new).** Pixel counts are the undilated label size (`npix`); "jumps" counts links
where the size changes by more than 25 % of the earlier hour's; `ΔG` in units of 10⁻¹⁴ s⁻⁴
over the window.

| track | anchor label | npix min / median / max | jumps > 25 % | splits / merges (task 4) | flagged links (task 4) | task-2 flag | links differing from plain (+ gaps filled) | `ΔG` label mean | `ΔG` material | `∫2F dt` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 81 | 104 / 178 / 413 | 15 | 9 / 10 | 18 | 9 | 35 | +0.56 | +2.41 | +6.92 |
| 1 | 84 | 44 / 141 / 328 | 13 | 14 / 7 | 20 | 14 | 2 (+22) | +34.4 | +26.8 | +27.9 |
| 2 | 202 | 133 / 238 / 311 | 12 | 14 / 11 | 24 | 14 | 0 | +1.47 | +4.00 | +7.91 |
| 3 | 100 | 188 / 227 / 303 | 0 | 9 / 10 | 19 | 10 | 0 | +0.38 | −2.04 | +2.44 |
| 4 | 187 | 169 / 286 / 437 | 8 | 15 / 18 | 30 | 16 | 0 | +0.26 | +1.27 | +1.48 |
| 5 | 5 | 79 / 212 / 267 | 11 | 14 / 14 | 27 | 15 | 34 (+29) | −4.26 | −0.28 | +2.66 |
| 6 | 96 | 106 / 151 / 250 | 4 | 2 / 3 | 5 | 4 | 0 | −1.66 | +1.82 | +2.52 |
| 7 | 133 | 74 / 113 / 218 | 4 | 4 / 3 | 7 | 7 | 0 | −0.27 | −0.17 | +0.37 |
| 8 | 161 | 98 / 145 / 321 | 10 | 11 / 9 | 17 | 12 | 0 | −0.76 | +1.26 | +3.51 |
| 9 | 218 | 61 / 115 / 127 | 1 | 1 / 0 | 1 | 1 | 0 | +0.27 | +0.86 | +0.78 |

Pooled over the 700 links (new): the label size changes by more than 25 % on 78; the median
hourly change of `log npix` is 0.10 on links adjacent to a task-2 flag and 0.03 elsewhere;
`|Δ log G|` of the label mean correlates with `|Δ log npix|` at 0.69. So the label mean moves
with the pixel count, which is the set-evolution term seen directly.

### 1.4 What the M4 analysis does *not* settle

These are the places where a change of pixel set can still enter the results. They are stated
as open, not as findings.

1. **The parcels are re-seeded every hour.** The matched rate at step `t` is material over one
   or two hours, but its parcels are the pixels of *that hour's* chosen label. The material
   curve `G_matched` chains these hourly rates by the trapezoid rule from the anchor. It is
   therefore a chain of short material rates over a changing parcel population, not the
   history of a fixed set of water parcels. After a split, merge or change of chosen piece, the
   rate is averaged over a different population. M3's followed-parcel study (report §5.3)
   showed a fixed parcel set leaves the front within about a day, so a fixed set is not the
   front either.
2. **Every hourly statistic inherits the hour's pixel set.** The frontogenetic fractions, the
   sign agreement, the 0.951 / 1.008 comparison and the per-front integrals are all means over
   the label of the hour. Whether they change when flagged hours are removed, or when the set
   is held fixed, has not been tested.
3. **"Flagged links reconcile no worse" used the narrower flag.** The reconciliation and the
   strength series carry task 2's flag (the advected mask overlaps two or more labels; 102
   links in the tracker's own count, 100–101 in the reconciliation tables), not task 4's
   split-or-merge flag (168 links, with a 5-pixel minimum and merges detected backwards).
   Figure 8's shaded hours are the task-2 flag too. The comparison has not been redone with
   the 168.
4. **Which piece is followed after a split is the tracker's decision.** The quality diagnostic
   is per link, against the previous found mask, so a tracker that follows the wrong piece
   consistently looks fine to it. The only cross-check is the plain-against-flow comparison.
   The score weights (flow term 1.0, dilation 1 cell) and the anchor (the ten largest at
   step 35) were choices, not tuned and not varied.
5. **Real change and tracking change are not separated.** A size jump may be a real
   lengthening, a threshold effect of the finder (a local 85th percentile, so a pixel can enter
   or leave as its neighbourhood changes), a split or merge, or a switch of chosen piece. No
   attribution of the 78 jumps (or the 168 flags) to these causes exists.
6. **The whole object is never followed.** After a split the unchosen piece is dropped; after
   a merge the incoming front's earlier history is ignored. No statistic has been computed on
   the union of pieces (the front's "family").
7. **Two band definitions coexist.** The label mean uses the `fronts` package's co-location
   with radius 1, which is a 4-connected band; the matched rates use the 3 × 3 (8-connected)
   band. The first is 1.127 times the second in the median. Both are in every row
   (`G_label`, `G_band8`).
8. **One discrepancy to check.** The task-5 log entry says the label-mean `ΔG` disagrees in
   sign with the material `ΔG` on 4 of 10 fronts and names labels 5, 96, 161 and 100.
   `m4_strength.json` gives both as negative for label 5 (−4.26 and −0.28), i.e. 3 of 10. The
   report does not quote this count.

---

## 2. Pointers: code, products, figures, tests

Paths are relative to `dev/frontogenesis/`. Run everything with the absolute interpreter
`/home/xavier/miniconda3/envs/frontogenesis/bin/python` from `py/` (`conda run -n
frontogenesis` silently uses another environment on this workstation).

### 2.1 Library — `py/tracking.py` (1,328 lines; coding doc §4.10)

| What | Function or object | Lines |
|---|---|---|
| Finding, one hour / all hours | `find_fronts`, `find_fronts_series` | 130, 174 |
| Label store I/O (dict keyed by midpoint time → `int32` label array) | `write_fronts_series`, `open_fronts_series`, `fronts_path` | 229, 267, 164 |
| Advecting a boolean mask (bilinear, threshold 0.5; labels are never interpolated) | `advect_mask_by`, `advect_mask` | 338, 352 |
| The hourly flow as displacement fields, cached in a store | `class Flow` (`displacement`, `advect`, `precompute`) | 363 |
| Dilated IoU and the score with the flow term | `iou_dilated`, `flow_weighted_score`, `FLOW_WEIGHTS` | 474, 498 |
| The flow-informed walk; the ten-largest driver | `follow_flow`, `track_top_n`, `largest_labels` | 521, 599, 591 |
| Track tables, plain-vs-flow comparison, JSON round trip | `track_table`, `compare_tracks`, `tracks_to_json`, `tracks_from_json` | 632–683 |
| Parcels: sampling and the forward map | `sample_at`, `forward_positions`, `class DerivedFields` | 732, 748, 764 |
| **One link: matched rate, stored rate, terms, `r_mask`, `r_label`, `set_evolution`** | `matched_link(…, centred=, band=)` | 803 |
| A whole track; pooled and per-track statistics against the tolerance | `reconcile_track`, `reconcile_summary` | 882, 916 |
| **Per-link quality, split and merge detection** | `tracking_quality`, `quality_summary`, `SPLIT_MIN_PIXELS` | 1001, 1048 |
| Label mean on the band; cross-front step and width | `label_mean`, `cross_front` | 1110, 1126 |
| **Per-step strength series and the integrated curves** | `front_strength_series`, `_integrate_from_anchor` | 1198, 1180 |
| Bootstrap over fronts and 6-h blocks; pooled summary | `bootstrap_fronts_hours`, `population_summary` | 1256, 1287 |

The `fronts` package is imported and never edited: `fronts/front_tracking.py` (`follow` at
line 505, `score_candidate` 416, `describe`, `Track`, `Link`, `Anchor`),
`fronts/properties/colocation.py` (`colocate_fronts_with_properties`), `fronts/viz/curtains.py`.
A `Track` holds `anchor`, `labels[step]`, `centres[step]` and `links[step]`; each link's
`terms` dict carries the score terms plus `_n_overlap` (task 2's flag when ≥ 2), `_n_pred` and
the predicted and chosen centroids.

Supporting modules: `py/semilag.py` (`departure_index`, `interp_to_departure` — the trajectory
integrator), `py/budget.py` (`derived_path(L)`), `py/m3_case.py` and `py/m3_parcels.py` (the
followed-parcel study of report §5.3 — the precedent for holding a parcel set fixed).

### 2.2 Runners and their products

| Runner (`py/`) | Wall time as logged | Writes (`data/`) |
|---|---|---|
| `m4_find.py` | 551 s | `tile330_fronts_D_L0.zarr` (`labels(time, j, i)`), `m4_find_done.json` |
| `m4_track.py` (`--n`, `--anchor-step`, `--flow-weight`) | 1,481 s | `tile330_flow.zarr` (`di_fwd`, `dj_fwd`, `di_bwd`, `dj_bwd` on `(transition, j, i)`), `m4_tracks.json` (both trackers, every link's terms), `m4_track_summary.json` |
| `m4_recon.py` | 1,276 s | `m4_recon.json` — four variants (`skeleton_forward`, `skeleton_centred`, `band1_forward`, `band1_centred`), each with `summary` and one row per link |
| `m4_quality.py` | 221 s | `m4_quality.json` — `flow` and `plain`, each with `summary` (incl. `per_track` split and merge steps) and one row per link |
| `m4_strength.py` (`--case k` picks Figure 8's track) | 729 s | `m4_strength.json` — `series[track]` (one row per step), `summary.per_front`, `summary.pooled`, `fig08`, `fig09` |

Row contents most relevant here:
- `m4_recon.json` rows: `n_parcels`, `n_alive`, `G_mean_t`, `G_mean_tp1_matched`,
  `G_mean_tm1_matched`, `r_matched`, `r_store`, `r_forward`, `two_F`, `subfilter`, `vertical`,
  `surface_flux`, `residual`, `n_mask`, `r_mask`, `n_label_tp1`, `r_label`, `set_evolution`,
  `split_merge` (task-2 flag).
- `m4_quality.json` rows: `label`, `prev_label`, `size`, `prev_size`, `n_pred`, `diff_cells`,
  `mask_dist_median`, `mask_dist_p90`, `flow_iou`, `split`, `merge`, `fwd_overlap_labels`,
  `bwd_overlap_labels`, `chosen_is_majority`, `chosen_overlap_frac`, `score`.
- `m4_strength.json` series rows: `label`, `npix`, `G_label`, `G_band8`, `two_F_label`,
  `r_matched`, `r_store`, `two_F`, the budget terms, `G_matched`, `G_pred`, `G_store`,
  `G_budget`, `delta_b`, `width_km`, `split_merge` (task-2 flag).

Other inputs: `data/tile330_derived_L0.zarr` (`G`, `two_F`, `DGDt_semilag`, `b`, the budget
terms, per pair), `data/tile330_raw_20120702T00_72h.zarr`, `data/tile330_grid.zarr`. `data/`
is git-ignored.

The loading recipe (from `m4_strength.py`, seconds to open):

```python
import json, budget, tracking as tr
from m4_track import FLOW_PATH, TRACKS_PATH
flow   = tr.Flow(store=FLOW_PATH)
labels = tr.open_fronts_series(tr.fronts_path(0))            # {time string: label array}
tracks = tr.tracks_from_json(json.load(open(TRACKS_PATH))['flow'])   # or ['plain']
fields = tr.DerivedFields(budget.derived_path(0))            # fields.get('G', step)
row = tr.matched_link(step, tracks[0].labels[step], tracks[0].labels.get(step + 1),
                      labels, flow, fields, centred=True, band=1)
```

### 2.3 Figures

| Figure | File (`figs/`) | Drawn by (`py/figures_m4.py`) | Registry entry (`figures.FIGURES`) |
|---|---|---|---|
| 8 — one tracked front: maps, strength curves, rates, `Δb`, width | `fig08_front_case.png` | `fig08_front_case` (line 126); from products: `fig08_from_products` (351) | `fig08` |
| 9 — the population: fractions, lifetime, net change, every link | `fig09_population.png` | `fig09_population` (229); `fig09_from_products` (370) | `fig09` |
| Tracking quality: centroid and mask distances, per-track strip with splits and merges, IoU | `fig_tq_tracking_quality.png` | `fig_tracking_quality` (29); `fig_tq_from_products` (377) | `fig_tq` |

Redraw from the products: `python make_figs.py --only fig08,fig09,fig_tq`. `make_figs.py
--recompute` re-runs the five M4 runners first (about 80 minutes). Style, the caption footer
(`figures.footer()`, which states `L`, mask, config and band) and `save` live in
`py/figures.py`; every figure function returns the dict of numbers it drew. PNGs are
git-ignored and tracked with `git add -f`. In the report these are Figures 11, 12 and 10.

### 2.4 Tests

`py/tests/test_tracking.py` (32 tests) with fixtures for a moving series, a growing front and a
strength case. The ones closest to this concern:
`test_matched_link_measures_the_material_rate_and_the_label_rate_does_not`,
`test_split_and_merge_flags`, `test_tracking_quality_plain_vs_flow`,
`test_flow_informed_walk_follows_the_fluid_where_plain_follow_cannot`,
`test_front_strength_series_curves`. Suite:
`cd py/tests && /home/xavier/miniconda3/envs/frontogenesis/bin/python -m pytest -q` —
**345 passed + 3 strict xfails**, 2 deselected, about 300 s (2026-11-18).

### 2.5 Where it is written up

- Log entries, `claude_prompts/frontogenesis_prompts.md`: "Execution prompt 5, task 1" to
  "task 5" (2026-10-26 to 2026-10-28), and "Execution prompt 6, task 1" (figure registry).
- `claude_prompts/frontogenesis_prompt_5.md`: the M4 tasks, criteria, and JXP's answers
  M4-Q1..Q4.
- `report/frontogenesis_report.md`: §3.6 (method), §9.1–9.5 (results), §10 (limitations),
  Appendix A (glossary: split, merge, matched rate, set-evolution term).
- `frontogenesis_coding.md` §4.10 (signatures as built), §8 (pitfalls).

---

## 3. Rules that carry over

- No slope or ratio of measured to predicted is quoted as an efficiency (the budget did not
  close). That covers 0.951 / 1.008 and the per-front `ΔG` ratios.
- Criterion 4 is never called met without saying in the same sentence that it failed as
  pre-registered and on which construction it was met.
- A mean over the label at `t` against the label at `t + 1` is not a material rate.
- Labels are never interpolated; only boolean masks are advected.
- `fronts/` is not edited from this work. Two known defects are worked around on the caller's
  side: the finder trio of M1 task 7a (rules in `tracking.py`'s header), and `follow()` losing
  its position veto when the shifted reference leaves the tile.
- One region, one season, one 72-hour window: nothing generalises beyond it.
- Declare any tolerance or pass/fail rule before the run that judges it, and record a result
  that fails as pre-registered beside any corrected construction.
- Documents get marked notes, not rewrites. Each task gets one log entry at the end of
  `claude_prompts/frontogenesis_prompts.md` and a Status paragraph here. Claude commits
  nothing; JXP commits between tasks.

---

## 4. Possible starting points *(suggestions only — JXP to edit or delete)*

Each addresses one item of §1.4; the first three need no new store and run from the JSONs or
in minutes.

- **(a) Condition on topology.** Redo the pooled statistics (frontogenetic fractions, sign
  agreement, the reconciliation) on unflagged links only, with task 4's 168-link flag and with
  a size-jump flag, against all links. *(§1.4 items 2, 3.)*
- **(b) Attribute the size changes.** For each link, split the label's pixels into kept
  (within a cell of the advected previous mask), gained and lost, and tabulate `G`, `2F` and
  the matched rate on each part. *(Items 1, 5.)*
- **(c) Decompose the set-evolution term.** Write `r_label − r_matched` as a gained-pixel and
  a lost-pixel contribution per link, and see which carries the 5–7×. *(Item 5.)*
- **(d) Follow the family.** After a split keep every piece; before a merge keep every
  parent; compute the strength series on the union and compare with the single-label track.
  *(Items 4, 6.)*
- **(e) Hold the parcel set fixed for longer.** Seed parcels on the label at the anchor and
  carry them 6, 12, 24 h (the machinery of `m3_parcels.py`), recording what fraction remains
  on any front pixel and on the tracked label. *(Item 1.)*
- **(f) Vary the tracker.** Re-run `m4_track.py` with other flow weights, a different anchor
  step, or the ten largest at several steps, and report which population statistics move.
  *(Item 4.)*

---

## Tasks

*(none yet — to be written by JXP)*

---

## Q&A

*(none yet)*
