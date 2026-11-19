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

**Status — Matched parcels, task 1 (2026-11-18, Fable 5.2): done, first iteration.**
`py/matched_parcels.py` (generic: `--track K`, or `--label N [--anchor-step S]`, following a
front that is not an M4 track on the fly) wrote `tracked/matched_parcels_track_81.md`, 11
figures in `tracked/figs/` and `data/matched_parcels_track_81.json`. The three rates equal
`m4_strength.json` exactly at all 69 steps. Three questions are open in Q&A (TA-Q1..Q3). See
the log entry at the end.

**Status — Matched parcels, task 2 (2026-11-18, Fable 5.2): done.** TA-Q1 (a), TA-Q2 (a) and
TA-Q3 (figures good) noted; nothing in the figures or the parcels changed.
`tracked/matched_parcels_track_81.md` now has the equations for `G`, `2F` and the stored
`DG/Dt` (§1), a section confirming the three rates use the same parcels in each pair, with two
qualifications (§2), and a section answering whether the agreement of `2F` and the stored
tendency means the front strengthens at the rate `2F` (§4: the reading is right; the
implication does not follow on its own). No new question. See the log entry at the end.

**Status — Matched parcels, task 3 (2026-11-18, Fable 5.2): done.** Yes, the surface-flux
term can be included: the derived store holds it for every pair, in both forms (1 m top cell,
primary; spread over the KPP boundary layer, the declared sensitivity), and it is read at the
seed cells of the matched parcels as `2F` is. `tracked/matched_parcels_track_81.md` has a new
§5 (definition, a new Figure 3, tables, six numbered findings), a fourth column in the
per-parcel figure, the term in every panel heading and two new columns in the table. On this
front neither form accounts for the gap between the measured rates and `2F`; and the 1 m
term's band mean follows the pixel set (0.94) where the three rates do not. No new question.
See the log entry at the end.

**Status — Surface-flux, task 1 (2026-11-18, Fable 5.2): done; one question open (TA-Q4).**
The same band means were run on all ten M4 tracks (`py/surface_flux_tracks.py`,
`data/surface_flux_tracks.json`). Short answer: **yes for the 1 m form, no for the surface
flux as such.** The flux as taken up in the 1 m top cell is not the rate at which any of the
ten fronts changes (multiplier −0.012 [−0.034, +0.033] on the matched rate, pooled; +1 would
be a correct term). The form spread over the boundary layer is about a tenth of `2F` and
smaller than the disagreement between the two measured rates, so it can neither be confirmed
nor ruled out. New: the sign of the 1 m term on a front is set by whether its warm side is the
dense side (4 of the 10 tracks, 38 % of the pooled front pixels). I recommend returning to the
report with marked notes on three points; nothing in its verdict changes. See the log entry.

---

## Tasks

### Matched parcels

1. I wish to dig in further on the tracked front analysis, especially the hourly rates for
matches pixels described in Section 9.4 of the report.  Let us track the same front (label 81,
pair 35) as in the M4 analysis.  I want to view for each of timestep `t` the pixels of the chosen label, 
and the pixels of the previous label advected by the flow.  Also report the evaluation of the 
hourly rates of these matched parcels: 2F, stored DG/dt and the matched rate.
Generate a markdown file named `tracked/matched_parcels_track_81.md` with the following content:
    - A series of figures to show the pixels of the chosen label, and the pixels of the previous label advected by the flow, i.e. the matched parcels.  Indicate the hourly rates for the matched parcels.
    - A short discussion of how the hourly rates for the matched parcels are calculated

I am likely to request addtional tracked fronts to analyze.  Be sure to write
generic Python scripts for this work.
We may need to iterate a few times on this task. If you have any questions, 
please put them in the "Q&A/Matched parcels" section below.
Use Fable 5.2.  Log your work in the Logs section below.

2. This is excellent.  I have a few requests for you for to make modifications to the markdown file:
   - Provide additional information on how each of the hourly rates are calculated.  Provide equations for 
    the stored G and for 2F.
   - Confirm (or refute) that for each of these calculations, the rates are calculated on the same set of parcels in each pair 
   - In this example, the rates for 2F and stored DG/dt appear to track one another well throughout the 
   window.  Am I reading that correctly?  Does this imply that the front is strengthening or weakening at 
   the rate predicted by 2F?

Consider each of the above and modify the markdown file accordingly.
Use Fable 5.2.  Log your work in the Logs section below.

3. None of the discussion we have had so far includes the surface-flux term,
which we examined at length in the main report and work that led to it.
Are you able to include it in this tracked front analysis?  If so, please do so.
If you have any questions, please put them in the "Q&A/Matched parcels" section below.
Use Fable 5.2.  Log your work in the Logs section below.

### Surface-flux

1. Given the inconsistency between surface-flux and the other hourly rates, might we conclude
that it is not impacting the front strengthening or weakening?  If so, should we return to the
main report on this topic?
If you have any questions, please put them in the "Q&A/Surface-flux" section below.
Use Fable 5.2.  Log your work in the Logs section below.

## Q&A

### Surface-flux

**TA-Q4 — How should the report take this up?** *(asked 2026-11-18, Surface-flux task 1)*
The ten-track result sharpens three statements of `report/frontogenesis_report.md` and
reverses none: (i) §2.4, §5.3 and §8 describe the term as frontolytic "at every hour" — that
is the net of two populations of opposite sign, set by whether the warm side of the front is
the dense side; (ii) §7.2 calls the 1 m term "near-orthogonal to the imbalance" — on front
means it is excluded at unit weight on every track, which is a stronger statement and supports
§7.3's reading; (iii) §12 lists a front-mean budget as an option not run — it has now been
run on ten fronts, not pre-registered.
- (a) Marked notes at §2.4, §7.2–§7.3, §8 and §12 pointing to one short new subsection (§9.6,
  "The surface-flux term on the tracked fronts"), with the table of the log entry. The 1 m
  form stays primary and the null result stands. *(recommended)*
- (b) Leave the report as it is; keep this in the track-analysis documents only.
- (c) As (a), and also make the boundary-layer form primary. This reopens M3-Q13 (c), and the
  evidence here does not show that form is right, only that the 1 m form is not.

### Matched parcels

**TA-Q1 — Which parcels should the rates be evaluated on?** *(asked 2026-11-18, task 1)*
The task describes "the pixels of the previous label advected by the flow, i.e. the matched
parcels". In M4 these are two different sets. The matched parcels of step `t` are seeded on
the label chosen **at `t`** (dilated by one cell) and traced one hour forward and one hour
back; the label at `t − 1` does not enter the rate at all. The previous label advected to `t`
is the tracker's prediction. The first iteration shows both sets in every panel and reports
the M4 rates (band 1 + centred, as in report §9.4) unchanged.
- (a) Keep it so: M4's parcels, with the advected previous label shown for comparison.
  *(what was done; recommended as the baseline)*
- (b) Add a second set of rates seeded on the **previous** label and carried forward one hour
  (`[G(t, x⁺) − G(t−1, x)] / Δt` on the band of the label at `t − 1`). This is M4's
  `band1_forward` variant, which reconciled less well than the centred one.
- (c) Add the rates split by pixel class — parcels of the chosen label that coincide with the
  advected previous label against those that are new — which is starting point (b) of §4.

>A. Keep it so, i.e. (a)

**TA-Q2 — "Previous" in time, or in the tracker's walk?** The figures use the previous pair
in time (`t − 1`) at every step. Before the anchor (pairs 0–34) the tracker walked backwards,
so its own reference for pair `t` was pair `t + 1`. Splits and merges are flagged in time
order here, so for pairs before the anchor they are the reverse of task 4's flags (the counts
on this track are 10 splits / 9 merges / 18 links here against task 4's 9 / 10 / 18).
- (a) Time order everywhere. *(what was done; recommended)*
- (b) The tracker's order (advect the label at `t + 1` backwards for pairs before the anchor).

>A. (a)

**TA-Q3 — Layout of the figures.** Eight panels per page, nine pages, a fixed crop for the
whole track, index axes (`j` east, `i` south), `log10 G` in grey behind the pixels. Say if you
want longitude / latitude axes, a tighter crop, fewer pairs, or a different background
(for example `2F`, or none).

>A. These figures look good

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

## Logs

### 2026-11-18 — Track analysis, Matched parcels task 1: the chosen label, the advected previous label and the hourly rates along the track of label 81 (Fable 5.2)

**Asked.** For the front of label 81 at pair 35 (M4 track 0): at each step, view the pixels of
the chosen label and the pixels of the previous label advected by the flow; report `2F`, the
stored `DG/Dt` and the matched rate on the matched parcels; write
`tracked/matched_parcels_track_81.md` with the figures and a short account of how the rates
are calculated; generic scripts.

**Written.**
- `py/matched_parcels.py` (new). `get_track` (an M4 track by index or by anchor label and
  step; any other label is followed with `tracking.follow_flow` and M4's settings),
  `pixel_sets` (chosen label, previous label advected, counts, split / merge in time order),
  `parcel_link` (`matched_link(centred=True, band=1)` kept per parcel), `analyse_track`,
  `track_summary`, `check_against_m4`, three figure functions, `write_markdown`, `run`, CLI.
  `tracking.py` and `fronts/` are not edited.
- `py/tests/test_matched_parcels.py` (new, 2 tests on `test_tracking`'s growing front).
- `tracked/matched_parcels_track_81.md`; `tracked/figs/matched_parcels_track_81_series.png`,
  `…_parcels.png` and nine `…_sets_AA_BB.png` pages (8 pairs each); and
  `data/matched_parcels_track_81.json` (every number in the markdown, one row per step).
- Command: `cd py && /home/xavier/miniconda3/envs/frontogenesis/bin/python matched_parcels.py
  --label 81 --anchor-step 35` (135 s). Other fronts: `--track K` or `--label N`.

**Checks.**
- At every step `parcel_link`'s three means are asserted equal to `matched_link`'s (rtol
  1e-9); the run passed.
- `r_matched`, `r_store` and `two_F` equal the `m4_strength.json` series of track 0 exactly at
  all 69 steps with a rate; the labels are identical at all 71.
- The on-the-fly path was run on a front that is not an M4 track (label 39 at pair 35, the
  11th largest): 71 steps, 201 s, output written to the scratch directory only.
- New tests: 2 passed (1.2 s). The full suite was **not** re-run; no existing file under `py/`
  was changed.

**Numbers for label 81** (band 1 + centred, 69 steps; units 10⁻¹⁹ s⁻⁵; none is an efficiency).
- Mean rate: `2F` +2.79, stored +3.01, matched +0.97. Fraction of hours positive: 0.81, 0.83,
  0.58.
- Hour by hour: matched against stored, correlation 0.968, r.m.s. difference 0.437 of the
  stored r.m.s.; matched against `2F`, 0.938, sign agreement 0.74; stored against `2F`, 0.947.
- Parcels: median 657 per step for a median label of 178 px; none dropped; median displacement
  0.23 cells per hour.
- Pixel sets over 70 links: a median 0.73 of the chosen label's pixels coincide cell for cell
  with the advected previous label; median dilated IoU 0.81; a median 0.04 of chosen pixels lie
  more than one cell from it. More than a quarter of the chosen label is new at 8 pairs (7, 11,
  12, 16, 21, 30, 39, 67); more than a quarter of the advected label is unmatched at 9 pairs
  (5, 10, 11, 13, 14, 18, 27, 38, 40). Splits 10, merges 9, links with either 18 (time order).

**Seen in the figures, not tested.**
- The matched rate sits below the stored rate and `2F` over most of pairs 5–35 (Figure 1) while
  following their hour-to-hour shape; over those pairs the mean gap to the stored rate is 2.1 × 10⁻¹⁹ s⁻⁵,
  against a stored mean of 3.2 and a matched mean of 1.1. This is the r.m.s. ratio 0.437 seen as an offset. No cause is established.
- From pair 41 to pair 66 the track is a 104–136-px piece with almost no change of pixel set,
  and that interval holds the largest swings of all three rates (+16 to −14). The big changes
  of pixel set (pairs 5–30, 38–40, 67) fall where the rates are small. So on this track the
  largest hourly rates are not produced by changes of the pixel set.
- Pairs 38–40: the label drops from 295 to 178 px (split, the western arm is left to another
  label), returns to 299 px (merge) and drops to 111 px (split). The three rates do not jump
  across these three links (matched +2.09, +3.49, +0.71, −3.5).

**Open.** TA-Q1 (which parcels the rates should be on — the task text equates the advected
previous label with the matched parcels, which M4 does not), TA-Q2 (previous in time or in the
tracker's walk), TA-Q3 (figure layout). Nothing committed; PNGs are git-ignored and need
`git add -f`.

### 2026-11-18 — Track analysis, Matched parcels task 2: equations, the same-parcels check, and what the agreement of `2F` and the stored tendency implies (Fable 5.2)

**Asked.** Three modifications to `tracked/matched_parcels_track_81.md`: more on how each rate
is calculated, with equations for the stored tendency and `2F`; confirm or refute that the
three rates are on the same parcels in each pair; and whether `2F` and the stored `DG/Dt`
track one another, and whether that implies the front strengthens or weakens at the rate
predicted by `2F`.

**Answers noted.** TA-Q1 (a) keep M4's parcels; TA-Q2 (a) time order; TA-Q3 the figures are
good. No figure and no parcel set was changed.

**Changed.** `py/matched_parcels.py` only (the markdown is generated, so every later front gets
the same sections): `parcel_link` now also returns the count of parcels entering `2F`, the
per-parcel correlations between the three quantities, and their means on the ridge (the
label's own pixels) and on the flanks; `analyse_track` copies the other budget terms from
`matched_link`; `track_summary` adds the same-parcels counts, r.m.s. differences, trapezoid
integrals and the ridge / flank means; `write_markdown` has new §1.1, §1.2, §2 and §4 (the
old §3 and §4 are now §5 and §6). Regenerated with `matched_parcels.py --label 81
--anchor-step 35`. The equations were written from `semilag.measured_DGDt`,
`operators._frontogenesis_discrete` and `budget.compute_budget`, not from memory.

**1. Equations (markdown §1).** `G` is `|L b^m|²` on the midpoint-mean buoyancy; the stored
`two_F` is twice the discretely consistent form (the commutator of the gradient stencil with
advection), on midpoint fields; `DGDt_semilag` is `[G(x, h_{t+1}) − G(x_d, h_t)] / Δt` with
hour-end buoyancy, `b` interpolated at the departure point; then the three means over the
`N_t` parcels of the band.

**2. Same parcels: confirmed, with two qualifications (markdown §2).** 55,213 parcels over 69
steps; 55,213 enter the matched and the stored rate and 55,213 enter `2F`, the same number at
every step, none dropped. Qualifications: (i) the stored tendency at a seed cell belongs to
the parcel arriving there at the end of the hour, half a displacement (median 0.12 cells) from
the matched parcel at the midpoint; (ii) the time windows differ (matched: two hours between
midpoint-mean fields; stored: one hour between hour-end fields; `2F`: instantaneous), all
centred on the midpoint. Between pairs the sets are not the same.

**3. `2F` against the stored tendency (markdown §4).** Units 10⁻¹⁹ s⁻⁵; none is an efficiency.
- The reading is right: correlation 0.947, r.m.s. difference 1.61 against an r.m.s. of 5.63
  (`2F`) and 5.32 (stored), sign agreement 0.96, mean difference +0.23. They are the closest
  of the three pairs of rates (matched − stored 2.33 r.m.s., matched − `2F` 2.49).
- It does not by itself imply the front strengthens at the rate `2F`:
  - The agreement is of band means, not of parcels. Median per-parcel correlation within a
    step: stored against `2F` 0.46, matched against stored 0.86, matched against `2F` 0.36.
    On the ridge (27 % of the band) the means are `2F` +3.82, stored +12.02, matched +5.08; on
    the flanks +2.41, −0.20, −0.49. The stored construction's known positive bias on one-cell
    ridges (report §9.3) may contribute; how much is not established.
  - The matched rate, the other measurement of the same Lagrangian rate, is below the stored
    tendency at **all 69 hours** (mean +0.97 against +3.01) and positive on 58 % of hours
    against 83 %.
  - Over 68 hours (trapezoid, 10⁻¹⁴ s⁻⁴): `∫2F` +6.76, stored +7.31, matched +2.31; the band
    mean of the label as found changed by +0.11 (6.63 to 6.74).
  - On the same parcels the surface-flux term averages +12.90, the vertical term −0.08 and the
    residual −12.59: the stored tendency is close to `2F` alone, which is not a closed budget.

**Correction to the task-1 entry.** It said the matched rate sits below the stored rate "over
most of pairs 5–35". It is below at every one of the 69 hours.

**Checks.** The run's assertion against `matched_link` passed at every step; the rates still
equal `m4_strength.json` exactly. `test_matched_parcels.py`: 2 passed. The full suite was not
re-run; no other file under `py/` was changed. Nothing committed.

**Open, not asked.** Why the matched rate is lower than the stored tendency at every hour, and
how much of the stored ridge value is the registration bias. A synthetic ridge moved with this
track's displacements, or the stored tendency recomputed with a centred two-hour step, would
separate them.

### 2026-11-18 — Track analysis, Matched parcels task 3: the surface-flux term on the matched parcels (Fable 5.2)

**Asked.** Whether the surface-flux term can be included in the tracked-front analysis, and
to include it if so.

**Answer: yes, with no new computation from model output.** `tile330_derived_L0.zarr` holds
`surface_flux` (the flux taken up in the 1 m top cell — the report's primary form) and
`surface_flux_kpp` (the flux spread over the KPP boundary-layer depth — the declared
sensitivity of report §7.3, M3-Q13 (c)), plus `vertical` and `KPPhbl`, for every pair at the
midpoint time. They are read at the seed cells of the matched parcels exactly as `2F` is, and
are finite on every parcel at every step, so their means are over the same `N_t` parcels.

**Changed.** `py/matched_parcels.py` only: `EXTRA_FIELDS` and per-parcel `S_1m`, `S_kpp`, `V`,
`hbl` in `parcel_link`; `surface_flux_summary` (means, r.m.s., residuals with and without the
term against both measured rates, the report's §7.2 multiplier and correlation, day / night,
ridge / flank, integrals); `set_change_summary` (does a band mean's hourly change follow the
change of label size); `fig_budget` (new Figure 3); a fourth column in `fig_parcels`; the term
in each panel heading of the pair-by-pair figures; markdown §5 (new), a row in §2's table, two
columns in the final table; the old §5 and §6 are §6 and §7. A store without these fields (the
test fixture) simply omits the section. Regenerated with `matched_parcels.py --label 81
--anchor-step 35`; one new file, `tracked/figs/matched_parcels_track_81_budget.png`.

**Numbers for label 81** (band means over 69 steps, 10⁻¹⁹ s⁻⁵; none is an efficiency).

| | mean | r.m.s. | hours positive |
|---|---|---|---|
| `2F` | +2.79 | 5.63 | 0.81 |
| stored `DG/Dt` | +3.01 | 5.32 | 0.83 |
| matched | +0.97 | 4.48 | 0.58 |
| vertical | −0.08 | 0.58 | 0.67 |
| `S`, 1 m | +12.90 | 13.62 | 1.00 |
| `S`, `h_bl` | +0.12 | 0.82 | 0.49 |

- Residual r.m.s., measured − predicted, against stored / matched: `2F` alone 1.61 / 2.49;
  `2F + V` 1.60 / 2.29; `2F + V + S(1 m)` 13.48 / 15.44; `2F + V + S(h_bl)` 1.61 / 2.52.
- The 1 m term is positive at all 69 hours, night (+13.14) and day (+12.63), on 92 % of
  parcels in the median step; ridge +19.84, flanks +10.43. Its correlation with the imbalance
  stored − `2F` − `V` is −0.09 and the residual-minimising multiplier +0.018 (matched
  imbalance: −0.20, −0.128); the report's pooled multiplier at `L = 0` is +0.048.
- The boundary-layer form is 0.060 of the 1 m form in r.m.s.; `h_bl` on the parcels has a
  median of 21.6 m (step means 4.0–28.0 m); the two forms' hourly band means correlate at
  +0.15.
- Integrals over 68 h (10⁻¹⁴ s⁻⁴): `S` 1 m +31.70, `S` `h_bl` +0.27, `2F` +6.76, stored +7.31,
  matched +2.31.
- **Pixel-set dependence (new, and on the concern of this document).** Correlation of the size
  of the hourly change of each band mean with `|Δ log npix|` over 68 consecutive steps: `2F`
  +0.01, stored −0.11, matched −0.05, `S` 1 m **+0.94**, `S` `h_bl` +0.43. Median change at the
  13 links where the label size jumps by more than 0.25 in the log, against the other links:
  `S` 1 m 5.65 against 0.40; `2F` 1.75 against 1.75; stored 1.64 against 1.72; matched 1.40
  against 1.76. The per-parcel map (Figure 2, fourth column) shows the 1 m term concentrated
  on the western arm of the front and near zero on the eastern arm, so its band mean changes
  when the tracker drops or regains an arm (pairs 14, 27, 30, 38–40, 67).

**Reading.** On this front the surface-flux term as stored does not account for the gap
between the measured rates and `2F` in either form — the report's pooled result (§5.2, §7.2,
§7.3) on one front; it is not reopened. For the pixel-set question: the three rates are
insensitive to the hour's change of pixel set and the 1 m surface-flux mean is not.

**Not established.** Why the 1 m term is positive at every hour here, when the report's sign
discussion (§2.4) describes heating towards the dense side as frontolytic; the split into
non-solar, shortwave and freshwater parts was not redone on these parcels.

**Checks.** The run's assertion against `matched_link` passed at every step, and
`matched_link`'s own `surface_flux` mean equals the new `S_1m_mean` (both are the mean of the
same field on the same parcels). `test_matched_parcels.py`: 2 passed. The full suite was not
re-run; no other file under `py/` was changed. Nothing committed.

### 2026-11-18 — Track analysis, Surface-flux task 1: is the surface-flux term acting on the tracked fronts, and should the report be revisited? (Fable 5.2)

**Asked.** Given the inconsistency between the surface-flux term and the other hourly rates,
may we conclude it is not affecting the strengthening or weakening of the front; and if so,
should we return to the main report.

**Done.** One front cannot carry that conclusion, so the band means of
`matched_parcels.py` (band 1 + centred, the same parcels for every term) were run on **all ten
M4 flow-informed tracks**, 69 hours each, 690 track-hours.
- `py/surface_flux_tracks.py` (new): per track and pooled, the front-mean residuals with and
  without each form of `S`; the report's §7.2 multiplier with a moving-block bootstrap over
  hours (6 h blocks, 2,000 draws, declared in the module before the run); the split of `S`
  into non-solar, shortwave and freshwater parts on the parcels
  (`m3_term_tests.split_components`); the thermal fraction of the buoyancy gradient,
  `R_T = ∇b·∇b_T / |∇b|²` (`R_T < 0`: the warm side is the dense side); and the same means
  on the budget's pooled front pixels. Product: `data/surface_flux_tracks.json` (1,039 s).
- `py/tests/test_surface_flux_tracks.py` (new, 2 tests). No other file changed; the report is
  not edited.

**Numbers** (band means, 10⁻¹⁹ s⁻⁵; `S₁` is the 1 m form, `S_h` the boundary-layer form; none
is an efficiency).

| label | `R_T` | mean `2F` | mean stored | mean matched | mean `S₁` | hours `S₁ > 0` | mean `S_h` | r.m.s. `S_h` / `2F` |
|---|---|---|---|---|---|---|---|---|
| 81 | −0.69 | +2.79 | +3.01 | +0.97 | +12.90 | 1.00 | +0.12 | 0.15 |
| 84 | +0.48 | +11.24 | +18.77 | +10.80 | −26.60 | 0.00 | −0.71 | 0.07 |
| 202 | +0.98 | +3.18 | +3.82 | +1.61 | −9.25 | 0.00 | −0.68 | 0.25 |
| 100 | −0.72 | +0.98 | +0.22 | −0.82 | +6.06 | 1.00 | −0.03 | 0.21 |
| 187 | −0.77 | +0.60 | +1.09 | +0.51 | +4.42 | 1.00 | +0.13 | 0.38 |
| 5 | −0.17 | +1.07 | −0.07 | −0.11 | +1.87 | 0.88 | −0.45 | 0.36 |
| 96 | +0.47 | +1.01 | +1.79 | +0.73 | −5.33 | 0.00 | −0.21 | 0.15 |
| 133 | +0.84 | +0.15 | +0.06 | −0.07 | −1.28 | 0.00 | −0.09 | 0.53 |
| 161 | +1.79 | +1.41 | +1.67 | +0.51 | −6.51 | 0.00 | −0.42 | 0.21 |
| 218 | +0.76 | +0.31 | +0.60 | +0.35 | −1.52 | 0.00 | −0.15 | 0.41 |

Pooled over the 690 track-hours:

| | stored | matched |
|---|---|---|
| r.m.s. of the measured rate | 8.41 | 5.68 |
| residual, `2F` alone | 3.61 | 1.89 |
| residual, `2F + V` | 3.66 | 1.90 |
| residual, `2F + V + S₁` | 14.84 | 12.45 |
| residual, `2F + V + S_h` | 3.90 | 1.87 |
| multiplier on `S₁` [95 %] | −0.200 [−0.247, −0.116] | −0.012 [−0.034, +0.033] |
| multiplier on `S_h` [95 %] | −1.79 [−2.61, −0.13] | +0.63 [+0.19, +1.46] |

R.m.s. of the terms: `2F` 5.72, `S₁` 12.16, `S_h` 0.63, `V` 0.33. Means: `2F` +2.27, stored
+3.10, matched +1.45, `S₁` −2.52, `S_h` −0.25.

**1. The 1 m form is not acting on these fronts. This can be concluded.**
- It is the largest term on every track (1.1 to 6.9 times the r.m.s. of `2F`) and of one sign at
  every hour on nine tracks of ten. Label 84's `S₁` averages −26.6 while the front
  strengthens at +10.8 to +18.8; label 81's averages +12.9 against +1.0 to +3.0.
- Adding it multiplies the residual by 4 (stored) and 6.5 (matched). Its multiplier lies
  between −0.66 and +0.17 on every track against either measured rate; no interval reaches
  +0.25, let alone +1.
- This is what §7.3 of the report expected for a physical reason: the mixing scheme spreads
  the flux through the boundary layer (median 17–40 m on these tracks) within the hour, so the
  top cell does not keep it. The statement is that the term **as formed over 1 m** is not the
  top cell's hourly tendency, not that the flux does nothing.

**2. The surface flux itself cannot be said to have no effect.**
- The boundary-layer form is the physically expected size. It is small: 0.11 of `2F` in
  r.m.s. pooled, 0.07 to 0.53 per track.
- It is not negligible in the mean on the weaker fronts: `S_h` is of one sign at every hour on
  four tracks, and its mean is −0.2 to −0.6 of the mean of `2F` on six.
- It is below what these data can detect. The two measured rates differ by 1.65 in the mean
  (stored above matched, the unexplained offset of the task-2 entry), against 0.25 for `S_h`,
  and they give opposite answers: the pooled multiplier is +0.63 [+0.19, +1.46] on the matched
  rate, which includes +1 and excludes 0, and −1.79 [−2.61, −0.13] on the stored rate, which
  excludes both. Per track, 8 of 10 multipliers are positive against the matched rate and 5 of
  10 against the stored. Until the stored–matched offset is understood, an effect of the size
  of `S_h` is neither shown nor excluded.
- Indirect effects of the flux (through the boundary-layer depth and mixing) are not in `S`
  at all and are not tested here.

**3. The sign of `S₁` is explained (the item left "not established" in the task-3 entry).**
- On every track the non-solar heat part carries the term (for label 81: +12.95 of +12.90;
  shortwave −1.25, freshwater +1.19).
- The sign follows `R_T`. The four tracks with `S₁ > 0` (81, 100, 187, 5) have `R_T < 0`: the
  warm side is the dense side and salinity sets the buoyancy gradient. The six with `S₁ < 0`
  have `R_T > 0`. On label 81's parcels `S₁` averages +13.97 where `R_T < 0` (92 % of them)
  and −1.75 elsewhere.
- So the term is the damping of the surface temperature contrast (more heat lost over warmer
  water). That weakens a front whose warm side is light and strengthens one whose warm side
  is dense.
- On the pooled front pixels (1.81 million, 69 pairs) 38 % have `R_T < 0` (35–40 % per hour);
  `S₁` averages +5.20 there and −8.67 on the rest, net −3.40, negative at every hour. The
  report's "frontolytic at every hour" is that net.

**4. Should we return to the report? Yes, with marked notes; the verdict does not change.**
Three statements are sharpened (TA-Q4): the sign of the term (§2.4, §5.3, §8); the 1 m form
excluded on front means rather than "near-orthogonal" (§7.2–§7.3); and the front-mean budget
of §12, now run on ten fronts. On front means `2F` alone leaves 0.43 (stored) and 0.33
(matched) of the measured r.m.s., against 0.74 per pixel in the report's pool. That comparison
was not pre-registered, is on ten fronts selected for size, and neither figure is a closed
budget or an efficiency.

**Also seen, not interpreted.** Pooled over tracks the stored imbalance is anti-correlated
with `S₁` (−0.65; multiplier −0.200), and the matched imbalance is not (−0.19; −0.012). Label
84 carries most of it (stored imbalance r.m.s. 10.8 with `S₁` at −26.6).

**Checks.** The recomputed 1 m term equals the stored `surface_flux` on front pixels to
2.7 × 10⁻⁸ relative r.m.s. at every pair. Track 0's means equal
`data/matched_parcels_track_81.json`. Tests: 4 passed (the 2 new and the 2 of
`test_matched_parcels.py`). The full suite was not re-run. Nothing committed.

**Limits.** Ten fronts, the largest at pair 35, one tile, 72 hours. `R_T` uses the midpoint
fields and a local linearisation of the equation of state. The bootstrap resamples hours
within tracks; it does not treat the ten tracks as a sample of fronts.
