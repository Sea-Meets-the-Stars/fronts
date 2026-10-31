# Frontogenesis execution prompt 6 — M5: Figures and report

**Milestone:** M5 (`frontogenesis_coding.md` §6).
**Prerequisites:** ~~M3 passed; M4 complete.~~ *(revised 2026-10-28, before execution.)* **M3 is
NOT closed** — criterion 1, the hard gate, failed at every judged `L` (residual r.m.s. over
measured 0.813 / 0.841 / 0.898 / 1.042 at `L = 0 / 2 / 4 / 8` against the pre-declared 0.50; 0 of
71 pairs; prompt 4 task 8, 2026-10-11). That is the planning §12 null result, and it is what M5
writes up. **M4 is complete** on JXP's instruction to proceed without closure (prompt 5 Status,
2026-10-26): criteria 1, 2, 3 and 5 met; criterion 4 met on the band-1 + centred construction
(M4-Q3 (a)) and failed as pre-registered. No slope has been quoted as an efficiency anywhere, and
M5 does not change that.
**Goal:** one command regenerates every figure from the stores, and a report that a reader can
check rather than take on trust.

**Reviewed 2026-10-28 (Fable 5.1), before execution.** This prompt was drafted with the rest of
the series (2026-09) before M3 was judged, and three of its instructions no longer fit the work:
(i) it describes Figure 2 and a "slope" section as if the measured-vs-predicted slope were a
reportable number — it is withheld (M3-Q15 (a), M3-Q24 (a)), and the prompt's own "Do not" list
forbids an efficiency before criterion 1 passes; (ii) it describes `figures.py` as something to
write — it exists since M3 task 7 (`figures.py` + `figures_maps.py` + `figures_stats.py`, a
`FIGURES` registry, a driver, ten tests), as do `figures_m4.py` (Figures 8, 9 and the
tracking-quality diagnostic) and `validate_figs.py` (V1–V6), so task 1 is consolidation, as
coding §6 M5 already says; (iii) its per-front section and limitations list predate M4's results
and M3's diagnosis. Every change below is marked *(revised 2026-10-28: …)*; the original wording
is in `git log` for this file. Three questions, **M5-Q1..Q3**, are in `## Q&A` at the end.
**Task 1 and task 3 can start on the recommendations; task 2 waits on M5-Q1 and M5-Q2.** The
Status paragraph convention of prompts 4–5 applies: one per task, added here when it is done.

---

## What exists *(added 2026-10-28)*

The inventory the tasks consolidate. Nothing here is to be redrawn for its own sake.

| Figure(s) | Written by | Driven by | Reads | Caption states `L` and mask? |
|---|---|---|---|---|
| 1, 2, 2b, 3, 3b, 4, 5, 6, 7, 10 → `figs/fig{NN}_*.png` | `py/figures.py` (style, loaders, `FIGURES`), `figures_maps.py` (1, 3b, 5, 10), `figures_stats.py` (2, 2b, 3, 4, 6, 7) | `py/m3_figs.py` → `data/m3_figs_summary.json` | the derived stores + `data/m3_closure_summary.json`; nothing recomputed | yes, via `figures.mask_note(L, mask)` and the four caveat strings (`FORCING_CAVEAT`, `WIND_CAVEAT`, `WINDOW_CAVEAT`, `SAMPLING_CAVEAT`) |
| 8, 9 → `figs/fig08_front_case.png`, `fig09_population.png` | `py/figures_m4.py` | `py/m4_strength.py` (computes the series, 729 s) | tracks + strength series + the `L = 0` store and label zarr | yes, as free text (`L = 0, mask_analysis, config D, band 1, centred`) |
| tracking-quality diagnostic → `figs/fig_tq_tracking_quality.png` (M4 criterion 3; not in planning §7's numbering) | `py/figures_m4.py` | `py/m4_quality.py` (221 s) | `data/m4_quality.json`'s tables | yes, free text |
| V1–V6 → `figs/V*_*.png` | `py/validate_figs.py` | `python validate.py` (its `__main__` runs the gates and writes all six) | synthetic fields + the grid | V-figures state their test; `L` is not meaningful for V1–V5, V6 states the mask geometry |
| two schematics → `report/figs/schematic_*.png` | `report/make_report_figs.py` | itself | nothing (drawn) | n/a |
| three case-study figures → `report/figs/case_*.png` | `report/make_case_figs.py` | itself | `data/m3_case_summary.json`, `m3_case_fields.npz` | yes |
| seven deck re-plots → `deck/figs_m3/` | `deck/make_m3_figs.py` | the deck build | the two M3 JSONs | **out of M5's scope** — decks are their own artefacts |
| working figures `figs/m3_closure.png`, `m2_*.png`, `m0_qa_*.png` | the milestone runners | — | — | **out of scope**: milestone QA, not report figures; they stay |

Tests: `tests/test_figures.py` (10) pins `FIGURES`' names and order to coding §4.10's ten and
asserts the drawn numbers equal the closure summary's; `tests/test_tracking.py` (32) covers
Figures 8, 9 and the diagnostic on a synthetic track; `tests/test_validate.py` covers V1–V6.
Suite as of task 5: **290 passed + 3 strict xfails**, 2 network tests deselected, ~265 s, with
the absolute interpreter `/home/xavier/miniconda3/envs/frontogenesis/bin/python`.

Report: `report/frontogenesis_M3_interim.md` (M0 → M3, 2026-10-12, 22-page PDF), written under
JXP's answers M3-Q16..Q26 (prompt 4 `## Q&A` → `### Report`), with `report/check_report_numbers.py`
asserting every JSON-backed number and the "Do not" list, `report/README.md` recording the pandoc
build, and one marked limitation paragraph added in §7 after M4 task 3 (M4-Q4 (a)). **No
`dev/frontogenesis/README.md` exists.** The environment is already exported:
`env/frontogenesis_env_linux-x86_64.yml` and `env/frontogenesis_pip_freeze_linux-x86_64.txt`
(prompt 4 task 2b).

---

## Tasks

### 1. `figures.py` — one function per figure

`fig01_maps` ... `fig10_term_budget`, plus `figV1..figV6` (or delegate those to `validate.py`,
which already writes them). **The figures themselves were produced in M1 (V1-V6), M3 (1-7, 10)
and M4 (8, 9); M5's job is consolidation** — uniform captions, one-command regeneration, and the
report. Every figure:

- regenerable from the zarrs by **one command**, with no manual steps;
- stating its filter scale `L` and its mask in the caption — a figure whose mask is ambiguous is
  not checkable;
- written to `dev/frontogenesis/figs/`.

*(revised 2026-10-28: what consolidation means, concretely.)*

- **One registry.** `figures.FIGURES` grows from the ten M3 entries to the full set — `fig08`,
  `fig09`, the tracking-quality diagnostic (keep its file name) and `figV1..figV6` — every entry
  callable as `fn(data_dir=None, fig_dir=None, **kw)` and returning the dict of numbers it drew,
  as the ten already do. Figures 8, 9 and the diagnostic get thin wrappers in `figures_m4.py`
  that load `data/m4_strength.json` / `data/m4_quality.json` (plus the `L = 0` store and the
  label zarr for Figure 8's maps) and call the existing drawing functions unchanged; the
  V-figures get wrappers that call `validate.py`'s gate functions, which draw through
  `validate_figs.py` as now (the figures stay there; coding §4.10's "`figV1..figV6` in
  `figures.py`" is satisfied by the registry, not by moving code — say so in a marked note).
  `tests/test_figures.py::test_figure_names_and_order_match_coding_4_10` pins the old list and
  must be updated to the new one, in order.
- **One command.** A new `py/make_figs.py` (keep `m3_figs.py` as it is — it is the provenance of
  `m3_figs_summary.json` and task 8's audit): `make_figs.py` renders every registry entry (or
  `--only`), writes `data/figs_summary.json` in the `m3_figs_summary.json` form, and with
  `--report` also runs `report/make_report_figs.py` and `report/make_case_figs.py`. Default
  behaviour reads the products (the four derived stores, the label and flow zarrs, the JSON
  summaries) — seconds to a few minutes; `--recompute` first re-runs the M4 runners
  (`m4_find.py` 551 s, `m4_track.py` 1,481 s, `m4_recon.py` 1,276 s, `m4_quality.py` 221 s,
  `m4_strength.py` 729 s) so the figures are regenerable from the stores alone, as the
  acceptance says; the M3 closure (`m3_closure.py`) is a product of M3 and is not re-run by
  this command — state that in the docstring and the README. Each entry in its own try/except
  (`m3_figs.render`'s pattern) and a non-zero exit if any fails.
- **Uniform captions.** One footer helper in `figures.py` — `L`, the mask, the config / band
  where a figure has one, then the caveats — used by every figure including 8, 9, the diagnostic
  and V6, replacing the free-text prefixes in `figures_m4.py`. The returned dict of **every**
  registry entry carries `L` and `mask` keys (V1–V5: `L = None`, `mask = 'synthetic'`), and a
  new test asserts that for every entry on the synthetic fixtures — the acceptance criterion
  made machine-checkable.
- **Nothing redrawn changes its numbers.** `test_the_real_figures_match_task_6_and_are_tracked`
  must still pass, and Figures 8, 9 and the diagnostic redrawn from the JSONs must carry the
  numbers the task-4 / task-5 log entries quote (`m4_strength.json`'s `fig08` / `fig09` blocks).
  Regenerated PNGs are git-ignored by `*.png` and need `git add -f`, as before.

Main figures 1, 2, 2b, 3, 3b, 4, 5, 6, 7, 8, 9, 10 and validation V1-V6 are specified in
planning §7. The two that carry the argument:

- **Figure 2** with the M1 discrete-null slope **drawn as a baseline line** — at **0.981 with its
  band [0.970, 0.994]** (the V3 real-velocity slope, `form='discrete'`; decided 2026-09-30,
  M1-Q4), with V3b's model-advection systematic band 0.954-1.003 beside it. *(revised
  2026-10-28: this is how `figures_stats.fig02_joint_pdf` already draws it, with the temporal
  systematic 0.972 ± 0.020 as a third band, and its slopes sit under an explicit **SLOPES NOT
  QUOTED** banner because criterion 1 failed (M3-Q15 (a)). Leave the figure as built. Whether
  it appears in the M5 report at all is **M5-Q1** — the interim report omitted it and its
  numbers entirely (M3-Q24 (a)).)*
- **Figure 2b**, the numerical-vs-diabatic discriminator, without which Figure 2 has no physical
  interpretation. *(revised 2026-10-28: as built. It does not earn the headline claim, it
  withdraws it — the partial correlations of the residual are +0.144 / +0.134 / +0.100 / +0.076
  on `lap2_b` and +0.056 / +0.090 / +0.146 / +0.219 on `KPPhbl` at `L = 0 / 2 / 4 / 8`, crossing
  over between `L = 2` and `L = 4` and never reaching 0.22, so no diabatic signal is isolated:
  planning §12's third null condition. Report §4 states that; it is M3's finding, not new.)*
- *(added 2026-10-28)* **Figure 9(d)** draws every link's matched rate against `2F` with decile
  means and the 1:1 line and **no fitted slope**. Keep it that way: the pooled correlation 0.951
  and amplitude ratio 1.008 on the band are recorded in the task-5 log and may be stated as a
  consistency check, never as an efficiency (M3 carry-forward, prompt 5 "Do not").

### 2. `frontogenesis_report.md`

*(revised 2026-10-28: location, lineage and rules.)* Written as `report/frontogenesis_report.md`,
beside the interim report, which is what it grows from (`report/README.md`, M3-Q17 (a)); its
relation to the interim is **M5-Q2**. Unless JXP changes them in the Q&A, **JXP's answers
M3-Q16..Q26 bind this report as they bound the interim**: an advanced PhD student in physical
oceanography as the reader, full physics with every symbol introduced in words and no code
identifiers in the running prose (Q16, Q25), Markdown with LaTeX equations and a pandoc PDF
build (Q17), M0 → M4 in proportion then the synthesis (Q18, extended by one milestone), numbers
**only** from the task log entries and the JSON summaries — `m3_closure_summary.json`,
`m3_figs_summary.json`, `m3_case_summary.json`, and now `m4_find_done.json`,
`m4_track_summary.json`, `m4_recon.json`, `m4_quality.json`, `m4_strength.json` — with no data
store opened by the report or its scripts (Q23), `check_report_numbers.py` extended (or a sibling
written) to assert the new report's JSON-backed numbers and the "Do not" list, the merged
glossary with units (Q25), the authors line, a dated status line and the circulation note (Q26).
The interim report is **not rewritten** — marked notes only, as for every document.

Structure it so the load-bearing numbers are findable, not buried:

1. **What we measured**, with the closure residual and its tolerance stated up front.
   *(revised 2026-10-28: 0.813 / 0.841 / 0.898 / 1.042 against 0.50, judged at `L >= 2`, `L = 0`
   reported not gated; 0 of 71 pairs; the tolerance fixed 2026-10-07 before any number existed,
   a module constant in `m3_closure.py` that `test_m3_closure.py` ties to `budget.py`'s copy.)*
2. **The budget**, term by term: `2F`, vertical, surface flux, subfilter `tau`, residual — with
   their relative sizes. *(revised 2026-10-28: all five are measured — the chunk terms were
   available (M2 source B, 72/72 hours) — and the `KPPhbl` depth-scale sensitivity of the
   surface-flux term (M3-Q13 (c)) is reported beside it: it fixes the term's magnitude and
   leaves its correlation with the imbalance at +0.02, which is M3's cause two.)*
3. **The slope**, every estimator, with feature-level bootstrap intervals, **quoted against the
   M1 discrete-null baseline** rather than against 1. *(revised 2026-10-28: this bullet predates
   the gate. What is always stated: the baseline 0.981 [0.970, 0.994], V3b's 0.954-1.003 and the
   temporal 0.972 ± 0.020, each with what it is; criterion 2's sampling-consistency slopes 0.691
   / 0.813 / 0.891 / 0.939 (corr 0.758 / 0.913 / 0.969 / 0.984; passes at `L = 4, 8`), which are
   not efficiencies; and that criterion 4's machinery — OLS, trimmed, orthogonal and ratio
   estimators with 1-h and 3-h block intervals, both forms, per width bin, every value relative
   to 0.981 — was built and run and its measured-vs-predicted slopes are held in the closure
   summary under `slopes_not_quoted` with `slopes_quotable = False`. Whether those four numbers
   are printed, banner or no banner, is **M5-Q1**; the recommendation is that they are not.)*
4. **Numerical vs diabatic** — what Figure 2b actually shows. This is the section where the
   headline claim is either earned or withdrawn. *(revised 2026-10-28: withdrawn, at M3 — see
   task 1's Figure 2b note; the section says so and gives the partial correlations.)*
5. **Per-front results**, including whether flow-informed tracking changed the linking.
   *(revised 2026-10-28: M4 is done, so this section has content, all of it in the prompt-5 log
   entries and the five M4 JSONs. It reports: the ten tracks (the ten largest fronts at pair 35,
   121–307 px, each found at 71 of 71 steps, 70 h, no gaps); **yes**, the flow changed the links
   on 3 of 10 tracks — 71 links chosen differently and 51 gaps filled, at every one the larger
   front, with the mechanism (a split followed by `follow()`'s centroid extrapolation) and the
   fact that the advected skeleton never beats persistence on IoU at hourly cadence, so the gain
   is the material *prediction*; tracking quality (168 of 700 links flagged split / merge; the
   chosen label within 1 cell of the advected prediction on 72 % of links); criterion 4 **both
   ways** — failed as pre-registered on the 1-px skeleton with a forward difference (corr −0.09,
   r.m.s. ratio 1.12, a registration artefact quantified on a synthetic ridge) and passed on the
   band-1 + centred construction JXP chose (0.953 / 0.415 against 0.8 / 0.5, 5 of 10 tracks
   individually), with the endpoint-label rate's set-evolution term 5–7× the material rate as
   the demonstration of why "front at `t` vs front at `t + dt`" is not material; the population
   (690 links): frontogenetic hours 0.64 [0.50, 0.77] by the matched rate against 0.81 [0.75,
   0.88] by `2F`, sign agreement 0.78 [0.67, 0.87], `∫2F dt > 0` on 10 of 10 fronts and material
   `ΔG > 0` on 7, lifetimes degenerate (every track spans the window), width 4.5 km median; and
   the contrast that is M3's finding at object level — hourly rates agreeing at corr 0.95 while
   the 70-h integrals per front scatter from −0.8 to 1.1 of the prediction — stated as that, not
   as an efficiency.)*
6. **Limitations**, stated plainly and without softening:
   - 72 h is 3 diurnal, 3.6 inertial and 5.8 M2 cycles — **those bands are not separable**, so
     Figure 6 is *indicative, not conclusive*. This is the strongest argument for extending to
     the full 504-hour series.
   - The `>= 100 km` offshore cut excludes the inner upwelling zone where CC frontogenesis is
     most vigorous, so the headline number describes the **offshore regime**, not the CCS.
     *(revised 2026-10-28: and M4's fronts are found on the same `mask_analysis` (M4-Q1 (a)), so
     the coastal zone's largest front — 1,179 px at pair 0, three times the largest offshore —
     is neither budgeted nor tracked.)*
   - LLC4320 tides are reported over-energetic; reversible tidal strain pushes the slope toward
     1 and can mask damping.
   - One region, one season, one 72-hour window.
   - *(added 2026-10-28, from M3 and M4)* **The hourly cadence** is cause one of the non-closure:
     two defensible estimates of the same tendency from the same hourly pair differ by 0.67 of
     the measured amplitude at `L = 0`, and only hourly output exists, so this cannot be tested
     at a shorter interval (M3-Q14 (a)). **The surface-flux term's near-orthogonality** is cause
     two, and its depth scale is a declared sensitivity, not a measurement. **The measured
     tendency carries a ridge-cell registration bias** (departure value interpolated, arrival
     value a cell: +0.6 to +1 % of `G` per hour on a translated 1-px ridge where the truth is 0;
     positive on 64 % of skeleton pixels against 50 % of all valid), quantified by M4 task 3,
     diluted but not measured on M3's `p90` pool (M4-Q4 (a); the one-day measurement is an §8
     option). **Figure 6's expectation was not met**: the composite residual is positive at
     every local hour, so the sign change the planning document predicted is not seen, and a
     single-signed residual is not damping either. **Track lifetimes are the window**, so
     "lifetime vs mean `F`" is not a measurement. **Two defects in the `fronts` package are
     described, not fixed** (the task-7a trio with its caller-side rules; `follow()`'s loss of
     its position veto when the shifted reference leaves the tile), and the report says which
     results would move if they were.
7. **What would have made this a null result** (planning §12) — and whether any of those
   conditions was in fact met. *(revised 2026-10-28: the verdict as it stands, condition by
   condition. (1) residual comparable to `2F` at all filter scales and the budget not closing:
   **met** — 1.4–1.6× in r.m.s., 0 of 71 pairs. (2) the measured slope not distinguishable from
   the discrete-null baseline: **not judged**, because the slope is withheld (M5-Q1; if JXP opts
   to print the slopes, judge it). (3) the residual tracks neither `grad^4 b` nor `KPPhbl` so no
   diabatic signal is isolated: **met** — Figure 2b. (4) the discrete null cannot pass at
   `1 ± 0.05`: **not met** — it passes, 0.981. Two of four met, the first decisive on its own;
   the conclusion is methodological and the limiting factor is named twice over (cause one at
   `L = 0–2`, cause two at `L = 4–8`).)*

### 3. Reproducibility

A `README` in `dev/frontogenesis/` giving the env spec, the exact commands for M0-M5 in order,
and the expected wall time of each. Someone should be able to reproduce the whole study from the
two docs plus this README.

*(revised 2026-10-28: what it must contain, given what exists.)* The env spec is the two files in
`env/` plus the interpreter path and the fact that `conda run -n frontogenesis` silently uses
another env on this workstation; the `fronts` package state the code needs (the `frontogenesis`
branch with `viz_tools` merged, M4-Q2; `dbof` importable, M0 task 3) and the two `fronts` defects
with their caller-side workarounds. The commands, in order, with the wall times **as logged** —
M0 (`m0_write.py`, `m0_qa_*`), M1 (`m1_write_masks.py`, `validate.py`), M2 (`m2_pull.py`,
`m2_chunk_pull.py` — the 42-hour chunk pull and the OSN pull are quoted from the logs and are
**not re-run to verify**; say so), M3 (`m3_run.py`, `m3_closure.py`, `m3_figs.py`, `m3_case.py`),
M4 (the five runners), M5 (`make_figs.py`, the report check, the pandoc build) and the test
suite with its expected count. Three warnings belong in it: `m0_write.py`, `m0_qa_checks.py` and
`m0_recon.py` must never be imported — they write stores on import (prompt 4 task 2b); `data/` is
git-ignored and holds ~10 GB of stores and products that the commands rebuild; every PNG is
git-ignored and tracked with `git add -f`.

---

## Acceptance criteria

- Every figure regenerable by one command from the stores.
- Every figure caption states its `L` and mask.
- The report states the closure tolerance, the null-test baseline, and the §12 null-result
  criteria **explicitly**.
- Limitations section written without hedging or softening.
- *(added 2026-10-28)* The test suite passes with the new figure tests (the registry's order,
  the `L` / mask keys on every entry, the M4 figures redrawn from the JSONs), and the extended
  number check passes on the new report. No figure's numbers change in consolidation.

## Do not

- Do not present a slope without its baseline.
- Do not describe the residual as "diabatic" — with the Q13 terms measured it is
  **numerical diffusion + interior KPP**, and if the chunk terms were unavailable it is a
  catch-all. Say which. *(revised 2026-10-28: the chunk terms were available, so say the first —
  and add the hourly sampling error, which M3 showed is as large as the imbalance at `L = 0–2`,
  so even "numerics + KPP" overstates what the residual isolates there.)*
- Do not claim the result generalises beyond one region, one season and 72 hours.
- *(added 2026-10-28)* Do not quote M4's hourly correlation and amplitude ratio (0.951 / 1.008),
  any per-front ratio of material `ΔG` to `∫2F dt`, or criterion 2's slopes as an efficiency.
  Do not say criterion 4 was met without saying in the same sentence that it failed as
  pre-registered and which construction it was met on. Do not rewrite the interim report, the
  planning document or the coding document — marked notes only.

## Log

Figure inventory with paths; the headline numbers as reported; and an explicit statement of
which planning-doc claims the results **confirmed** and which they **overturned**. That last
item matters most: this project has already overturned four of its own claims during planning
(the residual being purely diabatic, the unfiltered limit isolating it, interpolation error
being negligible, and heat fluxes being unavailable) and five more in M0 against real data
(land stored as 0 — it is NaN; `W[k_l=0] ~ 0` — it is `dEta/dt`; 1.8-2.3 km spacing — it is
1.7-2.1 km on a 90-degree-rotated face; rotation terms ~0.1% — they are identically zero; a
single `kappa_num` — it is strongly scale-dependent), plus four from M0's QA plot (a NaN rim on
the high tile edges only — it is finite and on all four edges, the low ones worst; a ~3-cell
stencil rim — it is 1 cell for `G`, 2 for `F`; `G` as the repo's `gradb2` — it must share
`b_x, b_y` with `F`, the stencils differ by 0.91x; and the interpolated Jacobian trace measured
at 0.80x the flux-form divergence; corrected 2026-09-28, M0 task 5), and the writeup should be
honest about which survived contact with data.

*(added 2026-10-28: the list continues through M1–M4, and the log entry compiles all of it from
the log entries named.)* M1: the discrete null passed (0.981) but the chain form's own baseline
is 0.79 and the finite-volume truth adds a systematic band (V3b, M1 task 6b); the planning risk
that semi-Lagrangian interpolation bias is 25–80 % of the signal — measured at 0.28–1.0 % of
`G` per hour at order 3 (V4, M1 task 5), small but signed, with the half-cell bias at a
maximum shown (V5). M2: the 6-hourly forcing makes the diurnal
shortwave a triangle (M2 task 6 item 4). M3: the budget was expected to close at `L >= 2` — it
does not; the residual was expected to change sign over the diurnal cycle (planning §7, Figure
6) — it is positive at every hour; the "diabatic damping" reading — refuted by Figure 2b; the
surface-flux term as a closing term — near-orthogonal to the imbalance; the semi-Lagrangian
estimate as *the* measured tendency — two estimates of it differ by 0.67 of the signal at
`L = 0`. M4: the pre-registered reconciliation on the 1-px skeleton — a registration artefact,
not the material rate; the advected mask as a better predictor than persistence — it never is on
IoU at hourly cadence, the gain is after splits; lifetime vs mean `F` — degenerate on a 72-h
window. Confirmed: one shared operator on both sides (M0 task 5, the 0.91× `gradb2` lesson), the
sub-filter term growing with `L` as the one-hour pilot predicted (0.35 / 0.55 / 0.75 of `2F`),
`KPPhbl`'s diurnal phase, flow-informed tracking changing the links (Lauren's question, planning
Q14), and the planning §11 warning that a mean over a changing pixel set is not material
(set-evolution 5–7× the matched rate).

---

## Q&A

### Claude, 2026-10-28 (review before task 1)

Three questions. Every number is from the prompt-4 and prompt-5 log entries and the JSON
summaries in `data/`; nothing was recomputed. Each has a recommendation; tasks 1 and 3 start on
them, task 2 waits.

**M5-Q1 — Do the withheld measured-vs-predicted slopes appear in the M5 report?** The interim
report omitted Figure 2 and its four slopes entirely, with one sentence saying they were computed,
are flagged unquotable in the closure summary, and are not reported because the budget they would
be read against did not close (M3-Q24 (a), 2026-10-12). Nothing has changed since: criterion 1
still fails, and M4 added a second number of the same kind (the band's 0.951 / 1.008) that is
likewise recorded and not quoted. This prompt's task 2 item 3 and planning §12's second null
condition both assume the slope is printed; they were written before the gate was judged.
- (a) **Carry M3-Q24 (a) over**: the report states the baseline and its bands, criterion 2's
  slopes, and that criterion 4's slopes exist and are withheld; Figure 2 stays in `figs/` under its
  banner and is not in the report; §12 condition 2 is recorded as not judged. **Recommended** —
  the M5 report is the one that will circulate, and a slope with a banner is still a slope.
- (b) An appendix only (M3-Q24 (c)): Figure 2 with the banner and the four numbers, out of the
  narrative, so that a reader who wants to see the shape of the cloud can; §12 condition 2 then
  judged there and nowhere else.
- (c) Print them in the body against the baseline, as the prompt's original item 3 says. This
  contradicts the "Do not" list of prompts 4, 5 and 6 unless you lift it, so it needs your word.

**M5-Q2 — What the M5 report is, relative to the interim.** The interim is 454 lines, 22 PDF
pages, M0 → M3, and over its Q19 length by a third (recorded, not accepted). M4 adds roughly a
section and a half (task 2 item 5 above) and the synthesis adds §12's verdict and the full
limitations list.
- (a) **A new document that supersedes the interim**: `report/frontogenesis_report.md`, reusing
  the interim's §2–§4 and §6 with light edits (and the trim Q19 asked for), adding M4, the
  limitations and the §12 verdict; the interim stays untouched as the 2026-10-12 record and the
  README says which is current. **Recommended** — one document to circulate, and the §12 verdict
  changes how the summary is written.
- (b) Extend the interim in place under a new title: least writing, but the 2026-10-12 record is
  lost to edits, and the planning rule is marked notes rather than rewrites.
- (c) A short M4 + synthesis addendum (~6 pages) that cites the interim for M0 → M3: smallest,
  but a reader then needs two documents and the limitations are split across them.

**M5-Q3 — The scope of "one command".** The acceptance says every figure regenerable from the
stores by one command. Figures 8, 9 and the diagnostic depend on the M4 products (tracks, flow
store, series), which take 71 minutes to rebuild from the stores; the ten M3 figures read the
derived stores and the closure summary in under two minutes; V1–V6 are synthetic and quick.
- (a) **`make_figs.py` draws every figure in `figs/` and, with `--report`, `report/figs/`, from the
  products by default and from the stores with `--recompute` (re-running the five M4 runners;
  the M3 closure is an M3 product and is not re-run)**; deck figures stay with the deck builds.
  **Recommended** — the M3 precedent is that figures read the summaries and never recompute, and
  the flag keeps the acceptance literally true.
- (b) The command always recomputes from the stores (71 min per run), no flag.
- (c) The command covers `figs/` only; `report/figs/` keeps its two scripts.

#### JXP
