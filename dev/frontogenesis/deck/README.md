# `dev/frontogenesis/deck/` — slide decks

Three decks live here. All are built by script from material already in the repo, so
none can drift from what it claims to summarise.

| Deck | Built by | Summarises |
|---|---|---|
| `Frontogenesis_Planning.pptx` | `build_deck.py` | The planning docs (13 slides, 2026-09-19) |
| `Frontogenesis_M0_Acceptance.pptx` | `make_m0_figs.py` + `build_m0_deck.py` | Milestone M0 (12 slides, 2026-09-29) |
| `Frontogenesis_M1_Acceptance.pptx` | `make_m1_figs.py` + `build_m1_deck.py` (+ `check_m1_deck.py`) | Milestone M1 (19 slides, 2026-09-30, extended 2026-10-01; no text below 20 pt) |

---

## M1 acceptance deck — work log, 2026-09-30

**Task.** `frontogenesis_prompt_2.md` task 8: a small deck for M1 acceptance — title, table
of contents, one slide per task, figures where possible, generating scripts kept here, and
**no text smaller than 20 pt**.

### What was built

```
deck/
  make_m1_figs.py                 five panel crops from ../figs/ + five large-font re-plots
  build_m1_deck.py                builds the .pptx (helpers copied from build_m0_deck.py)
  check_m1_deck.py                font floor (every run >= 20 pt) + text-box geometry check
  figs_m1/
    m1_crop_V1a.png                 V1 panel (a): G along parcels vs exp(2at)
    m1_crop_V3d.png                 V3 panel (d): the LLC gate, 0.981 [0.970, 0.994]
    m1_crop_V3b_c.png               V3b panel (c): OS7MP-like truth, 0.975 [0.954, 1.003]
    m1_crop_V5a.png                 V5 panel (a): the half-cell shift and its four biases
    m1_crop_V6a.png                 V6 panel (a): the mask stages on the whole tile
    m1_fig_operators.png            Jacobian vs flux-form strain regression, hour 0
    m1_fig_coarsegrain.png          budget closure with / without the term; term vs L
    m1_fig_gates.png                V1 error vs front width; V4 bias vs front width
    m1_fig_v3_changes.png           every change tried for gate V3 (LLC variant)
    m1_fig_v3b.png                  V3b slope per advection truth with CI (LLC)
  Frontogenesis_M1_Acceptance.pptx  19 slides, 1.3 MB  (15 at the 2026-09-30 build; task 9 added four)
```

**Slides** (numbering after the task-9 additions of 2026-10-01; the original 15 are
unchanged in content). 1 Title; 2 Contents; **3 M0 in one slide; 4 M1 in one slide** (task 9);
5 Task 1 masking + V6 (V6a crop); 6 Task 2 operators + oracle (operators re-plot); 7 Task 3
semilag + V5 (V5a crop); 8 Task 4 coarsegrain (closure re-plot); 9 Task 5 gates V1/V2/V4 (V1a
crop + width re-plot); 10 Task 6 gate V3 (V3d crop + changes-tried re-plot); 11 Task 6b V3b
(V3b-c crop + per-truth re-plot); 12 Task 7a `test_nan_finding.py`; 13 Task 7 decisions applied
+ the audit; 14 Acceptance (the seven criteria); 15 Findings for the writeup; 16 Decisions and
open issues (incl. the `fronts` fixes as *next steps*, per M1-Q9); **17-18 Glossary** (task 9);
19 Task 8, this deck. Task 7 is split into 7a and the audit because they were separate sessions
with separate deliverables.

### Where the content comes from

**Every number is quoted from the M1 task logs** in `../claude_prompts/frontogenesis_prompts.md`
("Execution prompt 2", tasks 1-7, 6b and 7a, 2026-09-28 .. 2026-09-30) and from the Status
paragraph, acceptance criteria and Q&A of `../claude_prompts/frontogenesis_prompt_2.md`.
Nothing is recomputed; `make_m1_figs.py` touches no network and opens no data store. As for
M0, the deck inherits any error in the log — it is a presentation artefact, not a check.

Where the logs updated a number, the latest is used:

- The Jacobian attenuation is quoted as **0.85x** (task 2, on `mask_analysis`), not M0's 0.80
  (whole ocean incl. the coast); slide 4 says so.
- The task-4 closure residuals (0.041 / 0.090 at 900 m) are the task-4 numbers, and slide 6
  notes that task 6's discrete `F` later improved them to 0.037 / 0.088.
- V3b's numbers are those of the task-7 regeneration of the figure, which the log records as
  identical to the task-6b table.
- The "0.80-0.85x attenuation biases M3 high" expectation (tasks 2, 6, M1-Q2) is reported as
  closed by V3b (task 6b / task 7: −0.006 ± 0.025), not as open.

### The 20 pt rule

`build_m1_deck.py` sets an explicit size on every run and clamps it at 20 pt (`MIN_PT`);
tags, captions, footers and the contents circles are all 20-21 pt, titles 30-38 pt.
`check_m1_deck.py` walks every text frame of the saved file: **minimum run size 20.0 pt, no
offender**, 15 slides (19 after task 9, same result). PNGs cannot be governed point by point, so:

- the five **re-plots** are drawn at the inch size they occupy on the slide (figure fonts
  13-21 pt, placed at or near 1:1), with explicit margins so the long y-labels do not inflate
  the image and shrink it on placement;
- the five **crops** are single panels of the 200-dpi V-figures placed at 4.3-5.5 in wide
  (0.75-0.9 of native), so their axis labels are ~9-11 pt equivalent — legible on a projector,
  but the smallest text in the deck. Every number they carry is repeated in the slide text at
  20 pt. The V3d crop has its clipped title and two stray panel-(f) tick labels painted out
  (`make_m1_figs.py`); the slide caption carries the title's content.

### Regenerating

```bash
PY=~/miniforge3/envs/frontogenesis/bin/python
cd dev/frontogenesis/deck
$PY make_m1_figs.py        # matplotlib + PIL only; reads ../figs/V*.png for the crops
$PY build_m1_deck.py       # python-pptx 1.0.2 (installed for M0)
$PY check_m1_deck.py       # font floor + geometry; prints the slide text
```

### QA performed

- **Font floor:** programmatic, as above — 20.0 pt minimum.
- **Render:** LibreOffice is now on this machine (`/opt/homebrew/bin/soffice`), so unlike the
  M0 and planning decks this one was rendered (`soffice --headless --convert-to pdf`, then
  `pdftoppm`) and **all 15 pages inspected**. Two overflows found and fixed before hand-over:
  the slide-7 stat cards (text shortened, cards taller, figures moved down) and the slide-15
  card (second sentence shortened). After the fix: no overflow, overlap or clipped text.
- **Geometry estimator:** `check_m1_deck.py` flags ~18 boxes as "overflow?" at 0.5 em per
  glyph; all are false positives at Calibri's real width (confirmed by the render). It stays
  as a coarse guard for anyone editing without a renderer.
- **Content:** slide text dumped by the checker and read against the task logs.

### Compromises, for the record

- Slide 8's closing paragraph ends ~0.2 in above the bottom edge — inside, but the tightest
  slide. Shorten it before adding anything there.
- V2 and V4 have no crop from `../figs/`: V2's map adds nothing a number does not say, and
  V4's panels are too dense at 20 pt-equivalent; both are carried by the re-plot (V4) and the
  stat cards (V2) on slide 7.
- The crops' own labels are below 20 pt (see above). A fully 20 pt deck would need the
  V-figures re-rendered with large fonts in `validate_figs.py`, which is outside this task.
- The `fronts` fixes (M1-Q9..Q15) are listed on slide 14 (16 after task 9) as next steps:
  **not applied** as of this build.

### Task 9 additions, 2026-10-01

**Task.** `frontogenesis_prompt_2.md` item 9 ("Simplify"): add a glossary of the main terms, a
one-slide summary of M0, and — as written — a second "one-slide summary of M0". The duplicate
line is read as a typo for **a one-slide summary of M1**, so both summaries were built; if a
second M0 slide really was meant, say so and it is a few lines in `build_m1_deck.py`.

**What was added** (only `build_m1_deck.py` edited; `make_m1_figs.py`, `figs_m1/`,
`check_m1_deck.py` and the M0 deck and its scripts untouched; 15 → 19 slides, 1.3 MB):

- **Slide 3, "Where we started: M0 in one slide"** — right after Contents. The goal; the five
  empirical answers with their numbers (land NaN, 0 mismatches in 518,400; `W(k_l=0) = dEta/dt`,
  corr 0.9936 / slope 1.037; 1.71 × 1.85 km, face 10 rotated 90°; rotation terms zero, metric
  0.1%; OS7MP, `diffKhT = 0`, linear free surface); the deliverables (`tile330_grid.zarr`
  1.8 MB, the two hours as a 21 MB raw store, the QA plot's 1 / 2-cell rim); the two overturned
  planning claims (§5.5, §2.2); the closing bar (five criteria PASS, four §2 traps, Jacobian
  0.80x flagged for M1; **closed 2026-09-28**). Quoted from the M0 task-3/4/5 logs,
  `build_m0_deck.py` and `frontogenesis_coding.md` §6 M0.
- **Slide 4, "The answer: M1 in one slide"** — before the per-task slides; the executive view
  rather than a copy of the acceptance or findings slides. Three stat cards (V3 1.004 / 0.981;
  discrete F ~0.79x the chain F; V3b 0.975 [0.954, 1.003], −0.006 ± 0.025), "what changed"
  (first attempt 0.950 / 0.758 → PASS; b interpolated at the departure point, never G), "handed
  to M3" (both forms of F, baseline and band, no upward correction, order 3 / 5, V4 bar, τ_δ),
  and the closing bar (seven criteria PASS, 84 passed / 3 strict xfailed, seven PNGs; **closed
  2026-09-30**). Quoted from the task-6, 6b, 7 logs and coding §6 M3 "Carried from M1".
- **Slides 17-18, Glossary** (two slides — one did not hold 15 terms at 20 pt without cramming),
  placed as an appendix before "This deck" and listed in Contents. Slide 17, physics and
  operators: `b, G`; `F`; `DG/Dt`; discrete vs chain-rule `F`; semi-Lagrangian step / departure
  point; `lowpass, L`; subfilter term τ, τ_δ, Germano. Slide 18, grid, gates and statistics:
  LLC4320 / face 10 / tile 330; C-grid and Jacobian; OS7MP; halo / tile-edge margin / analysis
  mask; front pixels (p90); V3 discrete null vs V3b finite-volume null; slope [lo, hi]; oracle
  and xfail.
- **Contents** now has 16 rows (two columns of eight, circles 0.5 in, no word-wrap so "M0" /
  "M1" stay on one line); **"This deck"** carries a task-9 tag and the slide count. The only
  in-deck cross-reference ("the numbers of the next slide", task 7 → acceptance) still holds.

**Glossary term selection.** The deck text was dumped with python-pptx and ~37 candidate terms
counted by the number of slides using them (scratch `scan_terms.py`, not kept). Kept: every
candidate on two or more slides, plus the headline terms of the findings slide. Dropped:
**JMD95** (on no slide — it now appears only inside the definition of `b`), `config D`, `DST3`,
`hFac`, `OSN` (one slide or none). Definitions use the slides' notation; the CI is described
as the log does it (OLS of `DG/Dt` on `2F` with intercept, 32-cell block bootstrap, 2.5-97.5%),
the OS7MP line uses the M0 log's "flux-limited, seventh-order, monotonicity-preserving (Daru &
Tenaud 2004)", and the Germano line follows the task-4 identity `T − lowpass(tau, L2) = Leo`.

**QA.** `check_m1_deck.py`: **minimum run size 20.0 pt, no offender, 19 slides.** Rendered with
LibreOffice (`soffice --headless --convert-to pdf`, `pdftoppm`) and every new or changed page
(2, 3, 4, 17, 18, 19) inspected. First render found five defects, all fixed before hand-over:
the Contents circles broke "M0" / "M1" onto two lines (word-wrap off); the M0 slide's DELIVERED
card and closing bar overflowed (raw-store file name replaced by "two hours ... 21 MB raw
store", bar text shortened); the M1 slide's stat-card labels and both bullet columns clipped
(cards taller, labels and bullets shortened, bar lowered); the glossary body collided with the
subtitle (subtitle dropped, list starts at 1.55 in); glossary 2 ran to the bottom edge
(definitions tightened, paragraph spacing 5 pt). After the fix: no overflow, overlap or clipped
text on any of the six pages. The twelve reordered task slides (old 3-14 → new 5-16) were
confirmed text-identical to the 2026-09-30 render with `pdftotext`; slide 19 differs only in
its tag and subtitle. The checker's 0.5-em estimator still flags the new boxes as "overflow?"
(as it does the old ones) — false positives at Calibri's real width, confirmed by the render.

**Compromises.** Glossary 2 is now the tightest slide (text ends ~0.35 in above the edge;
shorten before adding a term). The M1 summary repeats five numbers that also appear on the
acceptance and findings slides — unavoidable in an executive summary, but nothing on it is new.
The M0 summary names the raw store by content, not by its file name
(`tile330_raw_20120702T00_2h.zarr`), which would not fit its card at 20 pt in Courier New.

## M0 acceptance deck — work log, 2026-09-29

**Task.** `frontogenesis_prompt_1.md` task 6: a small deck for M0 acceptance — title, table
of contents, one slide per task, figures where possible, generating scripts kept here.

### What was built

```
deck/
  make_m0_figs.py                 generates three figures + downscales the QA plot
  build_m0_deck.py                builds the .pptx
  figs_m0/
    m0_fig1_numerical_diffusion.png   OS7MP implicit diffusion vs the kinematic rate
    m0_fig2_displacement.png          hourly displacement, ocean vs front pixels
    m0_fig3_overturned.png            the two overturned planning claims
    m0_qa_small.png                   ../figs/m0_qa_tile330_20120702T00.png at 1700 px
  Frontogenesis_M0_Acceptance.pptx    12 slides, 1.3 MB
```

**Slides.** Title; Contents; one slide each for tasks 1-6; plus three finding slides
(overturned claims, numerical diffusion, displacement); plus an acceptance slide. The three
extra slides are not padding — task 3 alone produced five results of unequal weight, and the
two contradicted planning claims are, by the prompt's own framing, *"the most valuable output
of this milestone"*. Burying them inside a five-row task slide would have understated them.

### Where the content comes from

**Every number is quoted from the M0 task logs** in `../claude_prompts/frontogenesis_prompts.md`
(entries of 2026-09-27 and 2026-09-28). Nothing is recomputed; `make_m0_figs.py` touches no
network and opens no data store. The consequence worth having is that the deck cannot silently
disagree with the log — but the flip side is that it inherits any error in the log, so it is a
presentation artefact, not an independent check.

The one exception is `m0_qa_small.png`, which is a straight LANCZOS downscale of the real QA
figure from `../figs/` (3800 x 2300 -> 1700 x 1029) so the deck stays near 1 MB instead of
carrying a 1.8 MB PNG.

### Regenerating

```bash
PY=~/miniforge3/envs/frontogenesis/bin/python
cd dev/frontogenesis/deck
$PY make_m0_figs.py        # matplotlib + PIL only
$PY build_m0_deck.py       # needs python-pptx
```

`python-pptx 1.0.2` was installed into the `frontogenesis` env for this task
(`pip install python-pptx`); it is the only dependency the env did not already have, and it
is a presentation-layer tool, so it was deliberately **not** added to
`../env/frontogenesis_env.yml` — nothing in the analysis path imports it.

### QA performed, and its limit

- **Geometry and overflow:** programmatic check over all 12 slides — no shape off-slide, no
  estimated text overflow, margins >= 0.5 in. Clean.
- **Content:** slide-by-slide dump verified against the task logs; 4 images placed as intended.
- **Not done: visual inspection.** There is no LibreOffice on this machine (`soffice` absent),
  so the usual render-to-image pass could not run — the same limitation recorded for the
  planning deck on 2026-09-19. The geometry check is a proxy, not a substitute; **a human eye
  on first open is still worth it.**

### Notes for whoever picks this up

- Slide 3 records that the brief's `xgcm<0.10` pin was wrong and was corrected to `>=0.10`.
  If the env is ever rebuilt from an older doc, that regression returns.
- Slide 7's figure is the one to look at before believing any future "frontogenesis
  efficiency" number: at 4dx the implicit diffusion is 30-100% of the kinematic rate, and at
  ~1.5-cell fronts, where the MP limiter engages, the numerics are the whole story.
- Slide 8 notes `follow(km_per_px=2.3)` is 25% too large for this tile (~1.8). That is an M4
  fix and is easy to forget.

---

## Planning deck — 2026-09-19

`build_deck.py` -> `Frontogenesis_Planning.pptx` (13 slides). Summarises
`../frontogenesis_planning.md`: the question, the budget, why surface-only works, what exists
vs what is new, the data, the method, the residual problem, the validation gates, the figures,
the milestones, risks, and the null-result criteria.

**Status: built and QA'd, never uploaded.** It was intended for the AIOcean Drive under
`data/HIINet/Frontogenesis/` as `Frontogenesis_Planning`. The upload stalled because the Drive
connector could only create a Slides file from inline base64, and 52 KB of base64 could not be
passed in one piece without risking a single-character corruption. Easiest completion is to
drag the `.pptx` into that folder and let Drive convert it (File > Save as Google Slides).

Note also that the planning deck predates M0 and therefore still states two claims that M0
overturned — land stored as 0, and `w` vanishing at the surface. **Rebuild it before showing
it to anyone**; `build_deck.py` would need those slides corrected first.
