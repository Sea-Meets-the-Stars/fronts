# `dev/frontogenesis/deck/` — slide decks

Two decks live here. Both are built by script from material already in the repo, so
neither can drift from what it claims to summarise.

| Deck | Built by | Summarises |
|---|---|---|
| `Frontogenesis_Planning.pptx` | `build_deck.py` | The planning docs (13 slides, 2026-09-19) |
| `Frontogenesis_M0_Acceptance.pptx` | `make_m0_figs.py` + `build_m0_deck.py` | Milestone M0 (12 slides, 2026-09-29) |

---

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
