# `dev/frontogenesis/report/` — the interim report

| File | What it is |
|---|---|
| `frontogenesis_M3_interim.md` | **The report.** M0 → M3, written for a reader outside the project — M3 did not close, and this is where that is explained |
| `make_report_figs.py` | Draws the two schematics in `figs/`. Everything else the report shows is re-used from `../figs/` and `../deck/figs_m3/` |
| `pandoc-header.tex` | LaTeX preamble for the PDF build: figure size caps, and hyphenation for the long JSON keys in Appendix B |
| `check_report_numbers.py` | Re-reads every JSON-backed number from the task-6 / task-7 summaries and asserts it appears in the report; also enforces the prompt's "Do not" list |
| `figs/schematic_pipeline.png` | The method as a picture: two stores → buoyancy → one operator set → the filter → the predicted and measured arms → the five-term budget → the residual |
| `figs/schematic_budget.png` | A front in cross-section, each process colour-matched to its term, with the measured term amplitudes beside it |
| `make_case_figs.py` | Draws the three §5.1 case-study figures from the case summary |
| `figs/case_front_maps.png` | The followed front over four hourly snapshots, and the predicted tendency on the same frames |
| `figs/case_strength.png` | Its strength along the trajectory against the prediction, and the hourly rates |
| `figs/case_budget.png` | Its five-term budget beside the 71-pair pooled one |

The project had no picture of its own method before these two. That is the one thing a reader
who is not embedded needs most, and it is why M3-Q20 (b) asked for them.

## The case study (§5.1)

`../py/m3_case.py` follows one strong front — 450 parcels seeded on the front pixels of a
corridor across a persistent filament at 37 °N, carried by the model's own velocity for a day
either side of the seed time — and writes `../data/m3_case_summary.json` and
`../data/m3_case_fields.npz`. It is the only part of the report that touches a data store, and
it touches it **once**: the figures and the prose read the summary, never the store.

Two checks are in the summary and both must hold. A parcel started at the departure point the
budget's own backward map gives for a grid point, stepped forward one hour, returns to that
point to a median 5.3e-4 cells — the integrator is the budget's, run backwards. And the case's
own Lagrangian rate agrees with the stored measured column at correlation 0.901 and amplitude
ratio 1.13.

**No front finding.** The prompt's "Do not" list reserves labelling, thinning and despurring for
M4. The front is a hand-drawn box; the parcels inside it are chosen by the same 90th-percentile
rule the rest of the report uses. What is followed is water, not an object.

An **unselected control** of 2,500 parcels is carried alongside, because parcels picked for
being strong now must weaken afterwards whatever the physics does. The report says so and
quotes both.

## Building the PDF

The `pandoc` binary is not on this workstation's `PATH`; the copy bundled with `pypandoc`
is what the build below uses. From this directory:

```
PANDOC=$(python3 -c 'import pypandoc; print(pypandoc.get_pandoc_path())')
$PANDOC frontogenesis_M3_interim.md -o frontogenesis_M3_interim.pdf \
        --pdf-engine=xelatex --include-in-header=pandoc-header.tex \
        -V geometry:margin=1in -V fontsize=11pt
```

`pandoc-header.tex` caps every figure to 0.95 of the text width and 0.78 of the text height,
keeping the aspect ratio — without it the two multi-panel page figures come out too small to
read and the wide schematics overrun. **Tested 12 October 2026: 22 pages, no missing glyphs.**
The PDF is not committed; it is rebuilt from the Markdown, which is the source of record.

Markdown is the format of record (planning §3, decision Q4, "reports in markdown"), and it is
what M5's `frontogenesis_report.md` will grow from. Equations are LaTeX, which GitHub, VS Code
and pandoc all render, so nothing is lost by not writing the source in LaTeX.

## The rules this report is written under

Set by JXP's answers to **M3-Q16..Q26** (`../claude_prompts/frontogenesis_prompt_4.md`,
`## Q&A` → `### Report`, answered 2026-10-12):

- **Audience** — an advanced PhD student in physical oceanography. Full physics, every symbol
  introduced in words, no code identifiers in the running prose.
- **Scope** — M0 → M3 in proportion, then M3 in full; the design document summarised from §1,
  §2, §5.3-§5.4, §11 and §12.
- **Numbers** — from the task log entries in `../claude_prompts/frontogenesis_prompts.md` and
  from `../data/m3_closure_summary.json`, `../data/m3_figs_summary.json` and
  `../data/m3_case_summary.json` **only**. No data
  store is opened, by the report or by either script here. `check_report_numbers.py` enforces
  the JSON-backed half; the M0-M2 numbers are log-only and the report's numbers appendix marks
  them as such.
- **The "Do not" list** — no frontogenesis efficiency is quoted, no diabatic damping is
  claimed, and the measured-versus-predicted slopes are omitted entirely rather than shown
  under a banner, because a banner does not stop a reader carrying the number away.

## Checking it

```
python check_report_numbers.py        # exit 0 if the report matches the summaries
python make_report_figs.py            # redraws the two schematics
```
