# Frontogenesis intuition

## Goals

I wish to generate further intuition on the process of frontogenesis, 
from its primary equations to its application in the context of tracked fronts.

## Context

- See the design docs, `frontogenesis_coding.md` and `frontogenesis_planning.md`.
- See the slide prompt doc `frontogenesis_slides.md`
- See the reports in report/
- See the code in py/
- See example tracked front in `tracked/matched_parcels_track_81.md`

We are going to generate a new file named `reports/frontogenesis_intuition.md`
It will include equations and figures to help intuitively understand the process of frontogenesis.

## Code

- Python in the `frontogenesis` conda environment
- **JXP runs all git, in both repos.**
- When possible reuse existing code 
- Place import commands at the top of the file.
- Try to reuse existing figures when possible, e.g. from the slides and Reports
- If you generate new code, put it in `py/intuition.py`

## Prompts

### 2F

1.  Read the context above.  We are going to begin by writing out the primary equations 
for 2F as introduced in the context files.  This should include 
its decomposition into a divergence and a strain term.  Include a schematic or two to
explain the terms and the angles involved in the derivation.
If you have questions, put them in the "Q&A/2F" section below.
Use Fable if you can.  Log your work.


## Q&A 

### 2F

*Fable 5.2, 2026-11-23, 2F prompt 1.*

**I-Q1 — The file's location.** The prompt names `reports/frontogenesis_intuition.md`, but the
existing folder is `report/` (no `s`). I wrote `report/frontogenesis_intuition.md`, beside the
other reports, with the new figures in `report/figs/intuition_2F_*.png`. **OK, or do you want a
new `reports/` folder?**

**I-Q2 — Three motions or two.** Figure 1 shows the rotation as a third panel beside
convergence and strain, although it contributes nothing to `2F`. I kept it because it explains
why `ζ` is absent and how it still acts (through `θ`). **Keep it, or show only the two terms?**

**I-Q3 — Slide versions.** The two schematics are sized for the note (12 in wide). They answer
2F-Q9 of `frontogenesis_slides.md` (a schematic of the pure flows). **Do you want
slide-format versions (10 × 4.45 in, larger fonts) in `slides/`?**

**I-Q4 — What comes next in this section.** Three candidates:
- (a) the equation for the angle itself, `Dθ/Dt` (strain turns `∇b` toward the compressional
  axis, vorticity turns it away), which explains why fronts sit near `θ ≈ 30°`. **Recommended**:
  it is the step §1.6 points to and does not yet derive;
- (b) the discrete `2F` of the store against the continuum form (why 0.82);
- (c) `2F` against the measured `DG/Dt` on the tracked front (2F-Q5 of the slides, deferred).

## Logs

### 2026-11-23 — 2F, prompt 1 (Fable 5.2)

**Done.** `report/frontogenesis_intuition.md` is started, with §1 on `2F`. Questions I-Q1 to
I-Q4 are in Q&A/2F.

- **Read:** planning §2.4; `operators.frontogenesis`, `strain_from_jacobian`,
  `strain_alignment`; `tracked/matched_parcels_track_81.md` §1.1; the 2F prompts, Q&A (Rounds
  1–3) and logs of `frontogenesis_slides.md`; `py/slides_2F.py`; the report's use of `2F`.
- **The note, §1:**
  - 1.1 the result: `DG/Dt = 2F`, `2F = −δG + |σ|G cos 2θ`, and the rate form
    `(1/G) DG/Dt = −δ + |σ| cos 2θ`;
  - 1.2 `G` as isopycnal spacing; 1.3 `D∇b/Dt = −(∇u)ᵀ∇b` and the compact form
    `(1/G) DG/Dt = −2 ∂u_n/∂n` (twice the cross-front convergence), which is new to the project's
    write-ups;
  - 1.4 the split of `∇u` into convergence + strain + rotation (Figure 1);
  - 1.5 the angles `α`, `ψ`, `θ` and the sign convention (Figure 2);
  - 1.6 what each term does, including that `θ = 45°` is not a steady state;
  - 1.7 the terms on label 83 at pair 36, reusing `slides/2F_method.png` and `2F_split.png`;
  - 1.8 the continuum and discrete forms in the code.
- **New code, `py/intuition.py`** (frontogenesis env):
  - `--what schematics` writes `report/figs/intuition_2F_terms.png` (one flow = convergence +
    strain + rotation, each acting on a ring of water and a set of isopycnals) and
    `report/figs/intuition_2F_angles.png` (the angles; one strain on fronts at θ = 0°, 45°, 90°;
    the `cos 2θ` curve with the parcel's 27° and the band median 29°). No data store is opened.
  - `--what numbers` (about 1 min) writes the growth rates of §1.7 to `data/intuition_2F.json`.
  - It reuses `operators.strain_from_jacobian` / `strain_alignment` for every rate on the
    schematics, and `slides_2F.compute` / `check` for the data.
- **Asserted:** on the schematics, `−δ + |σ| cos 2θ = −2 nᵀ(∇u)n` (1e-12), the three parts add
  to the total, rotation gives zero, and the drawn deformation equals `exp(rate × τ)` for the
  pure flows. On the data, the band means equal `slides_2F_36.json` (+1.29, +2.92, +4.21) and
  `⟨G⟩ ×` (G-weighted rate) `= ⟨2F⟩`.
- **Numbers** reproduce the slides' Round 3, now traceable to a JSON: parcel +2.18 + 0.87 =
  3.05 × 10⁻⁵ s⁻¹ (0.45 f, e-folding 9.1 h); band, G-weighted, +0.22 + 0.51 = 0.73 × 10⁻⁵ s⁻¹
  (0.11 f, 38 h); strain share 69%; 59% of cells converge; θ < 45° on 72%. Two small differences
  from Round 3: `f` is 6.82 × 10⁻⁵ s⁻¹ at the band's mean latitude (Round 3 quoted 6.80), and the
  G-weighted mean of `cos 2θ` is 0.30 (the 0.33 of Round 3 is the unweighted mean); the note
  gives both.
- **Visual check:** two passes. Fixed a clipped title in Figure 1, and in Figure 2 a clipped
  panel (d), a legend over the x label of (e) and oversized faint arrows in (a).
- Not done: no PDF build, no change to the main report, no tests added, nothing committed.
