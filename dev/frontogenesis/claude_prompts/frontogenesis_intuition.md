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

We are going to generate a new file named `report/frontogenesis_intuition.md`
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

2.  I have answered your questions.  Please read and respond to those, including
updating the Inuition file.
I will then pause for now.
Use Fable if you can.  Log your work.

3. Actually, I need to understand better the difference between the continuum and discrete forms of 2F.
Please add additional detail to Section 1.9 of the `report/frontogenesis_intuition.md` file.
Include equations and explain further why the discrete form is intended to be a better match for the semi-Lagrangian difference.
Use Fable if you can.  Log your work.

## Q&A 

### 2F

*Fable 5.2, 2026-11-23, 2F prompt 1.*

**I-Q1 — The file's location.** The prompt names `reports/frontogenesis_intuition.md`, but the
existing folder is `report/` (no `s`). I wrote `report/frontogenesis_intuition.md`, beside the
other reports, with the new figures in `report/figs/intuition_2F_*.png`. **OK, or do you want a
new `reports/` folder?**

>A. `report/` was correct. I have modified the file name in Context

**I-Q2 — Three motions or two.** Figure 1 shows the rotation as a third panel beside
convergence and strain, although it contributes nothing to `2F`. I kept it because it explains
why `ζ` is absent and how it still acts (through `θ`). **Keep it, or show only the two terms?**

>A. Keep it

**I-Q3 — Slide versions.** The two schematics are sized for the note (12 in wide). They answer
2F-Q9 of `frontogenesis_slides.md` (a schematic of the pure flows). **Do you want
slide-format versions (10 × 4.45 in, larger fonts) in `slides/`?**

>A. No, these are fine.  I'll make new figures for the Slides if needed.

**I-Q4 — What comes next in this section.** Three candidates:
- (a) the equation for the angle itself, `Dθ/Dt` (strain turns `∇b` toward the compressional
  axis, vorticity turns it away), which explains why fronts sit near `θ ≈ 30°`. **Recommended**:
  it is the step §1.6 points to and does not yet derive;
- (b) the discrete `2F` of the store against the continuum form (why 0.82);
- (c) `2F` against the measured `DG/Dt` on the tracked front (2F-Q5 of the slides, deferred).

>A. Yes, do your recommendation

#### Round 2 (Fable 5.2, 2026-11-23, 2F prompt 2): responses to the answers

- **I-Q1 (`report/`).** Nothing to change: the note and its figures are already in `report/`.
- **I-Q2 (keep the rotation panel).** Kept. Figure 1 is unchanged, and the rotation now has
  its own section (§1.7), so the panel leads somewhere.
- **I-Q3 (no slide versions).** None made.
- **I-Q4 (a) (the angle equation).** Done, as the new §1.7 "What sets the angle", with a test
  on the tracked front in §1.8. In brief:
  - the equation is `Dφ/Dt = (|σ|/2)(r − sin 2φ)`, with `φ` the signed angle of `∇b` from the
    compressional axis and `r = (ζ − 2 Dψ/Dt)/|σ|`;
  - for `|r| < 1` the angle locks at `sin 2φ* = r`, and the strain term falls to
    `|σ|√(1 − r²)`; for `|r| > 1` the gradient keeps turning and strain gives no net growth;
  - on label 83 at pair 36, with `r ≈ ζ/|σ|`: `φ` has the sign of `ζ` on 70% of the cells, and
    the binned medians of `sin 2φ` lie near the line `sin 2φ = r`. Cell by cell the scatter is
    large (correlation 0.50), and the measured mean `cos 2θ` is 0.35 against 0.82 for the
    steady angle.

**I-Q5 — Two open ends of the angle test, for when you resume.**
- The turning of the strain axes, `Dψ/Dt`, is not measured, so the test uses `r ≈ ζ/|σ|`.
  Measuring it needs a new operator (the change of `σn`, `σs` following the water across the
  hour). **Do you want it?** My view: worth doing only if the angle becomes part of the
  report's argument.
- The test is one hour of one front. Repeating it over the 71 pairs of track 81 would say
  whether the result is typical; it is about an hour of computing with the existing code.
  **Recommended as the next step if you continue this thread.**

>A. Let us hold off for now

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

### 2026-11-23 — 2F, prompt 2 (Fable 5.2)

**Done.** The answers to I-Q1 to I-Q4 are read and answered in Q&A/2F, Round 2. The note
gained §1.7 (the angle equation, I-Q4 (a)) and a test of it in §1.8. I-Q5 is new.

- **Answers:** I-Q1 `report/` is right; I-Q2 keep the rotation panel; I-Q3 no slide versions;
  I-Q4 (a). Only I-Q4 needed work.
- **The note:**
  - new §1.7 "What sets the angle": `Dφ/Dt = (|σ|/2)(r − sin 2φ)` (equation 8), derived from the
    cross-gradient component of `D∇b/Dt = −(∇u)ᵀ∇b`; the three cases `r = 0`, `|r| < 1`
    (steady angle, `cos 2θ* = √(1 − r²)`, the link to Okubo–Weiss) and `|r| > 1`;
  - the old §1.7 and §1.8 are now §1.8 and §1.9; the slide figures are Figures 4 and 5;
  - §1.8 gained "The angle against the vorticity" with Figure 6;
  - the notation table has `φ` and `r`; §1.1 and §1.6 point to §1.7.
- **Code, `py/intuition.py`:**
  - `dphi_dt`, `steady_angle`, `integrate_angle`, `check_angle_equation`, `fig_angle_theory`
    → `report/figs/intuition_2F_angle_theory.png` (in `--what schematics`);
  - `angle_numbers` (in `--what numbers`) and `fig_angle_data` (`--what data`, drawn from the
    JSON with no store) → `report/figs/intuition_2F_angle_data.png`.
- **One line added to `py/slides_2F.py`:** `compute` now also returns `zeta` (`v_x − u_y` from
  the Jacobian it already computes). I re-ran `slides_2F.py`; its assertions pass and it rewrote
  `2F_method.png` and `2F_split.png`. I had no earlier checksum to compare them with; the
  drawing code is untouched.
- **Asserted:**
  - equation (8) against the turning of a gradient carried by `exp(τ∇u)` for three velocity
    gradients (2 × 10⁻⁷);
  - pure strain: `tan φ = tan φ₀ e^(−|σ|t)` and `G ∝ cosh` (10⁻¹⁰);
  - integration reaches `φ* = ½ arcsin r`;
  - on the data, `|φ|` equals `strain_alignment`'s `θ` (6 × 10⁻¹⁴ rad).
- **Numbers** (`data/intuition_2F.json`, `numbers.angle`; band of label 83, pair 36, 1,143 cells):
  - `|ζ| < |σ|` on 73% of cells; median `|ζ|/|σ|` 0.61; 42% cyclonic; median `ζ` −0.03 f;
  - `φ` has the sign of `ζ` on 70%; median `φ` +26° (cyclonic, 476 cells), −15°
    (anticyclonic, 667);
  - binned median `sin 2φ` for `r` in (−1, −0.5), (−0.5, 0), (0, 0.5), (0.5, 1): −0.62, −0.23,
    +0.43, +0.83;
  - where `|r| < 1`: correlation 0.50; median θ 27° measured against 13° steady; mean `cos 2θ`
    0.35 against 0.82;
  - `1/|σ|` at the median strain: 19 h.
- **Caveats written into the note:** `Dψ/Dt` is not measured; one front, one hour; the
  Lapeyre, Klein & Hua (1999) reference is from memory and unchecked.
- **Visual check:** two passes on the theory figure (overlapping labels in (c), curves leaving
  (b)); the data figure was accepted as drawn.
- Not done: no PDF build, no tests added, nothing committed.

### 2026-11-23 — 2F, prompt 3 (Fable 5.2)

**Done.** §1.9 of `report/frontogenesis_intuition.md` is rewritten as "The continuum and the
discrete forms of `2F`" (it was two bullets). No new questions. I-Q5 is read ("hold off") and
nothing was done on it.

- **Read:** `operators._frontogenesis_discrete` and its docstring; `semilag.measured_DGDt`,
  `gradb2_at_departure`; `tracked/matched_parcels_track_81.md` §1.1 (the step-by-step
  semi-Lagrangian section); coding §(operators, V3); `figs/V3_discrete_null.png`.
- **The section:**
  - a table of the two forms (code name, what it is, where used, the pair-36 band mean);
  - the stencil gradient as the mean of the true gradient over two cells (equation 9);
  - the semi-Lagrangian difference written out for the two stencils under pure advection,
    giving `D(L_k b)/Dt` (equation 10) and the discrete `2F` (11), with the commutator named;
  - the one factor in which the forms differ, in 1-D with a uniform `u'`: the slope at the
    stencil's two ends (discrete) against the mean slope over the stencil (chain) (12), the
    expansion `1 − (2/3)(h/ℓ)²` (13), and the step-front limit;
  - the three choices in the code (same stencil, same centred velocity, fourth-order slope at
    the neighbours);
  - why that makes it the better match (an identity as Δt → 0 under pure advection, so the
    stencil cannot appear in measured − `2F`), the V3 slopes, and what is not claimed;
  - the tracked front: ratio 0.82 against 0.87 for a 2.2-cell front;
  - the old bullets on the strain set and `strain_alignment` are kept at the end.
- **Code, `py/intuition.py --what discrete`:**
  - `front_1d`: a tanh front in `u = −a x` on a grid, at the centre cell: the chain form,
    equation (10) with the exact slope, the code's fourth-order form, and one semi-Lagrangian
    step of exact advection;
  - `fig_discrete` → `report/figs/intuition_2F_discrete.png` (Figure 7);
  - `width_numbers`: the stored `front_width` on the label's pixels. It reads two stored
    fields; nothing is recomputed.
- **Reused:** `figs/V3_discrete_null.png` as Figure 8.
- **Asserted:** the 1-D semi-Lagrangian step equals equation (10) at every width drawn
  (tolerance 10⁻⁴; they agree to about 3 × 10⁻⁶); the label has the 307 px of the summary.
- **Numbers** (`data/intuition_2F.json`, `discrete`):
  - ratio to the chain form at ℓ/h = 1, 1.5, 2, 3, 4, 8: semi-Lagrangian step and equation (10)
    0.55, 0.76, 0.85, 0.93, 0.96, 0.99; the code's fourth-order form 0.65, 0.77, 0.85, 0.93,
    0.96, 0.99;
  - grid `G` / true `G` at the same widths: 0.58, 0.76, 0.85, 0.93, 0.96, 0.99;
  - label 83, pair 36: `front_width` median 2.19 cells (quartiles 1.85, 2.62); discrete / chain
    0.82 on the band, 0.81 on the front pixels, 0.83 beside them; the 1-D value at 2.19 cells
    is 0.87;
  - V3 slopes quoted from the existing figure and coding doc: chain 0.951 / 0.791, discrete
    1.004 / 0.981.
- **Two things I noticed and wrote into the note:**
  - the code's fourth-order slope is too high for the thinnest 1-D front (0.65 against 0.55
    at ℓ = h). The 1-D example uses the analytic `b` and interpolates nothing, so this does
    not test the code's match to its own cubic interpolation;
  - at pair 36 the band mean of the measured `DG/Dt` (+4.20) sits beside the chain value
    (+4.21), not the discrete one (+3.46). The note says a single band mean does not test the
    forms and points to V3.
- **Visual check:** two passes on Figure 7 (labels of (b) running into (c)).
- Not done: no PDF build, no tests added, no change to the main report, nothing committed.
