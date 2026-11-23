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

- Python in the `frontogenesis` conda environment
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

3. Now I wish to make a GIF of the tracking process for the same `label 81` front.  
This should span the entire analysis window, and show the movement of this front.
Unlike the previous figure, it is ok to just show the thin, skeleton front and not
the lassoed parcels.  
Use the same module (`py/slides_tracking.py`) and name the GIF `slides/tracking_81.gif`.
Show the `G` field as the background.  Zoom in on the front of interest.
If you have any questions, put them in the Q&A/Tracking section below.
Use Opus 5.5. Log your work.

4. This is very good.  Please make these modifications and regenerate the GIF:
    - Use red to trace the active front, instead of cyan
    - Do not show the previous hour, only the current front

5. Ok, now create a separate GIF that shows the tracking of the top 10 fronts; the
same ones that we focused on in the track analysis.  Use a different color for each front. 
Use the same module (`py/slides_tracking.py`) and name the GIF `slides/tracking_top10.gif`.
Show the `G` field as the background in grey scale. 
If you have any questions, put them in the Q&A/Tracking section below.
Use Opus 5.5. Log your work.

#### Method

1.  Now I want a graphic that shows how we track a front from time step to time step.
It should include the equations related to the process and ideally data, e.g. the `G` field
and arrows indicating the movement of the front.
We will add to the `py/slides_tracking.py` module.
Be sure to review the code that performs the tracking.
Before writing code, discuss this with me.  Put comments
and questions in the "Q&A/Tracking/Method" section below.
Use Opus 5.5. Log your work.

2. Read my answers to the Q&A/Tracking/Method section below and proceed to generate the figure.
If you have any additional questions, put them in the "Q&A/Tracking/Method" section below.
Use Opus 5.5. Log your work.

3. I am not following all of items presented in the Figure. Let us create a separate
doc that describes the tracking process in detail.  Call it `report/frontogenesis_tracking.md`.
Be sure that it defines and describes the equations and varaibles shown in the Figure,
e.g. M, IoU.  I also have a few questions of my own on items in the Figure:
   - Why does panel (a) refer to 307 pixels and then (b) refers to only 305?
   - Why was the "spur" near 125W and 28N not chosen in pair 36?
   - What are the green "allowed" pixels?

    Put your answers and any new questions in the "Q&A/Tracking/Method" section below.
    Use Opus 5.5. Log your work.

4. Ok, the report is helpful, thanks.  And I have answered all of the questions in the Q&A/Tracking/Method section below.
Please modify the figure to only show panels (a), (b) and (c).
No title.  Have it size to fit on a standard Google Slide that will include its own title.
Use dpi = 300.
Make the fonts no smaller than 16pt throughout unless absolutely necessary.
Use Opus 5.5. Log your work.

5. Ok, some of the text is a bit too large and crowd the figure. Please
    - Use small fonts for the legends.
    - Drop the "2 of 307 px dropped" text

   Use Opus 5.5. Log your work.

### dG/dt

1. I would like to create a figure that shows how we calcualte dG/dt for a tracked front
between two time stamps (1 hour).  Let us use the usual `label 81` front.  
Ideally, I wish show the data itself, but we may need to be graphical.
Discuss this with me before creating anything.  Put comments
and questions in the "Q&A/dG/dt" section below.
Use Opus 5.5. Log your work.

2. I have answered the first round of questions in Q&A/dG/dt.  Please read them
and continue the discussion.  Do not create a figure yet.
Use Opus 5.5. Log your work.

3. I have answered the second round of questions in Q&A/dG/dt.  Please read them
and proceed to generate the two figures with one script named `py/slides_dGdt.py`.
The fiures can be named `slides/dGdt_method.png` and `slides/dGdt_result.png`
unless you have a better suggestion.
Use Opus 5.5. Log your work.

4. Let us make a few more edits to the `slides/dGdt_method.png` figure:
    - Remove the title
    - Remove the footnote
    - Increase the fonts throughput
    - Move the contour labels in panel (b) to be away from the center pixel
    - Remove the extra text beneath panel (b)
    - Minimize white space
    - Note that this will inserted into a Google Slide
    - Indicate that panel (d) is the interpolated `G` field

    Use Opus 5.5. Log your work.

### 2F

1. I also wish to create a figure that shows how we calcualte 2F for a tracked front
between two time stamps (1 hour).  Let us use the usual `label 81` front.  
As with the dG/dt figures, I wish show the data itself, but we may need to be graphical.
Discuss this with me before creating anything.  Put comments
and questions in the "Q&A/2F" section below.
Use Opus 5.5. Log your work.

2. I have answered the first round of questions in Q&A/2F.  Please read them
and proceed to generate the two figures with one script named `py/slides_2F.py`.
The figures can be named `slides/2F_method.png` and `slides/2F_split.png`.
If you have any additional questions, put them in the "Q&A/2F" section below.
Use Opus 5.5. Log your work.

3. Let us make a few more edits to the `slides/2F_method.png` figure:
    - Remove the title
    - Remove the footnote
    - Remove the text giving G and 2F values at the center of the figure
    - Increase the fonts throughout

    If you have any additional questions, put them in the "Q&A/2F" section below.
    Use Opus 5.5. Log your work.

4. I am still trying to inuit the decomposition of 2F into a divergence and a strain term.
Please provide a further derivation in the Q&A/2F section below.  I will then ask to include
at least a portion of this in the main Report.
If you have any additional questions, put them in the "Q&A/2F" section below.
Use Opus 5.5. Log your work.

5. Back to the `slides/2F_method.png` figure.  Please make a few more edits:

## Q&A

### 2F

*Opus 5.5, 2026-11-19, 2F prompt 1. Nothing has been drawn yet. The method is from
`operators.frontogenesis` and `tracked/matched_parcels_track_81.md` §1.1. The numbers are band
means on the derived store at pair 36 (label 83, 1,143 parcels, the same as the dG/dt figures),
in 10⁻¹⁹ s⁻⁵.*

**What `2F` is.** In the continuum, `F = −(∇b)ᵀ(∇u)(∇b) = −(u_x b_x² + (u_y+v_x) b_x b_y + v_y b_y²)`,
and `DG/Dt = 2F` for adiabatic horizontal flow. Written with the divergence `δ`, the strain
magnitude `|σ|` and the angle `θ` between `∇b` and the compressional axis:

`2F = −δ G + |σ| G cos 2θ`

That is, *convergence* sharpens a front, and *strain* sharpens it when it compresses across
the front (θ < 45°). This is the picture a slide can carry.

**What is stored (`two_F`) is the discrete form.** It is the commutator of the 2-cell gradient
stencil with advection, `−2 Σ_k (L_k b)[L_k, u·∇]b`. This is the form the semi-Lagrangian
difference tends to on this grid, and it equals the continuum form for a linear `b`. On narrow
fronts it is smaller than the continuum form.

| pair 36, band of label 83 | mean | ridge / flank | r.m.s. |
|---|---|---|---|
| `2F` stored (discrete) | **+3.46** | +5.01 / +2.90 | 16.2 |
| `2F` continuum (`two_F_chain`) | +4.21 | +6.18 / +3.48 | 18.6 |
| divergence term `−δ G` (stored flux-form `δ`) | +1.30 | +1.59 / +1.20 | 13.5 |
| strain term `\|σ\| G cos 2θ` | +2.99 | +4.21 / +2.54 | 13.6 |
| `DG/Dt` (semi-Lagrangian, for comparison) | +4.20 | +12.4 / +1.2 | 22.0 |

- The discrete form is 0.82× the continuum form on this band (correlation 0.95 cell by cell).
- **Strain contributes about 70 %** of the continuum `2F`. The median θ is 29°, and 71 % of the
  parcels have θ < 45°, i.e. they are being compressed across the front.
- The median `|δ|` is 1.0 × 10⁻⁵ s⁻¹ (0.14 f) and the median `|σ|` is 1.6 × 10⁻⁵ s⁻¹ (0.23 f).
- With the stored flux-form `δ`, `σ`, the two terms sum to 4.29, not exactly 4.21. With the
  Jacobian-consistent set (`operators.jacobian` → `strain_from_jacobian`) the split is exact,
  so the script would recompute it that way.

**2F-Q1 — "Between two time stamps (1 hour)".** `2F` uses **no time difference**. It is
evaluated at one instant, the midpoint 12:30, from the hour-averaged fields
`b_mid = ½[b(12:00) + b(13:00)]` and `u_mid = ½[u(12:00) + u(13:00)]`. The two time stamps enter
only through that average. I would show this as a small inset (the two hours → the average →
`2F` at 12:30) and otherwise treat `2F` as an instant, in contrast to the one-hour
semi-Lagrangian difference. **OK?**

>A. Yes, I agree.  Do as you describe

**2F-Q2 — Which form do we explain?**
- (a) Explain with the continuum picture (`2F = −δG + |σ|G cos 2θ`: gradient, strain axes,
  convergence), quote the **stored discrete** value as the number (+3.46), and note on the slide
  that the discrete form is 0.82× the continuum form on these parcels. **Recommended.**
- (b) Draw the discrete commutator stencil (velocity differences to the neighbours × the
  gradient at the neighbours), parallel to the semi-Lagrangian stencil. It is exact, but too
  technical for a slide.
- (c) Both: (a) on the slide, (b) in a backup slide.

>A. Ok, let us do (a)

**2F-Q3 — The pair.** I suggest **pair 36** again: the same front, parcels, window and parcel as
`dGdt_method.png`, so the two slides compare directly (+3.46 against +4.20). **OK?**

>A. Yes, use `pair 36`.

**2F-Q4 — Layout.** Two figures, mirroring the dG/dt pair:
- **`slides/twoF_method.png`** (2 × 3):
  - (a) the front and the window, as before;
  - (b) the same parcel close up (5 × 5 cells): `b_mid` contours, the gradient arrow `∇b`, the
    velocity **relative to the parcel** (`u − u(x)`, quivers, which shows the strain), the
    compressional axis and the angle θ, with its numbers (`G`, `δ`, `|σ|`, θ, `2F`) underneath;
  - (c) `2F` on every parcel, with the band mean +3.46;
  - (d)–(f) the window: the relative velocity on `G`, `θ` (or `cos 2θ`), and `2F`.
- **`slides/twoF_split.png`** (2 × 3), divergence + strain = `2F` (continuum), every parcel on
  top and the window below: `−δG` + `|σ|G cos 2θ` = `2F_chain`, then a note → stored discrete
  +3.46. This parallels `dGdt_split.png`.

Alternatively, fold the split into the bottom row of the method figure and make one figure.
**Which?**

>A. It is fine for there to be two figures as with dG/dt.  Let's try that

**2F-Q5 — A comparison slide?** The obvious next figure is `2F` against `DG/Dt` on the same
parcels: side-by-side maps, or a per-parcel scatter. The band means are 3.46 vs 4.20 and the
cell-by-cell correlation is 0.59. The ridge (5.0 vs 12.4) and the flanks (2.9 vs 1.2) differ in
opposite directions. Do you want this now, or as its own prompt later? My suggestion is later,
as its own prompt, because it is a result rather than a method.

>A. Defer to later

**2F-Q6 — Showing the velocity.** The water moves 0.2 cells per hour, so the absolute velocity
is uninformative at this scale. What makes `2F` is the **velocity difference across the
front**. I'd show `u − u(x_parcel)` in the close-up and `u − ū_window` in the window, as
quivers, with the arrow scale stated (e.g. 5 cm s⁻¹). **OK?**

>A. Yes, that is ok.

#### Round 2 (Opus 5.5, 2026-11-19, 2F prompt 2): after making the figures

**2F-Q7 — The close-up parcel is convergence-dominated.** For continuity, `2F_method.png`
uses the same parcel as `dGdt_method.png` (j 126, i 653). At that cell, convergence gives +67.2
and strain +26.9 (sum 94.1, stored discrete 72.8). Over the band it is the other way round:
+1.29 + 2.92. The window is also convergence-led (16.2 + 20.5, so closer to even).
- (a) Keep the shared parcel, and say in the talk that this cell is convergence-led while the
  front overall is strain-led. **Recommended**: the two slides then show the same water.
- (b) Pick a strain-dominated ridge parcel for the 2F slide only (`--parcel j,i`).

>A. This is fine.  I wish to maintain our focus on this region.

#### Round 3 (Opus 5.5, 2026-11-23, 2F prompt 4): deriving the divergence + strain split

*The conventions are those of the code (`operators.frontogenesis`, `strain_from_jacobian`,
`strain_alignment`). The numbers are recomputed at pair 36 (label 83, 1,143 parcels, 12:30), on
the Jacobian-consistent set that `slides_2F.py` uses, and they reproduce `2F_split.png`. They come
from a throwaway session script built on `slides_2F.compute`; nothing new is drawn.*

**Step 0: what `G` measures.** `G = |∇b|²`. If two neighbouring isopycnals differ by `Δb` and
lie a distance `L` apart, then `|∇b| ≈ Δb/L` and `G ≈ Δb²/L²`. Since `b` is conserved by each
parcel (adiabatic), `Δb` is fixed, so **everything that changes `G` following the water is a
change in the spacing `L`**:

`(1/G) DG/Dt = −2 (1/L) DL/Dt`.

Sharpening a front means bringing isopycnals closer together. The question is only *which
part of the flow* does that.

**Step 1: the exact equation.** Take the horizontal gradient of `Db/Dt = 0`. For each
component, `D(b_x)/Dt = −(u_x b_x + v_x b_y)` and `D(b_y)/Dt = −(u_y b_x + v_y b_y)`. That is,

`D(∇b)/Dt = −(∇u)ᵀ ∇b`,  with  `∇u = [[u_x, u_y], [v_x, v_y]]`.

The gradient is rearranged only by the *velocity differences* across it; a uniform current
moves a front without changing it (hence the relative-velocity quivers on the slide). Then

`DG/Dt = 2 ∇b · D(∇b)/Dt = −2 (u_x b_x² + (u_y + v_x) b_x b_y + v_y b_y²) ≡ 2F`.

This is the `form='chain'` `F`.

**Step 2: split the velocity gradient into three motions.** Any 2 × 2 velocity-gradient tensor
can be written exactly as

```
∇u =  ½δ [[1, 0], [0, 1]]   +   ½ [[σn, σs], [σs, −σn]]   +   ½ζ [[0, −1], [1, 0]]
      (isotropic squeeze)       (pure strain, area-preserving)  (solid rotation)
```

with `δ = u_x + v_y` (divergence), `σn = u_x − v_y` (normal strain), `σs = v_x + u_y` (shear
strain) and `ζ = v_x − u_y` (vorticity). Multiplying out gives `u_x, u_y, v_x, v_y` back, so
the split is an identity and loses nothing. Inserting it into `F = −(∇b)ᵀ(∇u)(∇b)`:
- **The rotation drops out.** For any vector `g`, `gᵀ [[0, −1], [1, 0]] g = 0`. Rotation turns
  `∇b` without changing its length, so it cannot change `G`. It does change the direction of
  `∇b`, and so the angle `θ` below; that is how vorticity acts on fronts, indirectly.
- **Divergence gives `−½δ G`**, whatever the direction of `∇b`.
- **Strain gives `−½[σn(b_x² − b_y²) + 2σs b_x b_y]`**, which depends on the direction.

**Step 3: write the strain term as an angle.** Write `∇b = √G (cos α, sin α)`, so that
`b_x² − b_y² = G cos 2α` and `2 b_x b_y = G sin 2α`. Write the strain as
`σn = |σ| cos 2ψ`, `σs = |σ| sin 2ψ`, with `|σ| = √(σn² + σs²)`. Then

`σn(b_x² − b_y²) + 2σs b_x b_y = |σ| G cos 2(α − ψ)`.

`ψ` is the **extensional axis**: in a frame rotated by `ψ`, the strain matrix is
`½|σ| diag(1, −1)`, which stretches along `ψ` and squeezes along `ψ + 90°` (the **compressional
axis**). Let `θ` be the angle between `∇b` and the compressional axis, so `α − ψ = θ + 90°` and
`cos 2(α − ψ) = −cos 2θ`. Putting the pieces together:

`2F = −δ G + |σ| G cos 2θ`,  or, as a rate,  **`(1/G) DG/Dt = −δ + |σ| cos 2θ`**.

The second form is the most intuitive. **Each term is a growth rate of `G` in s⁻¹**, and
`1/rate` is an e-folding time.

**Step 4: what each term does to the spacing `L`.** Take the two pure flows separately, each in
its natural frame.
- **Pure divergence**, `u = (δ/2)x`, `v = (δ/2)y`. Every length scales at the rate `δ/2`, in
  every direction, so `DL/Dt = (δ/2) L` and `(1/G) DG/Dt = −δ`. **Convergence (`δ < 0`) packs
  isopycnals closer whatever the front's orientation.** A ring of parcels shrinks uniformly.
- **Pure strain**, `u' = +(|σ|/2)x'` along the extensional axis and `v' = −(|σ|/2)y'` along the
  compressional axis. Area is conserved, so a circle of parcels becomes an ellipse. From Step 1,
  `D(b_x')/Dt = −(|σ|/2) b_x'` and `D(b_y')/Dt = +(|σ|/2) b_y'`: the gradient component along
  the compressional axis grows and the one along the extensional axis decays. With `∇b` at
  angle `θ` from the compressional axis,
  `(1/G) DG/Dt = |σ|(cos²θ − sin²θ) = |σ| cos 2θ`.
  - `θ = 0°`: the isopycnals lie along the stretching direction and are squeezed together. The
    largest rate is `+|σ|`; this is the classic deformation-field front.
  - `θ = 45°`: compression and stretching cancel, so there is no change.
  - `θ = 90°`: the isopycnals are pulled apart (frontolysis), at the rate `−|σ|`.

  The same strain field can therefore sharpen or weaken a front. **What decides is the
  orientation of the front, not the strength of the strain.**

A useful comparison: the divergence term is like squeezing a sponge evenly from all sides; the
strain term is like stretching dough, which thins it in one direction only.

**Step 5: the numbers at pair 36.** Here `f = 6.80 × 10⁻⁵ s⁻¹` at 27.8°N.

*The close-up parcel (j 126, i 653).*
- Inputs: `δ = −2.18 × 10⁻⁵`, `|σ| = 1.48 × 10⁻⁵ s⁻¹`, `θ = 27°`, `cos 2θ = 0.59`.
- Rates: convergence `+2.18 × 10⁻⁵` and strain `+0.87 × 10⁻⁵`, total `3.05 × 10⁻⁵ s⁻¹ = 0.45 f`.
- **`G` would e-fold in 9.1 h.** Multiplying by `G = 3.08 × 10⁻¹³` gives the +67.2 and +26.9
  quoted earlier (×10⁻¹⁹ s⁻⁵).

*The band (1,143 parcels), weighted by `G`.* This is the weighting that makes the band means of
`2F` add up: `⟨2F⟩ = ⟨G⟩ × (G-weighted rate)`.

| | G-weighted mean rate [10⁻⁵ s⁻¹] | G-weighted mean of the magnitude | contribution to `⟨2F⟩` [10⁻¹⁹ s⁻⁵] |
|---|---|---|---|
| convergence `−δ` | +0.22 | `\|δ\|` 1.06 | +1.29 |
| strain `\|σ\| cos 2θ` | +0.51 | `\|σ\|` 1.77 | +2.92 |
| total | +0.73 (0.11 f, e-folding 38 h) | | +4.21 |

The table shows the intuition directly.
- **Divergence is large but nearly cancels.** `|δ|` is about 60 % of `|σ|`, but only 59 % of the
  parcels converge, so the convergence and divergence parcels mostly offset each other. The net
  rate is a fifth of the typical `|δ|`.
- **Strain has a steady sign.** `θ` is mostly below 45°: 72 % of the parcels, median 29°,
  `⟨cos 2θ⟩ = 0.33`. Strain is used at about 30 % of its maximum efficiency, and the sign
  rarely flips.
- **The result is that strain carries 69 % of the band's `2F`, even though the parcel shown is
  convergence-led** (2F-Q7). The split holds on the ridge (+1.93 + 4.25) and on the flanks
  (+1.05 + 2.43).

**Step 6: three points worth stating in the report.**
1. **Why strain sets the sign and convergence varies.** At a front, the large-scale (mesoscale)
   strain is slowly varying and coherent along the front. The convergence is mostly the
   *ageostrophic secondary circulation* of the front itself: the cross-front overturning that
   keeps the front near thermal-wind balance (Hoskins & Bretherton 1972; Hoskins 1982). It
   converges on one side of the front and diverges on the other, which fits the 59/41 split and
   the cancellation in the table.
   - In the semi-geostrophic picture, strain starts the sharpening and the convergence it
     induces speeds it up (the feedback that can collapse a front in finite time).
   - At the surface, `δ = −∂w/∂z`, so convergence comes with downwelling.
2. **Vorticity acts through `θ`.** `ζ` does not appear in `2F`, but it rotates `∇b`, while
   strain turns `∇b` toward the compressional axis. Their balance sets the typical `θ`
   (Lapeyre, Klein & Hua 1999, for passive tracers). A front sitting at θ ≈ 29°, rather than at
   0°, is the sign of that competition.
3. **The split applies to the continuum `2F`, not the stored discrete one.** The identity in
   Step 2 is exact for `F = −(∇b)ᵀ(∇u)(∇b)` (checked to 1 × 10⁻¹⁵ of the r.m.s. with the
   Jacobian-consistent `δ, σn, σs`). The stored discrete `2F` (+3.46) is a commutator of the
   stencil and has no unique split of its own; it is 0.82 × the continuum value here. With the
   flux-form `δ`, `σ` of the derived store, the two terms sum to 4.29 rather than 4.21. The report
   should therefore state the split for the continuum form and the comparison with `DG/Dt` for
   the discrete form.

**2F-Q8 — What goes in the report?** My suggestion is a short subsection containing:
- Step 0 (spacing);
- the boxed result of Step 3, `(1/G) DG/Dt = −δ + |σ| cos 2θ`;
- Step 4 (the two pure flows), in prose;
- the band table of Step 5;
- the three points of Step 6.

Steps 1–3 would go in an appendix. **OK?**

**2F-Q9 — A schematic?** Step 4 is the part a picture carries best: a circle of parcels across
two isopycnals under (i) convergence, (ii) strain with `θ = 0°`, and (iii) strain with
`θ = 90°`, after a short time. It would be a purely graphical panel (no data), usable both in
the report and as a slide before `2F_method.png`. **Do you want it, and if so, as its own prompt?**

### dG/dt

*Opus 5.5, 2026-11-19, prompt 1. Nothing has been drawn yet. The method is from
`tracked/matched_parcels_track_81.md` §1, and the numbers are from `data/matched_parcels_track_81.json`.
Rates are in 10⁻¹⁹ s⁻⁵.*

**How M4 gets dG/dt for a tracked front, at pair `t`:**
1. Take the label chosen at `t` (the skeleton) and dilate it by 1 cell. Each cell of this band
   is one parcel, at its cell centre. At pair 35 (label 81) that is 307 px and 1,119 parcels.
2. Carry every parcel one hour **forward** (to `x⁺`, in pair `t+1`) and one hour **backward**
   (to `x⁻`, in pair `t−1`) with the model surface flow.
3. `r_matched = [mean G_{t+1}(x⁺) − mean G_{t−1}(x⁻)] / 2Δt`. The labels at `t±1` play no
   part: the parcels are seeded only at `t`.

Next to it are the stored semi-Lagrangian `DG/Dt` (one hour, `h_t → h_{t+1}`, read at the seed
cells) and `2F` (instantaneous, read at the seed cells).

**DG-Q1 — Which "dG/dt", and over which interval?** Your prompt says "between two time stamps
(1 hour)". The rate M4 quotes (`r_matched`) is a **centred difference over two hours**
(`t−1 → t+1`), not one. The options:
- (a) Show the centred, two-hour construction exactly as M4 does it: three time levels, with
  the parcels traced back and forward. **Recommended**, since this is the number in the
  report's figures.
- (b) Show one hour only (`t → t+1`, forward). This is simpler to draw, but it is not a rate
  we actually quote.
- (c) Show the stored semi-Lagrangian `DG/Dt`. This one is genuinely one hour, but it works
  per cell: it traces back to departure points and interpolates `b`, not `G`. That is harder
  to show and less intuitive.

>A. I am most interested in the semi-Lagrangian `DG/Dt` (c).  We are finding that it is more closely
aligned with `2F` than the centered difference.

**DG-Q2 — Data or a cartoon?** You can't show the raw data at the scale of the whole front.
- The parcels move a median **0.23 cells per hour** (`disp_plus_median` 0.23, `disp_minus_median`
  0.26 at pair 35).
- The band-mean `G` changes by about **1 %** over the two hours (5.862 → 5.849 × 10⁻¹⁴ s⁻⁴).

So three `G` maps of the full front would look identical, and the arrows would be shorter than
a pixel. My proposal is a **hybrid**: real data in a small window, with the motion drawn
exaggerated and labelled as such.
- **Top row:** the full front at `t` on `G`, with the band (the parcels) lassoed as in
  `tracking_81.png`, and a box marking a window of about 15 × 15 cells on one arm.
- **Bottom row, in that window:**
  1. `G_{t−1}`, with each parcel's `x⁻` as a dot;
  2. `G_t`, with the seed cells and arrows `x⁻ → x_p → x⁺` (scaled ×10, stated on the panel);
  3. `G_{t+1}`, with each parcel's `x⁺` as a dot;
  4. the per-parcel `ΔG / 2Δt` map (as in `tracked/figs/..._parcels.png`), with the mean
     written as the result.

The alternative is (b): a pure schematic (a cartoon front, a few parcels, the formula). That is
cleaner, but it isn't the data. **Recommended: the hybrid.**

>A. I like a hybrid, but I want to focus on the semi-Lagrangian `DG/Dt` (c).
Let's discuss what hybrid might work best for it

**DG-Q3 — Where does 2F go?** The section is titled "dG/dt and 2F".
- (a) This figure covers dG/dt only, and 2F gets its own figure (next prompt). **Recommended**,
  because one slide with both methods is crowded.
- (b) Add a final panel: the per-parcel `2F` map on the same parcels, with its mean.

>A. Correct, `2F` will get its own figure and prompt.

**DG-Q4 — Which pair?** At the anchor pair 35 the rates are small and mixed:
- pair 35: matched −0.19, stored +1.52, `2F` +0.71;
- pair 36 (label 83): +2.38 / +4.20 / +3.46. All three are positive and roughly agree.

Pair 35 keeps continuity with the scene and tracking slides. Pair 36 gives a clearer message
("the front sharpened, and 2F predicts it"). **Recommended: pair 35** if the slide is about the
*method*, and 36 if it is about the *result*.

>A. Let us use `pair 36`.

#### Round 2 (Opus 5.5, 2026-11-19, prompt 2): a hybrid for the semi-Lagrangian `DG/Dt`, pair 36

**What the stored value is, at pair 36 (label 83, band of 1,143 parcels).** For each cell
`x` of the band, `DG/Dt(x) = [G(x, h₃₇) − G(x_d, h₃₆)] / Δt`. The hours are 12:00 and
13:00 UTC, and the front was found on the midpoint `G` at 12:30.
- `x_d` is the departure point at 12:00 of the water that arrives at the centre of `x` at 13:00.
  It comes from the midpoint velocity, using three midpoint-rule iterations.
- `G(x_d, h₃₆)` is computed by interpolating `b(h₃₆)` onto a 5-point "+" stencil centred at
  `x_d` and taking the gradient there. `G` itself is never interpolated.

I recomputed it from `tile330_raw_..._72h.zarr` (scratch, frontogenesis env). It matches the
stored `DGDt_semilag` to 2 × 10⁻⁷ of its r.m.s., and the band mean is **+4.20** (10⁻¹⁹ s⁻⁵),
the same as in the json. Facts that shape the figure:

| quantity on the band, pair 36 | value |
|---|---|
| departure distance `\|x − x_d\|` | median **0.20 cells**, p90 0.37, max 0.54 |
| band-mean `G`: `G(x_d, h₃₆)` → `G(x, h₃₇)` | 5.65 → 5.80 × 10⁻¹⁴ (+2.7 %) |
| per-cell change `\|G(x,h₃₇) − G(x_d,h₃₆)\| / G` | median **6.8 %** |
| mean of `DG/Dt`: ridge (307 skeleton px) / flank (836) | +12.4 / +1.2 |
| local part `[G(x,h₃₇) − G(x,h₃₆)]/Δt`: mean / r.m.s. | +2.30 / 37.7 |
| advective part `[G(x,h₃₆) − G(x_d,h₃₆)]/Δt`: mean / r.m.s. | +1.90 / 31.3 |
| their sum, `DG/Dt`: mean / r.m.s. | +4.20 / 22.0 |

Three things follow from this:
1. The departure point is always **inside the 3 × 3 neighbourhood** of its cell, so it can
   only be seen in a close-up a few cells across.
2. Side-by-side `G` maps change by about 7 % per cell, too little for the eye. Difference
   maps are needed.
3. The local and the advective parts are **each larger than their sum, and largely cancel cell
   by cell** (r.m.s. 38 and 31 against 22). This is the visual reason to follow the water, and
   probably why the semi-Lagrangian rate tracks `2F` better than the centred difference (the
   point you made in DG-Q1).

**DG-Q5 — Which hybrid?** Three layouts, all drawn from the data of pair 36:
- **(a) Zoom in, then add up** (*recommended*), one row of four panels:
  1. *Context.* `G` around label 83, with the band lassoed (as in `tracking_81.png`) and a box
     marking a window of about 15 × 15 cells.
  2. *One parcel, close up* (5 × 5 cells): `b(h₃₆)` contours, the arrival cell `x` as a square,
     its departure point `x_d` as a dot (true scale), and the 5-point stencil at `x_d`.
     The formula for that one parcel is written underneath with its numbers:
     `G(x,h₃₇)`, `G(x_d,h₃₆)`, and the rate.
  3. *The window:* `G(x_d, h₃₆)` and `G(x, h₃₇)`, either as two small maps or as one map with
     the arrows `x_d → x`.
  4. *The result:* the per-parcel `DG/Dt` map on the whole band, with the band mean **+4.20**
     printed as "the number we quote for this hour".
- **(b) Local + advective = semi-Lagrangian:** three maps of the window,
  `[G(x,h₃₇) − G(x,h₃₆)]/Δt`, `[G(x,h₃₆) − G(x_d,h₃₆)]/Δt`, and their sum, each with its
  band mean (2.30 + 1.90 = 4.20). This shows why following the water matters, but not *how*
  `x_d` is found.
- **(c) Both:** (a) as the method slide, and (b) as a second slide or a bottom row.

>A. Let us make both, as 2 separate figures.

**DG-Q6 — Arrow scale.** Departures are 0.2 cells (max 0.54).
- In the 5 × 5 close-up, true scale reads well: 0.2 cells is about 1/25 of the panel width.
  I'd draw it at true scale there.
- In the 15 × 15 window, true-scale arrows would be specks. I can exaggerate them ×5, stated on
  the panel, or leave arrows out of the window.

**Recommended:** true scale in the close-up, and no arrows in the window.

>A. Agreed, use that recommendation.

**DG-Q7 — `b` or `G` in the close-up?** The scheme really does interpolate **`b`** to the
stencil, not `G`.
- (a) Draw `b(h₃₆)` contours under the stencil in the close-up, so it is honest about what is
  interpolated, and use `G` everywhere else. **Recommended.**
- (b) Show `G` only, and describe the stencil in a caption.

>A. Yes, do (a)

**DG-Q8 — Which window and which parcel?** I'd let the script choose, and print what it chose.
- **Window:** the 15 × 15 window with the largest r.m.s. `DG/Dt` on the band is at
  j 119–133, i 652–666, on the northern part of the front. It has 38 parcels, 68 % of them
  positive, and a mean displacement of 0.12 cells.
- **Parcel:** a ridge cell with a positive rate and a displacement near the median.

You may prefer a window on a straight stretch of the front, where the cross-front structure is
cleaner. Say if you have one in mind.

>A. Start with the one you suggest, but we may search for a better one.

**DG-Q9 — Code note (not a decision unless you object).** The hour-end `b(h₃₆)`, `b(h₃₇)` and
the departure field are not in the derived store, and computing them needs `xgcm`, which
`ocean14` lacks.
- **Plan:** `py/slides_dGdt.py` gets a `--prep` step run in the **frontogenesis** env. It reuses
  `semilag.departure_index` and `gradb2_at_departure` and writes a small
  `data/slides_dGdt_36.npz` (the cropped `b`, `G`, `x_d` and `DG/Dt` fields).
- **Plotting** then runs in `ocean14` from that file.
- **Check:** the prep asserts the match with the stored `DGDt_semilag` (as above) and with the
  json's `r_store` = 4.20.

>A. Ah, my mistake. We should always run in the `frontogenesis` environment.
I have modified the Code section to reflect this.

#### Round 3 (Opus 5.5, 2026-11-19, prompt 3): after making the figures

**DG-Q10 — The window moved 6 cells west.** The suggested window (j 119–133, i 652–666) and an
unconstrained search both catch the front only across one corner of the box, because the
r.m.s. favours a few strong cells. The script now **centres the window on a front pixel**
(at least 40 parcels), and it chose **j 119–133, i 646–660**: 59 parcels, window mean +38.3,
with the front running corner to corner. Is this OK, or do you have a stretch in mind?
`--window j0,i0` overrides it, and `--parcel j,i` overrides the close-up parcel.

**DG-Q11 — File name.** I named the second figure `slides/dGdt_split.png` rather than
`dGdt_result.png`: it shows the local + advective split. The result itself (the band mean
+4.20) is panel (c) of `dGdt_method.png`. Rename it if you prefer.



### Tracking

#### Method

*Opus 5.5, 2026-11-21, Method prompt 1. Discussion only; no code was written. Every number
below was recomputed from `tile330_fronts_D_L0.zarr` and `tile330_flow.zarr` with
`tracking.Flow` / `tracking.flow_weighted_score` (frontogenesis env), and it matches
`m4_tracks.json` (flow track 0) link for link.*

**How the tracker works (`tracking.follow_flow` + `front_tracking.score_candidate`).** It starts
from the anchor (label 81, pair 35) and walks one pair at a time, forward to pair 70 and then
backward to pair 0. Each step from pair `t` to `t ± 1` does four things:

1. **Predict.** The mask $M_t$ of the label chosen at `t` is carried through the hour by the flow.
   The displacement is the one used by `semilag.departure_index`:
   $\mathbf{d} = \Delta t\,\mathbf{u}(\mathbf{x} - \tfrac12\mathbf{d})$, with three midpoint
   iterations, and $\mathbf{u}$ taken at the integer hour between the two pairs. The prediction is
   $\hat M_{t+1}(\mathbf{x}) = [\,M_t(\mathbf{x}-\mathbf{d}) \ge 0.5\,]$, where the 0/1 mask is
   interpolated bilinearly. The label field itself is never interpolated.
2. **Search.** Every label at `t + 1` with at least 5 px is a candidate $C$. The search radius is
   $R = \max(2,\ 0.6\ \mathrm{m\,s^{-1}}\,\Delta t/\Delta x) = 2$ cells, because the 1.87 km cells
   give 1.15 cells and the floor of 2 applies.
3. **Score.** Lower is better:
   $S = \sum_k w_k \tau_k / \sum_k w_k$, with the following terms.

   | term $\tau_k$ | definition | $w_k$ |
   |---|---|---|
   | position | $\min_{C}\,\mathrm{dist}(\cdot, \hat M)\,/\,R$ (nearest point, not the centroid) | 1.0 |
   | flow IoU | $1 - \mathrm{IoU}(\hat M \oplus 1, C \oplus 1)$ (both dilated by 1 cell) | 1.0 |
   | overlap | $1 - \mathrm{IoU}(M_t, C)$ (the **un-advected** mask, raw) | 0.8 |
   | length | $\lvert\ln(\ell_C/\ell_M)\rvert/\ln 2$, with $\ell = 4\sqrt{\lambda_{\max}}$ from the second moments | 0.6 |
   | orientation | $\Delta\theta/25^\circ$ (major axis, folded to 0–90°) | 0.6 |
   | area | $\lvert\ln(A_C/A_M)\rvert/\ln 2$ | 0.4 |

   **Veto:** if the position term is above 3 (more than 6 cells away), $S = \infty$.
4. **Choose.** The tracker takes $\arg\min S$ and accepts it if $S \le 2.5$. Otherwise the step is
   a gap, and the prediction keeps being advected to the next pair.

**The data of link 35 → 36 (label 81 → 83).**

- Over the hour the flow moves the parcels of label 81 by a median of **0.23 cells** (0.56 at
  most). The advected mask differs from $M_{35}$ in only **2 of 305 pixels**.
- Of the 261 candidates, **2** pass the veto:

  | candidate | px | position | flow IoU | overlap | length | orient. | area | **S** |
  |---|---|---|---|---|---|---|---|---|
  | **83 (chosen)** | 307 | 0.00 | 0.15 | 0.31 | 0.04 | 0.25 | 0.00 | **0.13** |
  | 61 | 32 | 2.50 | 1.00 | 1.00 | 1.57 | 1.77 | 3.26 | **1.73** |
  | 70 (next, vetoed) | 65 | 4.03 | — | | | | | ∞ |

- Label 83 has a raw IoU of only **0.69** with label 81 (0.86 once both are dilated), although
  the fluid moved the front by less than a quarter of a cell. **The front changes from hour to
  hour mainly because the finder redraws it (it is re-detected), not because the water moves.**
  That is why the score relies on shape terms and on a dilated IoU, and why the slide should show
  it.

**TR-M-Q1 — Which link?** I propose **35 → 36** (label 81 → 83). It is the first forward step from
the anchor, the same hour as the dG/dt and 2F slides, and a clean case (1 overlap, no split or
merge). The alternative is a link with a split or merge (pairs 38–40) to show a harder decision,
but the method is clearer in a clean case. *Recommendation: 35 → 36.*

>A. Use 35 → 36 as recommended.

**TR-M-Q2 — How do we show "arrows indicating the movement" when the motion is 0.23 cells/h?**
At true scale the arrows are invisible and the prediction lies on top of $M_{35}$. Options:
(a) a quiver of the 1-h displacement $\mathbf{d}$ along the front, **magnified** (e.g. ×10) with a
key that says so, and the predicted mask drawn at its true position;
(b) chain the prediction over 5 h (35 → 40) so the front visibly moves. But the tracker never
predicts more than 1 h ahead unless there is a gap, so this misrepresents the method;
(c) (a) for the main panels, plus a small inset of the 5-h chain labelled "for scale".
*Recommendation: (a).* It is honest, and the small motion is itself the point of the next panel.

>A. Let's try (a) with magnification ×10.

**TR-M-Q3 — Layout.** My proposal for one figure, `slides/tracking_method.png`, with 4 data panels
and an equation column:

- **(a) Pair 35:** `G` in grey (as in the other tracking slides), label 81 in cyan, and
  magnified $\mathbf{d}$ arrows. Equation: $\mathbf{d} = \Delta t\,\mathbf{u}(\mathbf{x}-\mathbf{d}/2)$.
- **(b) Prediction:** $M_{35}$ against $\hat M_{36}$ in a close-up, and the 2-px difference.
  Equation: $\hat M = [M(\mathbf{x}-\mathbf{d}) \ge 0.5]$.
- **(c) Pair 36:** `G`, the predicted mask, and contours of $R$ = 2 and the 6-cell veto. The
  candidates are coloured as chosen (83), allowed (61) and vetoed (grey). The pixels where 83
  differs from $\hat M$ are marked, to show the re-detection.
- **(d) Score:** a stacked bar of $w_k\tau_k/\sum w$ for 83 and 61, with the $S = 2.5$ acceptance
  line. The equation for $S$ and the term table go beside it.

Alternative: two figures, as for dG/dt: `tracking_method.png` with panels (a)–(c), and
`tracking_score.png` with (d) plus the full term table.
*Recommendation: one figure, if the term table is cut to a one-line formula plus the bar labels.
Otherwise two figures.*

>A. Go with one figure

**TR-M-Q4 — The backward walk and gaps.** The tracker also walks backward from the anchor, using
the flow with Δt = −3600 s, and it bridges gaps by advecting the prediction again. Should the figure
show either? (a) Forward only, with a one-line footnote; (b) a small timeline strip under the
panels: pairs 0–70, the anchor, the arrows of both directions, and gaps or splits marked.
*Recommendation: (a).* The timeline is already told by the GIF.

>A. Go with (a)

**TR-M-Q5 — Interpolation orders in the equations?** The prediction interpolates the velocity at
cubic order and the mask bilinearly (MASK_ORDER = 1, chosen so a 1-px skeleton never vanishes).
(a) Leave these details off the slide; (b) add them in a footnote. *Recommendation: (b), small.*

>A. Go with (b).  I may then remove it.

Code plan: as asked, new functions in `py/slides_tracking.py` (`compute_link`,
`fig_method`). They call `tracking.Flow(store=...)`, `advect_mask_by` and
`flow_weighted_score` directly, so the scores on the slide are the tracker's own. The script
asserts that they reproduce `m4_tracks.json`. It runs in the frontogenesis env, because it imports
`fronts.front_tracking`.

#### Round 2 (Opus 5.5, 2026-11-21, Method prompt 2): after making the figure

`slides/tracking_method.png` is made by `python slides_tracking.py --what method` (about 10 s).
The script asserts that the chosen label, its score and each of the six terms reproduce
`m4_tracks.json` to 1e-9.

**TR-M-Q6 — The close-up (b).** Both pixels that the advection changes are pixels *dropped*
from the mask (at (172, 697) and (176, 620)); none is gained. The box sits on the northern loop
around one of them. At ×10, the arrows in the 15-cell close-up are up to 5 cells long, so I show
only every 3rd cell there. Options: (a) keep it as is; (b) ×5 in (b) only (a different
magnification from (a)); (c) drop the arrows from (b) and let the blue field
$M(\mathbf{x}-\mathbf{d})$ show the shift. *Recommendation: (a).* The shared ×10 keeps panels a
and b consistent.

>A. Ok, I understand.  That is fine.

**TR-M-Q7 — The score bar of label 83 is tiny (S = 0.13).** Its term values are printed under
the bar in red. If you want the bar itself readable, the x-axis could be logarithmic, or the bar
could be shown in an inset. *Recommendation: keep it linear.* The contrast with 61 and with
the 2.5 line is the message.

>A. We are removing that part of the figure.

#### Round 3 (Opus 5.5, 2026-11-21, Method prompt 3): answers to JXP's questions

The detailed description is now in `report/frontogenesis_tracking.md`: notation table, the
finder, the four stages with equations, the worked step 35 → 36, the walk (backward, gaps,
splits / merges), and a panel-by-panel key to the slide.

**JXP-1 — Why does (a) say 307 px and (b) only 305?** The 307 are label 81 as found ($M_{35}$).
The 305 are the prediction $\hat M_{36}$, label 81 carried by the flow. The prediction moves the
0/1 mask by bilinear interpolation and keeps the cells $\ge 0.5$. A straight one-pixel front
always survives this. At a *diagonal* step of the skeleton, however, the weight is spread over
four cells and can fall below 0.5 everywhere. That happens at two pixels: (172, 697), with 0.457,
and (176, 620), with 0.494, the one circled in (b). No pixel is gained, so 307 − 2 = 305. The two
one-pixel breaks barely affect the score, which uses dilated masks and nearest-point distances
(report §3.2).

**JXP-2 — Why was the spur near 125° W, 28° N not chosen at pair 36?** It is the orange stub at
124.75° W, 28.0° N in (c): a 9-px side branch of label 81, up to 7 cells from label 83. The
tracker chooses whole labels, never pieces, and at pair 36 **the finder drew nothing there**. I
re-ran the finder (config D) on pair 36. Threshold and sharpen keep only 2 of the 9 pixels,
because `G` on the branch fell 9% over the hour (3.4 → 3.1 × 10⁻¹⁴ s⁻⁴) and most of it dropped
below the 85th-percentile threshold of its 64-cell window. The 3-px stub that is left is shorter
than `Lspur` = 10 and is removed as a spur. At pair 35 all 9 pixels are kept. The re-run
reproduces the stored labels exactly at both pairs. The branch only costs label 83 a little flow
IoU and overlap (report §4).

**JXP-3 — What are the green "allowed" pixels?** Label 61, a separate 32-px front to the west.
Its nearest pixel is 5 cells from the prediction, inside the 6-cell (3R) veto, so it is the only
other candidate that gets scored. It scores S = 1.73 against 0.13 for label 83 (no overlap, a
third of the length, a tenth of the size, turned 44°). "Allowed" meant only "not vetoed", and the
word was unclear.

**TR-M-Q8 — Rename "allowed" on the slide?** Options: (a) "scored, not chosen: 61"; (b) "not
vetoed: 61"; (c) keep "allowed". *Recommendation: (a).*  Keep (a)

**TR-M-Q9 — Annotate (c)?** Your questions suggest that two features of (c) need labels:
(a) a small callout on the orange stub ("branch of 81 not redrawn at 36") and one on label 61;
(b) leave the slide clean and let the report carry the explanation. *Recommendation: (a), one
short callout each.*

>A. No need to annotate (c) further

TR-M-Q6 and TR-M-Q7 (round 2) are still open.

#### Snapshots

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

>A. This is fine.  No need to add a note

**TR-Q5 — `tracking_top10.gif` is 20.7 MB.** It is 71 frames at 864 × 1125 px over nearly
the whole tile. Most of the bytes are the `G` texture, which changes every frame. That is fine
for PowerPoint / Keynote, but heavy for e-mail or a web page. Options:
(a) keep it as is;
(b) halve the frames (every 2nd pair, 0.6 s each): ~10 MB;
(c) export an MP4 next to the GIF (~2–3 MB, needs `ffmpeg` in ocean14);
(d) lower resolution (60 dpi): ~12 MB, but the fronts thin out.
*Recommendation: (a) for the slides; (c) if it must travel.*

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

### 2026-11-19 — Figures / Tracking, prompt 3 (Opus 5.5)

**Done.** I added `gif()` to `py/slides_tracking.py`, which writes `slides/tracking_81.gif`.
`main` now takes `--what fig|gif|both` (default `both`, ~14 s in ocean14) and writes every
number below to `data/slides_tracking_81.json` (keys `gif`, `fig`). TR-Q4's answer ("fine, no
note") needed no change. There are no new questions.

- **Frames:** one per pair over the whole window, **71 frames**, pairs 0–70 (`time_mid`
  2012-07-02 00:30 → 07-04 22:30 UTC). Each frame shows 300 ms, the last one held 5× longer,
  and the GIF loops: ~22 s per cycle, 960 × 601 px, 7.8 MB.
- **Each frame** shows `log10 G` (`L = 0`, grey as in the figure, TR-Q3 (a)), the chosen
  label's **skeleton** pixels as cyan dots, and the previous pair's skeleton as found
  (TR-Q2 (a)) as larger red dots underneath. Red that shows past the cyan is the hour's change.
  The title gives the time, pair, label ID, pixel count and the split / merge flag from
  `data/matched_parcels_track_81.json`, and marks the anchor.
- **Fixed for the whole GIF**, so nothing jumps between frames:
  - crop: the union of all 71 skeletons + 12 cells, j 0–216, i 592–719 (~26.7–28.9°N,
    123.5–128.0°W). It is wider than the figure's because pairs 0–~12 the label reaches west to
    ~127.8°W (j = 9, just outside the 7-cell edge rim);
  - colour limits: p2 / p99.5 of `log10 G` pooled over all frames in the crop (−16.76 /
    −12.49, with the same +0.6 dex stretch).
- **What the GIF shows:** label size min / median / max = **104 / 178 / 413 px**, matching
  `matched_parcels_track_81.md`. Ten frames are flagged SPLIT and nine MERGE. Pair 40, the
  third panel of `tracking_81.png`, is one of the splits: the tracker keeps the western arm.
  From pair ~40 to ~69 the label is the ~110–125 px western arm and loop, and at pair 70 it is
  long again (283 px).
- **Fixes along the way:**
  - the first step's split / merge flags are NaN in the JSON, and a plain truthiness test
    printed `[SPLIT MERGE]` on pair 0; the test is now `is True`;
  - colorbar label and y-label clipping fixed.

### 2026-11-19 — Figures / Tracking, prompt 4 (Opus 5.5)

**Done.** I regenerated `slides/tracking_81.gif` with `py/slides_tracking.py --what gif`.
`slides/tracking_81.png` is unchanged.

- The active front (the chosen label's skeleton) is now **red** (`COL_GIF = COL_PREV =
  '#ff2d55'`), with dots of 5 pt², up from 4.
- **The previous hour is no longer drawn**; each frame shows only the current front on
  `log10 G`. The legend has the one entry.
- Everything else is as in prompt 3: 71 frames, 300 ms, the same fixed crop and colour limits,
  and the title flags (10 SPLIT, 9 MERGE). The file is 7.8 MB, and `data/slides_tracking_81.json`
  (`gif`) has been updated.

### 2026-11-19 — Figures / Tracking, prompt 5 (Opus 5.5)

**Done.** `py/slides_tracking.py` has a new `gif_top10()`; `--what top10` (~35 s) writes
`slides/tracking_top10.gif`, and `--what all` builds the figure and both GIFs. The numbers are
in `data/slides_tracking_81.json` under the key `top10`. One question, TR-Q5 (file size), is
in Q&A/Tracking.

- **The ten fronts** are the ten M4 flow tracks of `data/m4_tracks.json` (`flow[0..9]`), the
  set the track analysis uses (`surface_flux_tracks.py`, all ten). They are anchored at the
  ten largest labels of pair 35 (anchor sizes 307 → 121 px). All ten are found at all 71
  pairs.

  | track | anchor label | colour | px min / median / max |
  |---|---|---|---|
  | 0 | 81 | red `#e6194b` | 104 / 178 / 413 |
  | 1 | 84 | green `#3cb44b` | 44 / 141 / 328 |
  | 2 | 202 | yellow `#ffe119` | 133 / 238 / 311 |
  | 3 | 100 | blue `#4363d8` | 188 / 227 / 303 |
  | 4 | 187 | orange `#f58231` | 169 / 286 / 437 |
  | 5 | 5 | purple `#911eb4` | 79 / 212 / 267 |
  | 6 | 96 | cyan `#42d4f4` | 106 / 151 / 250 |
  | 7 | 133 | magenta `#f032e6` | 74 / 113 / 218 |
  | 8 | 161 | lime `#bfef45` | 98 / 145 / 321 |
  | 9 | 218 | pink `#ff9f9f` | 61 / 115 / 127 |

- **Each frame** (71 frames, 300 ms, loop, last frame held) shows every track's chosen
  skeleton as dots in its colour. The track number sits at the front's centroid, in bold with a
  black outline, so the eye can follow each front. The background is grey `log10 G`
  (`Greys` squeezed to 0.12–0.78 so light and dark colours both read; p2 / p99.5 =
  −16.89 / −12.46, fixed). Land is tan and the `mask_analysis` boundary is dashed. The legend
  gives "track: anchor label at pair 35". The title gives the time and pair, and marks the
  anchor.
- **Crop** is the union of all ten tracks over all pairs + 15 cells: j 0–449, i 44–720
  (~26.7–37.5°N, 119–128°W). That is most of the analysis region, so the frame is portrait.
- **GIF palette.** The median-cut palette used for `tracking_81.gif` shifted the colours
  toward grey (cyan came out lavender). This GIF uses a **fixed palette** instead: the ten exact
  colours, land, and a 48-step grey ramp. At 80 dpi the dots are 5 pt², so the skeletons
  stay continuous. Size: 864 × 1125 px, **20.7 MB** (TR-Q5).
- **Refactor:** the GIF writing is now shared by both GIFs (`_write_gif`, with an optional
  fixed palette). `tracking_81.gif` regenerates byte-identical in size (7,818,378 bytes), and
  `tracking_81.png` is unchanged.

### 2026-11-19 — Figures / dG/dt and 2F, prompt 1 (Opus 5.5)

**Discussion only, as asked; no script or figure yet.** I read the method in
`tracked/matched_parcels_track_81.md` §1 and the step rows of `data/matched_parcels_track_81.json`,
pairs 34–36, and looked at the existing per-parcel maps in `tracked/figs/matched_parcels_track_81_parcels.png`.
Four questions are in Q&A/dG/dt:
- DG-Q1: the rate M4 quotes is centred over **2 h**, not 1 h;
- DG-Q2: the displacements (0.23 cells/h) and the change in G (~1 %) are too small to see at
  the scale of the whole front, so I propose real data in a zoomed window with arrows
  exaggerated ×10;
- DG-Q3: 2F in this figure or a separate one;
- DG-Q4: pair 35 or pair 36.

Plan, once answered: a new module `py/slides_dGdt.py` that reads the labels zarr, the derived
store (`G`, `two_F`) and the flow store directly, so it runs in `ocean14` without importing
`tracking.py`. It would reimplement the band and the forward/backward positions
(`tracking.forward_positions`, cubic `sample_at`), assert that the parcel mean equals
`r_matched` in the json, and write `slides/dGdt_81.png`.

### 2026-11-19 — Figures / dG/dt and 2F, prompt 2 (Opus 5.5)

**Discussion only, as asked; no figure.** I read JXP's answers:
- DG-Q1 (c): the semi-Lagrangian `DG/Dt`;
- DG-Q2: a hybrid, for (c);
- DG-Q3: 2F gets its own figure;
- DG-Q4: pair 36.

**Check.** I recomputed the stored tendency at pair 36 from the 72-h raw store, with a scratch
script in the frontogenesis env (reusing `m2_qa.open_inputs` / `hour_ds`, `inputs.filtered`,
`semilag.departure_index` and `gradb2_at_departure`). The result matches `DGDt_semilag`
(max difference 1.9 × 10⁻⁷ of the r.m.s.) and the band mean of label 83 is +4.20, equal to
`r_store` in `data/matched_parcels_track_81.json`.

**New numbers (Q&A round 2):**
- the departure distance is a median 0.20 cells (max 0.54);
- the per-cell change in `G` is a median 6.8 %;
- the local and advective parts (means +2.30 / +1.90) have r.m.s. 38 / 31 against 22 for their
  sum, so they cancel cell by cell.

New questions DG-Q5 to DG-Q9 are in Q&A/dG/dt:
- Q5, the layout: (a) zoom in, then add up, recommended;
- Q6, the arrow scale;
- Q7, `b` contours under the stencil;
- Q8, the choice of window and parcel;
- Q9, a prep step in the frontogenesis env writing a `.npz`.

### 2026-11-19 — Figures / dG/dt and 2F, prompt 3 (Opus 5.5)

**Done.** `py/slides_dGdt.py` (frontogenesis env, about 1 min) writes `slides/dGdt_method.png`
and `slides/dGdt_split.png`. Every number is in `data/slides_dGdt_36.json`. Follow-up questions
DG-Q10 (window) and DG-Q11 (file name) are in Q&A/dG/dt.

- **Data.** Pair 36 (hours 36 → 37, 12:00 → 13:00 UTC), label 83 of track 0, band of 1,143
  parcels (307 skeleton px). Everything is recomputed from the 72-h raw store with the
  pipeline's functions: `m2_qa.open_inputs` / `hour_ds`, `inputs.filtered` / `midpoint`,
  `semilag._centred_pair` / `departure_index` / `gradb2_at_departure`, `operators.gradb2`.
- **Asserted:**
  - the recomputation matches the stored `DGDt_semilag` on the band (max difference
    1.9 × 10⁻⁷ of the r.m.s.; tolerance 10⁻⁵);
  - the band mean is +4.2023 × 10⁻¹⁹ s⁻⁵, equal to `r_store` of
    `data/matched_parcels_track_81.json` (relative tolerance 10⁻⁶);
  - local + advective = total, to round-off.
- **`dGdt_method.png`** (2 × 3), all panels north-up in lon / lat. Fractional grid
  positions are mapped with bilinear `XC`, `YC`.
  - (a) The front on midpoint `G` (cividis, as on the scene slide), with the skeleton, the
    lassoed band and the window box.
  - (b) A 5 × 5-cell close-up of one parcel (j 126, i 653; a positive ridge cell whose
    displacement is nearest the band median). It shows `b(12:00)` shading and contours, the
    arrival cell, the arrival centre `x`, the departure point `x_d` with the arrow
    **at true scale** (|d| = 0.15 cells), and the 5-point stencil at `x_d` (DG-Q6, DG-Q7 (a)).
    Its numbers are written below: `G(x,13:00)` = 3.206, `G(x_d,12:00)` = 2.942
    (×10⁻¹³ s⁻⁴), rate +73.4.
  - (c) `DG/Dt` on every parcel, with the band mean +4.20 boxed as the number for the hour.
  - (d)–(f) The window: `G(x_d,12:00)`, `G(x,13:00)` (the same log scale), and their
    difference / 1 h (its own colour scale, ±p95 of the window).
- **`dGdt_split.png`** (2 × 3). The local part `[G(x,13)−G(x,12)]/1 h`, plus the advective
  part `[G(x,12)−G(x_d,12)]/1 h`, equals the semi-Lagrangian rate.
  - Top row: every parcel, with band means **+2.30 + 1.90 = +4.20** and r.m.s. 37.7 / 31.2 /
    22.0.
  - Bottom row: the window, on its own scale: means 25.5 + 12.8 = 38.3, r.m.s. 90 / 46 / 60.
  - The local and advective maps show opposite-signed bands either side of the ridge (the
    front moving), which cancel in the sum.
- **Fixes after the first look:**
  - the window is now centred on a front pixel (DG-Q10);
  - smaller markers in (b), so the 0.15-cell arrow shows, and the legend moved below the panel;
  - the window rows have their own colour scale (they saturated on the band scale);
  - the split figure has dedicated colour-bar axes (one overlapped panel c), and its footer is
    wrapped (it was clipped).
- **Env:** run in `frontogenesis`, per the revised Code section (DG-Q9). The imports from
  `slides_scene` / `slides_tracking` work there.

### 2026-11-19 — Figures / 2F, prompt 1 (Opus 5.5)

**Discussion only, as asked; no figure.** I read `operators.frontogenesis` /
`_frontogenesis_discrete`, `strain_from_jacobian` and planning §(F decomposition). I computed
band means at pair 36 (label 83, 1,143 parcels) from the stored `two_F`, `two_F_chain`, `delta`,
`sigma_mag`, `theta_align`, `G` and `DGDt_semilag`:
- discrete `2F` +3.46, continuum +4.21 (ratio 0.82, correlation 0.95);
- divergence term +1.30, strain term +2.99 (the strain term carries about 70 %);
- median θ 29°, 71 % of the parcels at θ < 45°;
- cell-by-cell correlation of `2F` with `DG/Dt`: 0.59.

Questions 2F-Q1 to 2F-Q6 are in Q&A/2F:
- Q1: there is no time difference in `2F`; it uses hour-averaged midpoint fields;
- Q2: explain with the continuum picture and quote the discrete value;
- Q3: pair 36;
- Q4: two figures, `twoF_method.png` and `twoF_split.png`;
- Q5: a 2F vs DG/Dt comparison, later;
- Q6: relative-velocity quivers.

The plan is `py/slides_twoF.py` (frontogenesis env), reusing the geometry and panel helpers of
`slides_dGdt.py`. DG-Q10 and DG-Q11 are still open.

### 2026-11-19 — Figures / 2F, prompt 2 (Opus 5.5)

**Done.** `py/slides_2F.py` (frontogenesis env, about 1 min) writes `slides/2F_method.png` and
`slides/2F_split.png`, with every number in `data/slides_2F_36.json`. One follow-up question,
2F-Q7, is in Q&A/2F. The answers to 2F-Q1 to Q6 all took the recommendations.

- **Data.** Pair 36 (12:30 UTC, from the 12:00 and 13:00 hours), label 83, 1,143 parcels. The
  window and the parcel are read from `data/slides_dGdt_36.json`, so they are the dG/dt slide's.
  Recomputed from the raw store with the pipeline's functions: `inputs.midpoint`,
  `operators.frontogenesis` (discrete and chain), `grad_b`, `jacobian`, `strain_from_jacobian`,
  `strain_alignment`, `centred_model_velocity` (rotated with `CS`/`SN`).
- **Asserted:**
  - the recomputed `two_F`, `two_F_chain` and `G` match the stored fields (≤ 2.1 × 10⁻⁷ of the
    r.m.s.);
  - the band mean of `2F` is +3.4636, equal to `two_F` in `matched_parcels_track_81.json`;
  - convergence + strain = continuum `2F` to 1.2 × 10⁻¹⁵ of the r.m.s. (Jacobian-consistent
    strain set; the stored flux-form `δ`/`σ` gave 1.30 + 2.99 = 4.29 in prompt 1).
- **`2F_method.png`** (2 × 3), with the midpoint inset across the top: "2F is an instant, not a
  difference: `b_mid`, `u_mid` → `2F` at 12:30" (2F-Q1).
  - (a) The front and the window (shared code).
  - (b) The parcel close up: `b_mid` shading and contours, the `∇b` direction, the compressional
    axis (inward arrows), and quivers of `u − u(x)` (2F-Q6, key 5 cm s⁻¹). Numbers:
    `G` = 3.08 × 10⁻¹³, `δ` = −2.18, `|σ|` = 1.48 (×10⁻⁵ s⁻¹), θ = 27°;
    2F = 67.2 + 26.9 = 94.1 (continuum), stored 72.8.
  - (c) The stored `2F` on every parcel: **mean +3.46**, on the colour scale of `dGdt_method.png`
    (c), with "DG/Dt on the same parcels: +4.20" beside it.
  - (d) The window: `G` with quivers of the flow relative to the window mean. Cross-front
    convergence onto the ridge is visible.
  - (e) `cos 2θ` on the parcels, with compressional-axis ticks on the ridge. 72 % of the band has
    θ < 45°, median 29°.
  - (f) The window's `2F` on the dG/dt window scale: mean +34.5, against `DG/Dt` +38.3.
- **`2F_split.png`** (2 × 3), reusing `slides_dGdt.fig_split`, generalised. Convergence `−δG` +
  strain `|σ|G cos 2θ` = continuum `2F`: **+1.29 + 2.92 = +4.21** (r.m.s. 11.1 / 12.5 / 18.6).
  The window gives 16.2 + 20.5 = 36.7. The footer says that strain carries 69 % and that the
  stored discrete value is +3.46 = 0.82 × continuum (correlation 0.95); 2F-Q2 (a).
- **Refactor of `slides_dGdt.py`** (for reuse): `panel_front()` was extracted, and `fig_split()`
  now takes `parts` / `suptitle` / `footer`. `dGdt_method.png` and `dGdt_split.png` regenerate
  **byte-identical** (md5 checked).
- **Fixes after the first look:** the quiver keys collided with a tick label and a title, and
  the title of (e) overlapped its colour bar.

### 2026-11-21 — Figures / Tracking / Method, prompt 1 (Opus 5.5)

- Discussion only; no code and no figure.
- Reviewed `py/tracking.py`: `Flow`, `advect_mask_by`, `iou_dilated`, `flow_weighted_score`,
  `follow_flow`. Also reviewed `fronts/front_tracking.py`: `describe`, `score_candidate`,
  `_radius_px`, `WEIGHTS`, `MAX_SCORE`, `MAX_POSITION_FACTOR`.
- Recomputed link 35 → 36 in a scratch script (frontogenesis env).
  - Scores: label 83 has S = 0.129 (it equals `m4_tracks.json`); runner-up 61 has S = 1.73.
    259 of the 261 candidates are vetoed.
  - Motion: |d| median 0.23 cells. The advected mask differs from M35 in 2 px.
  - Re-detection: raw IoU(81, 83) = 0.69.
- Wrote the method summary and TR-M-Q1–Q5 in Q&A/Tracking/Method.

### 2026-11-21 — Figures / Tracking / Method, prompt 2 (Opus 5.5)

- Read the answers: TR-M-Q1 35 → 36; Q2 (a) ×10; Q3 one figure; Q4 (a) forward only;
  Q5 (b) footnote.
- Added to `py/slides_tracking.py`:
  - `compute_link`: re-runs the decision with `tracking.Flow`, `advect_mask_by` (via
    `Flow.advect`) and `flow_weighted_score`, and asserts against `m4_tracks.json`.
  - geometry helpers `_ll`, `_corners`, `_cells`, `_outline`, `_dots`, `_arrows`, `_frame`,
    `_box`. They are local copies, because `slides_dGdt` imports `slides_tracking` and the
    import cannot go the other way.
  - `method_crop`, `zoom_crop`, `fig_method`.
  - `--what method` (also in `all`), `--step`, `--method`. The summary goes into
    `data/slides_tracking_81.json` under `method`.
- The module now imports `tracking` and `fronts`, so it needs the frontogenesis env. The
  docstring is updated. `slides_dGdt` and `slides_2F` still import cleanly.
- Output: `slides/tracking_method.png`.
  - |d| median 0.23 cells/h (max 0.56); the advected mask loses 2 of 305 px.
  - 261 candidates, 259 vetoed. Label 83 has S = 0.129, label 61 has S = 1.73.
  - Raw IoU(81, 83) = 0.69, dilated 0.86.
- Fixes during the visual check:
  - the close-up first landed between the two changed pixels, off the front. It is now
    centred on the changed pixel with the most front around it;
  - the score legend overlapped its title;
  - the quiver key overlapped the front.
- New questions TR-M-Q6 (the close-up arrows) and TR-M-Q7 (the bar scale).

### 2026-11-21 — Figures / Tracking / Method, prompt 3 (Opus 5.5)

- Wrote `report/frontogenesis_tracking.md`, the detailed tracking method.
  - It defines every symbol on the slide ($M$, $\hat M$, $C$, $\mathbf{d}$, $R$, IoU, $\oplus$,
    $\ell$, $A$, $\theta$, $\tau_k$, $w_k$, $S$) and describes the finder (config D).
  - It covers the four stages with equations, the worked step 35 → 36 (the arithmetic of
    $S_{83} = 0.129$), the walk (backward, gaps, R growth, fallback, split / merge flags) and
    track 81 in numbers.
  - It includes a panel-by-panel key, a list of what tracking does not do, and a code map.
- Answered JXP's three questions in Q&A/Tracking/Method round 3, with new questions TR-M-Q8
  ("allowed" wording) and TR-M-Q9 (callouts in (c)).
- Checks run (scratch scripts `q_method3.py` and `spur36.py` in the session scratchpad,
  frontogenesis env):
  - the dropped pixels and their interpolated values (0.457, 0.494); none gained;
  - the branch component (9 px at 124.74° W, 27.99° N; no label at pair 36; G median
    3.40e-14 → 3.11e-14);
  - the finder re-run with and without despur: `find_fronts` reproduces the stored labels at
    pairs 35 and 36. Without despur, 2 of the 9 branch px survive, and 3 px near the branch
    are pruned;
  - label 81 is the largest label at pair 35 (307 px; then 84 with 305 and 202 with 271);
  - track 81: 71/71 pairs, scores median 0.150, p90 0.472, max 0.567; position 0 at every
    link.
- No change to the slide or the code. The PDF is not built.

### 2026-11-21 — Figures / Tracking / Method, prompt 4 (Opus 5.5)

- Answers read: Q6 fine; Q7 moot (panel removed); Q8 (a) "scored, not chosen"; Q9 no callouts.
- Rewrote `fig_method` in `py/slides_tracking.py` for the slide:
  - only panels (a)–(c) and no title;
  - `SLIDE_SIZE` = 10 × 4.45 in (a 16:9 Google Slide of 10 × 5.625 in, less room for its own
    title) at `METHOD_DPI` = 300, which gives 3000 × 1335 px;
  - `FS_SLIDE` = 16 pt for all text: titles, ticks, colour bar, equations, legend. Only
    maths sub- and superscripts are smaller;
  - axes are placed in inches, and the titles and equations at fixed heights (`Y_TITLE`,
    `Y_EQ`);
  - the score panel, the term list and the footnote are removed. One shared legend
    (2 rows) sits at the bottom; green is "scored, not chosen: 61";
  - the quiver key is removed (the legend says "d ×10"); (b) has no ticks (its box is in (a)).
  - Removed `TERM_NAMES`, `TERM_COLOURS`, `_tag` and the `Patch` import (now unused).
- The checks in `compute_link` are unchanged and pass. The summary json still records all six
  terms of both scored candidates.
- Updated `report/frontogenesis_tracking.md` §6 (panel key: no (d), the green wording) and the
  module docstring.
- Visual check: three passes. They fixed equation boxes overlapping the tick labels and each
  other, and titles at mismatched heights.


### 2026-11-21 — Figures / Tracking / Method, prompt 5 (Opus 5.5)

- `py/slides_tracking.py` `fig_method`:
  - the legend is now `FS_LEGEND` = 12 pt (was 16), with column spacing 1.4;
  - the "2 of 307 px dropped" line under (b) is removed; the equation stays;
  - everything else is unchanged: 16 pt titles, ticks, colour bar and equations;
    10 × 4.45 in at 300 dpi (3000 × 1335 px).
- Re-ran `--what method`. The `compute_link` checks pass, the figure was inspected, and the
  docstring notes the legend size.

### 2026-11-21 — Figures / dG/dt, prompt 4 (Opus 5.5)

- Rewrote `fig_method` in `py/slides_dGdt.py` for a Google Slide:
  - no title and no footnote;
  - 10 × 4.45 in at 300 dpi (3000 × 1335 px), the same size as `tracking_method.png`
    (`SLIDE_SIZE` and `METHOD_DPI` come from `slides_tracking`). Axes are placed in inches;
  - fonts: panel titles `FS_DG` = 13 pt, ticks and colour bars 11 pt, legend
    `FS_DG_LEGEND` = 11 pt, the (c) mean 13 pt bold. These are 2–3× the old sizes at the
    placed size, since the old 17 in figure shrank to 10 in;
  - (b): the contour labels are placed by the new `_edge_labels`. It densifies each contour
    and puts its label at the point farthest from the parcel, inside an x / y margin of
    (0.66, 0.80) so nothing is clipped. There are 6 levels instead of 8, labelled with
    `%.1f`;
  - the text beneath (b) is removed: the G values, the DG/Dt arithmetic and |d|. The
    numbers stay in `data/slides_dGdt_36.json` (`method.parcel`);
  - (d) is titled "(d) Interpolated G(x_d, 12:00)";
  - less white space:
    - the panel legends of (a) and (b) are merged into one shared 2-row legend at the
      bottom (8 entries, including the close-up box and the b-contour units);
    - (d) and (e) share one colour bar;
    - lat/lon ticks are removed except the latitudes of (a). The window, the close-up box
      and (a) carry the geography;
    - the panel letters are in the titles.
- The checks pass (recompute vs stored and json, local + advective = total).
  `dGdt_split.png` is byte-identical (md5 1a3f1067…). `slides_2F` re-run: `2F_method.png`
  and `2F_split.png` are byte-identical too.
- Visual check: four passes. They fixed (a)'s longitudes colliding with (d)'s title, and the
  contour labels clipped at the edge or touching the centre cell.


### 2026-11-23 — Figures / 2F, prompt 3 (Opus 5.5)

**Done.** `slides/2F_method.png` is redrawn in the slide format of `dGdt_method.png` (DG prompt 4):
10 × 4.45 in at 300 dpi (3000 × 1335 px), for a 16:9 Google Slide under the slide's own title.
- **Removed:** the title, the footnote, and the `G`, `δ`, `|σ|`, θ and `2F` numbers under (b);
  those numbers are still in `data/slides_2F_36.json` (`method.parcel`). The (b) legend and the
  "DG/Dt on the same parcels" notes in (c) and (f) are also gone (the comparison is deferred, 2F-Q5).
- **Fonts:** panel titles 13 pt; ticks, colour bars, quiver key and the shared legend 11 pt
  (2–3× the old size at the slide's width).
- **Kept:** the "2F is an instant" box (2F-Q1), shortened to one line across the top:
  `b_mid = ½[b(12:00) + b(13:00)]`, likewise `u_mid` → `2F = −δG + |σ|G cos 2θ` at 12:30.
- **Layout:** one shared legend at the bottom; ticks removed except the latitudes on (a);
  short titles: (a) Label 83, G at 12:30; (b) One parcel at 12:30; (c) 2F, every parcel;
  (d) G, relative flow; (e) Alignment cos 2θ; (f) 2F, window. Contour labels in (b) are
  placed away from the parcel (`_edge_labels`, as on the dG/dt slide).
- **Quivers:** (b) and (d) now share one scale (10 cm s⁻¹ = 0.22 in; the maximum relative
  speed is about 10 cm s⁻¹ in both), with one key between (b) and (c). Before, each had its own key.
- **Reuse:** `slides_dGdt.py` gained `slide_front()` (panel a) and `slide_legend_handles()`,
  used by both slides. `dGdt_method.png`, `dGdt_split.png` and `2F_split.png` regenerate
  **byte-identical** (md5). All assertions in `slides_2F.check` pass.
- Note: matplotlib's quiver keys and arrows are sized in inches from the figure's own dpi, so the
  figure is created at 300 dpi.
- 2F-Q7 (the close-up parcel is convergence-led) is still open; the shared parcel is kept.

### 2026-11-23 — Figures / 2F, prompt 4 (Opus 5.5)

**Discussion only, as asked; no code or figure changed.** The derivation of
`2F = −δG + |σ|G cos 2θ` is in Q&A/2F, Round 3. Its steps:
- Step 0: `G ∝ 1/L²`, isopycnal spacing.
- Step 1: `D∇b/Dt = −(∇u)ᵀ∇b`.
- Step 2: split `∇u` into squeeze + strain + rotation; the rotation drops out.
- Step 3: write the strain term as an angle. The rate form is
  `(1/G) DG/Dt = −δ + |σ| cos 2θ`.
- Step 4: the two pure flows in their own frames.
- Step 5: the parcel and band numbers.
- Step 6: the ageostrophic convergence, the effect of vorticity through θ, and continuum vs
  discrete.

The conventions match `operators.strain_from_jacobian` / `strain_alignment`: ψ is the
extensional axis and θ is measured from the compressional axis.

New numbers at pair 36, from `slides_2F.compute` (Jacobian-consistent set):
- **The close-up parcel:** rates of +2.18 (convergence) + 0.87 (strain) = 3.05 × 10⁻⁵ s⁻¹
  = 0.45 f, an e-folding time of 9.1 h.
- **The band, G-weighted:** −δ +0.22, |σ| cos 2θ +0.51 (×10⁻⁵ s⁻¹), total 0.11 f, an e-folding
  time of 38 h.
- **Divergence against strain:** G-weighted |δ| is 1.06 against |σ| 1.77. Only 59 % of the
  parcels converge, so the convergence mostly cancels. Strain runs at about 30 % of its maximum
  efficiency (⟨cos 2θ⟩ = 0.33).
- **Ridge and flanks:** +1.93 + 4.25 on the ridge, +1.05 + 2.43 on the flanks.

New questions: 2F-Q8 (what goes in the report) and 2F-Q9 (a schematic of the pure flows).
