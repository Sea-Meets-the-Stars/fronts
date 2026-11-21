# How a front is tracked from hour to hour

**Authors:** J. X. Prochaska, with Claude (Opus 5.5)  
**Status:** Method note, 21 November 2026. A companion to the slide `slides/tracking_method.png` (frontogenesis_slides.md, Figures / Tracking / Method).  
**Circulation:** For the collaboration; not for citation.

---

## Summary

The tracker follows one front, label 81 at pair 35 (2012-07-03 11:30 UTC) on LLC4320 tile 330, through the 71 hourly pairs of the analysis window. The fronts at each pair are found independently by the front finder (config D), so a front has no identity from one hour to the next. The tracker supplies one. At every hour it carries the front it has with the model's own flow, looks at all the fronts the finder drew in the next hour, scores each one against the prediction, and keeps the best. This note defines every quantity on the slide (panels a–c), writes each step as an equation (including the score, which the slide leaves out), and works through the step shown there: label 81 at pair 35 to label 83 at pair 36. Every number was recomputed with the tracker's own functions. The script `py/slides_tracking.py --what method` asserts that the recomputation reproduces the stored track (`data/m4_tracks.json`, flow track 0) to 10⁻⁹.

Two facts about this front set the tone. First, **the water barely moves it in an hour**: the median displacement is 0.23 cells (0.4 km). Second, **the finder redraws it substantially every hour**: labels 81 and 83 share only 69% of their pixels (IoU 0.69). The tracker is therefore built to recognise a front whose outline changes more than its position, which is why most of the score is about shape.

---

## 1. Notation

| Symbol | Meaning | Value on the slide |
|---|---|---|
| pair $t$ | The two model hours $h_t$ and $h_{t+1}$. All derived fields (`G`, the fronts) are assigned to the midpoint, $t + \tfrac12$ h | $t = 35$: 11:30 UTC; $t = 36$: 12:30 UTC |
| $\Delta t$ | Time between consecutive pairs | 3600 s |
| $\mathbf{x} = (j, i)$ | Position on the model grid, in cells (row $j$, column $i$) | |
| $\Delta x$ | Mean cell size (median of `dxC`, `dyC`) | 1.87 km |
| $G$ | Front strength, $G = \lvert\nabla b\rvert^2$ [s⁻⁴], from the midpoint buoyancy | grey background of (a), (c) |
| $\mathcal{L}_t(\mathbf{x})$ | The **label field** of pair $t$: 0 off the fronts, and an integer $1, 2, \dots$ that names each separate front (8-connected) | |
| label | One front: the set of pixels where $\mathcal{L}_t$ equals a given integer. Label numbers are arbitrary and are reassigned every hour | 81 (pair 35), 83 and 61 (pair 36) |
| px | Pixel (grid cell). A front is one pixel wide (a skeleton) | |
| $M_t$ | The **mask** of the tracked front at pair $t$: a true/false map, true on the pixels of the chosen label, $M_t(\mathbf{x}) = [\mathcal{L}_t(\mathbf{x}) = \text{label}]$ | label 81: 307 px |
| $\mathbf{u}$ | Surface velocity of the model (centred on the cells) | |
| $\mathbf{d}(\mathbf{x})$ | Displacement over one hour of the water parcel that **arrives** at $\mathbf{x}$, in cells | median 0.23, max 0.56 cells on label 81 |
| $\hat M_{t+1}$ | The **predicted mask**: $M_t$ carried one hour by the flow | 305 px |
| $C$ | A **candidate**: the mask of one label at pair $t + 1$ | 261 candidates |
| $R$ | Search radius, in cells | 2 |
| $\mathrm{dist}(\mathbf{x}, A)$ | Distance from $\mathbf{x}$ to the nearest true pixel of mask $A$ (Euclidean, in cells) | |
| $\lvert A \rvert$ | Number of true pixels in mask $A$ | |
| $\mathrm{IoU}(A, B)$ | **Intersection over union**, $\lvert A \cap B \rvert / \lvert A \cup B \rvert$: 1 if the two masks are identical, 0 if they share no pixel | |
| $A \oplus 1$ | Mask $A$ **dilated** by one cell: every pixel of $A$ plus its 8 neighbours (a 3-pixel-wide ribbon around a 1-px front) | |
| $\ell$ | Length of a front: $4\sqrt{\lambda_{\max}}$, with $\lambda_{\max}$ the larger eigenvalue of the covariance of its pixel positions (the major-axis extent; equal to the true length for a straight line) | 110.6 (81), 114.1 (83) cells |
| $A$ | Area of a front: its pixel count | |
| $\theta$ | Orientation of a front's major axis (from the same covariance), in $[-90^\circ, 90^\circ)$ | −38.0° (81), −44.2° (83) |
| $\tau_k$ | One term of the score, dimensionless; 0 means identical, 1 means "as different as we would tolerate" | |
| $w_k$ | The weight of term $k$ | 1.0, 1.0, 0.8, 0.6, 0.6, 0.4 |
| $S$ | The **score** of a candidate: the weighted mean of its terms. Lower is better | 0.13 (83), 1.73 (61) |
| $S_{\max}$ | The largest score a link may have | 2.5 |

## 2. Where the fronts come from

The tracker does not find fronts; it chooses among the ones the finder drew. The finder (`tracking.find_fronts`, config D, `fronts/finding/configs/finding_config_D.yaml`) runs on each pair's `G` separately:

1. **Threshold.** A pixel is a front candidate if its `G` is above the 85th percentile of `G` in a 64 × 64-cell window around it.
2. **Sharpen** the binary map to a one-pixel-wide ridge.
3. **Remove spurs**: side branches shorter than $L_{\rm spur} = 10$ px are cut off.
4. Drop pieces smaller than 7 px, and **label** what remains: each 8-connected piece becomes one front with its own integer.

Fronts are found only in the analysis region (`mask_analysis`: ocean, more than 100 km from the coast, away from the tile edge).

Because every hour is processed on its own, the same physical front comes out with a different label number, slightly different pixels, and sometimes a branch more or less. §4 shows a branch that is present at pair 35 and absent at pair 36. This hour-to-hour redrawing is the main thing the tracker has to cope with.

## 3. One step of the tracker

Each step goes from the front chosen at pair $t$ to the front chosen at pair $t + 1$, in four stages. These are the four numbered panels of the slide.

### 3.1 Displace (panel a)

For every cell, the displacement over the hour solves

$$\mathbf{d}(\mathbf{x}) = \Delta t\ \mathbf{u}\!\left(\mathbf{x} - \tfrac12\,\mathbf{d}(\mathbf{x})\right),$$

the **midpoint rule**. The parcel that arrives at $\mathbf{x}$ at the end of the hour started at $\mathbf{x} - \mathbf{d}$, and its velocity is taken at the halfway point of that path. The equation is solved by three iterations starting from $\mathbf{d} = 0$, with $\mathbf{u}$ interpolated to the fractional halfway point at cubic order (`semilag.departure_index`). The velocity is the model's surface velocity at the integer hour that falls between the two pairs (hour 36 for pair 35 → 36), held fixed over the hour. The displacements of all 70 transitions, forward and backward, are precomputed in `data/tile330_flow.zarr` (`tracking.Flow`).

On label 81 the median $\lvert\mathbf{d}\rvert$ is 0.23 cells and the largest is 0.56. At true scale the arrows would be invisible, so the slide draws them ten times longer.

### 3.2 Predict (panel b)

The predicted mask is $M_t$ moved by the displacement:

$$\hat M_{t+1}(\mathbf{x}) = \left[\, M_t(\mathbf{x} - \mathbf{d}) \ \ge\ 0.5 \,\right].$$

$M_t$ is a 0/1 map, and $\mathbf{x} - \mathbf{d}$ falls between cells. So $M_t(\mathbf{x} - \mathbf{d})$ is interpolated **bilinearly** from the four surrounding cells, which gives a number between 0 and 1: the blue shading of panel (b). The prediction keeps the cells where that number is at least one half. Only the true/false mask is ever moved: interpolating the label numbers themselves would be meaningless.

Bilinear interpolation was chosen because a one-pixel front shifted by half a cell along a grid axis gives exactly 0.5 on both neighbouring cells, which the "≥ 0.5" keeps. So a straight front cannot vanish. **The exception is a diagonal step of the skeleton.** There, a shift that is not along the step spreads the pixel's weight over four cells, and none of them may reach 0.5. That happens twice on label 81 (both are shown in the slide):

| pixel $(j, i)$ | $\mathbf{d} = (d_j, d_i)$ cells | $M_t(\mathbf{x} - \mathbf{d})$ at the pixel | outcome |
|---|---|---|---|
| (172, 697) | (−0.24, +0.40) | 0.457 | dropped |
| (176, 620) | (+0.36, +0.24) | 0.494 | dropped |

No pixel is gained, so the prediction has 307 − 2 = **305** pixels. This is the answer to "why 307 in (a) and 305 in (b)": the 307 are label 81 as found, and the 305 are the prediction. The two lost pixels leave one-pixel breaks in the predicted front. They hardly matter, because the scores in §3.4 compare dilated masks and nearest-point distances.

### 3.3 Search (panel c)

Every label at pair $t + 1$ with at least 5 px is a candidate $C$: there are 261. The **search radius** is

$$R = \max\!\left(2,\ \frac{v_{\max}\,\Delta t}{\Delta x}\right), \qquad v_{\max} = 0.6\ \mathrm{m\,s^{-1}},$$

the farthest a front could plausibly move in the time since it was last found, with a floor of 2 cells. Here $0.6 \times 3600 / 1870 = 1.15$ cells, so $R = 2$.

The **position term** of a candidate is the distance from its *nearest* pixel to the *nearest* pixel of the prediction, in units of $R$:

$$\tau_{\rm position} = \frac{\min_{\mathbf{x} \in C}\ \mathrm{dist}(\mathbf{x}, \hat M_{t+1})}{R}.$$

The nearest-point distance is used, not the distance between centres. A front is long and curved, and its pieces can be close to the prediction while their centres are far apart.

**Veto.** A candidate with $\tau_{\rm position} > 3$, i.e. no pixel within $3R = 6$ cells of the prediction, is rejected outright, however similar its shape. Of the 261 candidates, **259 are vetoed** (blue in panel c). The two contours in (c) are the distances $R$ and $3R$ from the prediction.

The two survivors are:

- **label 83** (red, 307 px), which touches the prediction: $\tau_{\rm position} = 0$;
- **label 61** (green, "scored, not chosen", 32 px), a separate short front to the west whose nearest pixel is 5 cells from the prediction: $\tau_{\rm position} = 2.5$. It is close enough to be scored, and §3.4 shows it losing clearly.

### 3.4 Score and choose (panel d)

Each surviving candidate gets six terms. They are written here with the reference front $M = M_t$, the prediction $\hat M = \hat M_{t+1}$ and the candidate $C$:

| term $k$ | $\tau_k$ | what it asks | $w_k$ |
|---|---|---|---|
| position | $\min_{\mathbf{x}\in C}\mathrm{dist}(\mathbf{x}, \hat M)\,/\,R$ (veto if $> 3$) | Is it where the flow put the front? | 1.0 |
| flow IoU | $1 - \mathrm{IoU}(\hat M \oplus 1,\ C \oplus 1)$ | Does it lie on the prediction? | 1.0 |
| overlap | $1 - \mathrm{IoU}(M,\ C)$ | Does it lie on the front as last found (not moved)? | 0.8 |
| length | $\lvert\ln(\ell_C / \ell_M)\rvert\, /\, \ln 2$ | Is it as long? (1 = doubled or halved) | 0.6 |
| orientation | $\Delta\theta\, /\, 25^\circ$, with $\Delta\theta$ the angle between the two major axes (0–90°) | Is it aligned the same way? | 0.6 |
| area | $\lvert\ln(A_C / A_M)\rvert\, /\, \ln 2$ | Has it as many pixels? | 0.4 |

The score is their weighted mean:

$$S = \frac{\sum_k w_k\,\tau_k}{\sum_k w_k}.$$

All six terms are defined here, so $\sum_k w_k = 4.4$. A term whose inputs are missing (for example, the orientation of a front too small to have one) is skipped and the sum shrinks accordingly.

Why both an IoU on the moved mask and one on the unmoved mask? The flow IoU tests the prediction. The fronts are one pixel wide, so a raw IoU of two skeletons only one cell apart would already be 0; it is therefore taken on 3-pixel ribbons. Two parallel skeletons give a flow IoU of 1.0 at offset 0, 0.5 at one cell, 0.2 at two cells and 0 beyond. The overlap term is the original tracker's term (`fronts.front_tracking`), kept unchanged. It is raw, so it is the term that registers the finder's redrawing.

**The step 35 → 36:**

| candidate | px | position | flow IoU | overlap | length | orientation | area | **S** |
|---|---|---|---|---|---|---|---|---|
| **83** | 307 | 0.00 | 0.15 | 0.31 | 0.04 | 0.25 | 0.00 | **0.13** |
| 61 | 32 | 2.50 | 1.00 | 1.00 | 1.57 | 1.77 | 3.26 | **1.73** |

For label 83:

- **flow IoU 0.15**: the dilated IoU with the prediction is 0.85;
- **overlap 0.31**: the raw IoU with label 81 is 0.69;
- **length 0.04**: 114.1 against 110.6 cells;
- **orientation 0.25**: −44.2° against −38.0°, a 6.2° turn;
- **area 0**: 307 px in both;

so

$$S_{83} = \frac{1.0(0) + 1.0(0.146) + 0.8(0.309) + 0.6(0.045) + 0.6(0.247) + 0.4(0)}{4.4} = 0.129.$$

Label 61 shares no pixel with either mask, is a third as long, a tenth the size and turned 44°.

**Choose.** The candidate with the lowest $S$ is chosen, provided $S \le S_{\max} = 2.5$. Label 83 wins by a factor of 13. If no candidate passes, the step is a **gap** (§4).

## 4. The branch near 124.75° W, 28.0° N that "was not chosen"

In panel (c) a short orange stub of the prediction sticks out from the red front near 124.75° W, 28.0° N, with no red over it. It is a 9-pixel side branch of label 81 at pair 35, reaching up to 7 cells from label 83.

The tracker cannot choose or reject part of a front: it chooses whole labels. The question is therefore why label 83, the finder's front at pair 36, has no such branch. **At pair 36 the finder draws nothing there.** No label of any number covers those pixels. Re-running the finder on pair 36 shows the reason:

- **Threshold and sharpen.** These keep only 2 of the 9 branch pixels (3 within two cells of the branch). `G` along the branch fell by 9% over the hour (median $3.4 \times 10^{-14}$ to $3.1 \times 10^{-14}$ s⁻⁴), and most of it is no longer above the 85th-percentile threshold of its window.
- **Spur removal.** The 3-pixel stub that remains is shorter than $L_{\rm spur} = 10$ and is cut off.

At pair 35 the same steps keep all 9 pixels. Re-running the finder reproduces the stored labels exactly at both pairs, so this is the finder's behaviour, not a tracking error.

The cost to the tracker is small. The branch is part of the prediction but not of label 83, so it lowers the flow IoU and the overlap a little (both are in the 0.15 and 0.31 above). Conversely, label 83 has an 11-pixel stretch near 124.2° W, 28.55° N that is more than a cell from the prediction: the finder added a piece there.

## 5. The walk through the window

- **Anchor.** The track starts from a chosen label: label 81 at pair 35, the largest front at the midpoint of the window. Its score is 0 by definition.
- **Forward, then backward.** The tracker steps forward to pair 70, then backward from the anchor to pair 0. The backward walk is the same procedure with $\Delta t = -1$ h: the displacement is computed with the time step reversed, and the prediction is the front carried back one hour.
- **Gaps.** If no candidate passes at pair $t + 1$, nothing is recorded there. The prediction keeps being carried by the flow, hour by hour. The reference shape for the length, area, orientation and overlap terms stays the last front found. $R$ grows with the time since it was found (0.6 m s⁻¹ times the elapsed time, floor 2 cells). So the front can be picked up again where the water carried it. If the prediction is carried off the tile, the tracker falls back to the original tracker's prediction: the last front shifted by its recent centroid velocity.
- **Splits and merges** are flagged, never used in the choice. A link is a *split* if the previous front, carried forward, overlaps two or more labels at the new pair by at least 5 px each. It is a *merge* if the new front, carried back, overlaps two or more labels at the previous pair. These flags are what the GIF titles and the matched-parcel figures mark.

**Track 81 in numbers:**

- found at **all 71 pairs** (no gaps);
- link scores: median 0.15, 90th percentile 0.47, largest 0.57, all far below 2.5;
- the chosen label touches the prediction ($\tau_{\rm position} = 0$) at every link;
- 10 links are flagged as splits and 9 as merges (`tracked/matched_parcels_track_81.md`).

## 6. Reading the slide, panel by panel

The slide shows three panels (a)–(c). The score (§3.4) is in this note only.

- **(a) The front at pair 35.**
  - Grey: $\log_{10} G$ (dark = strong).
  - Cyan dots: the 307 pixels of label 81, i.e. $M_{35}$.
  - Yellow arrows: $\mathbf{d}$ at every 4th pixel along the front, drawn ×10.
  - Black square: the close-up of (b).
  - Below: the displacement equation and its size.
- **(b) The prediction, close up** (15 × 15 cells around one of the dropped pixels).
  - Blue shading: $M_{35}(\mathbf{x} - \mathbf{d})$, the moved mask before thresholding.
  - Cyan outline: $M_{35}$.
  - Dashed orange outline: $\hat M_{36}$, the cells where the shading is at least 0.5.
  - Red circle: a pixel of $M_{35}$ whose shading is 0.494, just under 0.5, so it is dropped (§3.2).
  - Arrows: $\mathbf{d}$ ×10 at every 3rd cell.
- **(c) The candidates at pair 36.**
  - Grey: $\log_{10} G$ at pair 36.
  - Translucent orange: $\hat M_{36}$.
  - Red: label 83, chosen.
  - Green: label 61, "scored, not chosen" (it passes the veto and loses on score, §3.3–3.4).
  - Blue: the 259 vetoed labels.
  - Dotted and solid orange contours: distance $R$ and $3R$ from the prediction.
  - Below: the veto rule and how many candidates it removes.

## 7. What tracking does not do

- It does not look at the strength of the front. `G` enters only through the finder; the score uses geometry alone.
- It does not choose pixels, only whole labels. Whatever the finder draws as one front at $t + 1$, branches included, is the tracked front at $t + 1$.
- It does not define the parcels on which rates are measured. Those are re-seeded every hour on the chosen label, dilated by one cell (`tracked/matched_parcels_track_81.md`, §1.2). The previous front is used only to choose the next one.

## Code

| What | Where |
|---|---|
| The walk | `py/tracking.py`: `follow_flow` |
| Displacement, prediction | `py/tracking.py`: `Flow`, `advect_mask_by`; `py/semilag.py`: `departure_index`, `interp_to_departure` |
| Score (not on the slide) | `py/tracking.py`: `flow_weighted_score`, `iou_dilated`; `fronts/front_tracking.py`: `score_candidate`, `describe`, `_radius_px`, `WEIGHTS`, `MAX_SCORE`, `MAX_POSITION_FACTOR` |
| Split and merge flags | `py/tracking.py`: `tracking_quality` |
| Finder | `py/tracking.py`: `find_fronts`; `fronts/finding/configs/finding_config_D.yaml` |
| The slide | `py/slides_tracking.py --what method`: `compute_link`, `fig_method` |

![How the tracker makes the link from label 81 (pair 35) to label 83 (pair 36).](../slides/tracking_method.png)
