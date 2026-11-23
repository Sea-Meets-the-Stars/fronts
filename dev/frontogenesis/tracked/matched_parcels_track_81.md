# Matched parcels along the track of label 81 (anchor pair 35)

*Generated 2026-11-23 by `py/matched_parcels.py` (track analysis, "Matched parcels" task 1). Regenerate with `python matched_parcels.py --label 81 --anchor-step 35`. Every number here is in `data/matched_parcels_track_81.json`.*

**Setting.** LLC4320 tile 330, `L = 0`, `mask_analysis`, finder config D, flow-informed track (m4_tracks.json flow track 0). The track is found at 71 pairs, 0–70 (2012-07-02 00:30 to 2012-07-04 22:30 UTC); the label has 104 / 178 / 413 pixels (min / median / max). Rates are on the band-1, centred construction (M4-Q3 (a)) and are quoted in units of 10⁻¹⁹ s⁻⁵. The M3 budget did not close, so **no ratio or slope of these rates is an efficiency**.

## 1. How the hourly rates of the matched parcels are calculated

### 1.1 The fields of one pair

Pair `t` joins the model hours $h_t$ and $h_{t+1}$ ($\Delta t = 3600$ s). Every field of the derived store (`tile330_derived_L0.zarr`) is on the cell centres and is assigned to the midpoint time $t + \tfrac12$ h. Buoyancy and velocity at the midpoint are the two-hour means,

$$b^{m} = \tfrac12\left[b(h_t) + b(h_{t+1})\right], \qquad \mathbf{u}^{m} = \tfrac12\left[\mathbf{u}(h_t) + \mathbf{u}(h_{t+1})\right].$$

**Front strength** (`G`). With $\mathsf{L} = (\mathsf{L}_x, \mathsf{L}_y)$ the centred gradient stencil of `operators.grad_b` (a difference over two cells),

$$G_t(\mathbf{x}) = \left|\mathsf{L}\, b^{m}\right|^2 = b_x^2 + b_y^2 \quad [\mathrm{s^{-4}}].$$

This is the field the fronts were found on and the field the matched rate samples.

**`2F`** (`two_F`), the kinematic prediction. In the continuum

$$F = -\left(u_x\, b_x^2 + (u_y + v_x)\, b_x b_y + v_y\, b_y^2\right), \qquad \frac{D_h G}{Dt} = 2F \ \text{for adiabatic horizontal flow}.$$

The stored field is twice the *discretely consistent* form (`operators.frontogenesis(form='discrete')`), evaluated on the midpoint fields $b^{m}$, $\mathbf{u}^{m}$:

$$2F_t(\mathbf{x}) = -2 \sum_{k \in \{x,\,y\}} \left(\mathsf{L}_k b^{m}\right) \left[\mathsf{L}_k,\ \mathbf{u}^{m}\!\cdot\!\nabla\right] b^{m},$$

$$\left[\mathsf{L}_k,\ \mathbf{u}\!\cdot\!\nabla\right] b = \frac{\left(\mathbf{u}(\mathbf{x}+\mathbf{e}_k) - \mathbf{u}(\mathbf{x})\right)\!\cdot\!\nabla b(\mathbf{x}+\mathbf{e}_k) + \left(\mathbf{u}(\mathbf{x}) - \mathbf{u}(\mathbf{x}-\mathbf{e}_k)\right)\!\cdot\!\nabla b(\mathbf{x}-\mathbf{e}_k)}{2 h_k}.$$

The bracket is the commutator of the gradient stencil with advection; $h_k$ is the grid spacing and the neighbour gradients are fourth order. It is what the semi-Lagrangian difference below tends to under exact advection on this grid. It equals the continuum expression for a linear $b$; at the centre of a front of width $\ell$ the continuum expression is larger by a factor $1 + \tfrac23 (\Delta x/\ell)^2$. `2F` is an instantaneous quantity: it uses one time level (the midpoint) and no time difference. §1.3 derives both forms and compares them on this track.

**Stored `DG/Dt`** (`DGDt_semilag`), the measured tendency. One semi-Lagrangian step over the hour of the pair (`semilag.measured_DGDt`):

$$\left.\frac{D_h G}{Dt}\right|_t(\mathbf{x}) = \frac{G\!\left(\mathbf{x},\, h_{t+1}\right) - G\!\left(\mathbf{x}_d,\, h_t\right)}{\Delta t}, \qquad \mathbf{x}_d = \mathbf{x} - \mathbf{d}(\mathbf{x}), \qquad \mathbf{d} \leftarrow \Delta t\ \mathbf{u}^{m}\!\left(\mathbf{x} - \tfrac12\mathbf{d}\right).$$

- $G(\mathbf{x}, h_{t+1}) = |\mathsf{L}\, b(h_{t+1})|^2$ at the cell: the hour-end buoyancy, not the midpoint mean.
- $\mathbf{x}_d$ is the departure point of the parcel that **arrives at the cell at $h_{t+1}$**; the displacement $\mathbf{d}$ is found by three iterations of the midpoint rule with the velocity interpolated at cubic order.
- $G(\mathbf{x}_d, h_t)$ is formed by interpolating $b(h_t)$ (cubic) onto the five-point stencil centred at $\mathbf{x}_d$ and applying the same $\mathsf{L}$. Buoyancy is interpolated, never $G$.

#### The semi-Lagrangian `DG/Dt`, step by step

*(added 2026-11-21, "Semi-Lagrangian dG/dt" task 1; written from `py/semilag.py`.)*

**What is being measured.** The rate of change of $G$ following the horizontal surface flow,

$$\frac{D_h G}{Dt} = \frac{\partial G}{\partial t} + \mathbf{u}\cdot\nabla_h G .$$

The two terms on the right are each much larger than their sum on a front that moves (the Eulerian form is kept in the store as `DGDt_euler`, a cross-check only). The semi-Lagrangian step avoids forming them: it takes one difference of $G$ along the path of the water, between where the water was at $h_t$ and where it is at $h_{t+1}$. The path is chosen to **end** on a cell centre, so the value at $h_{t+1}$ needs no interpolation and only the value at $h_t$ does. The result is a field on the grid, one value per cell, attached to the cell where the water arrives.

**Step 1 — velocity at the cell centres, in cells per hour.** The model velocities sit on the cell faces (`U` on the faces along the grid index $i$, `V` along $j$). They are averaged in time to the midpoint of the hour and in space to the cell centre, and divided by the local centre-to-centre spacing (`semilag.centre_velocities`, `departure_index`):

$$u_c(j,i) = \tfrac12\left[U^{m}(j,\,i_g{=}i) + U^{m}(j,\,i_g{=}i{+}1)\right], \qquad \hat u = \frac{u_c\,\Delta t}{\Delta x_C}, \qquad \hat v = \frac{v_c\,\Delta t}{\Delta y_C},$$

with $U^{m} = \tfrac12[U(h_t) + U(h_{t+1})]$ and $v_c$ likewise from `V`. Everything is done in index space along the model axes; nothing is rotated to east / north at this stage. $(\hat u, \hat v)$ is the displacement in cells over the hour if the velocity were uniform.

**Step 2 — the departure point.** For the cell centre $\mathbf{x} = (j, i)$ the displacement $\mathbf{d} = (d_j, d_i)$ of the water arriving there is found by the midpoint rule, iterated three times:

$$\mathbf{d}^{(0)} = \hat{\mathbf{u}}(\mathbf{x}), \qquad \mathbf{d}^{(n+1)} = \hat{\mathbf{u}}\!\left(\mathbf{x} - \tfrac12\mathbf{d}^{(n)}\right),\quad n = 0, 1, 2, \qquad \mathbf{x}_d = \mathbf{x} - \mathbf{d}^{(3)} .$$

The velocity is evaluated half-way along the path, at the middle of the hour, which makes the step second order in $\Delta t$. $\hat{\mathbf{u}}$ at the fractional position is interpolated with the cubic of step 3. $\mathbf{x}_d$ is a fractional grid position: where the water that is at the cell centre at $h_{t+1}$ was at $h_t$.

**Step 3 — the interpolation.** A field $f$ at a fractional position $\mathbf{p} = (p_j, p_i)$ is the tensor-product cubic Lagrange interpolant on the 4 × 4 nearest nodes (`semilag.interp_to_departure`). With $j_0 = \lfloor p_j \rfloor$, $\tau_j = p_j - j_0$ and the same along $i$,

$$\tilde f(\mathbf{p}) = \sum_{m=-1}^{2}\ \sum_{n=-1}^{2} \ell_m(\tau_j)\,\ell_n(\tau_i)\, f(j_0 + m,\ i_0 + n), \qquad \ell_k(\tau) = \prod_{\substack{q=-1 \\ q \ne k}}^{2} \frac{\tau - q}{k - q} .$$

It reproduces any cubic exactly and returns the node value at a whole-cell position. If any of the 16 nodes is land or outside the tile the result is NaN; nothing is filled.

**Step 4 — $G$ at the departure point.** $b(h_t)$ is interpolated at five points: $\mathbf{x}_d$ and its four neighbours one cell away along each axis, $\mathbf{x}_d \pm \mathbf{e}_i$ and $\mathbf{x}_d \pm \mathbf{e}_j$. The **same** displacement $\mathbf{d}(\mathbf{x})$ is used for all five, so the stencil is the arrival cell's own stencil moved rigidly to $\mathbf{x}_d$. Write the five values $b_0, b_E, b_W$ (along $i$) and $b_N, b_S$ (along $j$). Then the same gradient stencil $\mathsf{L}$ as everywhere else is applied (`semilag.grad_b_at_departure`):

$$g_i = \tfrac12\left[\frac{b_0 - b_W}{\Delta x_C^{-}} + \frac{b_E - b_0}{\Delta x_C^{+}}\right], \qquad g_j = \tfrac12\left[\frac{b_0 - b_S}{\Delta y_C^{-}} + \frac{b_N - b_0}{\Delta y_C^{+}}\right],$$

$$b_x = g_i \cos\theta - g_j \sin\theta, \qquad b_y = g_i \sin\theta + g_j \cos\theta, \qquad G(\mathbf{x}_d, h_t) = b_x^2 + b_y^2 .$$

$\Delta x_C^{\mp}$ are the spacings to the lower and upper neighbour and $\theta$ the grid angle (`CS`, `SN`), all taken at the arrival cell; they change by 0.02 % per cell. With $\mathbf{d} = 0$ this is bit for bit the ordinary $G$ of the cell.

**Step 5 — $G$ at the arrival cell,** $G(\mathbf{x}, h_{t+1}) = |\mathsf{L}\, b(h_{t+1})|^2$, the ordinary gradient of the hour-end buoyancy at the cell. No interpolation.

**Step 6 — the rate.**

$$\left.\frac{D_h G}{Dt}\right|_t(\mathbf{x}) = \frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}_d, h_t)}{\Delta t}, \qquad \Delta t = 3600\ \mathrm{s}.$$

It is stored at the arrival cell and labelled with the midpoint time of the pair. It can be written as a local change at the cell plus a change along the path at the first hour, which add to it exactly:

$$\frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}_d, h_t)}{\Delta t} = \underbrace{\frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}, h_t)}{\Delta t}}_{\text{local}} + \underbrace{\frac{G(\mathbf{x}, h_t) - G(\mathbf{x}_d, h_t)}{\Delta t}}_{\text{advective}} .$$

**Three choices in this construction, and why.**

- **Buoyancy is interpolated, never $G$.** $G$ has a sharp maximum on a front, and interpolating across a maximum cuts it: bilinear interpolation of $G$ is low by about $\Delta x^2\, G_{xx}/8$ there. A value at $h_t$ that is too low reads as strengthening that did not happen. On a test front 1.5 cells wide, moved half a cell, the error of $G$ at the maximum is −4.8 % with linear interpolation of $b$, −0.49 % with the cubic used here and −0.09 % with a quintic (the store keeps the quintic version as `DGDt_semilag_o5`).
- **The stencil is moved rigidly.** The alternative — carry the whole field $b(h_t)$ to the arrival grid with each cell\'s own displacement, then differentiate — gives $\nabla\!\left[b_t(\mathbf{x}_d(\mathbf{x}))\right] = (\mathsf{I} - \nabla\mathbf{d})^{\mathsf T}\,\nabla b_t$. The factor $\nabla\mathbf{d}$ is the strain of the flow over the hour, which is the frontogenesis itself. For water that keeps its buoyancy that carried field **is** $b(h_{t+1})$, so the difference would be zero when the true rate is $2F$. Holding $\mathbf{d}$ fixed across the stencil keeps the strain out of the value at $h_t$, so that it appears in the difference.
- **This is why it can be compared with `2F`.** If buoyancy is carried unchanged, $b(\mathbf{x}, h_{t+1}) = b(\mathbf{x}_d(\mathbf{x}), h_t)$, the two gradients differ only because neighbouring cells have different displacements. Expanding that difference to first order in $\Delta t$ gives the commutator form of `2F` written above, on the same stencil. Anything else that changes $b$ along the path — vertical exchange, surface fluxes, mixing, the model\'s own numerical diffusion — and the error of an hourly step, appears as a difference between the stored `DG/Dt` and `2F`.

**What limits it.**

- *The hourly step.* The path is built from two velocity snapshots an hour apart. The Eulerian and the semi-Lagrangian estimates of the same rate differ by 0.672 of its r.m.s. on the pooled front pixels at this scale (report §7.1); the interpolation order accounts for 0.108.
- *Ridge cells.* The value at $h_t$ is interpolated and the value at $h_{t+1}$ is read at a cell. On the crest of a ridge one cell wide the interpolated value is slightly low, which biases the rate positive: +0.6 to +1 % of $G$ per hour on a synthetic ridge whose true rate is zero (report §9.3, §10). §4 below shows the stored rate is much larger on the label's own pixels than on the flanks; how much of that is this bias is not established.
- *Time levels.* It uses hour-end buoyancy; `G` and `2F` use the midpoint mean. All three are centred on the midpoint of the pair.

**A worked example: pair 36 of this track** (2012-07-03, 12:00 → 13:00 UTC), drawn by `py/slides_dGdt.py`. The tracker chose label 83 at this pair: 307 pixels, 1,143 parcels after the one-cell dilation. Every field in the figure was recomputed from the raw store with the functions above and agrees with the stored `DGDt_semilag` to 2e-07 of its r.m.s. on the parcels.

![The semi-Lagrangian DG/Dt of pair 36](../slides/dGdt_method.png)

**Figure (from the slides): `slides/dGdt_method.png`.** Unlike the other figures of this document it is drawn north-up in longitude and latitude.

- **(a)** The front on $\log_{10} G$ at the midpoint (12:30 UTC): the label's pixels (red), the parcels (cyan outline) and a 15 × 15-cell window (orange) used for the lower row.
- **(b)** Steps 2 to 4 for one parcel, at true scale. The background and contours are $b$ at 12:00; the grey lines are cell edges. The white dot is the arrival cell centre $\mathbf{x}$ (grid cell j 126, i 653; black square). The red dot is its departure point $\mathbf{x}_d$: $\mathbf{d} = (d_i, d_j) = (-0.144, -0.035)$ cells, 0.15 of a cell in length. The four open circles are the stencil neighbours of step 4, one cell from $\mathbf{x}_d$; $b$ at 12:00 is interpolated at these five points.
  - For this parcel $G(\mathbf{x}, \text{13:00}) = 3.206$ and $G(\mathbf{x}_d, \text{12:00}) = 2.942$ (× 10⁻¹³ s⁻⁴), so $D_hG/Dt = (3.206 - 2.942) \times 10^{-13} / 3600 = +73.4$ × 10⁻¹⁹ s⁻⁵: a 9 % rise of $G$ in the hour.
- **(d), (e), (f)** The same three numbers for every parcel in the window: $G$ at the departure points at 12:00 (step 4), $G$ at the arrival cells at 13:00 (step 5), and their difference over one hour (step 6). The two $G$ maps are nearly identical to the eye; the rate is their small difference. The window holds 59 parcels with a mean of +38.3.
- **(c)** The rate at every parcel of the front. The mean over the 1,143 parcels is **+4.20** × 10⁻¹⁹ s⁻⁵. This is $r_{\rm store}$ of pair 36: the "stored" entry of that pair in the table of §7 and one point of the stored curve in Figure 1. Individual parcels reach several tens in either sign (r.m.s. 22.0), so the band mean is a small remainder of values of both signs.

Displacements on this band are a fraction of a cell: median 0.20, 90th percentile 0.37, largest 0.54 cells. In band means, $G(\mathbf{x}_d, h_t) = 5.647$, $G(\mathbf{x}, h_t) = 5.715$ and $G(\mathbf{x}, h_{t+1}) = 5.798$ (× 10⁻¹⁴ s⁻⁴). The local part is therefore +2.30 and the advective part +1.90, which add to +4.20. Parcel by parcel the two parts are larger than their sum (r.m.s. 37.7 and 31.2 against 22.0) and largely cancel; `slides/dGdt_split.png` shows them. For comparison, at this pair the matched rate is +2.38 and `2F` is +3.46.

### 1.2 The parcels of a step and the three means

At each pair `t` the tracker has chosen one label. `tracking.matched_link(centred=True, band=1)` then does the following, and `matched_parcels.parcel_link` repeats it parcel by parcel (the means agree to round-off at every step; asserted in the run).

1. **Seeds.** The chosen label at `t` — a one-pixel-wide skeleton — is dilated by 1 cell (3 × 3, 8-connected). Every cell $\mathbf{x}_p$ of this band is one parcel at its cell centre: the set $P_t$, of $N_t$ parcels. Median: 657 parcels per step, for a label of 178 pixels; the label's own pixels (the ridge) are a median 27% of the band and its flanks the rest.
2. **Trajectories.** Each parcel is carried forward to pair `t + 1` and backward to pair `t − 1` with the model's surface flow (`tile330_flow.zarr`: the same midpoint-rule displacement as above, over one hour, from `U`, `V` at the integer hour between the two pairs). `forward_positions` solves $\mathbf{x} = \mathbf{x}_p + \mathbf{d}(\mathbf{x})$ for the arrival points $\mathbf{x}_p^{+}$ (in pair `t + 1`) and $\mathbf{x}_p^{-}$ (in pair `t − 1`). The median displacement is 0.23 cells per hour on this track.
3. **The three rates** are means over the same $N_t$ parcels:

$$r_{\rm matched}(t) = \frac{1}{2\Delta t}\left[\frac{1}{N_t}\sum_{p \in P_t} G_{t+1}\!\left(\mathbf{x}_p^{+}\right) - \frac{1}{N_t}\sum_{p \in P_t} G_{t-1}\!\left(\mathbf{x}_p^{-}\right)\right],$$

$$r_{\rm store}(t) = \frac{1}{N_t}\sum_{p \in P_t} \left.\frac{D_h G}{Dt}\right|_t\!(\mathbf{x}_p), \qquad \overline{2F}(t) = \frac{1}{N_t}\sum_{p \in P_t} 2F_t(\mathbf{x}_p).$$

   $G_{t\pm1}$ are sampled at the fractional positions at cubic order; the stored tendency and `2F` are read at the seed cells, with no interpolation. Both samples of the matched rate are interpolated and the selection at `t` is symmetric in time, so the rate is centred on pair `t`.
4. **Parcels kept.** A parcel counts only if every sample is finite. The first and last pairs have no centred difference, so 69 of 71 steps have rates.

**The previous label does not enter any rate.** The parcels of step `t` are seeded on the label chosen at `t`; the label at `t − 1` is used only by the tracker, to choose the label at `t`. The parcel population is therefore re-seeded every hour. The figures of §6 show how the chosen label compares with the previous label carried by the flow, which is what decides how much the population changes from one hour to the next.

### 1.3 How `2F` is calculated: the continuum expression and its two forms on the grid

*(added 2026-11-23, "2F" task 1; written from `py/operators.py`, `frontogenesis` and `_frontogenesis_discrete`.)*

There is one continuum expression for $F$ and there are two ways of evaluating it on the model grid. The derived store holds both: `two_F_chain` and `two_F`. **Every `2F` in this document, and in the report, is `two_F`, the discretely consistent form.** In short:

| | what it is | exact for | in the store |
|---|---|---|---|
| continuum | the formula, for smooth fields | water that keeps its buoyancy and moves horizontally | — |
| chain-rule form | the formula with every derivative replaced by a grid difference | nothing on a finite grid; error of order $(\Delta x/\ell)^2$ for a front of width $\ell$ | `two_F_chain` |
| discretely consistent form | the rate of change of the **gridded** $G = \lvert\mathsf{L} b\rvert^2$ under advection | the quantity the stored `DG/Dt` measures, as $\Delta t \to 0$ | `two_F` (used here) |

So the continuum expression is exact as mathematics, but it cannot be evaluated exactly from gridded fields. Of the two gridded forms, the chain-rule form is the more direct transcription and the discretely consistent form is the one that matches what the measured side computes. Neither is "the exact one"; they answer slightly different questions, and on fronts one or two cells wide the difference is not small.

#### The continuum expression

If buoyancy is carried by the horizontal flow without change, $D_h b/Dt = \partial_t b + \mathbf{u}\cdot\nabla b = 0$. Take the gradient of that equation. The gradient does not commute with advection, and the extra term is the velocity gradient acting on the buoyancy gradient:

$$\frac{D_h}{Dt}\frac{\partial b}{\partial x_k} = -\,\frac{\partial \mathbf{u}}{\partial x_k}\cdot\nabla b, \qquad k \in \{x, y\}.$$

Multiply by $2\,\partial b/\partial x_k$ and sum over $k$, with $G = b_x^2 + b_y^2$:

$$\frac{D_h G}{Dt} = -2\,(\nabla b)^{\mathsf T}(\nabla\mathbf{u})\,\nabla b = 2F, \qquad F = -\left(u_x\, b_x^2 + (u_y + v_x)\, b_x b_y + v_y\, b_y^2\right).$$

The same quantity in terms of the divergence $\delta = u_x + v_y$, the strain magnitude $|\sigma|$ and the angle $\theta$ between $\nabla b$ and the compressional axis of the strain:

$$2F = -\,\delta\, G + |\sigma|\, G \cos 2\theta .$$

Convergence ($\delta < 0$) sharpens the front; strain sharpens it when the buoyancy gradient lies within 45° of the compressional axis. The only assumptions are that $b$ is conserved along the horizontal flow and that the fields are smooth. Vorticity does not appear: rotating the gradient does not change its length.

#### Form 1 — the chain-rule form (`two_F_chain`)

The formula for $F$ is evaluated with each derivative replaced by its grid estimate at the cell centre (`operators.frontogenesis(form='chain')`):

- $b_x, b_y$ from the gradient stencil $\mathsf{L}$ of `operators.grad_b`: differences of $b^m$ to the two faces of the cell, averaged back to the centre (a centred difference over two cells), rotated to east / north.
- $u_x, u_y, v_x, v_y$ from `operators.jacobian`: the face velocities $\mathbf{u}^m$ are averaged to the cell centres and rotated to east / north, then differenced and averaged back in the same way.
- the three products are formed at the cell and summed; the stored field is $2F$.

Each derivative is second-order accurate, which is very good for a feature many cells wide. The weakness is specific to this use. The measured side does not hold the true $|\nabla b|^2$; it holds the gridded $G = |\mathsf{L} b|^2$. A two-cell difference underestimates the gradient at the centre of a narrow front, so the gridded $G$ is smaller than the true one, and its rate of change is smaller too. The chain-rule form does not know this: it applies the product rule of calculus, which a finite-difference stencil does not obey. It therefore predicts a larger rate than the gridded $G$ can show, even for a perfectly advected field with no physics missing.

#### Form 2 — the discretely consistent form (`two_F`, the one used)

The idea is to repeat the continuum derivation with the stencil $\mathsf{L}$ in place of the derivative, keeping the commutator instead of applying the product rule. $\mathsf{L}$ is linear, so under $\partial_t b = -\mathbf{u}\cdot\nabla b$,

$$\frac{D_h}{Dt}\left(\mathsf{L}_k b\right) = -\,\mathsf{L}_k\!\left(\mathbf{u}\cdot\nabla b\right) + \mathbf{u}\cdot\nabla\!\left(\mathsf{L}_k b\right) \equiv -\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b, \qquad \frac{D_h}{Dt}\left|\mathsf{L} b\right|^2 = -2\sum_{k}\left(\mathsf{L}_k b\right)\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b = 2F_{\rm disc}.$$

In the continuum the commutator is $(\partial_k \mathbf{u})\cdot\nabla b$ and the chain-rule form is recovered. For the two-cell stencil, with $\mathsf{L}_k f = [f(\mathbf{x}+\mathbf{e}_k) - f(\mathbf{x}-\mathbf{e}_k)]/2h_k$, it is exactly

$$\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b = \frac{1}{2}\left[\frac{\mathbf{u}(\mathbf{x}+\mathbf{e}_k) - \mathbf{u}(\mathbf{x})}{h_k^{+}}\cdot\nabla b(\mathbf{x}+\mathbf{e}_k) + \frac{\mathbf{u}(\mathbf{x}) - \mathbf{u}(\mathbf{x}-\mathbf{e}_k)}{h_k^{-}}\cdot\nabla b(\mathbf{x}-\mathbf{e}_k)\right].$$

Read it as the chain-rule term with two changes: the velocity gradient is the pair of **one-sided** differences to the two neighbours, and each is multiplied by the buoyancy gradient **at that neighbour**, not at the cell itself. Across a narrow front the gradient at the neighbours is weaker than at the centre, which is how this form comes out smaller. In fact the second change is the only one: **the same sum with the centre gradient $g(\mathbf{x})$ in place of the neighbour gradients is the chain-rule form**, because the mean of the two one-sided velocity differences is the centred velocity gradient. `py/slides_2F.py` asserts this on the band of pair 36 (agreement to 7e-12 of the r.m.s.; *added 2026-11-23, slides 2F prompt 9*). At the close-up parcel of the slides (j 126, i 653) the four neighbour terms are +40.7 (north), +21.8 (south), +4.3 (east) and +6.0 (west), summing to `two_F` = +72.8; with the centre gradient they are +51.6, +32.2, +4.0 and +6.2, summing to `two_F_chain` = +94.1. Almost all of it is the cross-front (north–south) pair, where the neighbours read 0.75 and 0.78 of the centre's gradient. Figure: `slides/2F_discrete.png`. As coded (`_frontogenesis_discrete`), on the midpoint fields $b^m$, $\mathbf{u}^m$:

1. $g = (g_i, g_j) = \mathsf{L} b^m$ along the two model axes (`grad_b`, then the rotation undone). $F$ does not depend on the choice of axes, and the stencil is naturally written along the grid.
2. $\mathbf{u}_c = (u_c, v_c)$: the face velocities averaged to the cell centres, along the same axes. This is the same centred velocity the semi-Lagrangian departure point of §1.1 is built from.
3. The gradient at the neighbours is taken at fourth order, each component along its own axis: $g^{(4)}_i = \left[4 g_i - \tfrac12\left(g_i(i{+}1) + g_i(i{-}1)\right)\right]/3$, and likewise $g^{(4)}_j$ along $j$. This is also the node derivative of the cubic interpolant the semi-Lagrangian step uses.
4. For each axis $k$, with $h_k^{\pm}$ the spacings to the upper and lower neighbour,

$$C_k = \frac12\left[\frac{\left(\mathbf{u}_c(\mathbf{x}{+}\mathbf{e}_k) - \mathbf{u}_c(\mathbf{x})\right)\cdot g^{(4)}(\mathbf{x}{+}\mathbf{e}_k)}{h_k^{+}} + \frac{\left(\mathbf{u}_c(\mathbf{x}) - \mathbf{u}_c(\mathbf{x}{-}\mathbf{e}_k)\right)\cdot g^{(4)}(\mathbf{x}{-}\mathbf{e}_k)}{h_k^{-}}\right], \qquad 2F_{\rm disc} = -2\left(g_i\, C_i + g_j\, C_j\right).$$

**How the two forms differ.**

- They are equal when $b$ varies linearly across the stencil, and they converge as the front becomes wide compared with the grid.
- At the centre of a front with a tanh profile of width $\ell$, the chain-rule form is larger than the rate of the gridded $G$ by the factor $1 + \tfrac23(\Delta x/\ell)^2$. The discrete form over the chain-rule form is then 0.86 for $\ell = 2\Delta x$ and 0.96 for $\ell = 4\Delta x$. The discrete form is correct to order $(\Delta x/\ell)^4$.
- The form was not chosen in advance. The project's first validation test advected a tracer with the analysis's own semi-Lagrangian step, so that the truth was known, and regressed the measured rate on the chain-rule `2F`: the result on the real tile was 0.758 where 1 was required. With the discretely consistent form and a cubic departure velocity the same test gives 0.981 [0.970, 0.994] (report §4, "the discrete null"). That is a property of the operators on a synthetic truth, not a result about the model's fronts.
- What the discrete form gives up: it predicts the rate of the gridded $G$, which on a front one or two cells wide is not the rate of the true $|\nabla b|^2$. The comparison in this document is between two gridded quantities, which is what makes it a fair one.

**The two forms on this track** (means over the same parcels at every step: confirmed; × 10⁻¹⁹ s⁻⁵; 69 steps). The ratio of the two forms compares two predictions and is not a measured over predicted ratio.

| | chain-rule form | discrete form (`two_F`) |
|---|---|---|
| mean over the track | +3.87 | +2.79 |
| r.m.s. of the hourly band means | 6.41 | 5.63 |
| fraction of hours positive | 0.83 | 0.81 |
| mean on the ridge (the label's own pixels) | +5.21 | +3.82 |
| mean on the flanks | +3.37 | +2.41 |
| integral over the track (10⁻¹⁴ s⁻⁴) | +9.40 | +6.76 |
| against the stored `DG/Dt`: correlation of hourly means | 0.942 | 0.947 |
| against the stored `DG/Dt`: r.m.s. difference | 1.96 | 1.61 |
| against the stored `DG/Dt`: mean difference (stored − form) | -0.85 | +0.23 |
| against the matched rate: correlation of hourly means | 0.924 | 0.938 |
| against the matched rate: r.m.s. difference | 3.51 | 2.49 |
| against the matched rate: mean difference (matched − form) | -2.90 | -1.82 |
| per parcel within a step, correlation with the stored `DG/Dt` (median) | 0.41 | 0.46 |
| per parcel within a step, correlation with the matched rate (median) | 0.28 | 0.36 |

- The hourly band means of the two forms correlate at 0.997; the discrete form is 0.88 of the chain-rule form in r.m.s. and 0.72 in the mean. The chain-rule form is the larger in magnitude at 86% of hours.
- Parcel by parcel within a step they correlate at a median 0.95, with the discrete form a median 0.86 of the chain-rule form in r.m.s. They are the same pattern at different amplitude, not two different fields.
- The rows "against the stored `DG/Dt`" and "against the matched rate" show which form sits closer to each measured rate on this track. They are descriptive: the budget did not close, and no row is an efficiency.

**A worked example: pair 36 of this track** (label 83, 1,143 parcels, midpoint 12:30 UTC), the same hour, front, window and parcel as the `DG/Dt` figure of §1.1; drawn by `py/slides_2F.py`, which recomputes both forms from the raw store and agrees with the stored fields to 2e-07 of their r.m.s.

![2F of pair 36](../slides/2F_method.png)

**Figure (from the slides): `slides/2F_method.png`**, north-up. The picture is explained with the continuum expression (convergence plus aligned strain); the numbers in (c) and (f) are the stored discrete form. (a) the front and the window; (b) one parcel: contours of $b^m$, the direction of $\nabla b$ (red), the compressional axis of the strain (black) and the flow relative to the parcel (purple); (c) `2F` at every parcel; (d) the relative flow in the window, converging on the front; (e) the alignment $\cos 2\theta$; (f) `2F` in the window.

- **The parcel of panel (b)** (j 126, i 653): $G = 3.082$ × 10⁻¹³ s⁻⁴, $\delta = -2.18$ × 10⁻⁵ s⁻¹ (convergent), $|\sigma| = 1.48$ × 10⁻⁵ s⁻¹, $\theta = 27°$. Convergence gives $-\delta G = +67.2$ and strain $|\sigma| G \cos 2\theta = +26.9$, so the chain-rule form is +94.1. The discrete form at the same cell is **+72.8**, 0.77 of it.
- **The band.** The chain-rule form averages +4.21 (+1.29 from convergence and +2.92 from strain; `slides/2F_split.png`); the discrete form averages **+3.46**, 0.82 of it, and the two correlate at 0.95 over the parcels. 72% of the parcels have $\theta < 45°$.
- **Against the measured rates at this pair:** stored `DG/Dt` +4.20, matched rate +2.38. One pair shows how the numbers are formed; it is not evidence for either form.

## 2. Are the three rates calculated on the same set of parcels in each pair?

**Confirmed, with two qualifications.** Within each pair the three rates are means over one and the same set of seed cells $P_t$. Counted in the run: 55,213 parcels over the 69 steps; 55,213 enter the matched and the stored rate and 55,213 enter `2F` — the same number, at every step. No parcel was dropped.

In the code, one mask (`alive`: `G` finite at all three times and the stored tendency finite) selects the parcels of the matched and the stored rate; `2F` uses that mask and also requires `2F` finite.

The qualifications are about what the same seed cell means for each rate:

| rate | read at | built from | time span |
|---|---|---|---|
| `2F` | the seed cell, pair `t` | midpoint fields $b^m$, $\mathbf{u}^m$ | an instant, $t + \tfrac12$ h |
| stored `DG/Dt` | the seed cell, pair `t` | $b(h_t)$, $b(h_{t+1})$, $\mathbf{u}^m$ | one hour, $h_t \to h_{t+1}$ |
| surface-flux term $S$ (§5) | the seed cell, pair `t` | midpoint fluxes, $\Theta$, $\mathcal{S}$, $b^m$ | an instant, $t + \tfrac12$ h |
| matched | $\mathbf{x}_p^{-}$ in pair `t − 1`, $\mathbf{x}_p^{+}$ in pair `t + 1` | midpoint $G_{t-1}$, $G_{t+1}$ | two hours, $t - \tfrac12 \to t + \tfrac32$ h |

1. **A half-hour offset in the stored tendency.** The matched parcel sits on the seed cell at the midpoint time. The stored tendency at that cell belongs to the parcel that arrives there at the *end* of the hour, which at the midpoint is half a displacement upstream: a median 0.12 cells here. The parcels are the same to within that distance, not identically.
2. **Different time windows and time levels.** The matched rate differences midpoint-mean `G` fields two hours apart; the stored tendency differences hour-end `G` fields one hour apart; `2F` is instantaneous. All three are centred on $t + \tfrac12$ h.

Between pairs the sets are *not* the same: $P_t$ and $P_{t+1}$ are seeded on different labels (§6).

## 3. The track at a glance

![](figs/matched_parcels_track_81_series.png)

**Figure 1.** Top: the three hourly rates on the matched parcels. Bottom: pixel counts of the chosen label, of the previous label advected to the same pair, of their same-cell overlap, and of the pixels of each set farther than one cell from the other. ▲ split, ▼ merge.

| over the track | `2F` | stored `DG/Dt` | matched rate |
|---|---|---|---|
| mean [10⁻¹⁹ s⁻⁵] | +2.79 | +3.01 | +0.97 |
| fraction of hours positive | 0.81 | 0.83 | 0.58 |

Hour by hour (69 steps): matched against stored, correlation 0.968, r.m.s. difference 0.437 of the stored r.m.s.; matched against `2F`, correlation 0.938, sign agreement 0.74; stored against `2F`, correlation 0.947. These are consistency checks, not efficiencies.

Pixel sets over 70 links: the median fraction of the chosen label's pixels that coincide cell-for-cell with the advected previous label is 0.73; the median dilated IoU is 0.81. More than one cell from the advected previous label lie a median 0.04 of the chosen pixels; 8 links have more than a quarter of the chosen label new in this sense (pairs 7, 11, 12, 16, 21, 30, 39, 67) and 9 have more than a quarter of the advected label unmatched (pairs 5, 10, 11, 13, 14, 18, 27, 38, 40). Splits: 10; merges: 9; links with either: 18.

**Check against M4.** The three rates equal the series in `data/m4_strength.json` (the numbers behind the report's Figure 11) exactly at every step; the labels are identical.

![](figs/matched_parcels_track_81_parcels.png)

**Figure 2.** The matched parcels of pairs 30, 35, 67, one row per pair, each parcel drawn at its seed cell. Left: the per-parcel centred difference `[G(t+1, x⁺) − G(t−1, x⁻)] / 2Δt`; middle: the stored tendency; then `2F` and the surface-flux term at 1 m (§5). The rate quoted for a step is the mean of the map.

## 4. `2F` against the stored `DG/Dt`: what the agreement does and does not say

**Do the two track one another through the window?** The numbers, hour by hour on the band means (69 steps):

| pair of rates | correlation | r.m.s. difference [10⁻¹⁹ s⁻⁵] | sign agreement | mean difference [10⁻¹⁹ s⁻⁵] |
|---|---|---|---|---|
| stored − `2F` | 0.947 | 1.61 | 0.96 | +0.23 |
| matched − stored | 0.968 | 2.33 | 0.75 | -2.05 |
| matched − `2F` | 0.938 | 2.49 | 0.74 | -1.82 |

For scale, the r.m.s. of the hourly rates is 5.63 (`2F`), 5.32 (stored) and 4.48 (matched). So on this track the stored tendency and `2F` agree in the hour-to-hour shape, in sign on 96% of hours and in the mean to 0.23; the stored tendency is the larger on 68% of hours. Of the three pairs of rates, stored and `2F` are the closest in r.m.s.

**Does this imply that the front strengthens or weakens at the rate predicted by `2F`?** Not on its own. What the agreement establishes is narrow: averaged over the band of that hour's parcels, the one-hour semi-Lagrangian tendency of `G` has the sign, timing and amplitude of the kinematic prediction. Four things stand between that and a statement about the front.

1. **The agreement is between band means, not between parcels.** Parcel by parcel within a step, the stored tendency and `2F` correlate at a median 0.46; the matched rate and the stored tendency, two measurements of the same thing, at 0.86; the matched rate and `2F` at 0.36. Split by position in the band (means over the track, 10⁻¹⁹ s⁻⁵):

   | | `2F` | stored `DG/Dt` | matched |
   |---|---|---|---|
   | ridge (the label's own pixels, 27% of the band) | +3.82 | +12.02 | +5.08 |
   | flanks | +2.41 | -0.20 | -0.49 |
   | whole band | +2.79 | +3.01 | +0.97 |

   The band means of `2F` and the stored tendency are close, but they are made up differently: `2F` is spread over ridge and flanks, while the stored tendency is concentrated on the ridge. The report (§9.3) showed that the stored construction has a positive registration bias on one-cell ridges, which a band cancels to first order; how much of the ridge value here is that bias is not established.
2. **The other measurement of the same rate gives a different level.** The matched rate is a second estimate of the Lagrangian rate of the same parcels. It follows the same shape (correlation 0.968 with the stored tendency) but its mean is +0.97 against +3.01, it is below the stored tendency on 100% of hours and it is positive on 58% of hours against 83%. The two measurements differ from each other (2.33 r.m.s.) by more than the stored one differs from `2F` (1.61). Whether the parcels' `G` changes *at the rate* `2F` therefore depends on which measurement is taken; which is nearer the truth is not settled.
3. **Integrated over the window the three disagree.** Summed over the 68 hours between the first and last step with a rate (trapezoid rule, 10⁻¹⁴ s⁻⁴): `2F` +6.76, stored +7.31, matched +2.31. Over the same hours the mean `G` of the band of the label as found went from 6.63 to 6.74, a change of +0.11. None of these integrals is the history of one set of water: the parcels are re-seeded on a new label every hour (§6), and water followed for a day leaves the front (report §5.3). None of their ratios is an efficiency.
4. **The budget is not closed on these parcels.** On the same parcels the measured surface-flux term averages +12.90, the vertical term -0.08 and the residual -12.59 (10⁻¹⁹ s⁻⁵). The stored tendency is close to `2F` alone, not to `2F` plus the other measured terms, so the near-equality is not a closed budget (report §7.2). §5 takes the surface-flux term up in both of its forms.

**In short.** Hour by hour, the mean tendency of `G` on this front's parcels rises and falls with `2F`, and by the stored measurement it does so at about the predicted amplitude. That is a consistency check on band-mean rates. It does not show that the front, as an object, strengthens at the rate `2F`, and by the matched measurement even its water does not, in the mean. One front, one 72-hour window.

## 5. The surface-flux term on the matched parcels

**Can it be included?** Yes. The derived store holds the surface-flux term of the five-term budget for every pair, on the same grid and at the same midpoint time as `2F`, so it is read at the seed cells of the matched parcels exactly as `2F` is. Nothing new was computed from the model output.

### 5.1 What the term is

The budget of the report (§2.3) at `L = 0`, where the sub-filter term is identically zero, is

$$\frac{D_h G}{Dt} = 2F + V + S + R,$$

with $V$ the vertical term, $S$ the surface-flux term and $R$ the residual. The model forces its top cell with the net heat flux $Q_{\rm net}$, the shortwave flux $Q_{\rm sw}$ and the freshwater flux $F_w$ (positive into the ocean). Within a layer of thickness $h$ (report §2.4),

$$\left.\frac{\partial\Theta}{\partial t}\right|_{\rm sfc} = \frac{(Q_{\rm net} - Q_{\rm sw}) + f_{\rm sw}(h)\,Q_{\rm sw}}{\rho_c\, c_p\, h}, \qquad \left.\frac{\partial\mathcal{S}}{\partial t}\right|_{\rm sfc} = -\frac{\mathcal{S}\,F_w}{\rho_c\, h},$$

$$B_{\rm sfc} = \frac{g}{\rho_0}\left(-\rho\,\alpha\left.\frac{\partial\Theta}{\partial t}\right|_{\rm sfc} + \rho\,\beta\left.\frac{\partial\mathcal{S}}{\partial t}\right|_{\rm sfc}\right), \qquad S_t(\mathbf{x}) = 2\,\nabla_h b^{m} \cdot \nabla_h B_{\rm sfc}, \qquad \overline{S}(t) = \frac{1}{N_t}\sum_{p \in P_t} S_t(\mathbf{x}_p).$$

The fluxes, $\Theta$ and $\mathcal{S}$ are midpoint means of the two hours; the gradients use the same stencil $\mathsf{L}$ as `G` and `2F`. $S > 0$ sharpens the front and $S < 0$ weakens it. Two depth scales are stored, as in the report:

- **`surface_flux`, $h = \Delta z = 1$ m** (the top cell). The primary form of the report: the term the top cell obeys instant by instant.
- **`surface_flux_kpp`, $h = h_{\rm bl}$** (the KPP boundary-layer depth at each cell, floored at 1 m). The declared sensitivity of report §7.3: the hourly-mean tendency once the mixing scheme has spread the flux through the boundary layer. On these parcels $h_{\rm bl}$ has a median of 21.6 m (step means 4.0–28.0 m).

Like `2F`, the term is instantaneous at $t + \tfrac12$ h and is read at the seed cell with no interpolation. It is finite on every parcel at every step, so its mean is over the same $N_t$ parcels as the three rates of §2. The surface fluxes are six-hourly forcing interpolated by the model, so the shortwave cycle is a triangle, not resolved insolation.

### 5.2 What it gives on this track

![](figs/matched_parcels_track_81_budget.png)

**Figure 3.** Top: the measured rates (stored, matched) against `2F` alone and against the sum of the predicted terms with the boundary-layer form of $S$. Middle: the surface-flux term in its two forms, with the two imbalances it would have to explain (stored − `2F` − $V$ and matched − `2F` − $V$); the 1 m form is on the right-hand axis. Bottom: the boundary-layer depth on the parcels. Grey bands: local night (18–06 local solar time, UTC − 8 h).

| over the track [10⁻¹⁹ s⁻⁵] | mean | r.m.s. | fraction of hours positive |
|---|---|---|---|
| `2F` | +2.79 | 5.63 | 0.81 |
| stored `DG/Dt` | +3.01 | 5.32 | 0.83 |
| matched rate | +0.97 | 4.48 | 0.58 |
| vertical term $V$ | -0.08 | 0.58 | 0.67 |
| $S$, 1 m | +12.90 | 13.62 | 1.00 |
| $S$, $h_{\rm bl}$ | +0.12 | 0.82 | 0.49 |

The residual left on the band means, as the r.m.s. over the track of measured − predicted:

| predicted side | against stored `DG/Dt` | against the matched rate |
|---|---|---|
| `2F` alone | 1.61 | 2.49 |
| `2F` + $V$ | 1.60 | 2.29 |
| `2F` + $V$ + $S$ (1 m) | 13.48 | 15.44 |
| `2F` + $V$ + $S$ ($h_{\rm bl}$) | 1.61 | 2.52 |

1. **The 1 m form is large and one-signed here.** It averages +12.90 and is positive on 100% of hours, day and night (means by local time: night +13.14, day +12.63); 92% of the parcels are positive in the median step. Its r.m.s., 13.62, is 2.6 times that of the stored tendency. It is larger on the ridge (+19.84) than on the flanks (+10.43).
2. **Adding it at unit weight makes the budget of these parcels worse.** The residual against the stored tendency goes from 1.60 to 13.48, and against the matched rate from 2.29 to 15.44. The measured tendency shows no sign of a forcing of this size: this is the report's finding (§5.2, §7.2) on one front.
3. **It does not project onto what is missing.** Hour by hour, the 1 m term correlates with the imbalance stored − `2F` − $V$ at -0.09 and the multiplier on it that would minimise the residual is +0.018 (against the matched imbalance: -0.20 and -0.128); a correct term would give +1. Parcel by parcel within a step the correlation with stored − `2F` has a median of +0.15. The pooled value in the report is a multiplier of +0.048 at `L = 0`.
4. **The boundary-layer form is small and changes sign.** It averages +0.12 with an r.m.s. of 0.82 — 0.060 of the 1 m form — and is positive on 49% of hours. Adding it leaves the residual where `2F` + $V$ left it (1.61 against 1.60 for the stored tendency; 2.52 against 2.29 for the matched rate). Its correlation with the stored imbalance is +0.22 and with the matched imbalance -0.01. The two forms are not copies of one another: their hourly band means correlate at +0.15, because $h_{\rm bl}$ varies in space and through the day.
5. **The vertical term is negligible** (r.m.s. 0.58).
6. **The 1 m term follows the pixel set; the three rates do not.** This bears on the concern of this analysis. Over the 68 consecutive steps, the size of the hour-to-hour change of each band mean is compared with the size of the change of the label (`|Δ log npix|`); a jump is a change of more than 0.25 in the log (13 links):

   | band mean | correlation of its hourly change with the change of label size | median change at a jump | median change elsewhere |
   |---|---|---|---|
   | `2F` | +0.01 | 1.75 | 1.75 |
   | stored `DG/Dt` | -0.11 | 1.64 | 1.72 |
   | matched rate | -0.05 | 1.40 | 1.76 |
   | $S$, 1 m | +0.94 | 5.65 | 0.40 |
   | $S$, $h_{\rm bl}$ | +0.43 | 0.38 | 0.17 |

   The steps in the red curve of Figure 3 are the hours at which the tracker moves to a shorter or a longer label: the 1 m term has a different mean on different parts of the front, so its band mean changes when the set does (Figure 2, right-hand column, shows where on the front it is concentrated). The hourly changes of `2F` and of the two measured rates are no larger at a jump than elsewhere.

Integrated over the 68 hours (trapezoid, 10⁻¹⁴ s⁻⁴): $S$ at 1 m +31.70, $S$ over $h_{\rm bl}$ +0.27, against `2F` +6.76, stored +7.31 and matched +2.31. The 1 m integral is what carries the four-term curve off the axes in the report's Figure 11.

**Reading.** On this front the surface-flux term does not account for the difference between the measured rates and `2F`, in either form: at 1 m it is far too large and steady, and over the boundary layer it is too small to matter. That is the same result as the pooled budget of the report, and it does not reopen it: the depth scale is a declared sensitivity, not a measurement. It also means the gap between the matched rate and `2F` (§4, point 2) is not explained by the surface flux as stored.

## 6. The chosen label and the advected previous label, pair by pair

North is up (`j` runs east, `i` runs south); the crop is the same in every panel. Black: a cell in both the chosen label and the previous label advected by the flow. Blue: chosen label only. Orange: advected previous label only. Grey: other fronts at that pair. The thin green outline is the band on which the matched parcels are seeded. Each panel's heading gives the three rates of that pair's matched parcels. "Previous" is the previous pair in time (`t − 1`) at every step, including those before the anchor, where the tracker itself walked backwards from `t + 1`.

![](figs/matched_parcels_track_81_sets_00_07.png)

**Figure 4.** Pairs 0–7.

![](figs/matched_parcels_track_81_sets_08_15.png)

**Figure 5.** Pairs 8–15.

![](figs/matched_parcels_track_81_sets_16_23.png)

**Figure 6.** Pairs 16–23.

![](figs/matched_parcels_track_81_sets_24_31.png)

**Figure 7.** Pairs 24–31.

![](figs/matched_parcels_track_81_sets_32_39.png)

**Figure 8.** Pairs 32–39.

![](figs/matched_parcels_track_81_sets_40_47.png)

**Figure 9.** Pairs 40–47.

![](figs/matched_parcels_track_81_sets_48_55.png)

**Figure 10.** Pairs 48–55.

![](figs/matched_parcels_track_81_sets_56_63.png)

**Figure 11.** Pairs 56–63.

![](figs/matched_parcels_track_81_sets_64_70.png)

**Figure 12.** Pairs 64–70.

## 7. Table

Rates in 10⁻¹⁹ s⁻⁵. "same cell": pixels in both sets; "gained" / "lost": chosen / advected pixels more than one cell from the other set. Flag: S = split, M = merge. The last two columns are the surface-flux term of §5 at the two depth scales.

| pair | time (UTC) | label | px | prev label | advected px | same cell | gained | lost | flag | parcels | `2F` | stored `DG/Dt` | matched | $S$ (1 m) | $S$ ($h_{\rm bl}$) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 07-02 00:30 | 10 | 375 | — | — | — | — | — |  | — | — | — | — | — | — |
| 1 | 07-02 01:30 | 13 | 386 | 10 | 374 | 263 | 28 | 14 |  | 1432 | +4.40 | +4.75 | +3.05 | +9.85 | +1.19 |
| 2 | 07-02 02:30 | 11 | 391 | 13 | 386 | 294 | 20 | 12 |  | 1468 | +5.30 | +5.58 | +3.91 | +10.37 | +0.41 |
| 3 | 07-02 03:30 | 12 | 376 | 11 | 390 | 294 | 3 | 18 |  | 1403 | +6.54 | +6.00 | +4.58 | +10.52 | -0.08 |
| 4 | 07-02 04:30 | 10 | 380 | 12 | 377 | 287 | 8 | 5 |  | 1410 | +6.50 | +5.88 | +3.77 | +10.83 | -0.24 |
| 5 | 07-02 05:30 | 83 | 303 | 10 | 379 | 217 | 24 | 99 | S | 1157 | +1.99 | +1.35 | -0.74 | +11.40 | -0.31 |
| 6 | 07-02 06:30 | 76 | 315 | 83 | 294 | 201 | 31 | 24 |  | 1195 | +1.59 | +2.16 | -0.30 | +10.78 | -0.42 |
| 7 | 07-02 07:30 | 5 | 398 | 76 | 304 | 207 | 107 | 19 | M | 1483 | +3.47 | +4.39 | +2.95 | +10.14 | -0.66 |
| 8 | 07-02 08:30 | 7 | 403 | 5 | 375 | 270 | 33 | 27 |  | 1484 | +2.62 | +3.50 | +2.51 | +10.04 | -0.75 |
| 9 | 07-02 09:30 | 11 | 413 | 7 | 379 | 291 | 28 | 11 |  | 1523 | +2.21 | +3.64 | +0.34 | +9.86 | -0.74 |
| 10 | 07-02 10:30 | 11 | 241 | 11 | 399 | 165 | 4 | 176 | S | 884 | +1.83 | +3.11 | -0.72 | +14.21 | -0.62 |
| 11 | 07-02 11:30 | 80 | 305 | 11 | 236 | 109 | 158 | 93 | SM | 1149 | +3.18 | +3.76 | +2.22 | +9.84 | -0.11 |
| 12 | 07-02 12:30 | 8 | 402 | 80 | 302 | 237 | 115 | 16 | M | 1488 | +4.47 | +4.95 | +4.81 | +9.18 | -0.13 |
| 13 | 07-02 13:30 | 93 | 270 | 8 | 398 | 216 | 10 | 135 | S | 1011 | +5.01 | +5.57 | +4.15 | +9.68 | +0.25 |
| 14 | 07-02 14:30 | 140 | 176 | 93 | 267 | 130 | 8 | 110 | S | 657 | +3.06 | +3.69 | +2.73 | +3.87 | +0.12 |
| 15 | 07-02 15:30 | 135 | 169 | 140 | 174 | 133 | 4 | 8 |  | 634 | +2.58 | +3.38 | +2.46 | +3.87 | +0.35 |
| 16 | 07-02 16:30 | 89 | 289 | 135 | 166 | 134 | 121 | 3 | M | 1090 | +1.15 | +1.54 | -0.62 | +8.41 | +0.77 |
| 17 | 07-02 17:30 | 91 | 284 | 89 | 281 | 205 | 15 | 16 |  | 1056 | +1.86 | +2.10 | -0.86 | +8.17 | +0.71 |
| 18 | 07-02 18:30 | 88 | 140 | 91 | 279 | 102 | 8 | 151 | S | 532 | +4.20 | +2.40 | -1.28 | +13.69 | +1.56 |
| 19 | 07-02 19:30 | 87 | 130 | 88 | 136 | 92 | 7 | 14 |  | 490 | +8.89 | +9.20 | +2.81 | +13.71 | +1.40 |
| 20 | 07-02 20:30 | 93 | 122 | 87 | 127 | 91 | 1 | 10 |  | 457 | +8.83 | +5.84 | +1.45 | +14.01 | +1.05 |
| 21 | 07-02 21:30 | 87 | 253 | 93 | 121 | 82 | 130 | 2 | M | 931 | +3.79 | +1.80 | -1.39 | +8.15 | +0.12 |
| 22 | 07-02 22:30 | 87 | 253 | 87 | 246 | 190 | 2 | 5 |  | 940 | +3.19 | -0.78 | -1.95 | +8.68 | +0.47 |
| 23 | 07-02 23:30 | 87 | 265 | 87 | 250 | 165 | 24 | 9 |  | 995 | +5.50 | +4.10 | +2.52 | +8.92 | +0.81 |
| 24 | 07-03 00:30 | 86 | 289 | 87 | 264 | 190 | 29 | 5 | M | 1079 | +8.37 | +8.69 | +6.08 | +8.66 | +0.90 |
| 25 | 07-03 01:30 | 96 | 258 | 86 | 289 | 169 | 6 | 37 | S | 948 | +8.25 | +7.83 | +6.38 | +10.59 | +0.48 |
| 26 | 07-03 02:30 | 88 | 263 | 96 | 258 | 188 | 14 | 8 |  | 980 | +5.22 | +3.52 | +3.46 | +10.89 | -0.22 |
| 27 | 07-03 03:30 | 88 | 118 | 88 | 263 | 74 | 4 | 146 | S | 452 | +7.58 | +4.88 | +4.55 | +18.73 | -0.30 |
| 28 | 07-03 04:30 | 83 | 117 | 88 | 118 | 99 | 0 | 0 |  | 451 | +3.93 | +2.26 | +0.87 | +18.55 | -0.51 |
| 29 | 07-03 05:30 | 96 | 106 | 83 | 116 | 88 | 0 | 12 |  | 395 | +2.56 | +0.58 | -1.26 | +19.97 | -0.75 |
| 30 | 07-03 06:30 | 90 | 281 | 96 | 106 | 85 | 179 | 3 | M | 1039 | +2.55 | +2.22 | -0.32 | +10.32 | -1.02 |
| 31 | 07-03 07:30 | 95 | 301 | 90 | 280 | 204 | 24 | 9 |  | 1103 | +1.83 | +1.57 | -0.53 | +9.73 | -1.04 |
| 32 | 07-03 08:30 | 96 | 276 | 95 | 298 | 209 | 11 | 28 |  | 1032 | +0.34 | +0.87 | -1.62 | +9.82 | -1.08 |
| 33 | 07-03 09:30 | 98 | 299 | 96 | 268 | 235 | 18 | 0 |  | 1108 | -1.16 | -0.85 | -2.20 | +9.25 | -1.00 |
| 34 | 07-03 10:30 | 91 | 304 | 98 | 299 | 233 | 10 | 0 |  | 1124 | -1.37 | -0.05 | -1.88 | +8.80 | -0.83 |
| 35 (anchor) | 07-03 11:30 | 81 | 307 | 91 | 297 | 211 | 25 | 21 |  | 1119 | +0.71 | +1.52 | -0.19 | +8.61 | -0.67 |
| 36 | 07-03 12:30 | 83 | 307 | 81 | 305 | 249 | 11 | 9 |  | 1143 | +3.46 | +4.20 | +2.38 | +8.47 | -0.49 |
| 37 | 07-03 13:30 | 90 | 295 | 83 | 305 | 221 | 26 | 34 |  | 1106 | +2.92 | +3.60 | +2.09 | +9.08 | -0.32 |
| 38 | 07-03 14:30 | 133 | 178 | 90 | 294 | 139 | 10 | 123 | S | 663 | +3.84 | +4.11 | +3.49 | +3.61 | -0.26 |
| 39 | 07-03 15:30 | 88 | 299 | 133 | 177 | 152 | 119 | 0 | M | 1116 | +0.82 | +2.01 | +0.71 | +9.26 | +0.18 |
| 40 | 07-03 16:30 | 84 | 111 | 88 | 297 | 97 | 0 | 186 | S | 429 | -3.72 | -0.61 | -3.48 | +17.96 | +0.99 |
| 41 | 07-03 17:30 | 90 | 106 | 84 | 111 | 70 | 4 | 9 |  | 413 | -4.87 | -3.43 | -5.25 | +19.09 | +1.19 |
| 42 | 07-03 18:30 | 86 | 107 | 90 | 106 | 82 | 5 | 4 |  | 419 | -1.89 | -0.18 | -4.18 | +18.85 | +1.24 |
| 43 | 07-03 19:30 | 87 | 110 | 86 | 107 | 86 | 3 | 0 |  | 426 | +1.01 | +0.54 | -1.61 | +17.88 | +1.37 |
| 44 | 07-03 20:30 | 96 | 110 | 87 | 109 | 78 | 3 | 1 |  | 423 | +4.45 | +5.10 | +0.70 | +17.00 | +1.43 |
| 45 | 07-03 21:30 | 92 | 111 | 96 | 106 | 79 | 2 | 2 |  | 414 | +6.17 | +5.69 | +2.48 | +16.85 | +1.24 |
| 46 | 07-03 22:30 | 98 | 112 | 92 | 112 | 86 | 2 | 1 |  | 430 | +6.62 | +4.85 | +3.18 | +17.56 | +1.22 |
| 47 | 07-03 23:30 | 94 | 108 | 98 | 110 | 87 | 0 | 4 |  | 414 | +10.99 | +13.11 | +10.38 | +18.15 | +1.36 |
| 48 | 07-04 00:30 | 92 | 104 | 94 | 108 | 64 | 2 | 3 |  | 413 | +15.98 | +16.69 | +14.41 | +19.68 | +1.26 |
| 49 | 07-04 01:30 | 79 | 115 | 92 | 104 | 62 | 9 | 2 |  | 447 | +14.30 | +14.51 | +12.31 | +19.28 | +0.59 |
| 50 | 07-04 02:30 | 79 | 115 | 79 | 115 | 91 | 0 | 0 |  | 448 | +10.80 | +7.63 | +7.22 | +19.56 | -0.19 |
| 51 | 07-04 03:30 | 78 | 120 | 79 | 115 | 74 | 2 | 0 |  | 461 | +8.08 | +4.86 | +4.10 | +19.21 | -0.58 |
| 52 | 07-04 04:30 | 79 | 116 | 78 | 120 | 92 | 2 | 6 |  | 445 | +4.06 | +0.95 | +0.43 | +18.25 | -0.68 |
| 53 | 07-04 05:30 | 82 | 116 | 79 | 116 | 95 | 1 | 0 |  | 450 | +2.16 | +2.33 | -0.63 | +17.78 | -0.75 |
| 54 | 07-04 06:30 | 81 | 118 | 82 | 116 | 85 | 1 | 0 |  | 466 | -1.27 | -2.04 | -3.20 | +17.71 | -0.80 |
| 55 | 07-04 07:30 | 89 | 122 | 81 | 118 | 84 | 18 | 15 |  | 487 | -3.01 | -2.68 | -4.91 | +17.48 | -0.90 |
| 56 | 07-04 08:30 | 95 | 120 | 89 | 122 | 88 | 0 | 0 |  | 476 | -6.30 | -3.84 | -7.04 | +17.29 | -0.96 |
| 57 | 07-04 09:30 | 88 | 111 | 95 | 120 | 71 | 8 | 17 |  | 435 | -13.82 | -12.12 | -12.84 | +16.31 | -1.02 |
| 58 | 07-04 10:30 | 88 | 115 | 88 | 111 | 85 | 3 | 0 |  | 447 | -11.60 | -8.90 | -11.14 | +15.62 | -0.78 |
| 59 | 07-04 11:30 | 87 | 123 | 88 | 114 | 87 | 8 | 4 |  | 461 | -6.52 | -3.67 | -5.98 | +14.93 | -0.51 |
| 60 | 07-04 12:30 | 83 | 126 | 87 | 123 | 94 | 7 | 1 |  | 487 | -1.77 | +0.51 | -1.83 | +14.64 | -0.29 |
| 61 | 07-04 13:30 | 85 | 126 | 83 | 126 | 91 | 5 | 4 |  | 490 | +0.74 | +2.23 | -0.14 | +15.12 | -0.08 |
| 62 | 07-04 14:30 | 81 | 125 | 85 | 126 | 103 | 0 | 0 |  | 491 | +1.31 | +2.66 | +1.38 | +15.43 | +0.17 |
| 63 | 07-04 15:30 | 85 | 125 | 81 | 125 | 95 | 0 | 1 |  | 492 | +4.48 | +7.58 | +4.79 | +15.80 | +0.67 |
| 64 | 07-04 16:30 | 83 | 117 | 85 | 125 | 95 | 0 | 8 |  | 452 | +3.46 | +6.78 | +4.39 | +15.87 | +1.13 |
| 65 | 07-04 17:30 | 85 | 124 | 83 | 117 | 91 | 2 | 0 |  | 476 | -0.44 | +0.83 | +0.05 | +15.27 | +1.38 |
| 66 | 07-04 18:30 | 82 | 136 | 85 | 124 | 101 | 12 | 0 |  | 528 | +0.57 | +0.89 | -0.11 | +14.34 | +1.44 |
| 67 | 07-04 19:30 | 82 | 302 | 82 | 136 | 99 | 171 | 8 | M | 1141 | +2.32 | +3.70 | +0.70 | +7.92 | +0.14 |
| 68 | 07-04 20:30 | 87 | 282 | 82 | 287 | 215 | 4 | 16 |  | 1077 | +3.37 | +4.14 | +1.77 | +7.91 | +0.29 |
| 69 | 07-04 21:30 | 82 | 274 | 87 | 270 | 201 | 6 | 9 |  | 1048 | +4.64 | +5.00 | +2.44 | +8.59 | +0.74 |
| 70 | 07-04 22:30 | 78 | 283 | 82 | 262 | 204 | 7 | 0 |  | — | — | — | — | — | — |
