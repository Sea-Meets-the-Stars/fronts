# Frontogenesis intuition

**Authors:** J. X. Prochaska, with Claude (Fable 5.2)  
**Status:** Working note, begun 23 November 2026. Section 1 (`2F`) is written; later sections will follow from `claude_prompts/frontogenesis_intuition.md`.  
**Circulation:** For the collaboration; not for citation.

---

## Purpose

This note builds intuition for frontogenesis, from the primary equations to their use on tracked fronts. It is a companion to the report (`frontogenesis_report.md`) and to the slides, and it introduces no new result. Where it quotes numbers, they are for one front at one hour: label 83 (the tracked front "label 81") at pair 36, 2012-07-03 12:30 UTC, the case shown on the `2F` and `DG/Dt` slides.

## Notation

| Symbol | Meaning |
|---|---|
| $b$ | Surface buoyancy [m s⁻²] |
| $\nabla b = (b_x, b_y)$ | Horizontal buoyancy gradient; it points across the front, toward lighter water |
| $G = \lvert\nabla b\rvert^2$ | Front strength [s⁻⁴] |
| $\mathbf{u} = (u, v)$ | Horizontal velocity; $D/Dt = \partial_t + \mathbf{u}\cdot\nabla$ follows the water horizontally |
| $\nabla\mathbf{u}$ | Velocity-gradient tensor, $\begin{pmatrix} u_x & u_y \\ v_x & v_y \end{pmatrix}$ |
| $\delta = u_x + v_y$ | Divergence ($\delta < 0$ is convergence) |
| $\sigma_n = u_x - v_y$, $\sigma_s = v_x + u_y$ | Normal and shear strain; $\lvert\sigma\rvert = \sqrt{\sigma_n^2 + \sigma_s^2}$ |
| $\zeta = v_x - u_y$ | Vorticity |
| $\alpha$ | Direction of $\nabla b$, anticlockwise from east |
| $\psi$ | Direction of the **extensional** axis of the strain; the **compressional** axis is at $\psi + 90°$ |
| $\theta$ | Angle between $\nabla b$ and the compressional axis, between 0° and 90° |
| $F$ | Frontogenesis function [s⁻⁵]; the comparison is always $2F$ against $DG/Dt$ |

---

## 1. `2F`: the kinematic rate of change of a front

### 1.1 The result

For water that keeps its buoyancy and moves horizontally, the strength of a front changes at the rate

$$\frac{DG}{Dt} = 2F, \qquad F = -(\nabla b)^{\mathsf T}(\nabla\mathbf{u})(\nabla b) = -\left(u_x\,b_x^2 + (u_y + v_x)\,b_x b_y + v_y\,b_y^2\right). \tag{1}$$

It splits exactly into a divergence term and a strain term:

$$\boxed{\,2F = -\delta\,G \;+\; \lvert\sigma\rvert\,G\cos 2\theta\,} \qquad\text{or, as a growth rate,}\qquad \frac{1}{G}\frac{DG}{Dt} = -\delta + \lvert\sigma\rvert\cos 2\theta. \tag{2}$$

In words:

- **Convergence sharpens a front whatever its orientation.** The rate is $-\delta$.
- **Strain sharpens a front that it compresses across, and weakens one that it stretches across.** The rate is $\lvert\sigma\rvert\cos 2\theta$: it runs from $+\lvert\sigma\rvert$ ($\theta = 0°$) through zero ($\theta = 45°$) to $-\lvert\sigma\rvert$ ($\theta = 90°$).
- **Rotation does not appear.** Vorticity turns $\nabla b$ without changing its length.

Both terms of the second form are rates in s⁻¹, so they can be compared directly with each other and with $f$, and the inverse of their sum is an e-folding time for $G$.

The rest of this section derives (1) and (2) and shows what each term looks like.

### 1.2 What `G` measures

Take two neighbouring isopycnals that differ by $\Delta b$ and lie a distance $L$ apart. Then $\lvert\nabla b\rvert \approx \Delta b / L$ and $G \approx \Delta b^2 / L^2$. Each parcel keeps its buoyancy, so $\Delta b$ is fixed and **the only way to change `G` following the water is to change the spacing `L`**:

$$\frac{1}{G}\frac{DG}{Dt} = -\frac{2}{L}\frac{DL}{Dt}. \tag{3}$$

Frontogenesis is the flow bringing isopycnals closer together. The equations below say which part of the flow does it.

### 1.3 The exact equation

Start from conservation of buoyancy, $Db/Dt = 0$, and take its gradient. Differentiating $b_t + u\,b_x + v\,b_y = 0$ with respect to $x$ and to $y$ gives

$$\frac{D b_x}{Dt} = -(u_x b_x + v_x b_y), \qquad \frac{D b_y}{Dt} = -(u_y b_x + v_y b_y), \qquad\text{that is}\qquad \frac{D\,\nabla b}{Dt} = -(\nabla\mathbf{u})^{\mathsf T}\,\nabla b. \tag{4}$$

Only the *differences* of velocity across a parcel appear. A uniform current carries a front along without changing it, which is why the slides draw the velocity relative to the parcel. Then

$$\frac{DG}{Dt} = 2\,\nabla b\cdot\frac{D\,\nabla b}{Dt} = -2\,(\nabla b)^{\mathsf T}(\nabla\mathbf{u})(\nabla b) = 2F,$$

which is equation (1).

A more compact reading of the same expression: write $\nabla b = \sqrt{G}\,\mathbf{n}$, with $\mathbf{n}$ the unit vector across the front. Then

$$\frac{1}{G}\frac{DG}{Dt} = -2\,\mathbf{n}^{\mathsf T}(\nabla\mathbf{u})\,\mathbf{n} = -2\,\frac{\partial u_n}{\partial n}, \tag{5}$$

where $u_n$ is the cross-front velocity and $n$ the cross-front coordinate. **The front sharpens at twice the rate at which the cross-front flow converges on it.** Comparing with (3), $\partial u_n/\partial n$ is the rate $(1/L)\,DL/Dt$ at which the isopycnal spacing changes. Equation (2) is this one number, $\partial u_n/\partial n$, written in terms of quantities that do not depend on the front's orientation ($\delta$, $\lvert\sigma\rvert$) and one angle that does ($\theta$).

What (1) leaves out: vertical advection, surface fluxes and mixing, which all change $b$ following the horizontal flow. In the report these are the further terms of the budget $DG/Dt = 2F + V + S + R$.

### 1.4 Splitting the flow around a parcel into three motions

Any velocity-gradient tensor is the sum of three parts:

$$\nabla\mathbf{u} = \underbrace{\frac{\delta}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1\end{pmatrix}}_{\text{convergence / divergence}} + \underbrace{\frac{1}{2}\begin{pmatrix} \sigma_n & \sigma_s \\ \sigma_s & -\sigma_n\end{pmatrix}}_{\text{strain}} + \underbrace{\frac{\zeta}{2}\begin{pmatrix} 0 & -1 \\ 1 & 0\end{pmatrix}}_{\text{rotation}}. \tag{6}$$

Multiplying out returns $u_x, u_y, v_x, v_y$, so this is an identity. Each part is a simple motion of the water around the parcel (Figure 1):

- the first changes the area of a ring of water and keeps its shape;
- the second keeps the area and turns the ring into an ellipse;
- the third turns the ring and changes neither.

Put (6) into $F = -(\nabla b)^{\mathsf T}(\nabla\mathbf{u})(\nabla b)$, one part at a time:

- **Convergence:** $-\tfrac12\,\delta\,(b_x^2 + b_y^2) = -\tfrac12\,\delta\,G$, whatever the direction of $\nabla b$.
- **Strain:** $-\tfrac12\left[\sigma_n\,(b_x^2 - b_y^2) + 2\,\sigma_s\,b_x b_y\right]$, which depends on the direction of $\nabla b$.
- **Rotation:** $-\tfrac12\,\zeta\,(-b_x b_y + b_y b_x) = 0$.

So

$$2F = -\delta\,G - \left[\sigma_n\,(b_x^2 - b_y^2) + 2\,\sigma_s\,b_x b_y\right]. \tag{7}$$

![](figs/intuition_2F_terms.png)

**Figure 1.** One flow around a parcel (a) written as the sum of a convergence (b), a strain (c) and a rotation (d), as in equation (6). In each panel the flow has acted for a time $0.5/\lvert\sigma\rvert$ on a ring of water (grey dashed to brown) and on a set of evenly spaced isopycnals (grey dashed to blue). The red arrow is $\nabla b$ afterwards, drawn with its length proportional to $\lvert\nabla b\rvert$; before, it pointed east with the length shown in (d). The rate under each panel is the instantaneous $(1/G)\,DG/Dt$ of that flow. Example values: $\delta = -0.6\,\lvert\sigma\rvert$, $\zeta = 0.8\,\lvert\sigma\rvert$, $\theta = 25°$. A drawing, not data (`py/intuition.py`).

### 1.5 The angles

The strain term in (7) becomes a single cosine once two angles are introduced (Figure 2a).

1. **The direction of the gradient, $\alpha$.** Write $\nabla b = \sqrt{G}\,(\cos\alpha, \sin\alpha)$. Then $b_x^2 - b_y^2 = G\cos 2\alpha$ and $2\,b_x b_y = G\sin 2\alpha$.
2. **The direction of the strain, $\psi$.** Write $\sigma_n = \lvert\sigma\rvert\cos 2\psi$ and $\sigma_s = \lvert\sigma\rvert\sin 2\psi$. In axes turned by $\psi$ the strain matrix of (6) is $\tfrac12\lvert\sigma\rvert\begin{pmatrix} 1 & 0 \\ 0 & -1\end{pmatrix}$: the water is stretched along $\psi$, the **extensional axis**, at the rate $\lvert\sigma\rvert/2$, and squeezed along $\psi + 90°$, the **compressional axis**, at the same rate.

The bracket in (7) is then $\lvert\sigma\rvert\,G\,(\cos 2\psi\cos 2\alpha + \sin 2\psi\sin 2\alpha) = \lvert\sigma\rvert\,G\cos 2(\alpha - \psi)$.

3. **The angle that matters, $\theta$.** Let $\theta$ be the angle between $\nabla b$ and the compressional axis. Then $\alpha - \psi = 90° \pm \theta$, so $\cos 2(\alpha - \psi) = -\cos 2\theta$, and (7) becomes equation (2):

$$2F = -\delta\,G + \lvert\sigma\rvert\,G\cos 2\theta.$$

Three remarks on the angles.

- **Why twice the angle.** A strain axis is a line, not an arrow: turning it by 180° gives the same strain. The same holds for the front, since reversing the sign of $\nabla b$ does not change $G$. Everything therefore depends on $2\alpha$ and $2\psi$, and $\theta$ can be folded into 0°–90° with nothing lost.
- **The sign convention.** $\theta$ is measured from the *compressional* axis, so the strain term carries a plus sign and $\theta = 0°$ is the most frontogenetic. Measured from the extensional axis it would carry a minus sign (planning §2.4 records this correction). Check: for $u = -ax$, $v = ay$ and a front $b(x)$, the compressional axis is $x$, $\theta = 0°$, $\lvert\sigma\rvert = 2a$, $\delta = 0$, and $2F = 2a\,b_x^2 > 0$.
- **The same thing in the strain's own axes.** With $b_c$ and $b_e$ the components of $\nabla b$ along the compressional and extensional axes, the strain term is $\lvert\sigma\rvert\,(b_c^2 - b_e^2)$. From (4), strain makes $b_c$ grow at the rate $\lvert\sigma\rvert/2$ and $b_e$ decay at the same rate. This is the form that Figure 2 (b)–(d) draws.

![](figs/intuition_2F_angles.png)

**Figure 2.** (a) The angles of the derivation. The isopycnals (blue) are perpendicular to $\nabla b$ (red), which lies at $\alpha$ from east. The strain stretches the water along the extensional axis at $\psi$ (outward arrows) and squeezes it along the compressional axis at $\psi + 90°$ (inward arrows); the faint arrows are the strain flow. $\theta$ is the angle between $\nabla b$ and the compressional axis. Here $\psi = 20°$, $\alpha = 80°$, $\theta = 30°$. (b)–(d) One strain field (compression left–right, extension up–down) acting for a time $0.5/\lvert\sigma\rvert$ on fronts at $\theta = 0°$, 45° and 90°, drawn as in Figure 1. (e) The factor $\cos 2\theta$, with the two angles measured at pair 36: the close-up parcel of the `2F` slide and the median over the band of label 83. A drawing, apart from those two angles (`py/intuition.py`).

### 1.6 What each term does to a front

**Convergence** (Figure 1b). In the flow $u = \tfrac12\delta\,x$, $v = \tfrac12\delta\,y$ every length changes at the rate $\delta/2$, in every direction. The isopycnal spacing does too, and by (3) $G$ grows at the rate $-\delta$. A front in convergent flow sharpens whichever way it lies. At the surface $\delta = -\partial w/\partial z$, so convergence comes with downwelling.

**Strain** (Figure 1c, Figure 2b–d). The area of the ring is kept: what is squeezed along one axis is stretched along the other. What happens to a front depends on which of the two it lies across.

| $\theta$ | The front lies… | Isopycnal spacing | $(1/G)\,DG/Dt$ |
|---|---|---|---|
| 0° | along the extensional axis | squeezed | $+\lvert\sigma\rvert$ |
| 45° | half-way | unchanged at first | 0 |
| 90° | along the compressional axis | pulled apart | $-\lvert\sigma\rvert$ |

The same strain field can therefore sharpen or weaken a front. **The orientation of the front sets the sign; the strength of the strain sets only the size.**

The case $\theta = 45°$ is not a steady state. The strain also turns the isopycnals toward the extensional axis (Figure 2c: the blue lines are steeper than the grey ones), so $\theta$ falls and the front starts to sharpen. In pure strain every front ends up along the extensional axis and sharpening. In Figure 2c, $G$ has already grown by 13% after a time $0.5/\lvert\sigma\rvert$, although its initial rate is zero.

**Rotation** (Figure 1d). Solid rotation turns the isopycnals and leaves their spacing alone, so it does not appear in $2F$. It still matters, because it changes $\theta$. Strain turns $\nabla b$ toward the compressional axis ($\theta \to 0$) and rotation turns it away; the angle at which a front sits is the outcome of the two. Vorticity therefore acts on $2F$ through $\cos 2\theta$.

### 1.7 The two terms on a tracked front

The slide figures show the same quantities on data, at pair 36 on the band of label 83 (1,143 cells: the 307 front pixels widened by one cell).

![](../slides/2F_method.png)

**Figure 3.** `slides/2F_method.png`, reused. (b) is Figure 2a on data: one parcel with its $b$ contours, the direction of $\nabla b$, the compressional axis and the velocity relative to the parcel. (e) is $\cos 2\theta$ on every cell of the band. (c) and (f) are the stored `2F`.

![](../slides/2F_split.png)

**Figure 4.** `slides/2F_split.png`, reused: the convergence term $-\delta G$, the strain term $\lvert\sigma\rvert G\cos 2\theta$ and their sum, the continuum `2F`, on every cell of the band (top) and in the close-up window (bottom).

As growth rates (equation 2), with $f = 6.82 \times 10^{-5}$ s⁻¹ at 27.9°N:

| | $-\delta$ | $\lvert\sigma\rvert\cos 2\theta$ | total | e-folding time of $G$ |
|---|---|---|---|---|
| close-up parcel (j 126, i 653) [10⁻⁵ s⁻¹] | +2.18 | +0.87 | +3.05 (0.45 $f$) | 9.1 h |
| band, weighted by $G$ [10⁻⁵ s⁻¹] | +0.22 | +0.51 | +0.73 (0.11 $f$) | 38 h |
| band, contribution to the mean `2F` [10⁻¹⁹ s⁻⁵] | +1.29 | +2.92 | +4.21 | |

The parcel has $\delta = -2.18 \times 10^{-5}$ s⁻¹, $\lvert\sigma\rvert = 1.48 \times 10^{-5}$ s⁻¹ and $\theta = 27°$ ($\cos 2\theta = 0.59$). The band rates are weighted by $G$ because that is the weighting under which they add up to the band mean of `2F`: $\langle 2F\rangle = \langle G\rangle \times$ (the $G$-weighted rate), with $\langle G\rangle = 5.74 \times 10^{-14}$ s⁻⁴.

Three things to take from the table.

- **Strain carries 69% of the band's `2F`, although the parcel shown is led by convergence.** The share is the same on the front pixels (+1.93 and +4.25) and on the cells beside them (+1.05 and +2.43).
- **The divergence is not small; it cancels.** The $G$-weighted $\lvert\delta\rvert$ is $1.06 \times 10^{-5}$ s⁻¹, 60% of the $G$-weighted $\lvert\sigma\rvert$ ($1.77 \times 10^{-5}$ s⁻¹). But only 59% of the cells converge, so the net rate is about a fifth of the typical $\lvert\delta\rvert$.
- **The strain has a steady sign and is used at about 30% of its strength.** $\theta$ is below 45° on 72% of the cells, with a median of 29°. The $G$-weighted mean of $\cos 2\theta$ is 0.30 (the plain mean is 0.33). A front lying exactly along the extensional axis would have 1.

A caution on the first two points: they describe one front at one hour.

### 1.8 Two forms of `2F` in the code

Equation (2) is exact for the continuum expression (1), which the code calls the chain form (`operators.frontogenesis(form='chain')`, stored as `two_F_chain`). Two details matter when the numbers are compared with others in the project.

- **The stored `2F` is a different, discrete form** (`two_F`). It is built to match what the semi-Lagrangian $DG/Dt$ tends to on this grid, and on narrow fronts it is smaller than the continuum form. On this band it averages +3.46 against +4.21, a ratio of 0.82, with a cell-by-cell correlation of 0.95. It has no unique split into convergence and strain. So the split is stated for the continuum form, and the comparison with $DG/Dt$ is made with the discrete form.
- **The split closes only with the strain set taken from the same velocity gradient.** `operators.strain_from_jacobian` gives $\delta$, $\sigma_n$, $\sigma_s$ from the Jacobian that (1) uses, and with it the two terms sum to the continuum `2F` to $1 \times 10^{-15}$ of its r.m.s. The flux-form $\delta$ and $\sigma$ of the derived store are differenced on a different stencil; with them the terms sum to 4.29 instead of 4.21.

`operators.strain_alignment` computes $\theta$ exactly as in §1.5: $\cos 2\theta = -\left[\sigma_n\,(b_x^2 - b_y^2) + 2\,\sigma_s\,b_x b_y\right] / (\lvert\sigma\rvert\,G)$.

---

## Sources

- Equations and conventions: `frontogenesis_planning.md` §2.4; `py/operators.py` (`frontogenesis`, `strain_from_jacobian`, `strain_alignment`); `tracked/matched_parcels_track_81.md` §1.1; `claude_prompts/frontogenesis_slides.md`, Q&A/2F, Round 3.
- Figures 1 and 2: `py/intuition.py --what schematics`. Every rate written on them is computed with `strain_from_jacobian` and `strain_alignment` and asserted against $-2\,\mathbf{n}^{\mathsf T}(\nabla\mathbf{u})\,\mathbf{n}$.
- Figures 3 and 4: `py/slides_2F.py`.
- Numbers of §1.7 and §1.8: `data/intuition_2F.json` (`py/intuition.py --what numbers`), asserted against `data/slides_2F_36.json`. The 4.29 is from the slides log (2F prompt 1: 1.30 + 2.99).
