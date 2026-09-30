# Frontogenesis execution prompt 4 — M3: Field-level budget

**Milestone:** M3 (`frontogenesis_coding.md` §6). **THIS IS A HARD GATE.**
**Prerequisites:** M1 passed (all four gates, including the discrete-null slope), M2 source A
complete. Source B may arrive during this milestone.
**Goal:** close the surface buoyancy-gradient budget, per pixel, with no front finding involved.
This is the rigorous core of the study.

> **The exit criterion is budget closure, not a slope.** No efficiency number is quoted before
> the terms balance. If closure fails, that is the result (planning §12) — write it up as a
> methodological finding rather than reporting a slope anyway.

---

## The budget

```
D_h/Dt ( ½ G )  =  F  +  grad b . grad B          where G = |grad_h b|^2
```

Discretely, with everything through the M1 operators at one filter scale `L`:

```
measured  -  2F  -  subfilter  -  vertical  -  surface_flux   ~   numerical + interior KPP
```

`measured` is the semi-Lagrangian `DGDt`. **Compare `2F`, never `F`** — `F = ½ DG/Dt`.

## Modules to write

`vertical.py` (its physics functions — `load_chunk_levels` was M2's), `budget.py`, `stats.py`.
Signatures: `frontogenesis_coding.md` §4.6-§4.8. **`coarsegrain.py` belongs to M1** and should
already exist; if it does not, M1 was not finished.

### `vertical.py` — the terms that make this a real budget

- `b_z` from chunk `k = 0..2` Theta/Salt (`drF[0] = 1.0 m`, `Z[0] = -0.5 m`).
- `vertical_term` = `grad_h b . grad_h[ -W_k1 (b_k1 - b)/drF ]`, from the chunk **`W(k_l=1)`**
  (corrected 2026-09-28, M0 task 3; coding §4.6). Note the subtlety: the **continuum** tilting
  term vanishes at the surface because the velocity *relative to the free surface* vanishes
  there (`w = D eta/Dt`, not `w = 0` — the OSN `W(k_l=0)` is `dEta/dt`, ~5e-5 m s^-1, tidal),
  but `k=0` is a finite 1 m cell whose budget carries the flux through its *base*, where the
  model's `W(k_l=1)` combines that `dEta/dt` part with the convergence part `+drF delta`. Use
  the model's `W`, not `-drF delta` (wrong sign, missing part), and difference the tendency
  before taking `grad_h` so that `-w grad(b_z) . grad b` is not dropped; report the factorised
  `-b_z (w_x b_x + w_y b_y)` as a diagnostic. Expected ~30% of `F` by day and ~0 at night
  (M0's surface-only bracket: 0.4-14% rms across `b_z = 1e-5..4e-4`, order-one pointwise at
  fronts) — that diurnal signature, and its tidal phase, are themselves checks that the term
  is right.
- `surface_flux_term` = `grad b . grad B_sfc`. Convert heat and freshwater flux to a top-cell
  buoyancy tendency using thermal and haline expansion coefficients from **the same JMD95 EOS**
  as `operators.buoyancy`. **Treat `oceQsw` separately from `oceQnet`:** a large fraction of
  shortwave is absorbed *inside* the top cell where our `b` lives, and net flux alone blurs
  exactly the noon-peaking term Figure 6 is about.

### `budget.py`

`compute_budget(raw_ds, grid_ds, grid, masks, L_cells, dt=3600.0, chunk_ds=None)` returning
`measured, two_F, subfilter, vertical, surface_flux, residual`, plus `closure_report`.

**If `chunk_ds` is None, `closure_report` must say loudly that the vertical and surface-flux
terms are absent and the residual is a catch-all.** Silent degradation here would let a
two-term comparison masquerade as a closed budget.

### `stats.py` — and please read planning §11 before writing it

Several effects push the slope below 1 with **no physics involved**, and they all have to be
controlled before any efficiency is claimed:

- **Attenuation:** `2F` carries error, so OLS is biased by `var(X)/(var(X)+var(eta))`.
- **Selection on the outcome:** selecting front pixels on `G(t+dt)` biases the difference
  positive by regression to the mean; on `G(t)`, negative. *Fix:* select at the **midpoint
  time** on a field independent of both endpoints, and evaluate `F` there too.
- **Correlated errors:** evaluating `F` at `t+dt` makes both axes share `grad b(t+dt)`. The
  midpoint evaluation fixes this too.
- **Effective sample size:** 72 h on one tile is *tens of independent patches*, not 10^7
  independent pixels. **Bootstrap over blocks, never over pixels** — and at *this* milestone the
  block is a **contiguous spatial region plus hour**, because front objects do not exist yet.
  Feature-level bootstrap is M4's.

**How front pixels are selected here — without front finding.** Take `G` above a stated
percentile at the **midpoint time**, inside `mask_analysis`. That is independent of both endpoints
(which is what the selection-bias argument demands) and needs no labelling, no thinning and no
despurring. Keeping `tile_find` out of this milestone means the budget is not entangled with
front-detection choices; M4 is where labelled objects appear.

Report OLS, total-least-squares and bisector together; prefer **binned conditional means**
`E[Y|X]` computed **separately for `X > 0` and `X < 0`** (diabatic damping is asymmetric, and a
single slope averages over the asymmetry that carries the physics); quote ratio estimators
`sum(Y)/sum(X)` alongside; and quote every slope **relative to the M1 discrete-null baseline**,
never against 1.

**The baseline and its bands (decided 2026-09-30, M1-Q2 / M1-Q4).** The baseline is V3's
real-velocity slope **0.981 [0.970, 0.994]** (`form='discrete'`, hour-0 `b`, `mask_analysis`).
The systematic band comes from V3b (task 6b) — the slope the pipeline returns when the tracer is
advected as the model advects it (flux-form, OS7MP-like truth): **0.954-1.003, i.e. 0.975 ±
0.025**. Report M3's slope against 0.981 and quote the V3b interval as the model-advection
systematic. **No upward correction** for the 0.80-0.85x Jacobian attenuation — V3b measured its
effect on the slope at −0.006 ± 0.025. Report the slope **per front width** and subtract the
advection-numerics shortfall (**−2% at 2 dx, −4% at 1.5 dx, −11% at 1 dx**, discrete form) before
attributing anything on the sharpest fronts to diffusion. The ratio estimator is ill-conditioned
when the front pool has both signs of `2F` (M1 task 6): quote it separately for `X > 0` and `X < 0`.

## Runs

- The filter sweep, `L_cells` in `{0, 2, 4, 8}`, with `tau` computed explicitly at each.
- Semi-Lagrangian vs Eulerian, as independent estimates.
- Statistics restricted to `>= 100 km` offshore, **and** stratified by distance offshore.
- Residual composited by hour of day, with `KPPhbl`.
- **Both forms of `F`** (decided 2026-09-30, M1-Q1): `operators.frontogenesis(form='discrete')`
  (the default) gives the **primary** slope; `form='chain'` is run alongside, and the
  discrete-vs-chain difference is carried as a **stated systematic** (on the real hour the
  discrete `F` is ~0.79x the chain `F` on front pixels; the V3 baselines are 0.981 and 0.791).
- **Interpolation order** (decided 2026-09-30, M1-Q6): `order = 3` is the default; report the
  slope at **order 5 as a sensitivity** (task 3 saw ~5% between them on real front pixels).
  Quote V4's bar as **0.28-1.0% of `G` per hour (order 3)**, with the front width.

## Figures (see planning §7)

1, 2, **2b**, 3, **3b**, 4, 5, 6, 7, 10. In particular:

- **Figure 2** draws the M1 discrete-null slope as an **explicit baseline line**, not a caption
  note — at **0.981 with its band [0.970, 0.994]** (decided 2026-09-30, M1-Q4), with V3b's
  systematic band 0.954-1.003 beside it.
- **Figure 2b** — residual against high-order derivatives of `b` (a `grad^4`-like diagnostic) and
  against `KPPhbl`. This is what separates implicit numerical diffusion from genuine air-sea
  forcing. Without it, Figure 2's slope has no physical interpretation.
- **Figure 3b** — the filter sweep *shown*: rows `{b, G, 2F, tau}` x columns `{L = 0, 2, 4, 8}`.

---

## Acceptance criteria

1. **Closure** to a stated tolerance, with `vertical` and `surface_flux` **measured** from the
   chunk store rather than assumed.
2. Semi-Lagrangian and Eulerian estimates agree within a stated tolerance.
3. The filter sweep is interpretable — `tau` explicit, and the budget closing at each `L`.
4. Every slope quoted against the M1 baseline (0.981 [0.970, 0.994], with the V3b systematic band
   0.954-1.003; both forms of `F`, M1-Q1; order 5 as a sensitivity, M1-Q6 — decided 2026-09-30),
   with feature-level bootstrap intervals.
5. Figures **1, 2, 2b, 3, 3b, 4, 5, 6, 7, 10** written. They need not yet be wired into the
   one-command regeneration harness — that consolidation is M5 — but the plots themselves are
   produced here, where their data lives.

## Do not

- Do not quote a "frontogenesis efficiency" before criterion 1 passes.
- Do not call a slope below 1 "diabatic damping" without Figure 2b. Implicit numerical
  diffusion (OS7MP; planning §2.3, corrected 2026-09-28) damps `G` at 0.1-0.5 f at `4 dx` —
  the same order as the strain — and at 3-10% of the kinematic rate at 10 km, so a slope of
  0.5-0.8 at the grid scale is fully explicable with zero air-sea flux; quote `kappa_num` at
  the scale of the feature, not as one number.
- Do not run `tile_find`, label, thin or despur here. Front *pixels* are a percentile of `G` at
  the midpoint time (above); front *objects* are M4.

## Log

The closure residual and tolerance; the slope estimators with intervals, against the baseline;
whether semi-Lagrangian and Eulerian agreed; the relative sizes of all five budget terms; and —
plainly — whether closure passed. **A failure here is a publishable methodological result, not
an embarrassment.**
