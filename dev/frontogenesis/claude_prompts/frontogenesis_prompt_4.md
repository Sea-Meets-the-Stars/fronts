# Frontogenesis execution prompt 4 — M3: Field-level budget

**Milestone:** M3 (`frontogenesis_coding.md` §6). **THIS IS A HARD GATE.**
**Prerequisites:** M1 passed (all four gates, including the discrete-null slope), M2 source A
complete. Source B may arrive during this milestone. *(corrected 2026-10-07, M2 task 8: both
halves of M2 closed 2026-10-04 — source B is complete, 72/72 chunk hours on disk
(`data/tile330_chunk_20120702T00_72h.zarr`), so the "Do not block" path of prompt 3 is not
needed; `compute_budget` still has to run without `chunk_ds` and say so loudly, because that is
how its tests work offline.)*
**Goal:** close the surface buoyancy-gradient budget, per pixel, with no front finding involved.
This is the rigorous core of the study.

**Status 2026-10-07 — task 1 done** (`py/inputs.py` + `tests/test_inputs.py`, 14 tests; suite
**141 passed + 3 xfails**, 2 deselected, 180 s; the V3 / M2 numbers reproduce on the real stores —
`n_valid` 262,925, `n_front` 26,293, `drF` [1.0, 1.14, 1.30], the `k = 0` identity on all 72 hours;
log entry "Execution prompt 4, task 1"). **Task 2 done 2026-10-07** (the reader moved to
`py/chunk_store.py` with `vertical.load_chunk_levels` re-exported, M3-Q7; `py/vertical.py`
physics — `b_z`, `vertical_term` (+ `vertical_term_factorised`), `surface_flux_term`, all in F
units; `tests/test_vertical.py` 19 tests; suite **160 passed + 3 xfails**, 2 deselected, 257 s;
namelist verified: `f_sw = 0.521` (Jerlov IA, M3-Q10), `c_p = 3994`, `rhoConst = 1027.5`
(M3-Q12), `convertFW2Salt = −1`; the tendency's sign corrected, M3-Q11; smoke at `L = 0`:
vertical term **2.7-6.5 %** of `2F` in rms (diurnal shape right, the warm layer is 0.01 K not
0.1-0.3 K), surface-flux term **0.65-0.94** of `2F` at every hour, dominated by the non-solar
flux gradient (a damping of `G` at ~7e-6 s^-1), not peaking at 13 LST — findings, not tuned;
log entry "Execution prompt 4, task 2"; **M3-Q10..Q12 await JXP**, all implemented as
recommended). **Tasks 3-9 not started.** The task sequence below (tasks 1-9), the M3 carry-forward
cross-check table and the M3-Q1..Q9 questions were written by M2 task 8 (prompts 3, task 8; log
entry "Execution prompt 3, task 8"). **M3-Q6 is needed before task 1, M3-Q7 before task 2,
M3-Q3 and M3-Q4 before task 3, M3-Q1 / Q2 / Q5 / Q9 before task 6 (they are pre-declared there),
M3-Q8 before task 8.** Inputs on disk: `data/tile330_grid.zarr` (M0),
`data/tile330_masks.nc` (M1), `data/tile330_raw_20120702T00_72h.zarr` and
`data/tile330_chunk_20120702T00_72h.zarr` (M2). Suite at the start of M3: **127 passed + 3 strict
xfails** (2 network tests deselected); it must still pass at the end of every task.

**Q&A answered 2026-10-07.** All nine M3-Q questions are answered by JXP (`## Q&A` below); every
answer accepts the recommendation, and each decision is written into the task that uses it,
marked "(decided 2026-10-07, M3-Qn)":
- **M3-Q1** — closure tolerance, on `front & valid`, per `L`: `rms(residual)/rms(measured) <= 0.5`
  (explained fraction ≥ 0.75) **and** the residual's OLS slope on `2F` within ±0.10 (task 6).
- **M3-Q2** — semi-Lagrangian / Eulerian: OLS slope of `DGDt_euler` on `DGDt_semilag` within
  0.85-1.15 **and** corr ≥ 0.90 on front pixels at `L >= 2`; `L = 0` reported and interpreted,
  not gated (task 6).
- **M3-Q3** — `L_cells = {0, 2, 4, 8}` is the contract; `L = 1` as an extra column for Figure 3
  only if the pilot shows time to spare (task 4).
- **M3-Q4** — front pixels `G_mid >= p90` on `mask_analysis & finite` (primary); p80 and p95 as
  sensitivities; the trimmed estimator beside the OLS (tasks 3, 6).
- **M3-Q5** — `front_width = 2 sqrt(G/|lap G|)` stored in the derived product, binned
  `{<= 1, 1-1.5, 1.5-2, 2-3, 3-4, > 4}` dx, validated on `synthetic.py` tanh fronts in
  `test_budget.py`; the filter sweep as the cross-check; object widths noted for M4 (tasks 3, 6).
- **M3-Q6** — merge by **rename** (`Theta_k`, `Salt_k`, `W_k`; 16 vars) with accessors (task 1).
- **M3-Q7** — split `vertical.py` only: the reader goes to `chunk_store.py` with a re-export;
  `osn_tiles.py` / `validate.py` left whole (task 2).
- **M3-Q8** — yes: marked notes in the planning doc for the stale §4 / §2.3 / §5.3 numbers (task 8).
- **M3-Q9** — criterion 1 is judged at `L >= 2`; `L = 0` is reported and interpreted (its explicit
  subfilter term ≡ 0) (task 6).
**M3 is ready to start at task 1.**

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
percentile at the **midpoint time**, inside `mask_analysis` *(the percentile is **p90** on
`mask_analysis & finite`, with p80 / p95 as sensitivities — decided 2026-10-07, M3-Q4)*. That is independent of both endpoints
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

- The filter sweep, `L_cells` in `{0, 2, 4, 8}`, with `tau` computed explicitly at each
  *(confirmed as the contract 2026-10-07, M3-Q3; `L = 1` only as an optional extra Figure-3 column)*.
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

## Tasks

*(added 2026-10-07, M2 task 8)*

Run **one task per session**, in order, as in M0-M2, with one log entry per task (see **Log**).
Code goes in `dev/frontogenesis/py/`, tests in `dev/frontogenesis/py/tests/` (offline; those that
read the M0/M1/M2 stores are `@pytest.mark.needs_grid`; coding §5). Data goes in
`dev/frontogenesis/data/` (git-ignored), figures in `dev/frontogenesis/figs/` (`figs/.gitignore`
un-ignores `*.png`). The env is `~/miniforge3/envs/frontogenesis/bin/python`, and the M2 suite
(**127 passed + 3 strict xfails**) must still pass at the end of every task. Call
`operators.frontogenesis` and `semilag` with their **defaults** (`form='discrete'`, `order=3`,
`vel_order=3`, `n_iter=3`); anything in the older docs pinned to "`F = frontogenesis_tendency`"
means `form='chain'` (M1 task 6, flag 10).

**Why this shape (nine tasks).** The dependency chain is inputs → `vertical.py` → `budget.py` →
the sweep → closure → figures → audit → slides; `stats.py` needs only numpy and is placed *after
the sweep is launched* so it is written while the one long job runs. A separate input layer
(task 1) exists because M2 task 6 found that a plain `xr.merge` of the two stores fails, and
because the merge, the mask rule at `L = 8` and the hour-pair/midpoint logic are shared by
`budget.py`, the sweep, the closure and every figure — one tested module keeps `budget.py` under
coding §1.3's cap. The hour-0 smoke sits in task 3 so that a sign error costs minutes, not a
detached multi-hour run. Closure is its own task because it is the gate: tolerances are
**pre-declared** (M3-Q1, M3-Q2, M3-Q9) before any number is seen, as V3 did. Figures follow
closure because Figure 2 draws its slopes and Figure 6 its composites.

**Rules for long jobs (anti-stall).** From M1 (four agents stalled on foreground jobs) and M2
(one 28-min idle-sleep stall):
- Prefix every interactive python/pytest call with **`timeout 300`**.
- Never run the sweep (task 4) in the foreground. Launch it detached under **`caffeinate`**:
  `nohup caffeinate -i -s ~/miniforge3/envs/frontogenesis/bin/python m3_run.py > ../data/m3_run.nohup 2>&1 &`,
  with a progress log and a done-file, and poll the log.
- The sweep is **resumable per hour pair** (reuse `zarr_series.append_hour` / `present_times` /
  `repair_trailing`, as `pull_series` and `load_chunk_levels` do); an interrupted run is
  relaunched, not debugged. Expensive summaries are cached as JSON (`m2_qa.py`'s pattern).
- Scratch files (checksums, renders, object caches) go in the **session scratchpad**, never in
  `data/` or the repo; only the JSON caches a script itself owns live in `data/`.
- Start the log entry early ("entry started early; extended below") so a cut-off session leaves
  a record, and update the Status here at the end of each task.

**Pitfalls that apply to every task** (coding §8): compare **`2F`**, never `F`; interpolate `b`
not `G`, order ≥ 3; `F` evaluated and fronts selected at the **midpoint**; the same filter on
`b`, `U` **and** `V`; JMD95 (`calculate_fields.buoyancy_of_field`), never the legacy one; output
dims asserted after **every** dbof call; `expand_dims('face')` / `open_grid(with_face=True)`
before any dbof operator; face-10 orientation (`i`/`U`/`dxC` meridional, `i` southward; `j`/`V`/`dyC`
zonal); NaN-aware reductions; `float64` in the compute path, `float32` on disk; every slope
**against the M1 baseline, never against 1**; bootstrap over blocks, never pixels.

### 1. The merged input layer — `inputs.py`

- Write `py/inputs.py` (new, small; functions only). It is the **only** place M3 opens the stores.
  - `open_inputs(raw=vertical.OSN_RAW_ZARR, chunk=vertical.CHUNK_ZARR, grid=..., masks=...)
    -> (ds, grid_ds, grid, masks_ds)`. Opens `tile330_raw_20120702T00_72h.zarr` (§3.2) and
    `tile330_chunk_20120702T00_72h.zarr` (§3.3) and merges them by **rename** (decided
    2026-10-07, **M3-Q6**): `chunk.rename({'Theta': 'Theta_k', 'Salt': 'Salt_k', 'W': 'W_k'})`
    → 16 vars on dims `time, j, i, i_g, j_g, k, k_l`, keeping `k = 2` for the second-order `b_z`
    sensitivity and the `k = 0` bit-identity as an assertable invariant. (The **subset**
    `xr.merge([osn, chunk[['oceQnet', 'oceQsw', 'oceFWflx', 'drF']], chunk.W.isel(k_l=1,
    drop=True).rename('W_k1')])` → 14 vars was also verified by M2 task 6 but is **not** used.)
    **A plain
    `xr.merge([osn, chunk])` fails** (`MergeError` on `Salt`: 2-D vs 3-D under the same names).
    `grid_ds` via `osn_tiles.open_grid(with_face=True)`, `grid` via `osn_tiles.build_xgcm`,
    masks via `masking.open_masks`.
  - **Assert before merging**, every call: `time` (72, steps 3600 s), `niter`, `j`, `i`, `XC`,
    `YC`, scalar `face = 10` equal in both stores (M2 task 6 verified them equal); for the hours
    opened, chunk `Theta_k(k=0)`, `Salt_k(k=0)`, `W_k(k_l=0)` **bit-identical (NaN-aware)** to OSN
    `Theta`, `Salt`, `W` — the two sources are the same model output (M2 tasks 4-6), so this is a
    cheap, strong invariant, and it is what planning §4's "genuine cross-check" reduces to;
    the three flux fields carry `sign_convention` = positive downward and stored `oceQsw >= 0`
    everywhere in the hours opened — **refuse otherwise**, so a rewritten store can never be
    double-negated (M2 task 6, item 2).
  - Accessors, so the physics modules never index the store themselves: `W_k1(ds)` =
    `ds.W_k.isel(k_l=1)`, **the cell-base velocity** `vertical_term` takes (source `k_p1 = 1`,
    continuity to 7e-12 m/s) — never `k_l = 0`, which is `dEta/dt` (corr 0.998 with the centred
    `Eta` difference; 0.903 and 0.690 at `k_l` 1, 2); `drF(ds)` = `[1.0, 1.14, 1.30]` m and
    `Z(ds)` = `[−0.5, −1.57, −2.79]` m from the store; `fluxes(ds)` returning `oceQnet, oceQsw,
    oceFWflx` **as stored (no negation)**, with the `forcing_note` attr propagated; `wind(ds)`
    returning `oceTAUX`/`oceTAUY` **re-masked with `hFacW`/`hFacS`** (922 / 565 finite-on-land
    values otherwise; M0 task 3) and `KPPhbl`. *(corrected 2026-10-07, M3 task 1: the signature is
    `wind(ds, grid_ds)` — the `hFac` masks live in the grid store, not in the raw store.)*
  - `hour_pair(ds, t0) -> (hour_t, hour_tp1)`: the two snapshots with `expand_dims('face')`,
    `float64`; `midpoint(f_t, f_tp1)` = `semilag.midpoint_time`; `time_mid(ds, t0)` = `t0 + 30
    min` (the store coord for the derived product and Figure 6's local-solar axis: lon −120.5 →
    UTC − 8.0 h; `KPPhbl` max ~01 h solar, min ~13 h solar, M2 task 3).
  - `filtered(hour, L_cells) -> (b, U, V)`: `operators.buoyancy` (JMD95) then `operators.lowpass`
    at the **same** `L` on all three (coding §1.2; `L = 0` the identity).
  - `valid(masks_ds, *fields) -> bool array`: **`mask_analysis & isfinite(every field)`**. This is
    the reduction rule at every `L`, and it is **required at `L = 8`**, where the order-3
    departure support leaves the low-passed field's finite part for 7-34 analysis cells per pair
    (0 at `L <= 4`; worst pair 44, 07-03 20:00; M2 task 3). Return the count of cells lost so
    task 4 logs it per pair. *(as written 2026-10-07, M3 task 1: `valid(masks, *fields) ->
    (valid, n_lost)`; `masks` is the §3.5 Dataset or a bare bool array such as
    `masking.analysis_mask(grid_ds, edge_cells=13)`.)* `edge_cells` stays 7 (M2-Q7; the `L = 8` reach is exactly 7 at zero
    displacement, M1 task 6, so it **must not shrink**); the `edge_cells = 13` sensitivity is a
    second mask built at analysis time with `masking.analysis_mask(grid_ds, edge_cells=13)`
    (task 6), not a change here.
- **The displacement envelope this rests on** (M2 task 3, 71 pairs, `departure_index` defaults):
  ocean median 0.27-0.44 cells, p99 1.05-1.38, window max **4.05** (pair 59, Gulf of California
  tidal jet, outside `mask_analysis`); on `mask_analysis` median 0.25-0.44, p99 0.93-1.34, max
  **2.27** (pair 45), 0 NaN departures; front pixels median 0.33-0.56, p99 1.30-1.79, max 2.27.
  M1's 0.36 / 1.25 / 2.1 hold on the analysis domain within the tidal modulation.
- `tests/test_inputs.py`, offline, on synthetic two-store fixtures mimicking §3.2 / §3.3 (reuse
  the synthetic-store helpers of `test_pull_series.py` / `test_load_chunk_levels.py` where they
  fit): the merge succeeds with the expected var list and dims; the plain `xr.merge` of the raw
  pair raises (the M2 finding, pinned); the `k = 0` invariant catches a store shifted by one hour;
  the sign guard refuses an upward-positive store and a missing `sign_convention`; `W_k1` picks
  `k_l = 1` and never `k_l = 0`; `valid` drops NaN cells and counts them; `time_mid`; the wind
  re-mask leaves 0 finite values on `hFacW == 0`. One `needs_grid` test on the real stores: hour 0
  opens, 16 vars (rename, M3-Q6), the `k = 0` identity on all three levels, `drF`, `n_valid` at `L = 0`
  = 262,925 and `n_front` (p90) = 26,293 — the V3 / M2 numbers.
- *Honours:* M2 task 6 items 1 (merge), 2 (no re-negation, guard), 3 (`W.isel(k_l=1)`), 5
  (envelope), 6 (`isfinite` at `L = 8`, `edge_cells = 7`); M1 task 6 flag 10 (defaults).
- *Pitfalls:* `expand_dims('face')`; dims asserted; `oceTAU*` re-masked; `W(k_l=1)` not `delta`,
  not `W(k_l=0)`; §8's "loaded only `k = 0..2`" means *stored* — the fetch was never
  level-selective (M2 task 4).

*Discharges:* nothing on its own; it is the checked input every term of criteria 1-3 is computed
from.

### 2. `vertical.py` physics — `b_z`, `vertical_term`, `surface_flux_term`

- **Module split first (decided 2026-10-07, M3-Q7: option (a)).** `vertical.py` is 524 lines
  before any physics (M2 task 5 deviation 1). Move the chunk-store **reader** (`make_fs`, `_cat`,
  `_read_object`, `_load_levels`, `_load_hour`, `load_chunk_levels` and its helpers and constants)
  to a new `py/chunk_store.py`, keep `vertical.load_chunk_levels` as a one-line re-export so
  `m2_chunk_pull.py` keeps working, and repoint `test_load_chunk_levels.py`'s monkeypatch target
  (`vertical._cat` → `chunk_store._cat`) — its 23 offline tests must still pass unchanged
  otherwise. The physics then lands in a `vertical.py` that starts near empty. *(The "if JXP says
  leave it" branch is closed: the split is decided. `osn_tiles.py` (556) and `validate.py`
  (1,087) are **left whole** — M3-Q7, the M1-Q8 precedent; M5 may consolidate.)*
- Write, per coding §4.6 (signatures are the contract), all in **F units** (s^-5); the budget
  fields are `2 ×` these (task 3), exactly as `subfilter = 2 * subfilter_term` (coding §4.5, §1.1):
  - **`b_z(Theta, Salt, grid_ds, drF)`**: `b` at `k = 0` and `k = 1` from **the same
    `operators.buoyancy` call** (JMD95, potential density at `p = 0` for both levels — state
    it; the ~0.5 dbar in-situ difference is negligible, planning §5.1), `b_z = (b_k0 − b_k1) /
    (Z[0] − Z[1])` with `Z` from the store (`−0.5, −1.57` m → `dz = 1.07 m`). **Sign note:**
    code `b` *increases with density* (coding §1.1), so a warm layer over cooler water gives
    `b_z < 0` here where planning §2.2's textbook `b_z ~ +2-4e-4 s^-2` is positive. Record the
    convention in attrs; the terms below are built entirely in code `b`, so they are consistent.
    Offer `k = 2` as an optional second-order estimate for a sensitivity (the store has it).
  - **`vertical_term(b, b_x, b_y, b_k1, W_k1, drF, grid_ds, grid)`**: compute the **top-cell
    vertical advective tendency first**, `T_v = −W_k1 (b_k1 − b) / drF[0]` *(as written
    2026-10-07, M3 task 2: `T_v = −W_k1 b_z = −W_k1 (b − b_k1)/dz`, `dz = Z[0] − Z[1] =
    (drF[0] + drF[1])/2 = 1.07 m` — the formula here has the sign of `−w b_z` reversed relative
    to planning §2.2's own equation and to the factorised form below, and `dz` for `drF[0]`
    makes the two forms agree exactly for uniform `b_z`; M3-Q11)*, then
    `grad_h T_v` through `operators.grad_b`, dotted with `(b_x, b_y)`. **Not** the factorised
    `−b_z (w_x b_x + w_y b_y)`, which drops `−w grad(b_z) . grad b` (planning §2.2, coding
    §4.6); provide that form as **`vertical_term_factorised`**, a diagnostic only. `W_k1` is the
    chunk `W(k_l=1)` = source `W(k_p1=1)` — the model's own cell-base velocity, which already
    contains the `dEta/dt` part (~5e-5 m s^-1, tidal) and the convergence part `+drF delta`
    (coding §4.6; **do not** rebuild it from `delta`, wrong sign and missing part — planning §2.2,
    corrected 2026-09-28). Filtering at `L`: lowpass **`T_v` itself** (not its factors) with the
    same kernel as `b, U, V`, and dot with the filtered `grad bbar` — the subfilter correlation
    `mean(w b_z) − wbar bzbar` is then inside the term rather than silently dropped; state this in
    the attrs and in the log. Land and the `W`/`b_k1` NaN propagate.
  - **`surface_flux_term(b_x, b_y, oceQnet, oceQsw, oceFWflx, Theta, Salt, drF, grid_ds,
    grid)`**: the top-cell buoyancy tendency `B_sfc` from the stored, **downward-positive** fluxes
    (**no negation** — M2-Q6 (a); the store negated the source's upward-positive data at write,
    and `inputs.py` guards it): heat into the top cell `Q_top = (oceQnet − oceQsw) + f_sw(drF[0])
    · oceQsw`, i.e. **`oceQsw` treated separately** — `oceQnet` includes the shortwave (checked on
    M2 task 4's tile means: noon `oceQnet − oceQsw` ≈ the night-time non-solar ~+115-135 W m^-2
    cooling), and only the fraction `f_sw` absorbed inside the 1 m cell heats the `b` we measure.
    `f_sw` from the model's shortwave penetration (MITgcm `SWFRAC`, Paulson-Simpson two-band,
    Jerlov type I by default: `0.58 exp(z/0.35 m) + 0.42 exp(z/23 m)`, so **~0.56 at `z = −1 m`**)
    *(verified 2026-10-07, M3 task 2: `SHORTWAVE_HEATING` is defined, no `data.kpp` / `data.exf`
    override, and `swfrac.F` (checkpoint65v) hard-codes `jwtype = 2` = Jerlov **IA**,
    `0.62 exp(z/0.6 m) + 0.38 exp(z/20 m)`, so **`f_sw = 0.521`** in the 1 m cell, not 0.56;
    M3-Q10)*;
    **verify** the LLC4320 `data` namelist (planning §2.3's URL) for any override and record
    `f_sw`. Temperature tendency `dT/dt = Q_top / (rho0 c_p drF[0])` (`c_p = 3994 J kg^-1 K^-1`,
    MITgcm `HeatCapacity_Cp`; `rho0 = 1000` as in `buoyancy`) *(as written 2026-10-07, M3 task 2:
    the model divides the fluxes by `rhoConst = rhoNil = 1027.5` (`data`: `rhonil=1027.5`,
    `rhoConst` absent; `HeatCapacity_Cp` absent → 3994), so `rhoConst` is used here and
    `rho0 = 1000` only in the `g/rho0` of the buoyancy definition; M3-Q12)*, salinity tendency `dS/dt =
    −S oceFWflx / (rho0 drF[0])` with the local `Salt` (check `convertFW2Salt` in the namelist:
    `−1` means local salinity; a constant 35 otherwise — record which). Then `B_sfc = (g/rho0)
    [ −rho alpha dT/dt + rho beta dS/dt ]` in **code-`b` sign** (heating *lowers* `b`), with
    **`alpha`, `beta` by finite differences of the same JMD95 density** `operators.buoyancy`
    wraps (`dT = 0.01 K`, `dS = 0.01`, at the local `Theta, Salt`), and the term `grad_h b .
    grad_h B_sfc` via `operators.grad_b`, lowpassed as `T_v` is. Propagate the store's
    `forcing_note`: the fluxes are **6-hourly forcing linearly interpolated** (kinks at 03 / 09 /
    15 / 21 UTC), so the diurnal shortwave is a triangle peaking at 13 LST, not resolved
    insolation (M2 task 4; Figure 6 says so).
  - Expected sizes, which are **checks that the terms are right** (the prompt text above): the
    vertical term ~30 % of `F` by day and ~0 at night (planning §2.2; M0's surface-only bracket
    0.4-14 % rms across `b_z = 1e-5..4e-4`, order-one pointwise at fronts), with a tidal phase; the
    surface-flux term peaking at 13 LST with the triangular shape; `b_z` building a warm layer
    through the afternoon. Run the smoke on real hours 0 (16 LST), 9 (01 LST), 21 (13 LST) at
    `L = 0` on `mask_analysis` and front pixels, and put `rms(2·term)/rms(2F)` in the log.
- `tests/test_vertical.py` (offline, `synthetic.py` grids; one `needs_grid` smoke): `b_z` of a
  two-level analytic profile with the stated sign; the vertical term is 0 for `b_k1 = b`, equals
  the factorised form when `b_k1 − b` is uniform, and differs from it by exactly `−w grad(b_z) .
  grad b` when it is not (the dropped term, pinned); the surface-flux term is 0 for a uniform
  flux and sign-determinate for a flux gradient along `grad b` (a heating gradient towards the
  dense side is frontolytic in code `b` — state the expectation before coding); `oceQsw` enters
  with `f_sw` and `oceQnet − oceQsw` with 1; `alpha ≈ 2.4e-4 K^-1`, `beta ≈ 7.5e-4 psu^-1` at
  (17 °C, 33.6) within 2 % of JMD95; a 3-D `W` (length-3 `k_l`) passed where `W_k1` is expected
  raises; an upward-positive flux store raises (through `inputs`); NaN at land propagates; dims
  asserted after every dbof call.
- *Honours:* M2 task 6 items 2 (no re-negation), 3 (`W.isel(k_l=1)`), 4 (forcing caveat → attrs),
  9 (`vertical.py` split); coding §4.6's marked paragraph; planning §2.2 (tendency first).
- *Pitfalls:* JMD95, not the legacy `utils/physical_calculations` one, not TEOS-10;
  `vertical_term` from `W(k_l=1)`, not `delta`, not the OSN `W(k_l=0)`; dims asserted; the
  factor of two handled in task 3, not here (the functions return F units — say so in the docstring).

*Discharges:* the "**measured** from the chunk store" half of criterion 1 (the terms exist, are
tested and have the expected diurnal signature); supports criterion 5 (Figure 6, Figure 10).

### 2b. Workstation setup — the Python environment *(added 2026-10-08; run once, on the workstation, before task 3)*

M3 moves from the laptop (macOS arm64) to JXP's workstation from task 3 onwards. This session
builds and verifies the Python environment there; it writes **no** M3 code. JXP will already have
done the following on the workstation:
- pulled the `frontogenesis` branch of `fronts`;
- copied `dev/frontogenesis/data/` (~1.8 GB, git-ignored) from the laptop.

If either is missing, stop and say so.

- **Read first:**
  - the M0 task-1 log entry "Execution prompt 1, task 1: environment" in
    `frontogenesis_prompts.md` (the install commands, why Python 3.13 and not 3.14, the
    `dbof --no-deps` reason, the resolved-version table);
  - prompt 1 task 1, including its marked correction: **`xgcm>=0.10`**, not `<0.10`;
    `set_xgcm_grid` passes `padding='fill'`;
  - `env/frontogenesis_pip_freeze.txt` and `env/frontogenesis_env.yml`. These are the laptop's
    exports, macOS arm64, conda `file://` paths, so they **will not rebuild verbatim**; use them
    as the version reference only.
- **Platform.** Report the OS, architecture, CPU count, RAM, free disk, and whether conda/mamba
  (miniforge) is installed. If not, install miniforge in the user's home (no sudo) and say so.
- **Code.**
  - `fronts`: confirm the checkout is on `frontogenesis` and matches `origin/frontogenesis`.
  - `dbof`: clone `Sea-Meets-the-Stars/llc4320-native-grid-preprocessing` (or add a worktree if a
    clone exists) on branch `tiles-surface-only`, **pinned to commit `938bce1`**, the commit every
    M0-M3 log quotes line numbers against. If `origin/tiles-surface-only` has moved past it, check
    out `938bce1` anyway and report how far the branch has moved.
- **The env: `frontogenesis`, Python 3.13, conda-forge.**
  - Use the M0 package list with `xgcm>=0.10`, plus what M0-M3 added since: `python-pptx` (pip),
    `pytest`.
  - Pin the core packages to the laptop's working versions where conda-forge has them for this
    platform: **numpy 2.5.3, scipy 1.18.1, xarray 2026.7.0, zarr 3.4.0, dask 2026.8.0, xgcm
    0.10.1, scikit-image 0.26.0, skan 0.13.1, python-pptx 1.0.2**. Read the rest of the table in
    the M0 log and `pip freeze`, and pin those too where they matter: s3fs/fsspec, xmitgcm,
    scikit-fmm, h5netcdf, matplotlib.
  - If a pin is unavailable, take the nearest version and **record every deviation**.
  - Leave out what M3 does not need if it fights the solver (`healpy`, `pyvista`/`trame`,
    `PyQt6`, `pytorch`). Check first that nothing in `dev/frontogenesis/py/` or its tests imports
    it, and say what was dropped.
  - Then install:
    - `dbof` with `pip install -e . --no-deps`, because its `torch`/`timm` pins would otherwise
      downgrade torch (M0 task 1);
    - `fronts` with `pip install -e .`. If its deps drag in GUI or `timm` packages that fail on
      this platform, use `--no-deps` and install only what `fronts.finding` needs
      (`test_nan_finding.py` is the check).
  - Run `pip check` and explain each complaint. M0 found two metadata-only ones from `dbof`'s pins.
- **Non-Python tools.**
  - For task 9's slides: `soffice` (LibreOffice) and `pdftoppm` (poppler). Install without sudo
    (conda-forge or user-level) if possible; otherwise report the command JXP needs to run.
  - For long jobs: the prompts' `caffeinate -i -s` is macOS-only. On Linux use `nohup` inside
    `tmux`/`screen`, and check whether the machine suspends at all (a server normally doesn't).
    Record the equivalent the later tasks should use.
- **Network and credentials,** read-only checks, no secrets printed:
  - can this machine reach OSN (`https://mghp.osn.xsede.org`)?
  - can it reach Nautilus `s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/`, and does an AWS default
    profile exist?
  - time one small read from each.

  M3 needs neither, because the data is local. This only records whether a re-pull would be
  possible here.
- **Verify:**
  1. Imports: `dbof`, `fronts`, `xgcm`, `skfmm`, `skan`, `pptx`, and every module in
     `dev/frontogenesis/py/`.
  2. The data is present and intact. Run `series_verify.verify_series` on the OSN 72-hour store
     and `verify_chunk_series` on the chunk store, with `m2_pull.timestamps_72()`. Both must be
     `ok`.
  3. The full suite: `python -m pytest dev/frontogenesis/py/tests -q` must give **160 passed,
     3 xfailed, 2 deselected**, the laptop's count after task 2. The 3 strict xfails document
     `fronts` bugs (M1 task 7a). If they XPASS, the installed `fronts` differs; report it.
     `test_inputs`'s `needs_grid` test must reproduce **262,925** valid and **26,293** front
     pixels.
  4. Optionally, the two `-m network` smoke tests, if the network checks passed.
  5. Record the suite's wall time. It was 257 s on the laptop, close to the 300 s per-command
     timeout. If it is much faster here, say so, because later tasks may not need to run it in
     batches.
- **Write down what later sessions need:**
  - a portable env spec, **`env/frontogenesis_env_<platform>.yml`** (`conda env export
    --from-history` plus the pip-installed lines) and `env/frontogenesis_pip_freeze_<platform>.txt`.
    Leave the laptop's files untouched.
  - the absolute **Python path** of the new env;
  - the long-job wrapper (`nohup` + `tmux`, or `caffeinate`).

  Add a marked note to the Tasks preamble of this prompt: "(workstation, 2026-10-08, task 2b):
  interpreter `<path>`, long jobs `<wrapper>`; the `~/miniforge3/.../python` and `caffeinate`
  references below mean these on the workstation". **Do not** rewrite the per-task paths.
- **Log:** `### <date> — Execution prompt 4, task 2b: workstation environment (<model>)` at the
  end of `frontogenesis_prompts.md` `## Logs`. Include:
  - the platform;
  - the commits (`fronts`, `dbof`);
  - the install commands as run;
  - a resolved-version table against the laptop's, with every deviation;
  - what was dropped;
  - `pip check`;
  - the tool and network checks;
  - the verification results with numbers, and the suite's wall time.

  Update this prompt's Status with one line.
- *Rules:* do not edit any code, test, data store or `deck/`. Commit nothing. Prefix interactive
  commands with `timeout 300`; a long conda solve may need more, so run it under `nohup` with a
  log if it does.

*Discharges:* nothing in the criteria. It is the precondition for tasks 3-9 on the workstation.

### 3. `budget.py` — `compute_budget`, `closure_report`, the hour-0 smoke

- **`compute_budget(raw_ds, grid_ds, grid, masks, L_cells, dt=3600.0, chunk_ds=None, *, t0=0,
  forms=('discrete', 'chain'), order=3, order_sens=5, front_pct=90.0)`** (coding §4.7's
  signature plus keyword-only options; `front_pct=90.0` is the **primary** pool — decided
  2026-10-07, M3-Q4 (a); p80 / p95 are recomputed from the stored `G` in task 6, not here; `raw_ds` may be the merged dataset of task 1, in which case
  `chunk_ds` is implied) → one hour pair's budget as an `xr.Dataset` on `(time: 1, j, i)` with
  `time = t0`'s hour and a `time_mid` coord, the §3.4 fields plus the additions below (the §3.4
  var list is extended, not changed — task 8 marks coding §3.4):
  - `b`, `G` at the **midpoint**, filtered at `L` (`inputs.filtered`, `semilag.midpoint_time`,
    `operators.gradb2` = `b_x^2 + b_y^2` from the same `grad_b` as `F`, never
    `calculate_grad_squared_tracer`);
  - **`two_F`** (`form='discrete'`, the primary) and **`two_F_chain`** (`form='chain'`, run
    alongside; the difference is a stated systematic — M1-Q1; on real front pixels the discrete `F`
    is ~0.79x the chain `F`), both `2 * operators.frontogenesis(b_mid, U_mid, V_mid, ...)`;
  - **`DGDt_semilag`** = `semilag.measured_DGDt(b_t, b_tp1, U_mid, V_mid, grid_ds, grid, order=3)`
    (defaults; interpolate `b`, then differentiate — never `G`), **`DGDt_semilag_o5`** the order-5
    sensitivity (M1-Q6; task 3 of M1 saw ~5 % between the orders on real front pixels; it costs a
    6-node NaN rim per axis), **`DGDt_euler`** = `semilag.eulerian_DGDt` (the independent estimate);
    `measured` is `DGDt_semilag` (coding §1.1's three tiers);
  - **`subfilter` = `2 * coarsegrain.subfilter_term(b_bar, tau_x, tau_y, grid_ds, grid,
    tau_delta=coarsegrain.subfilter_bdelta(b, U, V, L, ...))`**, with `tau_x, tau_y =
    coarsegrain.subfilter_flux(b, U, V, L, ...)` from the **unfiltered** `b, U, V` (it filters
    internally). **`tau_delta` is mandatory** — the flux form alone overstates the term 2.2x in rms
    on this divergent surface flow (M1 task 4). At `L = 0` the explicit term is **identically
    zero** (planning §5.4, corrected): that column is `2F` vs measured with the whole numerical
    term in the residual, not a limit of `tau` — label it so;
  - **`vertical` = `2 * vertical.vertical_term(...)`** with `W_k1 = inputs.W_k1`, `b_k1` from
    `Theta_k, Salt_k` at `k = 1`; **`vertical_factorised`** (diagnostic); **`surface_flux` = `2 *
    vertical.surface_flux_term(...)`** from `inputs.fluxes` (no negation);
  - **`residual` = `measured − two_F − subfilter − vertical − surface_flux`**, its attrs listing
    `terms_present`. With the Q13 terms in, it is **numerical diffusion + interior KPP**, not
    "everything we could not compute" (planning §2.3);
  - `delta, sigma_n, sigma_s, sigma_mag` (`operators.strain_divergence` at the midpoint, corners
    to centres, model basis rotated by `2 alpha`), `theta_align` (`operators.strain_alignment`,
    compressional axis, folded to `[0, pi/2]`; sign final, M1-Q3);
  - diagnostics for tasks 6-7: **`lap2_b`** (the biharmonic of `b_mid`, the `grad^4`-like field
    Figure 2b regresses the residual on), **`front_width`** (the per-pixel width proxy, decided
    2026-10-07, **M3-Q5** (a): **`ell = 2 sqrt(G_mid / |lap G_mid|)`** in units of dx, exact for
    `G ∝ sech^4(x/ell)` at the maximum — i.e. for `synthetic.py`'s `b = b0 tanh(x/ell)` fronts;
    meaningful on front pixels, stored everywhere finite; task 6 bins it `{<= 1, 1-1.5, 1.5-2,
    2-3, 3-4, > 4}` dx), **`valid`** (`inputs.valid`, with `n_lost` in attrs), **`front`** (`G_mid >=
    percentile(G_mid[valid], front_pct)` — the selection at the midpoint time, independent of both
    endpoints, planning §11; **p90 primary, decided 2026-10-07, M3-Q4**), `KPPhbl` (midpoint mean)
    and `coast_distance_km` copied for the composites.
  - **If `chunk_ds` is None** (and the merged dataset lacks the chunk vars): `vertical`,
    `surface_flux` are **absent** (not zero), `residual.attrs['terms_missing'] = ['vertical',
    'surface_flux']`, and **`closure_report` says loudly** — in its printed summary and in
    `report['closed'] = None` with `report['warning']` — that the residual is a catch-all and
    this is **not** a closed budget. Silent degradation would let a two-term comparison
    masquerade as closure (prompt text above; coding §4.7, §8).
- **`closure_report(budget_ds, mask) -> dict`**: on `front & valid` and on `valid` separately:
  rms of each term relative to `rms(measured)` and to `rms(two_F)`; `rms(residual)/rms(measured)`
  and `/rms(two_F)`; the explained fraction `1 − var(residual)/var(measured)`; the OLS slope and
  corr of `residual` on `two_F` (a term missing in proportion to `2F` shows here); the OLS slope
  and corr of `DGDt_euler` on `DGDt_semilag`; `n_valid`, `n_front`, `n_lost`; the terms present;
  and the verdict `closed: True / False / None` against the **pre-declared tolerances of M3-Q1
  and M3-Q2** (module constants, with their source cited). Printing it must read like a verdict.
- **`write_derived(budget_ds, L, out=None)`** → `data/tile330_derived_L{L}.zarr` (§3.4) through
  `zarr_series.append_hour`: resumable and atomic per pair, `float32` on disk, one chunk per pair
  per variable, `time` encoded as §3.2, attrs recording `L_cells`, `forms`, `order`, `front_pct`,
  `edge_cells = 7`, the git commit and the two source stores. `present_times` decides "already
  present"; `clobber` rewrites.
- **Hour-0 smoke on real data** (the log, not a test): pair 0 (07-02 00-01 UTC, 16 LST) at
  `L = 0, 2, 4, 8` with the chunk terms; the five-term table (`rms` relative to measured and to
  `2F`, on `valid` and on `front`); `n_front` = **26,293** of 262,925 at `L = 0` (the V3 / M2
  pool); `two_F` **bit-for-bit** `validate.two_F(b_mid, U, V, grid_ds, grid, form='discrete')` and
  `DGDt_semilag` bit-for-bit the `measured` of `validate.null_step` **with the real `b_tp1`
  substituted** — same functions, same inputs, so any difference is a bug in the plumbing;
  `rms(vertical)/rms(2F)` at 16 LST within M0's bracket; `subfilter` = 0 at `L = 0` and
  **0.31 / 0.50 / 0.70 of `Fbar` in rms at `L = 2 / 4 / 8`** (M1 task 4's hour-0 numbers, which
  must reproduce; anti-correlation −0.66 / −0.60 / −0.54). Wall time per `L`, for task 4's plan.
- `tests/test_budget.py` (offline on `synthetic.py`'s exact advection solutions, as
  `test_coarsegrain.py`; one `needs_grid` smoke): with the semi-Lagrangian step as truth and
  `chunk_ds=None`, `residual` on front pixels < 2 % of measured (the V3 identity) and the report
  flags `closed=None` with the warning in its text; with a synthetic chunk dataset whose `W_k1 = 0`
  and uniform fluxes, `vertical = 0` and `surface_flux = 0` exactly and `terms_missing = []`;
  `subfilter` is 0 at `L = 0`, non-zero at `L = 2`, and differs from the no-`tau_delta` form;
  `two_F_chain != two_F`; `DGDt_semilag_o5` has the wider NaN rim; the §3.4 var list, dims,
  dtype and attrs; `front` is selected on `G_mid` (shifting `G_tp1` does not move it); `valid`
  excludes NaN; **`front_width` recovers `ell` on `synthetic.py`'s tanh fronts** at the front
  centre for `ell` in `{1, 1.5, 2, 3, 4}` dx to a stated tolerance (decided 2026-10-07, M3-Q5); `write_derived` resume / no-op / clobber through `zarr_series`; the `needs_grid`
  smoke asserts the pair-0 bit-for-bit identities above.
- *Honours:* M1 task 4 (`tau_delta`; `subfilter = 2 × term`; the `L = 0` column); M1-Q1 (both
  forms); M1-Q6 (order 5); M2 task 6 items 1-3 (through `inputs`), 6 (`isfinite` at `L = 8`);
  coding §6 M3 "Carried from M1"; planning §11 (midpoint selection); coding §4.7 (loud report).
- *Pitfalls:* `2F` not `F`; `b` not `G`; midpoint; same filter; land halo — land is NaN and
  `lowpass` propagates it, the halo and edge margin enter through `mask_analysis` at reduction, so
  nothing is differenced across a filled value; `vertical`/`surface_flux` present or loudly
  absent; dims asserted; `G` from `gradb2`, never `grad_b2`.

*Discharges:* criteria 1-3 **in code** (the budget, both estimates, the sweep machinery with
`tau` explicit); proven on data in tasks 4 and 6.

### 4. The `L_cells` sweep over the 72 hours — `py/m3_run.py` → `tile330_derived_L{L}.zarr`

- Write `py/m3_run.py`: for `L` in **`{0, 2, 4, 8}`** (coding §1.2; the contract — decided
  2026-10-07, **M3-Q3** (a)) and `t0` in
  `0..70`, `compute_budget` with the chunk terms, `write_derived`, and `closure_report` per pair
  cached to `data/m3_closure_L{L}.json` (resumable; the per-pair JSON is what tasks 6-7 read
  first). Progress log `data/m3_run.log` (per-pair wall time, `n_lost`, the five rms fractions),
  done-file `data/m3_run_done.json`, flags `--L 0,2,4,8`, `--pairs a:b`, `--dry-run`,
  `--no-chunk` (for the catch-all comparison of task 6), `--clobber`. One pair in memory at a
  time (720² float64 × ~30 fields ≈ 120 MB; M2's chunk pull plateaued at 1.4 GB RSS).
- **Pilot first, in the foreground under `timeout 300`:** `--pairs 0:2 --L 0,8`; record the
  per-pair wall time per `L` and extrapolate to 71 × 4. Task 3's smoke suggests the order of
  seconds to tens of seconds per pair per `L` (the V3 null step is ~2 s; the budget adds
  `coarsegrain`, the vertical and flux terms, the chain form, order 5 and the Eulerian estimate),
  i.e. **~1-4 h** for the sweep — a detached job either way. If the extrapolation exceeds ~6 h,
  run `L = 0, 4` first and `2, 8` in a second launch, and say so. *(decided 2026-10-07, M3-Q3:
  if instead the pilot shows time to spare, `L = 1` may be run as an **extra column for Figure 3
  only** — it needs no new mask — and is not part of the contract, the gate or the other figures;
  say so in the log and in `verify_derived_series`'s count if it is added.)*
- **Launch detached under `nohup caffeinate -i -s`** (rules above), poll the log, and verify the
  first pairs on disk. The session may end before the run does; the next session (task 5) checks
  `m3_run_done.json`, **re-runs `m3_run.py` and shows it is a no-op** (0 pairs computed; sha256 of
  the chunk files unchanged), and runs **`series_verify.verify_derived_series(out_zarr,
  timestamps, L)`** (new; add it to `series_verify.py`): 71 pairs, no gaps, the §3.4 (+ additions)
  schema, `float32`, one chunk per pair, `L_cells` attr, NaN ⊇ land every pair, `valid`'s
  `n_lost` = 0 at `L <= 4` and ≤ 34 per pair at `L = 8` (M2 task 3's bound). *(note 2026-10-07,
  M3 task 1: 34 is M2 task 3's pair-44 count with the **raw** midpoint velocity; with `U`, `V`
  low-passed at the same `L` as `b` (coding §1.2, `inputs.filtered`) pair 44 loses 26 cells and
  pair 0 loses 21 either way, so 34 stands as the upper bound.)*
- **Report** (criterion-7 style, as M2): pairs present per `L`, failures and relaunches, per-pair
  wall time (median, range) per `L`, volume on disk per store, and the extrapolation to the
  504-hour OSN series — extending M2's scale-up table (OSN ~23 s and 11 MB per tile-hour; chunk
  ~300 s and 175 MB fetched at 0.55 MB/s, 37 s at the 4.7 MB/s seen on 10-04; a server-side
  `k = 0..2` subset would cut the chunk fetch ~7x).
- *Honours:* M2 task 6 item 8 (scale-up numbers, `caffeinate`), item 6 (`n_lost` logged per
  pair at `L = 8`), item 5 (the envelope the loss follows); the anti-stall rules.
- *Pitfalls:* same filter at each `L`; `float32` on disk; the `L = 0` column labelled; never a
  foreground multi-hour job.

*Discharges:* the **data** for criteria 1-5 (every `L`, `tau` explicit, both estimates, both
forms, both orders, the measured chunk terms).

### 5. `stats.py` — estimators and the space-time block bootstrap

Written while the sweep runs (it needs only numpy); begin the session by checking
`m3_run_done.json` and the log, and do task 4's no-op re-run and `verify_derived_series` when the
run has finished. **Read planning §11 before writing it** (the stats paragraph above).

- Per coding §4.8 (signatures are the contract), plus what M1-M2 found necessary:
  - `slope_ols(x, y)` (with intercept — **the gate's definition**, as V3 declared it),
    `slope_tls(x, y)` (orthogonal; both axes share units), `slope_bisector(x, y)`;
    **`slope_trimmed(x, y, pct=1.0)`** — OLS with the top `pct` % of `|x|` dropped (M2 task 3's
    statistic: 0.987 ± 0.005 over 71 pairs against 0.972 ± 0.020 raw);
  - **`ratio_estimator(x, y, split_sign=True)`** → `{'all', 'pos', 'neg'}`: `sum(y)/sum(x)` is
    **ill-conditioned when the pool has both signs of `2F`** (0.85 on V3's synthetic pool against
    1.004 for every other estimator; M1 task 6), so it is quoted for `X > 0` and `X < 0`
    separately;
  - `binned_conditional_mean(x, y, bins, split_sign=True) -> DataFrame`: `E[Y|X]` per bin with a
    block-bootstrap band, **separately for `X > 0` and `X < 0`** (diabatic damping is asymmetric;
    a single slope averages over the asymmetry that carries the physics);
  - **`block_bootstrap(x, y, labels, estimator, n=1000, seed=0) -> (lo, hi, se)`**: resample
    **contiguous space-time blocks** with replacement, never pixels — at this milestone the block
    is a **32 × 32-cell square in one hour** (`block_ids(shape, B=32)` offset by the pair index),
    with an option for 3-hour × 32-cell blocks to show how much the hour-to-hour correlation
    widens the interval (M2 task 3 found χ²/dof 1.8 across pairs against the one-hour bootstrap:
    excess scatter beyond the within-hour CI). `feature_bootstrap(x, y, labels, estimator, n)` is
    the §4.8 name — a thin alias that M4 calls with front labels; here the labels are blocks;
  - **`slope_report(x, y, labels, *, baseline=0.981, baseline_ci=(0.970, 0.994), v3b_band=(0.954,
    1.003), temporal=(0.972, 0.020))`** → one dict with every estimator, its CI, and each slope
    **relative to the baseline** with the bands carried separately — the function that makes
    "never against 1" (coding §8) the default rather than a reminder.
- **Reuse, do not fork silently.** `validate.slope_estimators`, `validate.block_bootstrap_ols`
  and `validate._block_ids` already implement the OLS / orthogonal / GM / ratio set and the
  32-cell block bootstrap (M1 task 6), and `m2_baseline_stability.robust` the trimmed and
  leave-one-block-out statistics. `stats.py` generalises them (any estimator, time blocks,
  split by sign); `validate.py` is **not edited** (M1's, closed; M2 added only two keywords).
  Tests assert equivalence: `stats.slope_ols` ≡ `validate.slope_estimators()['ols']` and
  `stats.block_bootstrap(…, slope_ols, seed=0)` reproduces `validate.block_bootstrap_ols`'s CI
  for the same blocks (identical when the resampling is the same multinomial draw; otherwise
  within the bootstrap noise, stated). M5 may repoint `validate` at `stats`.
- `tests/test_stats.py` (offline, synthetic): a known slope recovered by every estimator on clean
  data; **errors-in-variables** — with noise on `x`, OLS attenuates by `var(X)/(var(X)+var(eta))`
  (to a stated tolerance) while TLS does not (planning §11); one heavy-tailed leverage point
  moves OLS and not the trimmed estimate (M2's day-3 mechanism in miniature); the ratio estimator
  on a two-signed pool is off while the split is right (pinned); the block bootstrap's CI is
  wider than a pixel bootstrap's on an autocorrelated field and wider again with time blocks on a
  temporally correlated one; `binned_conditional_mean` returns the asymmetry put in; the
  `validate` equivalences above; `slope_report` divides by the baseline and never by 1.
- *Honours:* M2 task 6 item 7 (trimmed and orthogonal beside the OLS gate; the window spread as
  the temporal systematic); M1 task 6 flag 5 (ratio split by sign); planning §11 (blocks, binned
  means, baseline); coding §8 (never against 1; features-vs-blocks by milestone).
- *Pitfalls:* bootstrap over blocks not pixels; the block is spatial + hour here, features only in
  M4; quote against 0.981, never 1.

*Discharges:* the machinery of criterion 4 (estimators and intervals); the numbers come in task 6.

### 6. Closure — the HARD GATE

- **Pre-declare before any number is seen**, as module constants in `py/m3_closure.py` and in
  the log entry's first paragraph *(all decided 2026-10-07 — the values below are the
  declaration; the M3-Q answers are their source)*:
  - the closure tolerance (**M3-Q1**), on **`front & valid`**, per `L`: **(a)
    `rms(residual) / rms(measured) <= 0.5`** (equivalently the explained fraction ≥ 0.75) **and
    (b) the residual's OLS slope on `2F` within ±0.10** (no term missing in proportion to `2F`);
    the five-term rms table and the with / without-chunk-terms comparison always reported;
  - the semi-Lagrangian / Eulerian tolerance (**M3-Q2**): OLS slope of `DGDt_euler` on
    `DGDt_semilag` within **0.85-1.15 and corr ≥ 0.90**, on **front pixels at `L >= 2`**; `L = 0`
    reported and interpreted, not gated;
  - which `L` the gate is judged at (**M3-Q9** (a)): **criterion 1 is judged at `L >= 2`** (every
    `L` in `{2, 4, 8}` reported with its own verdict); `L = 0` is reported and interpreted — its
    explicit subfilter term is ≡ 0, so its residual is the numerics-plus-KPP estimate Figure 2b
    is about. A failure at `L = 0` alone is **not** the planning §12 null; a failure at every `L` is;
  - the front percentile (**M3-Q4** (a)): **p90 on `mask_analysis & finite`** (the V3 / M2 pool,
    `n_front` = 26,293 at `L = 0`) is the gate's pool; **p80 and p95 as sensitivities**
    (recomputed from the stored `G` on `valid`), and the trimmed estimator beside the OLS;
  - the width proxy and its bins (**M3-Q5** (a)): the derived store's `front_width = 2 sqrt(G /
    |lap G|)` on front pixels, binned **`{<= 1, 1-1.5, 1.5-2, 2-3, 3-4, > 4}` dx**; the filter
    sweep (resolved width ≥ `L`) as the cross-check (b); per-object widths are M4's (c);
  - the hour sets (all 71 pairs; the **day-3 northern-front pairs 62-68 = 07-04 14-20 UTC shown
    separately, not dropped**; M2 task 3), the estimators (OLS gate, trimmed, orthogonal, ratio
    by sign, binned means) and the block definition (task 5's 32 × 32-cell square × hour).
  Nothing changes after the numbers.
- Write `py/m3_closure.py` → `data/m3_closure_summary.json` and a working figure
  `figs/m3_closure.png` (task 7 makes the publication figures from the JSON). It reads the four
  derived stores and the per-pair `m3_closure_L{L}.json`, and reports, per `L`:
  - **(a) Closure — criterion 1.** On `front & valid` and on `valid`, per pair and pooled: rms of
    the five terms relative to measured and to `2F`; `rms(residual)/rms(measured)` with its
    distribution over the 71 pairs; the explained fraction; the residual's OLS slope on `2F`;
    **with and without the chunk terms** (`--no-chunk` run or the identity `residual_catchall =
    residual + vertical + surface_flux`) — the measured terms' contribution is the headline of
    the Q13 decision; the `L = 0` column interpreted with `subfilter ≡ 0` (planning §5.4 — the
    whole numerical term sits in the residual there); the residual composited by local solar hour
    with `KPPhbl` (Figure 6's data; phase on the mixed-layer **minimum** ~13 h solar) and by
    `coast_distance_km` (Figure 7's data; also the primary statistic **restricted to `>= 100 km`**
    and stratified, planning §5.6); the **verdict per `L`** against M3-Q1's **0.5 / ±0.10**,
    judged at **`L >= 2`** (M3-Q9; `L = 0` interpreted, not gated — decided 2026-10-07).
  - **(b) Semi-Lagrangian vs Eulerian — criterion 2.** OLS slope and corr of `DGDt_euler` on
    `DGDt_semilag`, per `L`, on `valid` and on `front`. **Known starting point:** on the real hour
    at `L = 0` M1 task 3 measured **corr 0.74, slope 0.73** on `mask_analysis` — the two estimates
    are *not* yet known to agree at the grid scale; expect convergence with `L`. Verdict against
    M3-Q2's **slope 0.85-1.15 and corr ≥ 0.90 on front pixels at `L >= 2`** (decided 2026-10-07);
    `L = 0` is not gated — a failure there alone is interpreted (the Eulerian split's two nearly
    cancelling terms, planning §5.3), not hidden.
  - **(c) The filter sweep is interpretable — criterion 3.** `rms(subfilter)/rms(two_F)` and its
    correlation with `two_F` vs `L` over 71 pairs (M1 task 4 on hour 0: **0.31 / 0.50 / 0.70 at
    `L = 2 / 4 / 8`**, anti-correlated −0.66 / −0.60 / −0.54 — O(1) and *growing* with `L`, not
    constant, M1-Q8 (c) left this to the sweep); the budget closing at each `L` from (a).
  - **(d) Slopes — criterion 4, reported only where (a) passes** at that `L`; otherwise listed
    under "**what the slope would have been — not quoted**" (the "Do not" list, first item). Per
    `L`, per form (`discrete` primary; `chain` alongside with the discrete-vs-chain difference as
    a stated systematic — the V3 baselines are 0.981 and 0.791), order 3 and **order 5**, masks
    **`edge_cells = 7` and `13`** (`masking.analysis_mask(grid_ds, edge_cells=13)`, the p90 pool
    recomputed on it, as `m2_q7_edge_margin.py` did; the M2-Q7 row is 0.980 ± 0.012, 69/71),
    front pools **p90 (primary) with p80 and p95 as sensitivities** (M3-Q4, decided 2026-10-07), all
    71 pairs pooled and per pair: OLS (the gate's definition) with the **trimmed (top 1 % |2F|) and
    orthogonal fits beside it**, the ratio by sign, binned `E[Y|X]` by sign, space-time block
    bootstrap CIs; every slope **relative to 0.981 [0.970, 0.994]** with the V3b model-advection
    band **0.954-1.003 (0.975 ± 0.025; no upward correction for the Jacobian attenuation, V3b
    measured −0.006 ± 0.025)** and the **temporal systematic 0.972 ± 0.020 (0.987 ± 0.005
    trimmed)** as separate bands; **per front width** (the `front_width` bins `{<= 1, 1-1.5,
    1.5-2, 2-3, 3-4, > 4}` dx — M3-Q5, decided 2026-10-07; the per-`L` sweep as the cross-check),
    with the advection-numerics
    shortfall **−2 % at 2 dx, −4 % at 1.5 dx, −11 % at 1 dx** (discrete form; V3b) subtracted
    before anything on the sharpest fronts is attributed to diffusion; V4's bar **0.28-1.0 % of
    `G` per hour (order 3)** quoted with the width; the day-3 northern-front pairs separately.
  - **(e) Figure 2b's data.** The residual regressed on `lap2_b` (the `grad^4`-like diagnostic)
    and on `KPPhbl`, and composited by local hour, with partial correlations: this is what
    separates implicit numerical diffusion from air-sea forcing. **No "diabatic damping" without
    it** (the "Do not" list). For scale: planning §2.3's OS7MP `kappa_num` is a grid-scale
    estimate — V3b saw only +0.009 ± 0.03 of it in the resolved slope in one hour (the limiter a
    tail effect, p10 −1.9 %/h), while a third-order scheme would give −0.13 — so quote
    `kappa_num` at the scale of the feature, not as one number.
- **Verdict, plainly, per `L`: closure PASS or FAIL**, with the tolerance it was judged against
  (M3-Q1's 0.5 / ±0.10 on `front & valid`; the gate is the `L >= 2` verdicts, M3-Q9, with `L = 0`
  reported beside them — decided 2026-10-07), in the log and in the Status. If it fails, **that is the result**: write the methodological
  finding against planning §12's criteria (residual comparable to `2F` at all `L`; slope
  indistinguishable from the baseline; residual tracking `grad^4 b` not `KPPhbl`), name the
  limiting factor, and do **not** quote an efficiency. **Do not tune anything to pass.** If a bug
  is found, fix it, re-run the sweep (`m3_run.py --clobber`, detached) and re-evaluate, logging
  every change as V3's task 6 did — the change is a finding.
- *Honours:* M2 task 6 items 6 (`edge_cells = 13` row), 7 (trimmed / orthogonal; temporal
  systematic; day-3 hours separate), 4 (local-hour phasing), 11 (the hour 0-1 PASS vs 7/71 — the
  gate's definition is unchanged, this is its reporting); M1-Q1 / Q2 / Q4 / Q6 (both forms, V3b
  band, baseline, order 5); M1 task 4 (the `L = 0` column; the growth with `L`); M1 task 6 flag 5
  (ratio by sign); M1 6b (per-width shortfall; the `kappa_num` caveat); M1 task 3 (the Eulerian
  starting point); planning §5.6 (offshore stratification), §11, §12.
- *Pitfalls:* `2F` not `F`; never against 1; no "diabatic damping" without 2b; the residual is
  not purely diabatic; `kappa_num` at the feature scale; blocks not pixels.

*Discharges:* criteria **1, 2, 3, 4** (the numbers), and the "Do not" list's first two items.

### 7. Figures 1, 2, 2b, 3, 3b, 4, 5, 6, 7, 10

- Write `py/figures.py` — one function per figure, `fig01_maps(...)` … `fig10_term_budget(...)`
  (coding §4.10), each writing `figs/fig{NN}_*.png` (200 dpi, `validate_figs` style) and
  returning its numbers — driven by `py/m3_figs.py`. **From the derived stores and
  `m3_closure_summary.json` only**; no physics is recomputed, and a test asserts the numbers
  drawn equal task 6's. Every title or caption states its **`L` and its mask** (M5 standardises
  captions and wires the one-command harness; the plots themselves are produced here). If the
  module passes ~400 lines, split by figure family (`figures_maps.py`, `figures_stats.py`) and say
  so. The V-figures stay in `validate_figs.py` (M1 task 5); §4.10's `figV1..figV6` in `figures.py`
  is M5's consolidation.
  - **Figure 1** — `DGDt_semilag` beside `two_F` and `residual`, shared diverging scale, at
    `L = 0` and `L = 4`, for a daytime hour (07-03 21 UTC = 13 LST, the mixed-layer minimum) and a
    night-time one (07-03 09 UTC = 01 LST). If these do not look alike, nothing else matters.
  - **Figure 2** — joint PDF (2-D histogram) of measured vs `2F` on front pixels, all 71 pairs
    pooled, the 1:1 line, OLS / trimmed / orthogonal, binned `E[Y|X]` by sign; **the baseline
    drawn as an explicit line at 0.981 with its band [0.970, 0.994]** (M1-Q4), **V3b's band
    0.954-1.003 beside it**, and the temporal systematic 0.972 ± 0.020 as a third marker; one
    panel per `L`; the day-3 pairs as a separate marker. Only interpretable with 2b.
  - **Figure 2b** — the residual against `lap2_b` and against `KPPhbl` (binned means with CIs),
    and against local solar hour. The discriminator.
  - **Figure 3** — slope and correlation vs `L` (both forms, both orders, both masks), baseline
    and bands drawn; plus `rms(subfilter)/rms(2F)` vs `L` *(plus the optional `L = 1` column if
    task 4's pilot had time to spare — this figure only; M3-Q3, decided 2026-10-07)*.
  - **Figure 3b** — rows `{b, G, 2F, tau-term}` × columns `{L = 0, 2, 4, 8}` on one hour; the
    `L = 0` `tau` panel is **identically zero — label it so** (planning §5.4, corrected).
  - **Figure 4** — alignment PDF of `theta_align` on front pixels (compressional axis, folded to
    `[0, pi/2]`; `F = −½ delta G + ½ |sigma| G cos 2theta`, sign final M1-Q3).
  - **Figure 5** — sharpening timescale `G/(2F)` map with the `dt = 1 h` contour (call it
    `t_sharp` in code; the subfilter flux is already `tau`).
  - **Figure 6** — `residual`, `vertical` and `surface_flux` composited by **local solar hour**
    (UTC − 8.0 h), `KPPhbl` overlaid, phased on the mixed-layer **minimum** (~13 h solar;
    `KPPhbl` max ~01 h). The caption **must state**: the forcing is **6-hourly, linearly
    interpolated** (kinks at 03 / 09 / 15 / 21 UTC), so the shortwave is a **triangle peaking at
    13 LST, not resolved insolation**; the day-to-day deepening of the afternoon minimum (13 → 12
    → 8 m) is a **wind trend** (|tau| 0.11 → 0.06 N m^-2), not noise; and 72 h is three diurnal
    cycles — **indicative, not conclusive** (planning §7). The residual is expected to change sign
    over the cycle, which by itself refutes a single-signed "damping" reading.
  - **Figure 7** — statistics vs distance offshore (`coast_distance_km` bins, the `>= 100 km` cut
    marked): slope, residual fraction, `n` (planning §5.6: what the cut excluded stays visible).
  - **Figure 10** — the term budget: `2F`, `vertical`, `surface_flux`, `subfilter`, `residual`
    side by side (maps for one hour and rms bars per `L`), with the catch-all residual for
    comparison. With the Q13 terms this is a real budget, not a two-term comparison.
- `tests/test_figures.py`: offline smoke on a tiny synthetic derived store and summary (every
  function runs, writes its PNG, returns a dict; the baseline line is at 0.981, not 1).
  Check `git status` shows every PNG (`git check-ignore -v` if one is missing).
- *Honours:* M1-Q4 (Figure 2 baseline), M1-Q2 (V3b band), M2 task 6 items 4 (Figure 6 caveats),
  7 (temporal band; day-3 marker), 11 (planning §2.3 / §4's "noon-peaking term" is true at the
  6-hourly scale only — Figure 6's caption carries it); M1 task 4 (the `L = 0` `tau` column);
  planning §7.
- *Pitfalls:* every PNG written and in `git status`; baseline never at 1; no "diabatic damping"
  without 2b; `kappa_num` at the feature scale.

*Discharges:* criterion 5.

### 8. M3 acceptance audit

- Run the full suite offline (`timeout 300`) and the two network smoke tests detached; re-run
  `verify_series`, `verify_chunk_series` and `verify_derived_series` (all four `L`) fresh; re-run
  `m3_run.py` and show the no-op (0 pairs, sha256 unchanged).
- Go through **criteria 1-5 one by one with numbers**, in a table, as the M0 task-5, M1 task-7
  and M2 task-6 audits did; check the **"Do not"** list; check every task's *Discharges* line is
  honoured and nothing is claimed twice; **walk the carry-forward cross-check table below row by
  row** and mark each item done (with the task and log entry) or explicitly not applicable.
- Apply the M3-Q decisions and the audit's corrections to the docs with **marked notes**
  ("(corrected/added <date>, M3 task 8)"), no rewrites: coding **§3.4** (the derived store's
  additions — `two_F_chain`, `DGDt_semilag_o5`, `vertical_factorised`, `lap2_b`, `front_width`,
  `valid`, `front`, `time_mid`, the `(time, j, i)` layout); coding **§5** (`test_inputs.py`,
  `test_vertical.py`, `test_budget.py`, `test_figures.py` beside `test_stats.py`); coding
  **§4.6-§4.8** where signatures grew; coding **§4.9 / M1 task 6's "PASS"** (hour 0-1; 64/71 over
  the window, the gate's definition unchanged); coding **§8**'s "loaded only `k = 0..2`" (stored,
  not fetched); this prompt's acceptance 4 ("feature-level bootstrap intervals" — the block is
  spatial + hour here, M4's are features; planning §11); planning **§4 / §2.3 / §5.3** stale
  numbers — **yes, decided 2026-10-07, M3-Q8 (a): marked notes, the planning doc's own
  convention, nothing deleted**, for §4's "11 of the needed stores … 61 new, ~33 GB at ~539 MB"
  (72/72 exist; 306 MB compressed per hour, 22 GB), §2.3 / §4's `oceQsw` "noon-peaking term"
  (true at the 6-hourly scale only), §4's "different readers … genuine cross-check"
  (bit-identical at `k = 0`) and §5.3's displacement max 2.09-2.1 (hour-0, analysis-domain;
  window ocean max 4.05, 2.27 on the mask); coding §6 M3 **closed** if it is.
- **List what is carried to M4 / M5**: the per-pair closure JSONs and derived stores M4
  reconciles against (coding §6 M4 criterion 4, on the advected pixel set); the slopes M4's
  Figure 9 compares to; the `fronts` fixes and the caller-side NaN recipe (M1 task 7a; prompt 5
  still names the non-existent `build.tile_find` path); captions and the one-command harness
  (M5); the scale-up numbers for a longer window; module sizes.
- **Closure decision.** If every criterion passes, mark **M3 closed** here (Status) and in coding
  §6 M3. If criterion 1 fails, **M3 is not closed**: record every other criterion anyway, write
  the methodological result (planning §12) plainly, and note that nothing downstream starts
  (coding §7: everything after M3 is blocked on closure). A failure here is a publishable
  methodological result, not an embarrassment.

*Discharges:* closes criteria 1-5 formally; the carry-forward table.

### 9. Slides

A small M3 acceptance deck, with the same rules as M1's and M2's (`deck/`, python-pptx; a figures
script `deck/make_m3_figs.py` plus a builder `deck/build_m3_deck.py` with helpers from
`build_m2_deck.py`, `MIN_PT = 20`; **no text below 20 pt**, checked programmatically with
`check_m1_deck.py`; rendered with `soffice --headless` + `pdftoppm` to the scratchpad and **every
page inspected**). Title; contents; M3 in one slide (the budget equation and the verdict); one
slide per task; "Carried to M4 / M5"; a glossary; "this deck". Every number from the M3 log
entries only; where a figure's panel labels would be below ~20 pt equivalent, re-plot from the
JSON summaries at slide size rather than crop (the M2 precedent). Log the work in
`deck/README.md` (table row + work log) and update the Status here.

### M3 carry-forward cross-check

Every item of M2 task 6's "Carried forward to M3" list and its open items, and every M1
carry-forward item, mapped to the M3 task that handles it. **Nothing is dropped**; items needing
no M3 action say so. Task 8 walks this table and marks each row.

| # | Source | Item | M3 task | Note |
|---|---|---|---|---|
| C1 | M2 task 6, item 1 | The two stores and how they merge (plain `xr.merge` fails on `Theta`/`Salt`/`W`; rename or subset) | **1** | **rename** (decided 2026-10-07, M3-Q6); `inputs.open_inputs` is the one place |
| C2 | M2 task 6, item 2 | Fluxes already downward-positive; `surface_flux_term` must **not** re-negate; `load_chunk_levels` refuses raw `oceQsw > +1` | **1, 2** | guard on `sign_convention` and `oceQsw >= 0` in `inputs`; §4.6 marked paragraph |
| C3 | M2 task 6, item 3 | `W.isel(k_l=1)` is the cell-base velocity; `W(k_l=0) = dEta/dt` | **1, 2** | `inputs.W_k1`; a 3-D `W` raises in `vertical_term` |
| C4 | M2 task 6, item 4 | 6-hourly linearly interpolated forcing; shortwave a triangle at 13 LST; Figure 6 phased on the ML minimum; deepening 13 → 12 → 8 m a wind trend | **2, 6, 7** | `forcing_note` → term attrs (2); composites by local hour (6); Figure 6 caption (7) |
| C5 | M2 task 6, item 5 | Displacement envelope (ocean max 4.05 outside the mask; 2.27 on `mask_analysis`; front pixels p99 1.30-1.79) | **1, 4, 6** | the reason for `isfinite` at `L = 8` (1); `n_lost` logged (4); V4's bar quoted at the tail (6) |
| C6 | M2 task 6, item 6 | `edge_cells = 7` kept; `isfinite(DGDt) & mask_analysis` required at `L = 8`; `edge_cells = 13` sensitivity row | **1, 3, 4, 6** | `inputs.valid` (1, 3); counts per pair (4); the 13-cell mask at analysis time (6) |
| C7 | M2 task 6, item 7 | Baseline stability 0.972 ± 0.020 (0.902-0.997; 64/71), 0.987 ± 0.005 trimmed, `edge_cells = 13` row 0.980 ± 0.012; trimmed and orthogonal beside the OLS gate; window spread as temporal systematic; day-3 hours 07-04 14-20 UTC separate | **5, 6, 7** | `slope_trimmed`, `slope_report` (5); reporting (6); Figure 2's third band and marker (7) |
| C8 | M2 task 6, item 8 | Scale-up numbers; `caffeinate -i -s` for long jobs; server-side subset option | **4** | the anti-stall rules; the sweep extends the table |
| C9 | M2 task 6, item 9 | Module-size flags: `osn_tiles.py` 556, `vertical.py` 524, `m2_qa.py` 638, `validate.py` 1,087 | **2, 7, 8** | decided 2026-10-07, M3-Q7: `vertical.py` split, reader → `chunk_store.py` (2); `figures.py` split if needed (7); `osn_tiles`/`validate` **left whole** (M1-Q8 precedent; 8 records the sizes) |
| C10 | M2 task 6, item 10 | Note to Lauren (`note_to_lauren_flux_signs.md`) pending JXP's forwarding; stale llc-repo docs are hers | **none** | no M3 action; task 8 records its status |
| C11a | M2 task 6, item 11 | Planning §4 "11 of the needed stores … ~33 GB at ~539 MB" stale / uncompressed | **8** | marked note — yes (decided 2026-10-07, M3-Q8) |
| C11b | M2 task 6, item 11 | Planning §2.3 / §4 `oceQsw` "the noon-peaking term" — true at the 6-hourly scale only | **7, 8** | Figure 6 caption (7); marked note — yes (decided 2026-10-07, M3-Q8) (8) |
| C11c | M2 task 6, item 11 | Planning §4 "different readers … a genuine cross-check" — sources bit-identical at `k = 0` | **1, 8** | the identity is asserted as an input invariant (1); marked note — yes (decided 2026-10-07, M3-Q8) (8) |
| C11d | M2 task 6, item 11 | Planning §2.3 / coding §4.6 written against the documented `+=down` sign; resolved by storing downward-positive | **2** | coding §4.6 already marked; `surface_flux_term` reads the documented convention |
| C11e | M2 task 6, item 11 | Prompt 3 task 1 "record it and move on" vs "no gaps" — resolved as stop-at-gap | **none** | record only |
| C11f | M2 task 6, item 11 | Coding §4.9 / M1 task 6 "PASS" on hour 0-1 — 7/71 pairs fail the OLS gate over the window | **6, 8** | reported as the temporal systematic (6); coding §4.9 marked (8); the gate's definition unchanged |
| C11g | M2 task 6, item 11 | Planning §5.3 / M1 task 3 displacement max 2.09-2.1 is hour-0 / analysis-domain; window ocean max 4.05 | **8** | marked note — yes (decided 2026-10-07, M3-Q8); the envelope (C5) supersedes |
| C11h | M2 task 6, item 11 | M2-Q3 "about a minute per pair" — ~2 s | **none** | Q&A record |
| C11i | M2 task 6, item 11 | M1 task 6 "edge reach at `L = 8` exactly 7, no slack" — zero-displacement statement; 7-34 cells exceed it | **1** | `isfinite` at `L = 8`; `edge_cells` must not shrink; log record, no doc edit |
| C11j | M2 task 6, item 11 | `test_load_chunk_levels.py` network docstring "~5 min" — link-speed statement | **none** | M2 code; re-time before a longer window (open item 4) |
| C11k | M2 task 6, item 11 | Task 5 log "so `xr.merge` works" — superseded by item 1 | **1** | = C1 |
| O1 | M2 task 6, open item 1 | Planning §4 / §2.3 marked notes or pre-execution record — JXP's call | **8** | **M3-Q8: marked notes** (decided 2026-10-07) |
| O2 | M2 task 6, open item 2 | The M3 prompt should state the merge recipe, no-negation rule, `isfinite` at `L = 8`, trimmed / orthogonal / 13-cell reporting, Figure 6 caveats | **this restructure** | done 2026-10-07 (M2 task 8): tasks 1, 2, 3, 6, 7 |
| O3 | M2 task 6, open item 3 | Split `vertical.py` / `osn_tiles.py` | **2, 8** | **M3-Q7** (decided 2026-10-07): `vertical.py` yes (2); `osn_tiles.py` no (8 records) |
| O4 | M2 task 6, open item 4 | Network docstring "~5 min" and the 0.55 MB/s scale-up assumption; re-time before a longer window | **4** | the sweep's report re-times the local side; the chunk link is re-timed only if a longer window is planned (none in M3) |
| O5 | M2 task 6, open item 5 | Note to Lauren pending | **none** | = C10 |
| O6 | M2 task 6, open item 6 | M2 code already in `HEAD`; only docs uncommitted | **none** | record |
| M1a | M1-Q1 (task 6, 7) | Both forms of `F`: `discrete` primary, `chain` alongside, the difference a stated systematic (~0.79x on front pixels; V3 0.981 vs 0.791) | **3, 6, 7** | `two_F`, `two_F_chain` (3); reporting (6); Figures 2, 3 (7) |
| M1b | M1-Q2 / task 6b | V3b band 0.954-1.003 (0.975 ± 0.025) as the model-advection systematic; **no upward correction** for the 0.80-0.85x Jacobian attenuation (−0.006 ± 0.025); per-width shortfall −2 / −4 / −11 % at 2 / 1.5 / 1 dx subtracted first; chain form's own baseline 0.79 | **6, 7** | M3-Q5 width proxy `front_width = 2 sqrt(G/|lap G|)`, binned (decided 2026-10-07; computed in 3, binned in 6); `slope_report` bands (5) |
| M1c | M1-Q4 | Figure 2's baseline drawn at 0.981 with its band [0.970, 0.994], V3b band beside | **5, 7** | `slope_report` default; `fig02` |
| M1d | M1-Q6 | Order 3 default; **order 5 as a sensitivity**; V4's bar 0.28-1.0 % of `G`/h (order 3) quoted with the front width | **3, 6** | `DGDt_semilag_o5` (3); reporting (6) |
| M1e | M1 task 4 | `tau_delta` must be passed (flux form alone overstates 2.2x); budget field `subfilter = 2 * subfilter_term` | **3** | mandatory in `compute_budget`; test pins the difference |
| M1f | M1 task 4 / M1-Q8 (c) | Subfilter term 0.31 / 0.50 / 0.70 of `Fbar` at `L = 2 / 4 / 8`, anti-correlated ~−0.6, growing with `L`; explicit term ≡ 0 at `L = 0` (the `L = 0` column holds the whole numerical term) | **3, 6, 7** | hour-0 smoke reproduces (3); (c) over 71 pairs (6); Figure 3 / 3b (7) |
| M1g | M1 task 6, flag 5 | Ratio estimator ill-conditioned with both signs of `2F`; quote by sign | **5, 6** | `ratio_estimator(split_sign=True)` |
| M1h | M1 task 6 | Tile-edge reach of the default `F` at `L = 8` is exactly 7 = `edge_cells`; it must not shrink | **1, 6** | mask unchanged; 13 only as the wider sensitivity |
| M1i | M1 task 6, flag 10 | Call `operators.frontogenesis` and `semilag` with their defaults; "`F = frontogenesis_tendency`" means `form='chain'` | **1, 3** | the Tasks preamble |
| M1j | M1 task 6, flag 6 | The llc CI excludes 1 (0.970-0.994): draw 0.981 with its band, not 1 | **7** | = M1c |
| M1k | M1 task 6b, flags 1-2 | The "open systematic" (attenuation) closed as a recorded bias; planning §2.3's OS7MP `kappa_num` an over-estimate by an order of magnitude as a caveat on the *resolved* slope (+0.009 ± 0.03 in one hour; DST3 −0.13 shows a diffusive scheme) | **6** | (e) interpretation; `kappa_num` at the feature scale |
| M1l | M1 task 3 | Real hours: Eulerian vs semi-Lagrangian corr 0.74, slope 0.73 at `L = 0` | **6** | (b)'s known starting point; M3-Q2 (decided 2026-10-07: gated at `L >= 2`, slope 0.85-1.15 / corr ≥ 0.90; `L = 0` interpreted) |
| M1m | M0 task 3 / planning §2.2 | Vertical term bracket 0.4-14 % rms (order-one pointwise at fronts), ~30 % of `F` by day, ~0 at night | **2, 3** | the diurnal check on the smoke hours |
| M1n | M1-Q8 (a, b) | `tile330_masks.nc` stays git-ignored; `validate.py` not split | **none / 8** | (a) no action; (b) stands — `validate.py` left whole (decided 2026-10-07, M3-Q7) |
| M1o | M1 task 7, open items 1-3 | `fronts` bugs and the xfails; prompt 5's non-existent `build.tile_find` path; the caller-side NaN recipe | **none** | M4's; task 8 lists them in "carried to M4 / M5" |
| M1p | M1 task 7, open item 6 | The M0 planning deck still asserts two overturned claims | **none** | record |
| M1q | coding §6 M3 "Carried from M1" | The paragraph's items (both forms; baseline and band; no upward correction; per-width shortfall; order 5; V4's bar) | **3, 6** | all covered by M1a-M1d above |

---

## Acceptance criteria

1. **Closure** to a stated tolerance, with `vertical` and `surface_flux` **measured** from the
   chunk store rather than assumed. *(decided 2026-10-07, M3-Q1 / M3-Q9: the tolerance is, on
   `front & valid`, per `L`, `rms(residual)/rms(measured) <= 0.5` **and** the residual's OLS
   slope on `2F` within ±0.10; judged at `L >= 2`, with `L = 0` reported and interpreted — task 6.)*
2. Semi-Lagrangian and Eulerian estimates agree within a stated tolerance. *(decided 2026-10-07,
   M3-Q2: OLS slope of `DGDt_euler` on `DGDt_semilag` within 0.85-1.15 and corr ≥ 0.90 on front
   pixels at `L >= 2`; `L = 0` reported and interpreted, not gated — task 6.)*
3. The filter sweep is interpretable — `tau` explicit, and the budget closing at each `L`.
   *(decided 2026-10-07, M3-Q9: "each `L`" is judged at `L >= 2`; at `L = 0` the explicit
   subfilter term is ≡ 0 by construction, so that column is reported and interpreted, not gated.)*
4. Every slope quoted against the M1 baseline (0.981 [0.970, 0.994], with the V3b systematic band
   0.954-1.003; both forms of `F`, M1-Q1; order 5 as a sensitivity, M1-Q6 — decided 2026-09-30),
   with feature-level bootstrap intervals. *(corrected 2026-10-07, M2 task 8: at this milestone
   the bootstrap block is a **contiguous spatial region plus hour** — the `stats.py` paragraph
   above and planning §11; feature-level intervals are M4's, where front objects exist.
   "Feature-level" here reads as "block-level". Also quoted beside the baseline: the temporal
   systematic 0.972 ± 0.020 (0.987 ± 0.005 trimmed) and the `edge_cells = 13` row 0.980 ± 0.012,
   M2 task 6 item 7, with the trimmed and orthogonal fits beside the OLS gate.)*
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

## Q&A

### Claude, 2026-10-07 (before task 1)

Numbered M3-Qn, so they do not collide with the planning Q-numbers in `frontogenesis_prompts.md`
or with M1-Qn / M2-Qn. Written by M2 task 8 from the docs and the M1-M2 logs; each has a
recommendation. **M3-Q6 is needed before task 1, M3-Q7 before task 2, M3-Q3 and M3-Q4 before
task 3; M3-Q1, M3-Q2, M3-Q5 and M3-Q9 are pre-declared in task 6 and so must be answered before
it; M3-Q8 before task 8.**

*(2026-10-07)* **All nine answered by JXP the same day; every answer accepts the recommendation.**
Each decision is written into the task that uses it, marked "(decided 2026-10-07, M3-Qn)", and
summarised in the Status paragraph; an *Applied* pointer under each answer says where (log entry
"2026-10-07 — M3 Q&A applied to prompt 4").

##### Questions

**M3-Q1 — The closure tolerance (criterion 1).** Planning §12 says a null result is "the residual
comparable to `2F` at all filter scales", and nothing in the docs states a number. The residual
legitimately contains the model's OS7MP numerics and interior KPP (planning §2.3), so a tight
tolerance would fail by construction at `L = 0`; and M1 task 4's synthetic coarse-grained budget
closed to 4-9 % rms at `dt = 3600` from the midpoint-field time discretisation alone (2-3 % at
`dt = 900`), so a few per cent is the floor even with perfect physics. Proposal, pre-declared on
`front & valid`, per `L`:
- (a) **`rms(residual) / rms(measured) <= 0.5`** (equivalently the explained fraction ≥ 0.75), and
- (b) the residual's OLS slope on `2F` within **±0.10** (no term missing in proportion to `2F`),
- with the five-term rms table and the with/without-chunk-terms comparison always reported, and
  the verdict stated per `L` (see M3-Q9 for which `L` the gate is judged at).
Alternatives: a tighter 0.3 on (a) (M1's synthetic closure suggests it is reachable only if the
numerical term is small on the resolved `G`, which V3b hints — +0.009 ± 0.03 — but which real
data has not shown); or (a) alone. I recommend (a) + (b) at 0.5 / 0.10.

> **JXP:**  Go with your recommendation.

*Applied (2026-10-07):* → task 6 (pre-declaration, (a), verdict); acceptance criterion 1; Status.

**M3-Q2 — The semi-Lagrangian / Eulerian agreement tolerance (criterion 2).** The docs say
"within a stated tolerance". The known starting point is poor: M1 task 3 measured **corr 0.74,
slope 0.73** between the two estimates on the real hour at `L = 0` on `mask_analysis` — the
Eulerian split's two large, nearly cancelling terms (planning §5.3). Proposal: OLS slope of
`DGDt_euler` on `DGDt_semilag` within **0.85-1.15 and corr ≥ 0.90 on front pixels at `L >= 2`**,
with `L = 0` reported and interpreted rather than gated. Alternative: gate at every `L`, which
the hour-0 number suggests fails at `L = 0` for a reason that is not a bug. I recommend the
proposal.

> **JXP:**  Go with your recommendation.

*Applied (2026-10-07):* → task 6 (pre-declaration, (b)); acceptance criterion 2; table row M1l; Status.

**M3-Q3 — Which `L_cells` to sweep.** Coding §1.2 fixes `{0, 2, 4, 8}`. M1 task 4 found the
subfilter term growing with `L` (0.31 / 0.50 / 0.70 of `Fbar`), and at `L = 8` the budget loses
7-34 analysis cells per pair and `F`'s NaN reach is exactly the edge margin. Options: (a) keep
`{0, 2, 4, 8}` as the contract; (b) add `L = 1` and/or `L = 16` (16 would need `edge_cells`
and `halo_cells` ≥ 11, i.e. a new mask, so it is not free); (c) drop 8. The sweep costs ~25 % per
extra `L`. I recommend **(a)**; if the pilot shows time to spare, `L = 1` as an extra column for
Figure 3 only (it needs no new mask).

> **JXP:** Use (a)

*Applied (2026-10-07):* → task 4 (the sweep set; the optional `L = 1` after the pilot); task 7 Figure 3; "Runs"; Status.

**M3-Q4 — The front-pixel percentile.** V3 (M1 task 6), M2 task 3 and M2-Q7 all used
**`G_mid >= p90`** on `mask_analysis & finite` (n 26,293 of 262,925), and the baseline 0.981 and
its temporal band were measured on that pool. Changing it breaks comparability; keeping it keeps
the OLS gate's known leverage sensitivity (top 1 % of |2F| moves day-3 pairs by 0.03-0.08).
Options: (a) **p90, with p80 and p95 as a sensitivity** in task 6 and the trimmed estimator
beside the OLS; (b) p95 as primary. I recommend **(a)**.

> **JXP:** Use (a)

*Applied (2026-10-07):* → task 3 (`front_pct=90.0`, `front`); task 6 (pre-declaration, (d) pools); the `stats.py` paragraph; Status.

**M3-Q5 — A per-pixel front-width proxy, for "slope per front width".** M1-Q2 asks for the slope
per front width with V3b's shortfall (−2 / −4 / −11 % at 2 / 1.5 / 1 dx) subtracted, but V3b's
widths were synthetic; real pixels have no label. Options: (a) a local proxy from the fields
already in the derived store — e.g. **`ell = 2 sqrt(G / |lap G|)`** evaluated on front pixels
(exact for `G ∝ sech^4(x/ell)` at the maximum), binned into `{<= 1, 1-1.5, 1.5-2, 2-3, 3-4, > 4}`
dx; (b) use the filter sweep as the width axis (at `L` the resolved front width is ≥ L); (c)
defer per-width reporting to M4, where `curtains.path_metrics` gives a width per front object.
I recommend **(a)** stored as `front_width` in the derived product, validated on `synthetic.py`'s
tanh fronts in `test_budget.py`, with (b) as the cross-check and (c) noted for M4.

> **JXP:**  Go with your recommendations

*Applied (2026-10-07):* → task 3 (`front_width` field and its `test_budget.py` check); task 6 (pre-declaration, (d) bins); table row M1b; coding §3.4 note; Status.

**M3-Q6 — Merge strategy for the two stores.** M2 task 6 verified both recipes: **rename**
(`Theta_k`, `Salt_k`, `W_k`; 16 vars, keeps `k`, `k_l` dims and all three levels) or **subset**
(fluxes + `drF` + `W.isel(k_l=1)` as `W_k1`; 14 vars, flat 2-D). Rename keeps `k = 2` for a
second-order `b_z` sensitivity and makes the `k = 0` bit-identity an assertable invariant on
every open; subset is simpler to use and cannot pass a 3-D `W` by mistake. I recommend
**rename**, with `inputs.W_k1` / `inputs.fluxes` accessors so the physics never indexes the store.

> **JXP:**  Go with your recommendation.

*Applied (2026-10-07):* → task 1 (`open_inputs`, the `needs_grid` test's 16 vars); table row C1; Status.

**M3-Q7 — Module splits (coding §1.3's ~400-line cap).** `vertical.py` is 524 lines of
chunk-store reader before any physics, `osn_tiles.py` 556, `validate.py` 1,087 (M1-Q8 (b) left it
whole). Options for `vertical.py`: (a) **move the reader to `chunk_store.py`**, keep
`vertical.load_chunk_levels` as a re-export, repoint `test_load_chunk_levels.py`'s monkeypatch
target (one line) — the physics then lands in a near-empty `vertical.py` as coding §4.6 names it;
(b) leave the reader and put the physics in `vertical.py` anyway (~800 lines); (c) put the
physics in a new `vertical_terms.py` (contradicts §4.6's module name). For `osn_tiles.py` and
`validate.py`: leave (the M1-Q8 precedent; M5 may consolidate). I recommend **(a)** and leave.

> **JXP:**  Go with your recommendation.

*Applied (2026-10-07):* → task 2 (module split first); table rows C9, O3, M1n; coding §4.6 note; Status.

**M3-Q8 — Marked notes in the planning doc for the stale numbers (M2 task 6, open item 1).**
Planning §4's "11 of the needed stores … 61 new, ~33 GB at ~539 MB per timestep" (72/72 exist;
306 MB compressed per hour, 22 GB), §2.3 / §4's `oceQsw` "noon-peaking term" (true at the
6-hourly scale only), §4's "different readers … genuine cross-check" (bit-identical at `k = 0`),
and §5.3's displacement max 2.09-2.1 (hour-0, analysis-domain; window ocean max 4.05, 2.27 on
the mask). Options: (a) **marked notes** in task 8, the doc's own convention, nothing deleted;
(b) leave the planning doc as the pre-execution record and let the log carry the corrections. I
recommend **(a)**: the planning doc is what a reader opens first.

> **JXP:** Go with your recommendation.

*Applied (2026-10-07):* → task 8 (the marked-notes list); table rows C11a, C11b, C11c, C11g, O1; Status. The planning doc itself is edited by task 8, not now.

**M3-Q9 — At which `L` is the gate judged?** Criterion 3 says "the budget closing at each `L`",
but planning §5.4 (corrected, M1 task 4) says the explicit subfilter term is identically zero at
`L = 0`, so the `L = 0` residual holds the model's whole numerical term by construction — and
planning §12's null criterion is the residual comparable to `2F` **at all** filter scales. Options:
(a) **judge criterion 1 at `L >= 2`**, report `L = 0` and interpret it (its residual is the
numerics-plus-KPP estimate Figure 2b is about); (b) judge at every `L` including 0; (c) judge at
`L = 4` only. A failure at `L = 0` alone would then not be the planning §12 null; a failure at
every `L` would. I recommend **(a)**.

> **JXP:** Go with your recommendation.

*Applied (2026-10-07):* → task 6 (pre-declaration, (a), verdict); acceptance criteria 1 and 3; Status.

### Claude, 2026-10-07 (during task 2)

Three judgement calls met while writing `vertical.py` (log entry "Execution prompt 4, task 2").
Each was implemented as recommended (all three are one-line constants or a sign, trivially
undone); the real-data smoke numbers in the log entry were computed with these choices.

**M3-Q10 — `f_sw`: Jerlov type IA (0.521), not type I (0.56).** Task 2 assumed MITgcm's
`SWFRAC` with "Jerlov type I by default" (`0.58 exp(z/0.35) + 0.42 exp(z/23)`, 0.565 absorbed in
1 m). The LLC4320 `code/CPP_OPTIONS.h` defines `SHORTWAVE_HEATING`, `data.kpp` / `data.exf` /
`data.pkg` set no penetration option, and `model/src/swfrac.F` at checkpoint65v **hard-codes
`jwtype = 2` = type IA** (`0.62 exp(z/0.6) + 0.38 exp(z/20)`), giving **`f_sw = 0.521`** for
`drF[0] = 1.0 m`. Options: (a) **0.521** (the model's own profile; `vertical.F_SW`,
`sw_fraction_absorbed(dz, jwtype=2)`); (b) 0.565 as the prompt wrote. The difference is 8 % of
the shortwave part, which the smoke shows is only 0.6-18 % of the surface-flux term, so this
is < 2 % of the term. I recommend **(a)**, with the type exposed as a keyword so (b) is a
sensitivity.

> **JXP:** Use your recommendation.

**M3-Q11 — The vertical tendency's sign and denominator.** Prompt 4 task 2, coding §4.6 and
planning §2.2 ("`-w_base (b_base - b)/drF`") write `T_v = −W_k1 (b_k1 − b)/drF[0]`. Planning
§2.2's own budget equation, `D_h b/Dt = B − w b_z` with the factorised term `−b_z (w_x b_x +
w_y b_y)`, requires `T_v = −W_k1 b_z`; with `b_z = (b − b_k1)/dz` (the `b_z` contract) that is
`−W_k1 (b − b_k1)/dz` — **the opposite sign** (an upwelling `W_k1 > 0` of denser water,
`b_k1 > b` in code `b`, must raise the top-cell `b`), and `dz = Z[0] − Z[1] = 1.07 m` rather
than `drF[0] = 1.0`. As literally written, the task's own test "equals the factorised form when
`b_k1 − b` is uniform" cannot pass (the two differ in sign and by 1.07). Options: (a)
**`T_v = −W_k1 (b − b_k1)/dz`**, `dz = (drF[0] + drF[1])/2` (derived from the `drF` the
signature takes; the factorised identity then holds exactly and the dropped term is exactly
`−w grad(b_z) . grad b`); (b) the same sign with `drF[0]` (7 % larger, the identity holds to 7 %);
(c) the model's face-value form `W_k1 (b_face − b)/drF[0]` with a linear face value, which is
0.535x (a) and would need OS7MP's actual near-surface reconstruction to be more than a guess. I
recommend **(a)** (implemented; the sign error is marked in coding §4.6 and planning §2.2 as a
plain correction, the `dz` choice is this question).

> **JXP:** Use your recommendation.

**M3-Q12 — `rhoConst` (1027.5) or `rho0` (1000) in the flux-to-tendency conversion.** The
task says `dT/dt = Q_top/(rho0 c_p drF[0])` with "`rho0 = 1000` as in `buoyancy`". The model
converts its fluxes with `mass2rUnit = 1/rhoConst`, and the `data` namelist has `rhonil = 1027.5`
with `rhoConst` absent (so `rhoConst = rhoNil = 1027.5`; `HeatCapacity_Cp` absent → 3994;
`convertFW2Salt = −1` → local salinity, `useRealFreshWaterFlux = .TRUE.`). Options: (a)
**`rhoConst = 1027.5`** for `dT/dt` and `dS/dt` (what the model does; `rho0 = 1000` stays only in
the `g/rho0` of `b = g sigma0/rho0`); (b) `rho0 = 1000` throughout as written. The difference is a
uniform 2.7 % of the surface-flux term. I recommend **(a)**.

> **JXP:** Use your recommendation.

## Log

Append to `frontogenesis_prompts.md` under `## Logs`, one entry per task, titled
`### <date> — Execution prompt 4, task N: <title> (<model>)` *(added 2026-10-07, M2 task 8: the
M0-M2 convention)*. Across the milestone, record:
the closure residual and tolerance; the slope estimators with intervals, against the baseline;
whether semi-Lagrangian and Eulerian agreed; the relative sizes of all five budget terms; and —
plainly — whether closure passed. **A failure here is a publishable methodological result, not
an embarrassment.** *(added 2026-10-07, M2 task 8)* Also: the pre-declared tolerances and pools
(task 6) before any number; the sweep's wall times and volumes (task 4); every change made to
pass, if any, as V3's task 6 logged them; and the walk through the carry-forward table (task 8).
