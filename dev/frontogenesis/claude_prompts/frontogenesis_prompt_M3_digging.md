# Frontogenesis execution prompt M3-digging — the vet report's tests

**Milestone:** an addendum to M3 (`frontogenesis_coding.md` §6), run **after** M4 and before
M5's report is written. It does not reopen the M3 gate.
**Prerequisites:** M3 judged (criterion 1 failed at every `L`, prompt 4 task 8); the interim
report (`report/frontogenesis_M3_interim.md`, 2026-10-12); the vet-idea report
(`claude_outputs/frontogenesis_M3_vet_idea_report.md`, 2026-10-26); M4 complete (prompt 5);
JXP's answers **M3-Q27..Q30** (prompt 4 `## Q&A` → `### Digging`, 2026-10-31).
**Goal:** give the M3 null result the one thing the vet report says it lacks — an
**independent noise floor** — and answer the report's five ranked changes with runs rather
than argument, on the stores already on disk, without touching the M3 product or its verdict.

**Written 2026-10-31 (Fable 5.1), prompt 4 task 11 (ii).** Every feasibility claim below was
checked against the stores and the source on 2026-10-31 (prompt 4 task 11 (i), logged); the
wall times are from `data/m3_run.log` and `data/m3_closure_summary.json`, not estimates, except
where a line says "estimate". Five questions, **M3D-Q1..Q5**, are in `## Q&A` at the end.
~~**Tasks 1-3 can start now on the recommendations; task 4's closure is not *interpreted* until
M3D-Q1 is answered (the run itself can start); task 5 waits on M3D-Q2.**~~ *(revised
2026-10-31, before task 1: **all five answered** — M3D-Q1 (a) the three-way reading, Q2 (a) tile
329, Q3 (a) both velocity columns, Q4 (a) a separate checker, Q5 (a) the same 72 h — so every
task can run as written; the decisions are marked where each task cites its question.)* The
Status paragraph convention of prompts 4-6 applies: one per task, added here when it is done.

**Status 2026-10-31 — task 1 done** (Fable 5.1). `budget.derived_path(L, tag=)`;
`inputs.open_inputs(chunk=None)` with the two invariants skipped *and recorded*; the stores'
scalar `face` read and required to agree (grid and masks included), `TILE_FACE` a reference
only; `utc_offset_for_lon` to the nearest half hour (tile 330 stays at −8.0 exactly);
`m3_run.py --raw/--chunk none/--grid/--masks/--tag`, `--clobber` refused on the product;
`m3_closure.py --tag/--grid/--masks`, `closure()` tolerant of absent chunk terms (the catch-all
is then the residual, said so); `series_verify` verifies a chunk-less store. **Regression:**
pair 0 at `L = 0` through `--tag` equals the product on all 25 variables (NaN-aware, same
dtypes); the closure re-run on the product matches `m3_closure_summary.json` on **10,895 of
10,895** numeric leaves; `check_report_numbers.py` OK; nothing in `data/` changed. Suite **300
passed + 3 xfails** (was 290), 2 deselected, 270 s. Found along the way: `m3_run --no-chunk` had
never been runnable (`hour_pair` raised on the stripped dataset) — fixed by the same change.
Log entry "Execution prompt M3-digging, task 1".

---

## What JXP decided (prompt 4, M3-Q27..Q30)

| Q | Decision | Consequence here |
|---|---|---|
| M3-Q27 | **(c)** both: the cheap set first, then the tracer | tasks 2-3, then task 4 |
| M3-Q28 | **(a)** a short digging note, `report/frontogenesis_M3_digging.md`, **referenced from the interim report** | task 6; one marked paragraph in the interim's §8 |
| M3-Q29 | **(a)** the tracer starts from the model's own fields at hour 0, the edge margin clamped | task 4's design is fixed |
| M3-Q30 | **(a)** a surface-only second tile from OSN | task 5; the tile is M3D-Q2 |
| M3D-Q1 | **(a)** the three-way tracer reading, 0.50 / 0.25 *(decided 2026-10-31)* | task 4 |
| M3D-Q2 | **(a)** tile 329, west, face 7; 354 the fallback *(decided 2026-10-31)* | task 5 |
| M3D-Q3 | **(a)** both velocity columns, linear and piecewise constant *(decided 2026-10-31)* | task 4 |
| M3D-Q4 | **(a)** a separate `check_digging_numbers.py` *(decided 2026-10-31)* | task 6 |
| M3D-Q5 | **(a)** the same 72 h window for the second tile *(decided 2026-10-31)* | task 5 |

## What exists, and what each test needs

Nothing below is new physics. The physics modules are M3's; this prompt adds **drivers**.

| Test (vet §6) | Reuses | Needs | Data |
|---|---|---|---|
| 1a shuffled / lagged / rolled controls | `m3_closure.load_pool`, `closure`, `stats.block_bootstrap` | `py/m3_controls.py` | the four derived stores |
| 1b offline tracer, full pipeline | `fvadvect.fv_step_arrays` (OS7MP, `dt_sub = 100 s`), `m3_run --no-chunk`, `m3_closure` | `py/m3_tracer.py`, a `tag` on the store paths (task 1) | the raw store's `U`, `V`, `Theta`, `Salt` |
| 3 `V` from the second cell | `vertical.vertical_term`, `b_z(order=2)`, the chunk store's `k = 2`, `k_l = 2` | an `inputs.W_k2` accessor, a diagnostic script | the chunk store |
| 4 split `S` | `vertical.surface_flux_term` (zero the other fluxes) | the same script as 3 | the chunk store |
| 5a other selections; 5b 12-24 h blocks | `load_pool`'s selection dict; `stats.space_time_block_ids(hours=)` | lines in `m3_controls.py` | the derived stores |
| 5a case re-seeded | `m3_case.run` | two parameters | the `L = 0` store |
| 5c second tile, surface-only | `osn_tiles.tile_spec`, `load_grid`, `pull_series`, `masking.build_masks`, `m3_run --no-chunk` | `py/m3_tile2.py`; the `tag` and a `face` generalisation (task 1) | a 72-h OSN pull, ~1.3 GB |

Fixed facts the tasks rely on: `tile330_derived_L{L}.zarr` has 71 pairs × 25 fields including
`DGDt_euler`, `DGDt_semilag_o5`, `surface_flux_kpp`, `sigma_mag`, `KPPhbl`, `lap2_b`, `valid`,
`front`; the chunk store has `Theta`/`Salt` at `k = 0, 1, 2` and `W` at `k_l = 0, 1, 2`
(`drF` 1.00 / 1.14 / 1.30 m); `rect_ij_to_tile` resolves offline (checked: tile **329** west
= face 7, `j` 3600:4320, `i` 2880:3600, lon −143 to −128; tile **354** north = face 10, `j`
0:720, `i` 2160:2880; tile 330 itself spans lon −128 to −113, lat 26.7 to 38.3). The
workstation: 24 cores, 755 GB RAM, 8.4 TB free on `/mnt/tank`. Numbers in the record that the
tests are compared with: residual / measured **0.813 / 0.841 / 0.898 / 1.042** at `L = 0 / 2 /
4 / 8`; the two-term catch-all is lower by the factor `chunk_terms_help.rms_ratio` = 1.10 /
1.16 / 1.26 / 1.52; `rms_two_F / rms_measured` 0.52 (`L = 0`) and 0.65 (`L = 8`);
`measured_on_2F.corr` +0.70 / +0.67 / +0.62 / +0.54; `residual.slope_on_2F` 0.44 / 0.45 /
0.46 / 0.53; the Eulerian − semi-Lagrangian gap 0.672 / 0.408 / 0.252 / 0.183; order 5 − order
3 0.108 / 0.053 / 0.027 / 0.021; the case study's control ×0.89 against the selected ×0.19.

## Environment and house rules (unchanged from prompts 4-6)

- The interpreter is **`/home/xavier/miniconda3/envs/frontogenesis/bin/python`**, by absolute
  path; `conda run -n frontogenesis` silently resolves to another env on this machine.
- `py/` is not a package: run scripts by path from `dev/frontogenesis/`; modules `sys.path`
  their own directory. Long runs go detached (`nohup … > ../data/<name>.nohup 2>&1 &` inside
  `tmux`), resumable at pair granularity, with a `.log`, a `_done.json` and an ETA line, as
  `m3_run.py` does.
- `data/` is git-ignored by design; JSON summaries are products but never committed. PNGs are
  git-ignored and **force-added** (`git add -f`); say so when a figure is drawn.
- Tests: `pytest py/tests` from `dev/frontogenesis`; offline by default; `needs_grid` tests
  skip without the stores. Suite before this prompt: **290 passed + 3 strict xfails**, 2
  network tests deselected, ~265 s (prompt 6 "What exists"). Every task reports the count.
- One log entry per task in `claude_prompts/frontogenesis_prompts.md` under `## Logs`, titled
  `### <date> — Execution prompt M3-digging, task N: <title> (<model>)`.

---

## Tasks

Execute in this order. Tasks 2 and 3 can run while task 4's advection computes; task 5's
pull can run detached any time after task 1.

### 1. The plumbing — a `tag` on the stores, a chunk-less open, a `face` that is read not assumed

Nothing in tasks 2-5 can run without this, and none of it changes a number. The three
hard-coded facts to generalise, each in **one** place:

1. **Store paths.** `budget.derived_path(L, tag=None)` → `tile330_derived_L{L}.zarr` when
   `tag is None` (the product, unchanged), else `data/{tag}_derived_L{L}.zarr`.
   `m3_run.py` gains `--raw`, `--grid`, `--masks`, `--chunk` (a path, or `none`), `--tag`;
   `--no-chunk` keeps its present meaning and its `noChunk` store name. The closure JSON
   becomes `m3_closure_{tag}_L{L}.json`, the log `m3_run_{tag}.log`, the done-file likewise;
   with no tag every name is what it is today. `m3_closure.py` gains `--tag` →
   `load_pool(L, …, path=)` and `OUT_JSON` → `data/m3_closure_{tag}_summary.json`, the working
   figure `figs/m3_closure_{tag}.png`.
2. **A chunk-less open.** `inputs.open_inputs(chunk=None)` opens the surface store alone: the
   merged-variable check becomes `OSN_VARS`, the k = 0 identity and the flux-sign guard are
   **skipped and recorded** (`ds.attrs['chunk'] = 'absent'`), `hour_pair` follows, and
   `compute_budget` takes its existing `has_chunk=False` path (the vertical and surface-flux
   terms *absent*, the residual a labelled catch-all — rule 1 of `budget.py`). `--chunk none`
   is this; `--no-chunk` remains "open the merge and strip it" so the product's catch-all
   comparison is untouched.
3. **The face.** `inputs.TILE_FACE = 10` is asserted today. Replace the constant with a check
   that the raw store, the grid and the masks agree on their scalar `face` coord, and record
   it in `ds.attrs`. `osn_tiles.ORIENTATION` documents face 10's rotation (`CS = 0, SN = −1`);
   the operators carry `CS`/`SN` from the grid and were tested in both orientations
   (`test_operators`), so nothing else should depend on the face — **task 5 verifies that on
   real data**, this task only removes the assertion. `inputs.local_solar_hour`'s
   `UTC_OFFSET_H` becomes a function of the tile's median `XC` (tile 330's value unchanged to
   the minute).

`m3_closure.closure()` must also **tolerate absent chunk terms**: `terms` and
`term_contribution` over the terms present; `catchall` = the residual itself when `vertical`
and `surface_flux` are absent (with a note saying so); `fig2b`'s `KPPhbl` partials,
`depth_scale_check` and `surface_flux_kpp` sections written as `None` with a reason. The
verdict logic is unchanged.

**Regression — the product is bit-identical through the new flags.** (i) Pair 0 at `L = 0`
recomputed with the defaults (`--tag` unset) into a scratch store equals the product's pair 0
field for field (`np.array_equal` on every variable, NaN-aware); (ii) `m3_closure.py` re-run on
the product writes a summary whose every number the interim quotes is unchanged —
`report/check_report_numbers.py` passes against the *re-run* JSON, then the original JSON is
restored (`data/m3_closure_summary.json` is never overwritten by this prompt; run the re-run
with `--out` to a scratch path); (iii) the suite passes with the same count plus the new
tests (`tests/test_inputs.py`: the chunk-less open and the face check; `tests/test_m3_run.py`:
the tag resolves and the defaults do not move).

Deliverables: the edits above; tests; log entry with the regression results.

### 2. The controls — a number for "no skill", and the pooling checks (vet §3, §6 items 1a, 5a, 5b)

`py/m3_controls.py` → `data/m3_controls_summary.json` + `figs/m3_controls.png`;
`tests/test_m3_controls.py`. Everything is computed from the four derived stores; nothing is
recomputed from the raw stores; the product is read, never written.

**Pre-declared before any number is looked at** (write these into the JSON's `declared` block
as `m3_closure.py` does):

- The real residual ratio is said to show **skill over no-skill** at a given `L` if it lies
  below the **5th percentile** of the permutation distribution *and* below the 24-h-lag
  control by more than the real value's block-bootstrap half-width. This is a reference, not
  a gate; nothing in the M3 verdict changes either way.
- Seeds: `numpy.random.default_rng(0)`; 20 random permutations; spatial rolls of at least 64
  cells (two blocks) in each axis; the lags 24 h and 48 h without wrap-around (pairs `n` and
  `n + lag` both inside 0..70).

**(a) The three controls**, each at every `L` in {0, 2, 4, 8}, on the gate's pool (`front &
valid` of the *measured* pair):

1. *Random pair permutation.* Keep `DGDt_semilag`, `DGDt_euler`, `valid` and `front` of pair
   `n`; take `two_F`, `subfilter`, `vertical`, `surface_flux` from pair `π(n)` at the same
   pixel (require `valid` of both). Recompute the residual, its ratio to measured, the
   explained fraction, the residual's slope and correlation on `2F`, `measured_on_2F`, the
   two-term catch-all, and `term_contribution`'s optimal multipliers. 20 permutations → the
   distribution (median, 5-95 %).
2. *The 24-h lag* (and 48 h as a check): the same with `m = n + 24`. This keeps the local
   solar hour, so `vertical` and `surface_flux` keep their diurnal phase and only the flow is
   decorrelated; it is **the control quoted beside the real number**. Its composite by local
   hour is written too (`composites.by_local_hour` on the lagged pool).
3. *The spatial roll*: predicted fields of the *same* pair rolled by `(dj, di)` drawn with
   `|dj|, |di| ≥ 64`; 20 draws. Pixels rolled onto land or outside `valid` are dropped.

Report, per `L` and per statistic: the real value with its 32-cell / 3-h block-bootstrap
interval (`stats.block_bootstrap` with `estimator=lambda x, y: rms(y)/rms(x)` for the ratio,
`slope_ols` for the slope), the three controls' distributions, and the gap real − control in
units of the control spread and of the real interval. The back-of-envelope reference for
independent fields, `sqrt(1 + (rms 2F / rms measured)²)` = ~1.13 at `L = 0`, ~1.19 at `L = 8`,
is written beside the measured controls so the reader sees whether the permutation reproduces
it (it should, within the selection's effect).

**(b) Other selections** (vet 5a) — the closure statistics of task 6 recomputed on:
`G_prev` (the previous pair's `G` at the same pixel, p90 over `valid` — breaks
selection-on-the-outcome; pair 0 dropped), `sigma_mag` p90 (strain, not gradient),
`persist` (pixels that are `front` in pairs `n − 1`, `n` and `n + 1`), beside the existing
p80 / p90 / p95 and `edge13` rows. One table per `L`.

**(c) Longer blocks** (vet 5b) — `block12` and `block24` labels added to the pool
(`space_time_block_ids(…, hours=12 / 24)`); the bootstrap intervals of the residual ratio and
of the residual's slope on `2F` for 1 / 3 / 12 / 24-h blocks, per `L`. The expected widening
is the point; if 24-h blocks do not widen the interval, say so (M3 task 5 found 3-h blocks
widen it 1.58× only when the series is correlated).

**(d) The case re-seeded** (vet §7) — `m3_case.run` called twice more, written to
`data/m3_case_controls.json` (the original `m3_case_summary.json` untouched): seeded at
`t0 − 12 h` on the same corridor, and seeded on persistence (`front` in ≥ 5 of the 7 pairs
around `t0`) rather than on `G ≥ p90` at `t0`. Report each run's `change_factor`,
`control_change_factor`, `predicted_over_observed_change` and
`budget_over_observed_change` beside the original's 0.19 / 0.89 / −0.08 / 0.99.

Figure `figs/m3_controls.png`: four panels (`L`), each the residual ratio, explained fraction
and residual-on-`2F` slope as points with intervals for the real pool, and the three controls
as bands. Caption states `L`, mask, pool, the seeds and the pre-declaration.

Tests (offline, synthetic pools of a few hundred pixels across a handful of "pairs"): the
permutation of a pool with `two_F ≡ measured` gives a residual ratio of `sqrt(2)` ± the
sampling; the 24-h lag leaves the local-hour label unchanged; the roll never maps a pixel
onto itself; `G_prev` selection on a pool where `G` is constant in time equals the p90
selection; the declared block is written before the stores are opened.

Log: the full table; which `L` show "skill" by the declaration; the lagged control's diurnal
composite against the real one (does the "positive at every local hour" survive the
shuffle?); the three selections' residual ratios beside the p90 row; the interval widths by
block length; the two re-seeded case runs; wall time (expect ~1-2 min per `L` per control
family — estimate).

### 3. The two term tests — `S` split into its parts, `V` from the second cell (vet §6 items 3, 4)

`py/m3_term_tests.py` → `data/m3_term_tests_summary.json`, `figs/m3_term_tests.png`;
`tests/test_m3_term_tests.py`. Both tests read the raw and chunk stores through
`inputs.open_inputs` and write **no field to disk** — only the JSON — so the product is not
touched and no schema changes.

**Pre-declared:** the split is said to reveal a **cancellation** if the three-multiplier fit
reduces the imbalance's rms below the unit-sum residual by more than the 32-cell / 3-h
bootstrap half-width *and* every fitted multiplier lies in [0.5, 2]; otherwise §6.2's
orthogonality stands. `V` from the second cell is a **sensitivity**, with no threshold: what
is reported is whether its diurnal phase and sign agree with the top cell's.

**(a) Split `S`.** For every pair and every `L` (71 × 4; the chunk read dominates, so expect
the sweep's ~7 s per pair → ~35 min detached — estimate), the three components of the
surface-flux term, by calling `vertical.surface_flux_term` with the other fluxes zeroed:

- non-solar heat: `oceQnet → oceQnet − oceQsw`, `oceQsw → 0`, `oceFWflx → 0`;
- shortwave: `oceQnet → oceQsw`, `oceQsw → oceQsw`, `oceFWflx → 0` (so `Q_top = f_sw oceQsw`);
- freshwater: `oceQnet → 0`, `oceQsw → 0`, `oceFWflx` as stored;

in both depth forms (`drF[0]` and `depth=KPPhbl`). **Identity, asserted per pair:** the three
sum to the stored `surface_flux` (resp. `surface_flux_kpp`) to 1e-10 relative in rms — this is
the test that the split is the stored term and nothing else; a zero array must pass
`_check_flux_sign` (add a test). Pool on the stored `front & valid`; the imbalance is
`I = DGDt_semilag − two_F − subfilter − vertical` from the store. Report per `L`: each
component's rms over measured, its correlation with `I`, its composite by local hour; the OLS
fit `I ~ a_ns S_ns + a_sw S_sw + a_fw S_fw` with block-bootstrap intervals on the `a`; the rms
of `I − Σ a S` against `I − S` (unit sum, the product's residual) and `I` (no `S`); the same
with the `KPPhbl` form. Also the two-component version (heat = non-solar + shortwave, and
freshwater), since task 2 of prompt 4 found the non-solar gradient dominates.

**(b) `V` from the second cell.** On the three smoke pairs of prompt 4 task 2 (0 at 16 LST,
9 at 01 LST, 21 at 13 LST) plus pair 30 (the case seed), at `L = 0` and `L = 4`:

- `V_top`: the stored term (recomputed, asserted equal to the store);
- `V_top,o2`: the top-cell term with `b_z(order=2)` — the three-level sensitivity the store's
  third level allows (`vertical_term_factorised` with the order-2 `b_z`);
- `V_cell2`: the tilting term of the **second** cell's budget — `vertical_term(b_k1, ∂b_k1,
  b_k2, W_k2, drF[1], …)` with a new `inputs.W_k2` accessor (`W_k(k_l=2)`, refused if 3-D as
  `W_k1` is) and the gradient of `b_k1` filtered at `L`.

Report rms over measured, the correlation and the OLS slope of `V_cell2` on `V_top`, and the
sign of each term's pool mean at the three local hours. State the reading the vet report
proposed ("if it moves, it is a grid artefact") and the interim's (§2.4: it *is* a
discretisation term; moving with the cell is the prediction), and which the numbers support
— without choosing a threshold after the fact.

Figure: (left) the three components' local-hour composites per `L` with `I`'s; (right) the
four smoke pairs' `V_top`, `V_top,o2`, `V_cell2` rms and the slope.

Tests (offline, synthetic grids from `synthetic.py` + the `test_vertical` fixtures): the split
identity on random fluxes; the zero-flux guard; `W_k2` refuses 3-D; `V_cell2` on a field with
`b_k1 = b_k2` is zero; the fit recovers `(1, 1, 1)` on a pool built as the unit sum plus noise.

Log: the pre-declaration; the identity's max relative error; the component table; the fit
and whether the declaration's condition is met; the `V` table and the reading; wall times.

### 4. The offline tracer — the pipeline on a case with a known answer (vet §3, §6 item 1b)

`py/m3_tracer.py` → `data/tile330_tracer_20120702T00_72h.zarr`; then `m3_run.py --raw <that>
--chunk none --tag tile330tracer` → `data/tile330tracer_derived_L{L}.zarr`; then `m3_closure.py
--tag tile330tracer` → `data/m3_closure_tile330tracer_summary.json`; `figs/m3_tracer.png`;
`tests/test_m3_tracer.py`.

**The design (M3-Q29 (a)), fixed before the run.** Two passive tracers, `Theta` and `Salt` —
the model's own two, so `operators.buoyancy` (JMD95) is applied to advected fields exactly as
to the model's and the pipeline reads the store unchanged — start from the model's hour 0 and
are advected for 72 h by the model's hourly `U`, `V`:

- the step is `fvadvect.fv_step_arrays` (OS7MP, `form='advective'`, `integrator='split'`,
  `dt_sub = 100 s`, `fill='nearest'` for land) — V3b's truth, M1 task 6b;
- the velocity at each sub-step is **linear in time between the hourly snapshots** (the face
  transports `uT`, `vT` and the Courant rates interpolated, which is the same thing); a
  second column with the velocity **piecewise constant over each hour** (the hour's midpoint
  velocity, as the budget's semi-Lagrangian step sees it) is M3D-Q3 — **decided (a), both
  columns, 2026-10-31**: `--vel linear` and `--vel piecewise`, two stores, two sweeps;
- after every sub-step the cells **outside `mask_edge`** (the 7-cell margin, already excluded
  from every statistic) are reset to the model's own `Theta`, `Salt` interpolated in time —
  the clamp of M3-Q29. Record the clamp width and the number of clamped cells in the attrs;
- the store has the raw store's layout (`osn_tiles.write_raw` / `zarr_series.append_hour`,
  one chunk per hour per variable, same coords and attrs): `Theta`, `Salt` are the tracer;
  `U`, `V`, `W`, `Eta`, `KPPhbl`, `oceTAUX`, `oceTAUY` are **copied from the raw store** so
  `inputs`, the masks and the wind readers work unchanged. `attrs` state the construction,
  `dt_sub`, the interpolation, the clamp, the git commit, and that `Theta`/`Salt` are *not*
  the model's.

Cost: 36 sub-steps × 72 h = 2,592 steps of a 720² limited stencil, two tracers — of order
10-30 min in numpy (estimate; time one hour first and log the rate); the sweep ~30 min (the
product's took 1,807-1,975 s for four `L`); the closure ~3 min. Run the advection and the
sweep detached, resumable per hour and per pair respectively.

**The tracer's true budget is exactly** `DG/Dt = 2F + τ + (OS7MP dissipation)`: no `V`, no
`S`, no KPP, no sub-hourly flow. So the residual the pipeline reports on it is the pipeline's
own floor — the hourly semi-Lagrangian sampling of a tracer that evolved on a within-hour-smooth
flow, plus the scheme's dissipation at the grid scale — and the Eulerian − semi-Lagrangian
gap, the order-5 − order-3 gap and the explained fraction on the tracer are what those
numbers *would be* if the model's `b` evolved only by resolved advection.

**Checks, written into the summary and all required:**
1. *Hour 0 → 1 against V3b.* With piecewise-constant velocity, the first hour of the driver
   equals `fvadvect.fv_advect` applied once (bit-identical) — the driver is the V3b step
   iterated, nothing more.
2. *Conservation.* The ocean-mean and the variance of each tracer inside `mask_analysis`,
   every hour: the mean must drift by less than the clamp's inflow can explain (report both
   numbers); the variance decays (OS7MP is dissipative) — report the 72-h factor.
3. *The front population.* `G`'s p50 / p90 / p99 on `mask_analysis` at hours 0, 24, 48, 72 for
   the tracer and for the model: the tracer's fronts sharpen under the strain and decay by
   numerics only, so they will differ from the model's — report the ratio, because the
   pipeline's p90 selection is relative and the comparison of residual *ratios* survives it,
   while a comparison of absolute rms does not.
4. *The pipeline's invariants* on the tracer store: `verify_series` passes; `open_inputs
   (chunk=None)` opens it; `n_lost` at `L = 8` is logged as for the product.

**The pre-declared reading (M3D-Q1 — decided (a), 2026-10-31; the thresholds below are
fixed).** With `r_tr(L)`
the tracer's residual ratio on its own `front & valid`:
- at `L = 0`, `r_tr ≥ 0.50` reads "the hourly cadence limits the **analysis** at the grid scale
  even on a within-hour-smooth flow"; `r_tr ≤ 0.25` reads "the limit is the model's own
  sub-hourly evolution — **the data**"; between, both, in the proportion the numbers show;
- at `L = 8`, `r_tr ≤ 0.25` against the product's two-term catch-all of 0.685 reads "the
  filtered-scale failure is not the pipeline";
- the tracer's Eulerian − semi-Lagrangian gap against the product's 0.672 / 0.183 says how
  much of that gap is the sampling of a smooth flow and how much is the model's sub-hourly
  evolution.

Figure `figs/m3_tracer.png`: residual ratio, explained fraction, the two coherence gaps and
the subfilter fraction against `L`, tracer beside product (and beside the piecewise-constant
column if M3D-Q3 is yes); a second row with check 3's `G` percentiles against hour.

**The tracer's slopes.** Criterion 1 may well pass on the tracer. If it does, criterion 4's
slopes on the tracer are **quotable as the pipeline's calibration** — labelled "tracer, not
the model" everywhere they appear, compared with V3b's 0.975 [0.954, 1.003] — and **never as a
frontogenesis efficiency** of LLC4320. The product's slopes stay withheld (M3-Q24 (a)).

Tests (offline, ≤ 64² synthetic grids): check 1 on a synthetic flow; the time interpolation at
the hour endpoints returns the endpoint velocity; the clamp resets exactly the margin cells and
no others; the written store round-trips through `open_inputs(chunk=None)` and `hour_pair`;
`--vel piecewise` and `--vel linear` differ on a time-varying synthetic flow and agree on a
steady one.

Log: the rate per hour and the total wall; the four checks; the tracer's five-number table
per `L` beside the product's; the two gaps; the reading under M3D-Q1 **in the declared terms
only**; the slopes, labelled as calibration, if criterion 1 passed on the tracer.

### 5. A second tile, surface-only (vet §6 item 5c; M3-Q30 (a))

`py/m3_tile2.py` (resolve, grid, masks, pull, verify) → `data/tile{N}_grid.zarr`,
`tile{N}_masks.nc`, `tile{N}_raw_20120702T00_72h.zarr`; then `m3_run.py --raw … --grid … --masks
… --chunk none --tag tile{N}` → `tile{N}_derived_L{L}.zarr`; `m3_closure.py --tag tile{N}` →
`data/m3_closure_tile{N}_summary.json`; `figs/m3_tile2.png`. The tile is **M3D-Q2 — decided
(a), tile 329, 2026-10-31**, so `N = 329` below, face 7, no land; tile 354 is the fallback only
if the orientation QA fails for more than a one-line fix. The window is the same 72 h (M3D-Q5,
decided (a)) so the local-hour composites line up.

Steps, in order, each logged with a number:
1. **Resolve** with `osn_tiles.tile_spec(i_rect, j_rect)` and print the `TileInfo`; pull the
   static grid (`load_grid(tile=…)`, `write_grid(out=…)`); print `CS`, `SN`, the spacing
   (`masking.spacing_km`) and the lon/lat extent. **Orientation QA:** if the face differs from
   10, state whether `CS`/`SN` match face 10's (`CS = 0, SN = −1`) or not, and run the V3
   discrete null on the new tile's hour 0 (`validate`'s llc variant, generalised to a tile
   argument) — slope within [0.95, 1.05] is the acceptance that the operators read this face
   correctly. Nothing else in this task runs until that passes.
2. **Masks** with `masking.build_masks` (same `halo_cells = 7`, `min_km = 100`, `edge_cells =
   7`). On a tile with **no land** `coast_distance_km` must be defined (all-infinite or a
   sentinel, recorded) and `mask_offshore` all-true; the Gulf-of-California check of
   `m1_write_masks` is tile 330's and is not run. Report the analysis-cell count beside tile
   330's 262,925.
3. **Pull** the 72 h with `pull_series(timestamps_72(), out, tile=…)` detached (M2 task 2's
   driver generalised with `--tile`), `verify_series` after; log the per-hour wall time and
   the total (the smoke test ran 9 s per hour; estimate 15-60 min).
4. **Sweep and closure** with `--chunk none`: the three-term budget (`measured`, `2F`, `τ`),
   the residual a labelled catch-all, criterion 1 judged against the product's **two-term
   catch-all** (0.739 / 0.728 / 0.713 / 0.685 — the product's `catchall.over_measured`), not
   against its five-term ratio. Criterion 2 and the two coherence gaps are judged as for the
   product. The `by_coast_km` composite is skipped on a tile without land, with a note.
5. **Compare**: one table per `L` with tile 330 (five-term and catch-all) and the new tile
   (catch-all): residual ratio, explained, the two gaps, `subfilter / 2F`, `measured_on_2F`,
   the p90 `G` threshold (the front population differs — report it), `n_front`.

Do **not** pull a chunk store for this tile; the five-term budget on a second region is a
separate decision (M3-Q30 (b), declined for now).

Tests: the tile resolver round-trips for 330 and the chosen tile; the masks on a synthetic
grid without land; the `--tile` driver's dry run lists 72 timestamps and the store name.

Log: the `TileInfo`, the orientation QA numbers, the mask counts, the pull's wall time and
volume, the sweep's wall time, the comparison table, and one paragraph on what a second
regime does to the `L = 0` failure and to the diagnosis's cause one.

### 6. The digging note, and its reference from the interim (M3-Q28 (a))

`report/frontogenesis_M3_digging.md` + `report/check_digging_numbers.py` (M3D-Q4, decided (a):
a separate checker, the interim's frozen) + figures
copied or referenced from `figs/`; one marked paragraph in the interim report; one marked
edit in prompt 6.

- **Audience and rules:** the interim's (M3-Q16..Q26): an advanced PhD student in physical
  oceanography; every symbol introduced in words; no code identifiers in the running prose;
  numbers from the new JSONs **only** (`m3_controls_summary.json`,
  `m3_case_controls.json`, `m3_term_tests_summary.json`,
  `m3_closure_tile330tracer_summary.json`, `m3_closure_tile{N}_summary.json`) and the product's
  `m3_closure_summary.json` for the comparison columns; the "Do not" list below.
- **Length:** ~2,000 words of body prose plus the four figures and a numbers appendix in the
  interim's format (value, source, JSON key).
- **Structure:** (1) what the vet report objected to, in two paragraphs; (2) the noise floor
  — the controls and the tracer, with the pre-declared readings and what each gave; (3) the
  two term tests; (4) the pooling and selection checks and the re-seeded case; (5) the second
  tile; (6) **a closing table answering the vet report's §6 items one by one** — item, test,
  outcome, what it changes in the interim's diagnosis (§6.1, §6.2) and what it does not; (7)
  what remains untestable (sub-hourly, a second full-depth region) and why.
- **The interim is not rewritten.** It gets **one** marked paragraph at the end of §8,
  `*(added <date>, M3 digging.)*`, naming the note, listing the tests in one sentence each
  with their outcome in the pre-declared terms, and stating that the verdict of §5.2 is
  unchanged. `report/check_report_numbers.py` must still pass after the edit (the paragraph
  quotes no withheld slope; the tracer's calibration slope, if quoted, is labelled and is not
  one of the four forbidden strings — add it to the checker's forbidden list only if it
  collides).
- **`check_digging_numbers.py`** does for the note what `check_report_numbers.py` does for the
  interim: every JSON-backed number re-read and required verbatim; the four withheld product
  slopes forbidden; the tracer's slopes allowed only with the words "tracer" or
  "calibration" in the same sentence. Exit 0 / 1.
- **Prompt 6, task 2:** the digging note and its JSONs are added to the allowed-source list
  for the M5 report, as a marked edit *(added <date>, M3 digging)*; the M5 report absorbs the
  note's closing table into its limitations and its answer to planning §12.
- **PDF:** built as the interim's (`report/README.md`), every page inspected, the page count
  logged; `report/README.md` gains the note's rows.

Log: the word count; the number of checked numbers; the checker's result; the interim's new
paragraph verbatim; the prompt-6 edit; the PDF page count; the suite count; which PNGs need
`git add -f`.

---

## Acceptance criteria

1. **The product is untouched.** The four `tile330_derived_L{L}.zarr`, `m3_closure_summary.json`,
   `m3_figs_summary.json` and `m3_case_summary.json` have the same bytes / numbers after this
   prompt as before (task 1's regression; `check_report_numbers.py` passes throughout).
2. **Every number in the note has a JSON behind it**, written by a script in `py/`, and
   `check_digging_numbers.py` passes.
3. **Pre-declarations precede numbers** — tasks 2, 3 and 4 each write their declared block
   before a store is opened, and the log entry quotes it before any result.
4. **The tracer's identity checks hold** (task 4, checks 1-4) — without them the floor is not a
   floor.
5. **No efficiency is quoted** for LLC4320; the tracer's slopes appear only as calibration,
   labelled; the product's four slopes remain withheld.
6. **The interim references the note** in one marked paragraph; the verdict of §5.2 is
   unchanged in words.
7. The suite passes with the new tests; the figures are force-added; one log entry per task.

## Do not

- Do not quote a "frontogenesis efficiency" (prompt 4) — on the product, the tracer, or the
  second tile.
- Do not call a slope below 1 "diabatic damping" (prompt 4).
- Do not run `tile_find`, label, thin or despur here (prompt 4); the controls' selections are
  percentiles of stored fields.
- Do not overwrite the product: `--clobber` is refused when `--tag` is unset; the controls,
  term tests and tracer write their own JSONs and stores.
- Do not change the M3 verdict in this prompt. The note reports what each test found; what
  it means for planning §12 is M5's to write.
- Do not tune after the fact: the seeds, lags, roll widths, `dt_sub`, the clamp width, the
  interpolation, the fit's bounds and the tracer's reading thresholds are fixed in the
  declared blocks before the runs, and a change is a second run with a second log entry.
- Do not pull a full-depth chunk store for the second tile (M3-Q30 (b) declined).
- Do not extend the window or the cadence; sub-hourly fields do not exist (M3-Q14).

## Q&A

### Claude, 2026-10-31 (before task 1)

**M3D-Q1 — The tracer's pre-declared reading (task 4).** The thresholds proposed are
`r_tr(0) ≥ 0.50` → "analysis-limited at the grid scale", `r_tr(0) ≤ 0.25` → "data-limited",
between → both; and `r_tr(8) ≤ 0.25` → "the filtered-scale failure is not the pipeline". The
0.50 is the gate's own tolerance; the 0.25 is M1 task 4's synthetic floor (4-9 % at
`dt = 3600`) with a wide margin for the real flow and the real front population. Options:
(a) **these thresholds**; (b) one threshold only, at 0.50, with everything below it read as
"the pipeline would have closed"; (c) no threshold — report the number and let M5 read it.
I recommend **(a)**: a three-way reading is what the vet report's objection needs, and the
numbers are the record's own.

> **JXP:** Yes, use (a)

**M3D-Q2 — Which tile (task 5).** Resolved offline: **tile 329** (west; face 7; lon −143 to
−128, lat 26.7 to 38.3; open NE Pacific, **no land**) and **tile 354** (north; face 10; lat
~38 to 50, lon −128 to −113; the northern California and Oregon coast, mostly offshore after
the 100 km cut). Options: (a) **329** — the regime the interim's §8 asks for ("flat with
distance offshore" gets a reading with no coast at all; weaker fronts), at the cost of a face
change that task 5's orientation QA must pass; (b) **354** — the same face and orientation,
no QA risk, a second coastal regime; (c) both, 329 first. I recommend **(a)**, with (b) the
fallback if the orientation QA fails for a reason that is not a one-line fix.

> **JXP:** (a) is fine.

**M3D-Q3 — A piecewise-constant-velocity column for the tracer (task 4).** The driver can
also advect with the hour's midpoint velocity held constant over the hour — exactly the flow
the budget's semi-Lagrangian step assumes — at the cost of a second 72-h advection and sweep
(~1 h). The difference between the two columns isolates the within-hour velocity
interpolation from everything else. Options: (a) **yes, both columns**; (b) linear only.
I recommend **(a)**: it is cheap, and without it the tracer's residual at `L = 0` conflates two
things the reading in M3D-Q1 wants separated.

> **JXP:** Yes, use (a)

**M3D-Q4 — The number checker (task 6).** Options: (a) **a separate `check_digging_numbers.py`**
for the note, the interim's checker frozen with the interim; (b) extend
`check_report_numbers.py` to both documents. I recommend **(a)**: the interim's checker is
part of the interim's record, and the note's rules (the tracer's labelled slope) differ.

> **JXP:** Yes, use (a)

**M3D-Q5 — The second tile's window (task 5).** Options: (a) **the same 72 h**, 2012-07-02 00
to 07-04 23 UTC, so the local-hour composites and the day-3 pairs line up; (b) a different
72 h in the same season, for independence in time as well as space. I recommend **(a)**:
the question is whether the `L = 0` failure is Monterey's, and one variable at a time answers
it; (b) would need its own `DAY3_PAIRS` and loses the diurnal alignment.

> **JXP:** (a) is fine.

## Log

Append to `claude_prompts/frontogenesis_prompts.md` under `## Logs`, one entry per task,
titled `### <date> — Execution prompt M3-digging, task N: <title> (<model>)`. Across the
prompt, record: every pre-declared block before its numbers; the product's regression (task
1); the controls' distributions and the "skill" reading per `L` (task 2); the split identity
and the fit, the `V` table and the reading (task 3); the tracer's four checks and its table
beside the product's, with the reading in M3D-Q1's terms (task 4); the orientation QA, the
pull's wall time and the two-tile table (task 5); the note's checker result, the interim's
new paragraph and the prompt-6 edit (task 6); wall times and volumes for every run; the suite
count after every task; and — plainly — **which of the vet report's five changes each task
answered, and whether the interim's diagnosis moved.**
