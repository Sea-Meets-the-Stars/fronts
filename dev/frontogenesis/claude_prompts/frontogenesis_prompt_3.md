# Frontogenesis execution prompt 3 — M2: Data pull

**Milestone:** M2 (`frontogenesis_coding.md` §6).
**Prerequisite:** M0 complete (static grid on disk). **M2 genuinely does not depend on M1** — it
pulls raw fields and needs no physics, no operators and no masks. Masks are M1's; the static grid
is M0's. So M1 and M2 can run in parallel.
**Goal:** the 72-hour window on disk, in one time-dimensioned store per source, resumably.

**Status 2026-10-03: tasks 1-3 done.** Task 1: `osn_tiles.pull_series` on `zarr_series`'s
per-hour atomic append with repair-on-resume, stop-at-gap policy; `series_verify.verify_series`;
`tests/test_pull_series.py` 20 offline + 1 network smoke, 22 s for one real hour; suite 104 passed
+ 3 xfailed (log entry 2026-10-03). Task 2: `py/m2_pull.py` pulled all **72 hours** into
`data/tile330_raw_20120702T00_72h.zarr`, detached, in **27.3 min** (median 22.4 s/hour, range
20.7-33.1 s; 0 retries, 0 failures, 0 repairs), **765 MB** on disk (11.1 MB/hour);
`verify_series` OK (no gaps, §3.2 schema, land-NaN == `hFac` masks in all 72 hours, `niter` steps
144, `KPPhbl` present); the re-run is a no-op (0 pulled, all 834 files sha256-identical); hours
0-1 identical to M0's 2-hour store, which is kept (task-2 log entry). **Task 3 done** (log entry 2026-10-03): `py/m2_qa.py` → `figs/m2_qa_series.png` — tide 2.0 m range at 12.4 h (M2 0.62 m + K1 0.49 m), `KPPhbl` diurnal amplitude 6.4 m with its maximum at ~01 h local solar, land-NaN fraction constant, `oceTAU*` 922/565 finite-on-land → 0 after re-masking, no frozen field / NaN change / outlier; displacement over 71 pairs: ocean median 0.27-0.44, p99 1.05-1.38, window max 4.05 cells (Gulf of California tidal jet, outside `mask_analysis`; 2.27 on the analysis mask); edge support leaves the finite tile for 0 analysis cells at `L ≤ 4` and 7-34 per pair at `L = 8` (`edge_cells = 7` kept, `isfinite` required at `L = 8`). M2-Q3 extra step (`py/m2_baseline_stability.py` → `figs/m2_v3_stability.png`, two backward-compatible keywords `store=`/`t0=` in `validate.py`): the V3 null on all 71 pairs reproduces 0.9806 on hour 0-1, gives 0.902-0.997 (mean 0.972, weighted 0.983; 64/71 pass the gate; every CI overlaps the baseline band; 0.971-1.002 with the top 1 % |2F| pixels trimmed) — the baseline is a property of the operators with a leverage-driven tail from a sharp front at the northern tile edge on 07-04; Figure 2's baseline is unchanged. Suite 104 passed + 3 xfailed. **Task 4 done** (log entry 2026-10-03, `py/m2_chunk_recon.py`): `s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/` on Nautilus (credentialed, present here), zarr v3, one store per hour — **72/72 hours present**, 0 missing; `Theta, Salt, U, V, W, Eta, oceQnet, oceQsw, oceFWflx, oceTAU*, SIarea` in all; 51 levels, `W` on `k_p1` (52; `k_p1 = k_l` by continuity); exactly tile 330 (grid bit-identical to `tile330_grid.zarr`); one 51-level object per variable per hour, so a `k = 0..2` read fetches 174 MB/hour (12.5 GB, ~6.3 h at the 0.55 MB/s measured here); `drF[0] = 1.0`, `Z[0] = −0.5` confirmed; `k = 0` fields and `W(k_p1=0)` **bit-identical** to OSN (`Eta` in all 72 hours); flux attrs say `+=down` but the data are upward-positive, and the forcing is 6-hourly, linearly interpolated. No blocker for task 5. Tasks 5-7 not started.
M0 closed 2026-09-28; M1 closed
2026-09-30 (prompt 2, task-7 log entry). M1 being closed does not change M2's scope; M2 still needs no physics. It does
fix the operator defaults M3 will use on this data (`form='discrete'`, cubic departure velocity),
which is why task 3's QA reuses them.

---

## The window

**2012-07-02 00:00 -> 2012-07-04 23:00 UTC**, 72 consecutive hourly steps.

This window was chosen, not defaulted. The full-depth `monterey_bay` chunk store already holds
**11 hours inside it**, because someone had already picked 2012-07-03 as a dense 3-hourly day:

- 2 daily 12:00 stores — 07-02 T12 and 07-04 T12
- 8 from the dense day — 07-03 T00/03/06/09/12/15/18/21
- 1 extra — 07-04 T00

Starting at the beginning of the 504-hour OSN series would have captured only 3. Do not change the
window without re-checking that overlap.

## Two sources

### A. OSN surface — the primary fields. Start immediately.

- `llc_surf`: `Theta, Salt, U, V, W, Eta` at `k=0`, hourly.
- `llc_wind`: `KPPhbl, oceTAUX, oceTAUY` (also `k=0`, hourly; coverage 2011-11-01 -> 2012-07-15,
  so our window sits inside it). `KPPhbl` is the key interpretive variable for the diurnal
  residual, so it is not optional.
- Static grid once (already written by M0 as `tile330_grid.zarr`, §3.1): `XC, YC` (as coords,
  like the hourly stores, so `xr.merge([hour, grid])` works plainly — corrected 2026-09-28, M0
  task 5), `dxC, dyC, dxG, dyG, rA, rAz, CS, SN, hFacC, Depth` plus `hFacW, hFacS` and the 0-d
  `drF, Z, Zl`; `face = 10` is a scalar coord in both stores (`expand_dims('face')` /
  `open_grid(with_face=True)` before any dbof operator).
  `oceTAUX`/`oceTAUY` come masked with the centred mask; store as-is, re-mask with
  `hFacW`/`hFacS` at use (M0 task 3).

Write `osn_tiles.pull_series(timestamps, out_zarr, ..., clobber=False)` — schemas in
`frontogenesis_coding.md` §3.1-§3.2. **This concat step exists nowhere in either repo**: the
repo's `run_series` writes one NetCDF per timestamp with no time dimension. We write it.

**It must be resumable:** skip timestamps already present in `out_zarr` unless `clobber`. A
72-step network pull will be interrupted.

### B. Chunk store — the extra budget terms. Gated on Lauren's transfer.

From the hourly full-depth `monterey_bay` transfer (decision Q13): all 51 levels are being
written, plus `oceQsw` and `oceFWflx` added to `transfer.variables`.

Use **`vertical.load_chunk_levels(window, k_max=2, out_zarr=...)`**. Load only `k = 0..2` and only
`Theta, Salt, W, oceQnet, oceQsw, oceFWflx` — schema in `frontogenesis_coding.md` §3.3. The store
holds 51 levels at ~539 MB per timestep; nothing obliges us to read them.

**Also capture `drF` for `k = 0..2` from the chunk store's 3-D grid** (`process_llc4320_3d_grid`,
which adds `Z, Zl, Zu, Zp1, drF`). `vertical.py` needs the `k = 1, 2` values; OSN carries only
the `k = 0` scalar (`drF = 1.0`, already in `tile330_grid.zarr` — corrected 2026-09-28, M0).
Cross-check `drF[0] = 1.0 m` and `Z[0] = -0.5 m` here. **Load `W` on interfaces `k_l = 0..2`**,
not just `k_l = 0`: `vertical_term` takes the cell-base `W(k_l=1)` (coding §4.6), because the
model's linear free surface makes `W(k_l=0) = dEta/dt`, a free-surface signal rather than a
flux (planning §2.2).

These three levels are what turn the finite-top-cell vertical term and the surface-flux part of
the diabatic term from *inferred* into *measured* (planning §2.2-§2.3). That is the single
biggest improvement to the budget since the first draft, so it is worth waiting for — but see
"Do not block" below.

## What M2 does *not* write

`tile330_grid.zarr` came from **M0**; `tile330_masks.nc` comes from **M1**. M2 writes neither —
that separation is exactly what lets M1 and M2 proceed in parallel. If you find yourself needing
a mask here, you are doing M3's job early.

---

## Tasks

Run **one task per session**, in order, as in M0 and M1, with one log entry per task (see
**Log**). Code goes in `dev/frontogenesis/py/`, tests in `dev/frontogenesis/py/tests/`
(offline, except one `@pytest.mark.network` smoke test, coding §5). Data goes in
`dev/frontogenesis/data/` (git-ignored). The env is `~/miniforge3/envs/frontogenesis/bin/python`,
and the M1 suite (84 passed + 3 strict xfails) must still pass at the end of every task.

Two halves, as above: **A** (tasks 1-3) needs nobody. **B** (tasks 4-5) depends on Lauren's
transfer; task 4 finds out how far it has got, and nothing in A waits for it.

**Rules for long pulls.** These come from M1: four agents stalled on long foreground jobs.
- Never run a multi-hour pull in the foreground. Launch it as a detached script (`nohup ... &`)
  that writes a progress log, and check the log.
- Prefix every interactive python/pytest call with `timeout 300`.
- Because `pull_series` is resumable, an interrupted pull is restarted, not debugged.

### 1. `osn_tiles.pull_series` — resumable concat, tested offline

- Write `pull_series(timestamps, out_zarr, tile=None, endpoint=OSN_ENDPOINT,
  include_wind=True, clobber=False) -> str` per coding §4.1, on top of the existing
  `load_hours` / `write_raw` (M0 task 4). The product is the §3.2 schema exactly, including:
  - `float32` on disk;
  - one `(1, 720, 720)` chunk per hour per variable;
  - `time` encoded as `seconds since 2011-09-10`;
  - `niter(time)`, scalar `face`, and `XC`/`YC` as coords;
  - `U` on `i_g`, `V` on `j_g` (no pre-interpolation);
  - no halo applied;
  - the attrs §3.2 lists.
- **Resumability, as designed.** Append one hour at a time (`to_zarr(append_dim='time')`), and
  decide "already present" from the store's own `time` coord, not from a side file.
  - An hour must be **atomic**: core and wind fields are written together or not at all. A crash
    mid-write must not leave a half-hour that a re-run then skips.
  - Out-of-order or duplicated timestamps raise.
  - `clobber=True` rewrites from scratch.
  - Retry transient OSN errors (a few attempts with backoff); if an hour still fails, record it
    and move on, so that one bad hour cannot stall 72.
- **Verification helper.** `verify_series(out_zarr, timestamps) -> dict` checks:
  - gaps and duplicates;
  - the schema against §3.2;
  - the land-NaN pattern against `hFacC`/`hFacW`/`hFacS` in every hour (the M0 task-3 property);
  - consecutive `niter` steps of 144;
  - `KPPhbl` present.

  Task 2 runs it, and so does the audit.
- `tests/test_pull_series.py`, **offline**: monkeypatch the loaders with synthetic hours, and
  cover resume after a simulated crash (including a crash mid-hour), the no-op re-run, clobber,
  the gap report, the duplicate/out-of-order error, and dtype/chunking. Add one
  `@pytest.mark.network` smoke test that pulls one real hour into a temp store. Register the
  `network` marker if pytest.ini lacks it.

*Discharges:* criterion 2 (resumability) in code; tested here, proven on real data in task 2.

### 2. Pull the 72 OSN hours

- Write `py/m2_pull.py`, a script that pulls the 72 timestamps
  (`2012-07-02 00:00:00` … `2012-07-04 23:00:00`, dbof format) into
  `data/tile330_raw_20120702T00_72h.zarr`. It logs per-hour wall time and failures to
  `data/m2_pull.log`. Launch it **detached**.
- Once complete, run `verify_series`, then **re-run `m2_pull.py` and show it is a no-op**:
  zero hours pulled, the store byte-identical (mtime or checksum of the chunk files).
- Report: hours present, failures and retries, total volume on disk, total and per-hour wall time
  (median and range; M0 measured 40-90 s per hour across both stores, network-bound), and the
  extrapolation to a longer window.
- Spot-check that hours 0-1 of the new store are **identical** to M0's
  `tile330_raw_20120702T00_2h.zarr`, which M1's tests use. Do not delete the 2-hour store.

*Discharges:* criteria 1 (72 steps, no gaps, §3.2 schema), 2 (no-op re-run), 3 (`KPPhbl`), and
the OSN half of 7 (volume and wall time).

### 3. QA of the series

This is a time-series sanity pass, not physics, and is cheap. Write `py/m2_qa.py` →
`figs/m2_qa_series.png`, covering:
- tile-mean and percentile time series of `Eta` (the tide should be visible), `KPPhbl` (the
  diurnal cycle; Figure 6 depends on it), `Theta`, and `|u|`;
- the land-NaN fraction per hour, which must be constant;
- `oceTAUX`/`oceTAUY` re-masked with `hFacW`/`hFacS` (M0 task 3), with any finite values on
  land counted, which should be 0 after re-masking;
- the **hourly displacement distribution** over all 71 hour pairs, using
  `semilag.departure_index` with the default `vel_order=3`, against M1's assumptions (median 0.36,
  p99 1.25, max 2.09 cells on hour 0, task 3). Report the max over the window, and whether any
  departure leaves the order-3 interpolation support near the tile edges. That would grow the NaN
  rim beyond `edge_cells = 7`; M1 task 6 found the edge reach at `L = 8` is exactly 7.

Flag anything anomalous, e.g. a frozen field, a missing tide, or a NaN-pattern change. Do **not**
compute `F`, `G` budgets or slopes; that is M3.

*Discharges:* supports criteria 1 and 3; gives M3 its displacement envelope.

### 4. Chunk-store reconnaissance (source B)

Find out, before writing any loader:
- where the hourly `monterey_bay` transfer lives (endpoint and path; `dbof`'s chunk-store
  readers, `process_llc4320_3d_grid`);
- **which of the 72 hours exist** today (M2 planned for 11);
- whether `oceQsw` and `oceFWflx` are present (Q13 asked Lauren to add them);
- that all 51 levels are there;
- how the store is laid out (chunking per level and per variable), so a `k = 0..2` read touches
  only those levels.

Read the 3-D grid's `drF`, `Z` and `Zl` for `k = 0..2` and **confirm `drF[0] = 1.0 m` and
`Z[0] = -0.5 m`**, the OSN values.

**Consistency check** for one hour present in both sources (e.g. 07-03 T00): is chunk `k = 0`
`Theta`/`Salt`/`U`/`V` identical to OSN's? Is chunk `W(k_l=0)` identical to OSN's `W` (which
is `dEta/dt`, M0 task 3)? Either answer is a finding. The budget combines the two sources, so
they had better be the same model output on the same tile.

Write the inventory to the log. Do not bulk-load anything yet.

*Discharges:* criterion 6 (`drF[0]` confirmed), and the hour inventory for criterion 5.

### 5. `vertical.load_chunk_levels` → the §3.3 store

- Write `py/vertical.py` with **only** `load_chunk_levels(window, k_max=2, out_zarr=None)` per
  coding §4.6. The physics functions in that section are M3's.
- Use the same resumable, atomic, per-hour append design as `pull_series`. Reuse it; do not copy
  it. Load **only** `k = 0..2` and `Theta, Salt, W (k_l = 0..2), oceQnet, oceQsw, oceFWflx`, plus
  `drF(k)`, to the §3.3 schema, `float32` on disk.
- Run it, detached, for every hour task 4 found. Record the **missing hours explicitly**.
- Re-run it to show it is a no-op. Add `tests/test_load_chunk_levels.py`, offline, with
  synthetic stores.
- **If the transfer is still incomplete,** write what exists and stop. When more hours land,
  re-running this task is just re-running the script, because it is resumable. M3 does not wait
  (see "Do not block").

*Discharges:* criteria 5 and 6, and the chunk half of 7.

### 6. M2 acceptance audit

- Run the full suite, offline and with the network smoke test.
- Re-run `verify_series` on both stores.
- Go through the criteria one by one with numbers, as the M0 task-5 and M1 task-7 audits did,
  and list what is carried forward to M3: missing chunk hours, the displacement envelope, any
  source inconsistency, and the volume and wall-time numbers for scale-up.
- If every criterion passes, mark **M2 closed** here (Status) and in coding §6 M2. If only the
  chunk half is incomplete because of the transfer, close the OSN half explicitly and leave the
  chunk half open, as "Do not block" intends.

### 7. Slides

A small M2 acceptance deck, with the same rules as M1's (`deck/`, python-pptx; a figures script
plus a builder script kept in `deck/`; **no text below 20pt**, checked programmatically; rendered
and inspected). It has a title, contents, a one-slide M2 summary, and one slide per task.
Log the work in `deck/README.md`.

---

## Acceptance criteria

- 72 timesteps present in the OSN store, **no gaps**; schemas match §3.2-§3.3.
- Re-running `pull_series` is a **no-op** (resumability actually works, not just coded).
- `KPPhbl` present — easy to forget, and Figure 6 needs it.
- ~~Masks written, and consistent with the M0 QA plot.~~ *(Corrected 2026-10-01: masks are M1's,
  written as `tile330_masks.nc` in M1 task 1; M2 writes none. This item contradicted "What M2
  does not write". It is replaced by **the land-NaN pattern in every hour matching
  `hFacC`/`hFacW`/`hFacS`**, as `verify_series` checks, which is what makes M1's masks valid for
  all 72 hours.)*
- Chunk store: `k=0..2` + three flux fields + `drF` for as many of the 72 hours as exist, with
  the missing hours recorded explicitly in the log.
- `drF[0]` confirmed and recorded.
- Report the total volume and wall time, so later scale-up decisions are informed.

## Do not block

**Do not hold M3 development on source B.** `budget.compute_budget` runs without `chunk_ds` —
it simply omits the vertical and surface-flux terms and reverts to a catch-all residual, and
`closure_report` must say so loudly rather than quietly. Develop M3 against source A, then add
source B's terms when the transfer completes.

## Do not

- Do not pre-interpolate `U`/`V` onto cell centres on disk. Keep `U` on `i_g`, `V` on `j_g`;
  staggering is information and the budget needs it.
- Do not apply the halo mask to the stored raw fields. Store raw, mask at compute time — that
  way the halo width stays a tunable rather than being baked into the archive.
- Do not store `float64`. `float32` on disk, `float64` in the compute path.

## Q&A

### Claude, 2026-10-01 (before task 1)

Numbered M2-Qn. **M2-Q1 decides whether task 5 can finish in M2**; the others can be answered
any time before the task that needs them.

##### Questions

**M2-Q1 — State of Lauren's transfer.** Is the hourly full-depth `monterey_bay` transfer
running or complete? Does it include `oceQsw` and `oceFWflx` (Q13)? Where does it live (endpoint
and path)? Task 4 will find out empirically either way, but if you already know, it saves a
search, and it tells us whether task 5 can close in M2 or stays open into M3.

> **JXP:** . I believe all of the data is traferred.  Please check

> *Claude's finding (task 4, 2026-10-03):* confirmed complete. All 72 hourly stores are in
> `s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/` (endpoint `https://s3-west.nrp-nautilus.io`),
> each with all 51 levels and with `oceQsw` and `oceFWflx` (transfer config
> `run_chunks_monterey_72h.yaml`, llc branch `transfer-monterey-72h`). Task 5 can close in M2.

**M2-Q2 — Local disk for the 72-hour store.** The OSN product should be ~0.75 GB, from M0's
~10.5 MB per hour on disk; the chunk product for three levels is of the same order. Both go in
`dev/frontogenesis/data/` (git-ignored), on this machine. Is that the right place, or should
they live elsewhere (an external disk, or the workstation; there is a "save for workstation"
commit) and be symlinked?

> **JXP:** . That is fine

**M2-Q3 — Stability of the M1 baseline across the window (optional).** Figure 2's baseline
(0.981 [0.970, 0.994]) is V3's real-velocity slope from **one** hour pair, 07-02 T00-T01. With 71
pairs on disk after task 2, re-running `validate.test_discrete_null(velocities='llc')` on, say,
6 pairs spread over the tidal and diurnal cycle would show whether 0.981 is a property of the
operators or of that hour. It costs about a minute per pair and uses no new physics. Do it as an
extra step in task 3, leave it to M3, or skip it? I lean towards task 3: M3's headline is quoted
against this number.

> **JXP:** Do it as an extra step

**M2-Q4 — The 2-hour M0 store.** M1's tests read `tile330_raw_20120702T00_2h.zarr`. Keep it
(my default, and task 2 checks that it equals hours 0-1 of the 72-hour store), or repoint the
tests at the 72-hour store and delete it?

> **JXP:** Keep it.

## Log

Append to `frontogenesis_prompts.md` under `## Logs`, one entry per task, titled
`### <date> — Execution prompt 3, task N: <title>`. Across the milestone, record: timesteps pulled per source, any gaps or failures, total volume, wall time, and the confirmed
`drF[0]` / `Z[0]`. If the chunk transfer is incomplete, say exactly which hours are missing.
