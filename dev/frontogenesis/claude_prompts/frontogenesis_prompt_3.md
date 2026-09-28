# Frontogenesis execution prompt 3 — M2: Data pull

**Milestone:** M2 (`frontogenesis_coding.md` §6).
**Prerequisite:** M0 complete (static grid on disk). **M2 genuinely does not depend on M1** — it
pulls raw fields and needs no physics, no operators and no masks. Masks are M1's; the static grid
is M0's. So M1 and M2 can run in parallel.
**Goal:** the 72-hour window on disk, in one time-dimensioned store per source, resumably.

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

## Acceptance criteria

- 72 timesteps present in the OSN store, **no gaps**; schemas match §3.2-§3.3.
- Re-running `pull_series` is a **no-op** (resumability actually works, not just coded).
- `KPPhbl` present — easy to forget, and Figure 6 needs it.
- Masks written, and consistent with the M0 QA plot.
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

## Log

Timesteps pulled per source, any gaps or failures, total volume, wall time, and the confirmed
`drF[0]` / `Z[0]`. If the chunk transfer is incomplete, say exactly which hours are missing.
