# Frontogenesis — Coding / Execution Document

**Companion to:** `frontogenesis_planning.md` (physics, decisions, figures). This document
is the *implementation* contract: modules, signatures, data schemas, milestones, and
acceptance criteria.

**How to use this document.** Each milestone in §6 (M0-M5) has **one execution prompt doc**.
A milestone is not "done" until its acceptance criteria pass; M1 and M3 are hard gates — do not
proceed past them on a failure, because everything after inherits the error silently.

| Milestone | Prompt doc | Gate |
|---|---|---|
| M0 Access and reconnaissance | `claude_prompts/frontogenesis_prompt_1.md` | |
| M1 Operators and validation | `claude_prompts/frontogenesis_prompt_2.md` | **HARD** |
| M2 Data pull | `claude_prompts/frontogenesis_prompt_3.md` | |
| M3 Field-level budget | `claude_prompts/frontogenesis_prompt_4.md` | **HARD** |
| M4 Fronts and flow-informed tracking | `claude_prompts/frontogenesis_prompt_5.md` | |
| M5 Figures and report | `claude_prompts/frontogenesis_prompt_6.md` | |
| M6 Depth | deferred — not yet written | |

M1 and M2 may run in parallel; everything after M3 is blocked on closure.

**Created:** 2026-09-12. **Updated:** 2026-09-26 (Q13-Q15 closed; Lauren's review folded in).

**Branch prerequisite (Q15, resolved).** Work proceeds against the feature branches
`tiles-surface-only` (llc) and `viz_tools` (fronts) — both usable today. PR #24 is still open;
once the merge sequence in planning §10 lands, **§2's API table must be re-verified**, since
its line numbers are pinned to those branches.

---

## 1. Conventions

Fix these once. Most of the failure modes in planning §2.3 and §11 are convention slips.

### 1.1 Physical

| Quantity | Symbol in code | Units | Notes |
|---|---|---|---|
| buoyancy | `b` | m s^-2 | `b = +g sigma0/rho0` via `calculate_fields.buoyancy_of_field` (**JMD95**, not TEOS-10). `g=9.81`, `rho0=1000.0`. **Increases with density** — the negative of textbook `b`. Harmless (`G`,`F` quadratic; alignment enters as `cos 2theta`). Do not 'fix' it. |
| front strength | `G` | s^-4 | `G = |grad_h b|^2 = b_x^2 + b_y^2` from the **same** `grad_b` (`calculate_native_gradient_tracer`) that feeds `F`. **Not** the repo's `gradb2` / `calculate_grad_squared_tracer` (squares on the staggered points first): the two differ by **0.911x** in the interior median, and the identity `F = (1/2) DG/Dt` only holds discretely when `G` and `F` share `b_x, b_y`. The repo's `gradb2` is for front *finding* (M4) only (corrected 2026-09-28, M0 task 5). |
| frontogenesis tendency | `F` | s^-5 | `F = -(u_x b_x^2 + (u_y+v_x) b_x b_y + v_y b_y^2)` |
| measured tendency | `DGDt` | s^-5 | `D_h G / Dt` |

**The factor of two.** `F = (1/2) DG/Dt`. Every comparison, axis label and regression uses
**`2F` vs `DGDt`**. Name the predicted variable `two_F` in code so it cannot be confused.

**Naming of the measured tendency, deliberately three-tiered** (so the distinction is visible
rather than accidental): `DGDt` in prose; `DGDt_semilag` and `DGDt_euler` as the two *estimates*
on disk (§3.4); and `measured` as the budget dataset's field name for whichever estimate is
primary (the semi-Lagrangian one).

The sign of `b` cancels in both `G` and `F`, but fix it anyway so plots are interpretable.

### 1.2 Numerical

- **Basis:** `grad b` and the Jacobian both come from the repo helpers, which **both rotate
  to geographic** via `CS`/`SN`. What matters is that they are in the *same* basis — they
  are. `G` and `F` are rotational invariants, so this changes neither.
  **Departure points, by contrast, stay in native index space** (raw `U`,`V` → `di`,`dj`):
  no rotation, no round-trip.
- **dtype:** `float64` throughout the compute path; `float32` only on disk.
- **Masks:** boolean, **`True` = valid/retained** (matches `halo_mask`'s convention).
  Masked cells are `NaN` in float fields. Every reduction uses NaN-aware ops.
- **Filter scale:** integer `L_cells` in `{0, 2, 4, 8}`; `L_cells = 0` is the identity.
  The **same** filter is applied to `b`, `U` and `V` (planning §5.4). Kernel (added
  2026-09-29, M1 task 2): a separable, normalised **top-hat of half-width `L_cells/2`**
  (support `L_cells + 1` cells) in index space, applied along whichever of `j`/`j_g`,
  `i`/`i_g` the field carries; **NaN propagates and is never renormalised** (a cell whose
  footprint touches land, a stencil rim or the tile edge is NaN), so the filter stays
  shift-invariant and commutes with the discrete gradient wherever it is finite.
- **Time:** `dt = 3600.0` s. Fields needed at the trajectory **midpoint** are formed as
  `0.5*(f_t + f_tp1)` (planning §5.3, item 3).
- **Array layout:** native face-local `(j, i)` per snapshot; stored with time first,
  `(time, j, i)`. `U` keeps dim `i_g`, `V` keeps `j_g` — do not pre-interpolate on disk.

### 1.3 Style

Follow the `sharpen` effort's conventions: **methods, not classes**; reuse existing code;
inline comments that explain the *physics*, not the syntax. No module may exceed ~400 lines;
split rather than nest.

---

## 2. External APIs

Read directly from source on the branches we will work from
(`llc4320-native-grid-preprocessing @ tiles-surface-only`, `fronts @ viz_tools`).
**Do not guess these.** Note especially the **four** marked ***TRAP***.

### 2.1 Data access (`dbof`)

```python
# dbof/llc4320_ingestion/date_iterations.py
DATE_FMT = '%Y-%m-%d %H:%M:%S'          # L31; also TS_PER_HOUR=144, FIRST_WIND_RECORD_OFFSET=10368
def mit_date_to_iteration(date_str: str) -> int          # L38
def osn_date_to_iteration(date_str: str) -> int          # L64  = mit + 10368  <- USE THIS for OSN

# dbof/llc4320_ingestion/get_raw_data.py
def get_remote_llc_data(endpoint_url, it, face_range)    # L21  -> lazy Dataset
      # Eta,U,V,W,Theta,Salt; already .isel(time=0,k=0,k_l=0)
      # dims (face,j,i) / (face,j,i_g) / (face,j_g,i)
def get_remote_llc_wind_data(endpoint_url, it, face_range)  # L151 -> KPPhbl,PhiBot,oceTAUX,oceTAUY,SIarea
def get_remote_gridfile(endpoint_url)                    # L241 -> all 13 faces, untrimmed

# dbof/preprocessing/preproc_llc_core_data.py
def process_llc4320_grid(grid_ds)                        # L37
      # -> XC,YC,dxC,dyC,dxG,dyG,rAz,rA,Depth,hFacC,SN,CS
      # *** TRAP: uses reset_coords(), which can DROP comodo attrs.
      #     Always ensure_comodo_attrs() before set_xgcm_grid(). ***
```

### 2.2 Tile indexing and xgcm (`dbof`)

```python
# dbof/tiles/tile_mapping.py
def rect_ij_to_tile(i_rect: int, j_rect: int) -> TileInfo   # L112  NOTE ARG ORDER (i, j)
@dataclass(frozen=True) class TileInfo:                     # L48 (re-verified 2026-09-28, M0 task 5)
    tile_idx, tile_j_rect, tile_i_rect, rect_j_slice, rect_i_slice,
    face_idx, j_face_slice, i_face_slice
# ours: rect_ij_to_tile(13320, 9720) -> face_idx=10, j 0:720, i 2880:3600

# dbof/llc4320_ingestion/grid.py
COMODO_COORD_META = {                                       # L11 (re-verified 2026-09-28, M0 task 5)
    'j':   {'axis': 'Y'},
    'j_g': {'axis': 'Y', 'c_grid_axis_shift': -0.5},
    'i':   {'axis': 'X'},
    'i_g': {'axis': 'X', 'c_grid_axis_shift': -0.5},
}
def ensure_comodo_attrs(ds, *, strict=False, source=None)  # L46 -> Dataset (public; not in tile_utils)
def set_xgcm_grid(ds_grid, use_connections: bool = True)    # L121 -> xgcm.Grid; we pass False
      # calls xgcm.Grid(..., padding='fill') -> REQUIRES xgcm>=0.10 (verified 2026-09-28)

# dbof/tiles/tile_utils.py  -- short private helper; copy rather than import if preferred
def _tile_indexer(ds, tile) -> dict                         # L334
      # {'j','j_g'} -> tile.j_face_slice ; {'i','i_g'} -> tile.i_face_slice, for dims present.
      # Staggered dims take the SAME slice -> the derivative rim is invalid on ALL FOUR
      # tile edges, and FINITE, not NaN: xgcm padding='fill' pads the missing neighbour
      # with 0, so the low edges (j=0, i=2880) difference against 0 (G ~1e6x) and the
      # high edges (j=719, i=3599) interpolate with 0 (G ~0.5x).  Crop test: G 1 cell on
      # every edge; Jacobian 1 cell on the low edges, 2 on the high.  The land halo does
      # not remove it -- masking.analysis_mask(edge_cells=...) does
      # (corrected 2026-09-28, M0 task 5).
```

### 2.3 Physics operators (`dbof`)

```python
# dbof/utils/native_gradient.py                -- ALL return geographic-basis quantities
#   (line numbers re-verified 2026-09-28 at tiles-surface-only 938bce1, M0 task 5: the
#   file grew by 77 lines since the survey; the earlier L13/L53/L126/L224/L301 are stale)
def calculate_native_gradient_tracer(ds_value, ds_grid, grid)   # L203 -> (zonal, merid)
def calculate_jacobian(u_x, v_y, ds_merge, grid)                # L130 -> (du_dx, du_dy, dv_dx, dv_dy)
      # *** TRAP: args named u_x, v_y but they ARE U and V (the raw staggered fields). ***
      # Confirmed on real data (M0 task 5): a numpy replica from U/V is bit-identical.
      # Swapping them does NOT raise on numpy-backed inputs -- xgcm interps V along X
      # (it has i), the CS/SN multiply broadcasts (j_g, i_g) x (j, i) to 4-D and the
      # process is OOM-killed; only dask-backed inputs raise KeyError.  Assert dims.
def calculate_grad_squared_tracer(ds_value, ds_grid, grid)      # L301 -> |grad s|^2 at centres
      # squares on the staggered points BEFORE interpolating: 1.10x the component form
      # (b_x^2 + b_y^2) in the interior median -- front FINDING only, never our G (§1.1)
def calculate_native_strain_vorticity(u_x, v_y, ds_grid, grid)  # L378 -> dict, NOT a tuple:
      #   'strain_normal_center', 'divergence_center'   on (j, i)
      #   'vorticity_corner',     'strain_shear_corner' on (j_g, i_g)  <- interp to centres!
def rotate_vector_to_geographic(u_x, v_y, ds_merge, grid, *, interpolate=True)   # L91

# dbof/preprocessing/calculate_fields.py       -- NOTE: on tiles-surface-only the file is
#   calculate_fields.py.  "calculate_additional_fields.py" is the name on the OLD llc4320_v2
#   branch; earlier notes citing it are stale.
def buoyancy_of_field(ds_merge)                                 # L96  -> b = G*sigma0/RHO0
def potential_density_anomaly(ds_merge)                         # L66  -> sigma0 (JMD95, p=0)
VelocityJacobian  = namedtuple(..., ['du_dx','du_dy','dv_dx','dv_dy'])   # L122
BuoyancyGradients = namedtuple(..., ['zonal','merid'])                   # L134
def compute_velocity_jacobian(ds_merge, grid)                   # L145
def compute_buoyancy_gradients(ds_merge, grid)                  # L168
def _frontogenesis_formula(du_dx, du_dy, dv_dx, dv_dy, grad_bx, grad_by)   # L627
def frontogenesis_tendency(ds_merge, grid, *, jacobian=None, buoyancy_gradients=None)  # L654
def grad_b2(ds_merge, grid)                                     # L254

# dbof/preprocessing/physical_constants.py
G = 9.81            # m s^-2
RHO0_REFERENCE = 1000.0   # kg m^-3
```

### 2.4 Masking (`dbof`) — contains a confirmed bug

```python
# dbof/preprocessing/static_masks.py
def generate_halo_land_mask(ds_grid, target_km_res, DXC=None, DYC=None, stitched=True)  # L5
      # land_mask = (hFacC == 0); halo_km = target_km_res  <- KM, NOT CELLS
      # stitched=False -> native (face, j, i).  Returns bool ndarray, True = KEEP.

# dbof/preprocessing/halo_mask.py
def llc_native_grid_halo_mask(mask, dxC, dyC, halo_km)   # L5
      # input True = masked-out; output True = retained; needs skfmm; mask must be xarray.
```

***TRAP — confirmed bug at `halo_mask.py:74-75`:***

```python
        elif (phi==-1).any():          # the whole face is masked out
            return mask_f              # <-- returns a single 2-D face, ALL-TRUE ("keep"),
                                       #     from inside the loop, aborting remaining faces
                                       #     and inverting the convention.
```

Should be `halo_mask[face] = np.zeros_like(mask_f, dtype=bool)`. Our `masking.halo_mask`
must not hit this path silently — assert the returned shape, and collapse any `k` dim on
`hFacC` first (a `k`-carrying `hFacC` makes the mask 4-D and breaks `skfmm`).

### 2.5 Front finding, tracking, properties (`fronts @ viz_tools`)

```python
# fronts/finding/algorithms.py
def fronts_from_gradb2(gradb2, window=40, thin=False, rm_weak=None, dilate=False,
                       sharpen=False, despur=False, Lspur=None, connectivity=2,
                       threshold=90, thresh_mode='generic', n_workers=None,
                       min_size=7, verbose=False, debug=False)          # L12 -> bool ndarray

# fronts/finding/pyboa.py
def front_thresh(array, wndw=64, prcnt=90, mode='vectorized', n_workers=None, chunks='auto')  # L708
def cropping(array, min_size=7, connectivity=2)                          # L882

# fronts/front_tracking.py
def anchor_at(labels, step, label, *, pad=0.5) -> Anchor                 # L335
def anchor_at_point(labels, lon, lat, point_lon, point_lat, step, *,
                    max_km=80.0, pad=0.5) -> (Anchor, km)                # L346
def follow(labels_at, times, anchor, *, km_per_px=2.3, max_score=2.5,
           weights=None, min_pixels=5) -> Track                          # L505
      # km_per_px default 2.3 is 25% too large for tile 330 (1.7-2.1 km; median dxC 1.80,
      # dyC 1.95 km) -- pass it from the measured grid (corrected 2026-09-28, M0 task 3)
@dataclass class Track: anchor, labels, centres, links                   # L239
      # .label_at(step), .steps(), .gaps(n_steps), .weakest(n=3)
```

***TRAP — timestamp format.*** `front_tracking.parse_time` (L116) uses
`"%Y-%m-%dT%H_%M_%S"`, e.g. `'2012-07-02T00_00_00'` — **underscores, not colons**. This is
*not* `DATE_FMT` from `dbof` (`'%Y-%m-%d %H:%M:%S'`). Two formats are live in this project;
`tracking.py` must convert.

`labels_at` is `Callable[[int], np.ndarray]` — `step -> (H, W)` int label array, 0 = background.

```python
# fronts/properties/colocation.py
def colocate_fronts_with_properties(labeled_fronts, properties: Dict[str, np.ndarray],
                                    stats=None, percentiles=None, min_npix=1,
                                    nan_policy='propagate', dilation_radius=0,
                                    checkpoint_dir=None) -> pd.DataFrame      # L101
      # stats default ['mean','std','median']; also min/max/count
      # returns one row per front: flabel, npix, {prop}_{stat}, {prop}_p{pct}
      # NOTE: pass nan_policy='omit' -- our fields are NaN over land/halo.

# fronts/properties/algorithms.py
def group_fronts(fronts_binary, lat, lon, fronts_file, output_dir,
                 n_workers=None, skip_curvature=False) -> pd.DataFrame        # L84
      # fronts_file must contain YYYY-MM-DDTHH_MM_SS

# fronts/runs/prototypes/one_full/build_v5.py   -- the tile front-finding entry point
#   step 1 -> build.tile_find -> fronts.preproc.gradb2.generate_tile_gradb2
#   config keys: build.tile_find{name, lon/lat | i_rect/j_rect, property, pipeline}
#   finding config "D" = fronts/finding/configs/finding_config_D.yaml -- READ IT rather than
#   assuming its window/percentile; the defaults in fronts_from_gradb2 are NOT config D.
#   (noted 2026-09-30, M1 task 7 audit, from task 7a) neither `build.tile_find` nor
#   `generate_tile_gradb2` exists at this checkout (grep over fronts/); build_v5.py step 1 calls
#   generate_for_channels / export_channels and gradb2.py has only generate_gradb2. M4's entry
#   point is fronts_from_gradb2 directly (as finding/run.py does), with config D's values and an
#   explicit n_workers, then `fronts &= isfinite(gradb2)` (task 7a's safe recipe).

# fronts/viz/curtains.py
def extract_main_axis(front_mask) -> (L,2) int32 (j,i)                        # L103
def path_metrics(path, XC_rect=None, YC_rect=None, *, smooth=False,
                 smooth_window=5) -> dict                                     # L237
      # keys: dist_px, dist_km, tangents, normals, smoothed
def perpendicular_path(path, normals, idx, half_width) -> (2*hw+1, 2) float   # L367
def transect_front_crossings(axis_path, normals, front_mask, half_width)      # L526
```

---

## 3. Data contracts

All under `dev/frontogenesis/data/`.

### 3.1 `tile330_grid.zarr` — static, pulled once

```
dims   : (j: 720, i: 720) plus staggered i_g, j_g
coords : XC(j,i), YC(j,i),                    # COORDS, as in the raw gridfile and §3.2, so
                                              # xr.merge([hour, grid]) works; face = 10 as a
                                              # scalar coord, open_grid(with_face=True)
                                              # restores the (face, j, i) dim the dbof
                                              # operators expect (corrected 2026-09-28,
                                              # M0 task 5; process_llc4320_grid demotes
                                              # them, load_grid re-promotes)
vars   : dxC, dyC, dxG, dyG, rA, rAz, CS, SN, hFacC, Depth,
         hFacW(j,i_g), hFacS(j_g,i),          # U/V land masks; raw gridfile, dropped by
                                              # process_llc4320_grid -- re-attach
         drF, Z, Zl                           # 0-d, k=0 only: 1.0, -0.5, 0.0 (OSN gridfile)
attrs  : face_index=10, j_face_start=0, i_face_start=2880, rect_i=13320, rect_j=9720,
         source='OSN', git_commit, created,
         orientation='CS=0, SN=-1: j/V/dyC zonal (eastward), i/U/dxC meridional
                      (i increasing southward); u_east=V, v_north=-U',
         dx_km_37N=1.71, dy_km_37N=1.85, land_fill='NaN'
```
Comodo attrs (`axis`, `c_grid_axis_shift=-0.5`) must be present on `i`,`i_g`,`j`,`j_g`.
*(`hFacW`, `hFacS`, `drF`, `Z`, `Zl` and the orientation/spacing/land attrs added 2026-09-28,
M0 tasks 2-3.)*

### 3.2 `tile330_raw_20120702T00_72h.zarr` — the Phase-1 product

```
dims   : (time: 72, j: 720, i: 720) + i_g, j_g
vars   : Theta(time,j,i), Salt(time,j,i), U(time,j,i_g), V(time,j_g,i),
         W(time,j,i), Eta(time,j,i),
         KPPhbl(time,j,i), oceTAUX(time,j,i_g), oceTAUY(time,j_g,i)
coords : time (datetime64), XC, YC, niter(time), face (= 10, scalar), k, k_l
attrs  : iterations (list), endpoint, stores=['llc_surf','llc_wind'], git_commit
```
`face` is kept as a scalar coord in both stores; `expand_dims('face')` on a snapshot (or
`open_grid(with_face=True)` for the grid) restores the `(face, j, i)` layout the dbof operators
expect. `XC`/`YC` are coords in both, so an hour merges with the grid plainly (corrected
2026-09-28, M0 task 5).
`KPPhbl`/`oceTAU*` come from the **second** OSN store (`llc_wind`), which covers our window.
Heat fluxes are in neither *OSN* store — they come from the chunk store (§3.3), which is what
makes the diabatic term measurable rather than inferred (Q13). Land is **NaN** in every field
(M0 task 3): centred fields exactly where `hFacC == 0`, `U` where `hFacW == 0`, `V` where
`hFacS == 0`. Exception: `oceTAUX`/`oceTAUY` arrive masked with the *centred* mask despite
living on `i_g`/`j_g` — store them as they come, but re-mask with `hFacW`/`hFacS` (§3.1)
before any stress-divergence.

### 3.3 `tile330_chunk_20120702T00_72h.zarr` — the extra budget terms (Q13)

From the hourly full-depth `monterey_bay` transfer. **Store only what we need** — the source
holds all 51 levels; we keep three. *(corrected 2026-10-04, M2 task 6: the source writes each
variable as **one 51-level zstd object per hour** (`(51, 1, 720, 720)`; 306 MB compressed per
hour, 22 GB for the 72), so a `k = 0..2` read is not level-selective — it fetches the whole
object, 174 MB per hour / 12.5 GB over the window, and keeps ~14 MB per hour; M2 task 4.)*

```
dims : (time: 72, k: 3, k_l: 3, j: 720, i: 720)   # no staggered dims: the store has no U/V
                                                  # (corrected 2026-10-04, M2 task 6)
vars : Theta(time,k,j,i), Salt(time,k,j,i), W(time,k_l,j,i),   # W on interfaces k_l=0..2;
                                                              # k_l=1 is the top-cell base.
                                                              # The SOURCE puts W on k_p1 (52
                                                              # interfaces); k_p1=n is the top
                                                              # face of cell n = k_l=n (continuity
                                                              # to 7e-12 m/s, M2 task 4), renamed
                                                              # at write (corrected 2026-10-04,
                                                              # M2 task 6)
       oceQnet(time,j,i), oceQsw(time,j,i), oceFWflx(time,j,i), drF(k)
coords: time (seconds since 2011-09-10, as §3.2), niter(time) = the OSN iteration,
       mit_iteration(time) = niter - 10368 (the source's selected_iteration), face (= 10),
       k, k_l, j, i, XC, YC, Z(k), Zl(k_l)          # (added 2026-10-04, M2 task 6)
attrs : source='CHUNKS/monterey_bay', levels='k=0..2, k_l=0..2', git_commit,
       flux_sign_convention='positive downward', provenance (records the negation below)
```
**Flux sign (corrected 2026-10-04, M2 task 6; decided M2-Q6 (a)).** The source's `oceQnet`,
`oceQsw`, `oceFWflx` attrs say `+=down` but the data are **upward-positive** (`oceQsw <= 0`
everywhere). The three are **negated at write time**, so this store holds them
**downward-positive** (stored `oceQsw >= 0`, `oceQnet > 0` = ocean warming, `oceFWflx < 0` = net
evaporation), each with `sign_convention` / `source_sign_convention` / `sign_conversion` attrs.
§4.6's `surface_flux_term` therefore reads the documented convention and must **not** negate
again. The forcing is 6-hourly, linearly interpolated (`forcing_note`).

### 3.4 `tile330_derived_L{L}.zarr` — per filter scale

```
vars : b, G, two_F, DGDt_semilag, DGDt_euler, subfilter, vertical, surface_flux,
       residual, delta, sigma_n, sigma_s, sigma_mag, theta_align
```

### 3.5 `tile330_masks.nc`

```
vars : mask_ocean, mask_halo, mask_offshore, mask_edge, mask_analysis, coast_distance_km
       # mask_edge: the tile-edge margin (§4.2; added 2026-09-28, M0 task 5)
```

---

## 4. Module specifications

Signatures below are the **contract**. Write them exactly; deviations ripple into the
prompt docs.

### 4.1 `py/osn_tiles.py` — library-route data access

```python
TILE_RECT_I, TILE_RECT_J = 13320, 9720          # tile 330; planning §4
OSN_ENDPOINT = "https://mghp.osn.xsede.org"

def tile_spec():                                  -> TileInfo   # rect_ij_to_tile wrapper
def load_grid(endpoint=OSN_ENDPOINT, tile=None):  -> xr.Dataset # tile-subset, comodo-annotated, computed
def build_xgcm(grid_ds):                          -> xgcm.Grid  # use_connections=False
def load_hour(ts, tile, endpoint=OSN_ENDPOINT):   -> xr.Dataset # Theta,Salt,U,V,W,Eta (lazy)
def load_wind_hour(ts, tile, endpoint=...):       -> xr.Dataset # KPPhbl,oceTAUX,oceTAUY,...
def pull_series(timestamps, out_zarr, tile=None,
                endpoint=OSN_ENDPOINT, include_wind=True,
                clobber=False):                   -> str        # THE MISSING CONCAT STEP
```
`pull_series` is the piece that **exists nowhere in either repo** (planning §9). It must be
resumable: skip timestamps already present in `out_zarr` unless `clobber`.
*(corrected 2026-10-03, M2 task 1)* Written with keyword-only extras after the signature above
(`grid_ds, attempts, backoff, sleep, log, report`; `report` is a dict filled in place because the
return is the path). The per-hour atomic append / repair-on-resume machinery is
`py/zarr_series.py` (`append_hour`, `repair_trailing`, `present_times`, `with_retries`) — §4.6
`load_chunk_levels` reuses it — and the §3.2 checker is
`py/series_verify.py: verify_series(out_zarr, timestamps, grid_ds=None) -> dict`.

### 4.2 `py/masking.py`

```python
def ocean_mask(grid_ds):                          -> np.ndarray  # bool, True=ocean, from hFacC
def halo_mask(grid_ds, halo_cells=7):             -> np.ndarray  # bool, True=retained
def coast_distance_km(grid_ds):                   -> np.ndarray  # float, km to nearest land
def offshore_mask(grid_ds, min_km=100.0):         -> np.ndarray  # bool
def edge_mask(grid_ds, edge_cells=7):             -> np.ndarray  # bool, False within edge_cells of ANY tile edge
def analysis_mask(grid_ds, halo_cells=7,
                  min_km=100.0, edge_cells=7):    -> np.ndarray  # halo & offshore & ocean & edge
```
**`edge_mask` exists because the tile-edge rim is finite, not NaN** (corrected 2026-09-28,
M0 task 5; §2.2, planning §5.5): xgcm's `padding='fill'` pads the missing high staggered point
with 0, so `G` is wrong by ~1e6x on the low edges (`j = 0`, `i = 2880`) and ~0.5x on the high
edges (`j = 719`, `i = 3599`), and the Jacobian is wrong 1 cell deep on the low edges and 2 on
the high. The land halo cannot see it (`skfmm` measures distance from `hFacC == 0`) and the
offshore cut leaves the open-ocean west and north edges alone. Minimum `edge_cells` for the raw
operators is 2; the default 7 matches the land halo once the filter half-width (4) is counted.
`halo_mask` wraps `llc_native_grid_halo_mask` and **must handle two known defects**
(planning §5.5): the 2-D early return when a face is entirely land, and a `k`-carrying
`hFacC` that makes the mask 4-D and breaks `skfmm`. Collapse `k` first; assert output shape.

**Our API takes cells; the underlying helper takes km.** `generate_halo_land_mask(ds_grid,
target_km_res, ...)` uses `target_km_res` directly as `halo_km`, so `masking.halo_mask` must
convert: `halo_km = halo_cells * median(dxC)`. Our requirement is **7 cells** (3 for the
Jacobian+interp stencil — measured reach is 1 cell for `G` and 2 for the Jacobian/`F`, so this
is one cell conservative (M0 task 5); 4 for the widest filter half-width), which is ~12-14 km across the
tile's 1.7-2.1 km spacing (corrected 2026-09-28, M0 task 3; `dxC` is the *meridional* spacing
on this face, `dyC` the zonal — see §3.1 orientation). **Convert from the measured `dxC`;
never hard-code the km value.** Both `halo_mask` and `analysis_mask` therefore take
`halo_cells`. Land is already NaN in the OSN fields (planning §5.5), so `ocean_mask` from
`hFacC` and the finite pattern of `Theta` must agree cell for cell — assert it — and the
halo's job is filter support and `skfmm` distance, not removing a `b(0,0)` ribbon.

### 4.3 `py/operators.py` — the single shared operator

This module is the reason the study is trustworthy: **both sides of the comparison go
through it** (planning §5.1).

```python
def buoyancy(ds):                                 -> xr.DataArray  # wraps calculate_fields.buoyancy_of_field
def lowpass(field, L_cells):                      -> same type     # L_cells=0 -> identity
def grad_b(b, grid_ds, grid):                     -> (b_x, b_y)    # via calculate_native_gradient_tracer (geographic)
def gradb2(b, grid_ds, grid):                     -> G             # = b_x^2 + b_y^2 from grad_b, NOT
      # calculate_grad_squared_tracer (a different stencil, 0.911x; V3 would start biased
      # by ~0.9). Both sides of the comparison share b_x, b_y (§1.1; M0 task 5).
def jacobian(U, V, grid_ds, grid):                -> (u_x, u_y, v_x, v_y)
def frontogenesis(b, U, V, grid_ds, grid,
                  form='discrete'):               -> F             # inputs ALREADY filtered
      # (corrected 2026-09-29, M1 task 6) form='discrete' (the default) is the discretely
      # consistent F = -sum_k (L_k b) [L_k, u.grad] b -- the commutator of the gradient stencil L
      # with advection, which is what the semi-Lagrangian [G(x,t+dt) - G(x_d,t)]/dt tends to under
      # exact advection; the neighbour gradient is 4th order. The continuum chain-rule product
      # -(grad b)^T (grad u)(grad b) fails on the grid by (2/3)(dx/ell)^2 at a front of width ell
      # (V3 null slope 0.95 synthetic / 0.76 real hour); the consistent form gives 1.00 / 0.98.
      # form='chain' is the repo-equivalent path (bit-for-bit frontogenesis_tendency), kept as
      # criterion 7's oracle. Also added: centred_model_velocity (shared with semilag) and
      # model_basis_gradient. NaN reach of the default: taxicab L/2 + 4 from land (chain L/2 + 3;
      # chessboard L/2 + 2 for both); finite on all of mask_analysis at every L.
      # (decided 2026-09-30, M1-Q1) M3 runs BOTH forms: 'discrete' is the primary slope, 'chain'
      # runs alongside, and the discrete-vs-chain difference is a stated systematic.
def strain_divergence(U, V, grid_ds, grid):       -> (delta, sigma_n, sigma_s, sigma_mag)
      # calculate_native_strain_vorticity returns a DICT with sigma_s and vorticity on
      # CORNERS (j_g, i_g) -- interpolate to centres before combining. See §2.3.
      # Its strain pair is MODEL-basis: rotate (sigma_n, sigma_s) by 2*alpha (cos alpha = CS,
      # sin alpha = SN; on face 10 a sign flip of both) so it shares grad_b's geographic
      # basis -- confirmed on the real face, slopes +0.85 / +1.00 against the Jacobian
      # (added 2026-09-29, M1 task 2).
def strain_alignment(b_x, b_y, sigma_n, sigma_s): -> theta         # radians, for Figure 4
      # theta from the COMPRESSIONAL axis, folded to [0, pi/2]:
      # F = -(1/2) delta G + (1/2) |sigma| G cos 2theta (plus sign; planning §2.4,
      # corrected 2026-09-29, M1 task 2; sign FINAL, decided 2026-09-30, M1-Q3)
```
`frontogenesis` takes **pre-filtered** inputs — filtering happens once, at the call site, so
it cannot silently differ between the two sides.

### 4.4 `py/semilag.py`

```python
def centre_velocities(U, V, grid_ds, grid):       -> (u_c, v_c)   # interp to cell centres
def departure_index(u_c, v_c, grid_ds, dt=3600.0,
                    n_iter=3, vel_order=3):       -> (di, dj)     # index-space displacement
      # (corrected 2026-09-29, M1 task 6) the velocity at the trajectory midpoint is interpolated
      # at order 3, not bilinearly: the displacement is insensitive (< 0.03 cell) but the strain
      # of the departure map is not -- bilinear smoothing of grid-scale velocity structure made
      # it 0.965 of the Jacobian's on real front pixels (0.992 cubic), V3 'llc' 0.94 -> 0.98.
      # measured_DGDt takes the same vel_order.
def interp_to_departure(field, di, dj, order=3):  -> same shape   # cubic+ REQUIRED
def measured_DGDt(b_t, b_tp1, u_mid, v_mid, grid_ds, grid,
                  dt=3600.0, order=3):            -> DGDt
def eulerian_DGDt(G_t, G_tp1, u_mid, v_mid, grid_ds, grid,
                  dt=3600.0):                     -> DGDt         # independent cross-check
```
**`measured_DGDt` interpolates `b`, then differentiates** — it does *not* interpolate `G`.
Interpolating `G` bilinearly biases it negative at maxima by 25-80% of the signal, i.e. it
fabricates frontogenesis (planning §5.3). `order >= 3` is not optional.
*(Clarified 2026-09-29, M1 task 3.)* "Interpolate `b`, then differentiate" means: `b_t`
interpolated onto the five-point tracer stencil **centred at the departure point** (the
displacement held fixed across the stencil), then the `operators.grad_b` stencil —
`semilag.gradb2_at_departure`. It does **not** mean the gradient of the shifted field
`b_t(x_d(x))`: that field is the adiabatic prediction of `b_{t+dt}`, so its `G` cancels the
kinematic term and `[G_{t+dt} - G(b_t(x_d(x)))]/dt` measures the *residual* (~0 under pure strain,
`test_semilag.py`), not `DG/Dt`. The interpolation is a local Lagrange kernel of odd order
(1 / 3 / 5 = 2 / 4 / 6 nodes per axis; NaN wherever the support touches NaN or leaves the tile),
not a prefiltered B-spline (`map_coordinates`), whose recursive prefilter leaks a filled NaN
46% / 12% / 3.3% / 0.9% into the coefficients 1 / 2 / 3 / 4 nodes away. Departures: `di = U dt/dxC` along `i`,
`dj = V dt/dyC` along `j`, verified against the haversine centre distances on the tile grid.
*(Decided 2026-09-30, M1-Q6.)* `order = 3` stays the default; M3 reports its slope at `order = 5`
as a sensitivity (order 5 costs a 6-node NaN rim per axis against 4). V4's bar is 0.28-1.0% of `G`
per hour at order 3 (1.5-cell / 1-cell front).

### 4.5 `py/coarsegrain.py`

```python
def subfilter_flux(b, U, V, L_cells, grid_ds, grid):   -> (tau_x, tau_y)  # mean(ub) - ubar bbar
      # on the U/V points, model basis, from the UNFILTERED b, U, V (filters internally;
      # L_cells may be a sequence = the composite filter, for the Germano identity)
def subfilter_bdelta(b, U, V, L_cells, grid_ds, grid): -> tau_delta       # mean(b delta) - bbar deltabar
def subfilter_term(b_bar, tau_x, tau_y, grid_ds, grid,
                   tau_delta=None):                    -> term            # -grad(bbar).grad(div tau - tau_delta)
```
Without this the filter sweep is uninterpretable (planning §5.4).
*(Corrected 2026-09-29, M1 task 4.)* **Units:** `subfilter_term` returns the term in **F units**
(`D/Dt(Gbar/2) = F + term + ...`, planning §5.4's equation); the §3.4 budget field `subfilter`
is **`2 * subfilter_term`** so that it sits beside `two_F` and the measured `DGDt`. **Divergence:**
`-grad(bbar).grad(div tau)` is the non-divergent form; the surface flow is divergent, and the
exact subfilter advection is `sigma = div tau - tau_delta` with `tau_delta = mean(b div u) -
bbar div ubar` (`subfilter_bdelta`). On hour 0 the flux form alone overstates the term 2.2x in
rms at every `L` — M3 must pass `tau_delta`. **Placement:** `u b` is formed on the staggered
velocity points (`b` averaged to the U/V point, flux form, as the model advects), `div` is the
model's flux-form divergence at the centres, and both gradients are `operators.grad_b`.

### 4.6 `py/vertical.py` — the extra budget terms from the chunk store (Q13)

`load_chunk_levels` is an **M2** step (it is a data pull); the physics functions below are **M3**.

```python
def load_chunk_levels(window, k_max=2, out_zarr=None):  -> str | xr.Dataset  # §3.3, M2
def b_z(Theta, Salt, grid_ds, drF):                     -> xr.DataArray      # top-cell b_z
def vertical_term(b, b_x, b_y, b_k1, W_k1, drF,
                  grid_ds, grid):                       -> xr.DataArray      # grad_h b . grad_h[ -W_k1 (b_k1 - b)/drF ]
def surface_flux_term(b_x, b_y, oceQnet, oceQsw, oceFWflx,
                      Theta, Salt, drF, grid_ds, grid):  -> xr.DataArray     # grad b . grad B_sfc
```
`vertical_term` (corrected 2026-09-28, M0 task 3) takes the chunk store's **`W(k_l=1)`** — the
model's own cell-base vertical velocity, which already contains both the `dEta/dt` part
(~5e-5 m s^-1, tidal; LLC4320 has a *linear* free surface, so `W(k_l=0) = dEta/dt` and is a
coordinate-relative flux the tracer equation sets to zero) and the convergence part
(`+drF*delta`). Do **not** rebuild it from `delta`, and do not use the OSN `W(k_l=0)` as a
flux. Compute the top-cell vertical advective tendency `-W_k1 (b_k1 - b)/drF` first and take
the horizontal gradient of that, rather than the factorised `-b_z (w_x b_x + w_y b_y)`,
which drops `-w grad(b_z) . grad b`; report the factorised form as a diagnostic only.
`surface_flux_term` must convert heat and freshwater flux into a buoyancy tendency for the top
cell (thermal + haline expansion coefficients from the same JMD95 EOS as `operators.buoyancy`),
and must treat the **shortwave absorbed inside the top cell** separately from `oceQnet` —
that is why `oceQsw` was requested. `drF[0] = 1.0 m`, `Z[0] = -0.5 m` (confirmed; also carried
as 0-d scalars by the OSN gridfile and written to §3.1).
*(corrected 2026-10-04, M2 task 6)* The §3.3 store already holds the three fluxes
**downward-positive** — negated at write from the source's upward-positive data, whose `+=down`
attrs are wrong (M2-Q6 (a), task 5) — so `surface_flux_term` takes them as documented here and
**must not flip the sign**; the chunk `W(k_l=1)` is the source's `W(k_p1=1)`; and `drF[0..2] =
1.0, 1.14, 1.30 m`, `Z[0..2] = -0.5, -1.57, -2.79 m` were read from the chunk `grid.zarr` (task 4)
and are in the store as `drF(k)`, `Z(k)`, `Zl(k_l)`.

### 4.7 `py/budget.py`

```python
def compute_budget(raw_ds, grid_ds, grid, masks, L_cells, dt=3600.0,
                   chunk_ds=None):                      -> xr.Dataset
      # chunk_ds (§3.3) supplies the MEASURED vertical and surface-flux terms.
      # Without it the budget still runs but those terms are absent and the
      # residual reverts to a catch-all -- say so loudly in closure_report.
def closure_report(budget_ds, mask):                    -> dict
```
`compute_budget` returns `measured, two_F, subfilter, vertical, surface_flux, residual` and is
the object on which the Phase-2 exit criterion is evaluated. With the Q13 transfer the residual
reduces to **numerical diffusion + interior KPP**, not "everything we could not compute".

### 4.8 `py/stats.py`

```python
def slope_ols(x, y);  def slope_tls(x, y);  def slope_bisector(x, y)
def ratio_estimator(x, y):                              -> sum(y)/sum(x)
def binned_conditional_mean(x, y, bins, split_sign=True)-> DataFrame
def feature_bootstrap(x, y, labels, estimator, n=1000)  -> (lo, hi)
```
**Bootstrap over frontal features and hours, never over pixels** (planning §11). Every slope
is reported relative to the M1 discrete-null baseline.

### 4.9 `py/validate.py` — four gates plus two supporting figures (six PNGs)

```python
# the four gates
def test_cartesian_deformation(alpha=1e-5, png=True)   -> dict   # V1; G ~ exp(2 alpha t)
def test_native_metric(grid_ds, png=True)              -> dict   # V2; analytic f(XC,YC)
def test_discrete_null(velocities='strain', png=True)  -> dict   # V3; MUST give slope = 1 +/- 0.05
      # expect a ~0.8 attenuation of the interpolated Jacobian before co-location (M0 task 5:
      # trace vs flux-form divergence slope 0.80, corr 0.97); G from the same b_x, b_y as F
      # (corrected 2026-09-29, M1 task 6) the semi-Lagrangian null is BLIND to that attenuation:
      # the Jacobian trace is exactly D_h u_c (the (1,2,1)/4 mean of the flux-form divergence)
      # and the departure map is built from the same u_c, so both sides see 0.85x of the flux
      # form. What the null measured was the chain-rule violation, (2/3)(dx/ell)^2: first attempt
      # 0.950 (strain) / 0.758 (llc); with the consistent F and the cubic departure velocity
      # 1.004 [0.995, 1.017] / 0.981 [0.970, 0.994] -- the slope Figure 2 draws is `slope`.
      # (corrected 2026-09-30, M1 task 7) the attenuation does NOT bias the slope: V3b below
      # measured its effect at -0.006 +/- 0.025 with a flux-form truth.
def test_fv_null(velocities='strain', png=True)        -> dict   # V3b (task 6b): RECORDED BIAS, not a gate
      # the same pipeline with a flux-form finite-volume truth (fvadvect; OS7MP-like, MITgcm
      # conventions): 0.975 [0.954, 1.003] on the real hour, 0.985 [0.977, 0.996] strain --
      # M3's systematic band 0.954-1.003 (0.975 +/- 0.025) around the 0.981 baseline; no upward
      # correction; per-width shortfall -2% / -4% / -11% at 2 / 1.5 / 1 dx. `res['bias']`.
def test_interpolation_bias(png=True)                  -> dict   # V4; uniform flow, true DGDt = 0
      # (decided 2026-09-30, M1-Q6) the bar: 0.28-1.0% of G per hour at order 3 (1.5-cell / 1-cell
      # front, real-hour displacements); order 3 the default, order 5 an M3 sensitivity (0.06%).
# two supporting figures
def demo_interp_half_cell(png=True)                    -> dict   # V5; the figure Lauren asked for
def qa_land_halo(grid_ds, png=True)                    -> dict   # V6; coastline before/after halo,
      # plus the finite tile-edge rim and the edge_cells margin that removes it (M0 task 5)
```

**Four gates, six PNGs (V1-V6).** Every one writes to `dev/frontogenesis/figs/` (the fronts
`.gitignore` ignores `*.png`; `figs/.gitignore` un-ignores them with `!*.png`, added 2026-09-28,
M0 task 5 — verify with `git check-ignore -v` if a figure fails to show up) — part of
acceptance, not an extra. `demo_interp_half_cell` (V5) renders a synthetic front shifted half a
cell and plots truth vs `G` from bilinear-`G` vs `G` from cubic-`b`, annotating the negative bias
at the maximum. `test_discrete_null` must also **return the fitted slope**, because Figure 2
draws it as a baseline line — at **0.981 with its band [0.970, 0.994]** (decided 2026-09-30,
M1-Q4). V3b (`test_fv_null`, task 6b) is a seventh PNG, `figs/V3b_fv_null.png`, a recorded bias.

`test_native_metric` (V2) and `qa_land_halo` (V6) need the real tile grid, so they are **not**
pure-offline: mark them `@pytest.mark.needs_grid` and point them at `tile330_grid.zarr`, which
is an **M0** deliverable.

### 4.10 `py/tracking.py`, `py/figures.py`

```python
def find_fronts_series(derived_zarr, mask, config='D')  -> dict[time -> label array]
def advect_mask(mask_bool, u_c, v_c, grid_ds, dt=3600.0) -> np.ndarray
      # flow-predicted mask at t+dt: advect as float through semilag, threshold at 0.5.
      # NEVER interpolate the integer label field -- only the boolean mask.
def flow_weighted_score(candidate, reference, predicted, radius,
                        predicted_mask=None, weights=None,
                        **kw)                           -> (float, dict)
      # wraps front_tracking.score_candidate, adding IoU(predicted_mask, candidate) as an
      # extra scored term via its existing `weights` dict. Additive, not a rewrite.
      # `predicted_mask` comes from advect_mask(); `weights` gains one key, e.g. 'flow_iou'.
def track_top_n(labels_by_time, times, n=10, flow=None) -> list[Track]
      # `times` MUST be '%Y-%m-%dT%H_%M_%S' (underscores) for front_tracking.parse_time,
      # NOT dbof's '%Y-%m-%d %H:%M:%S'. Convert here; see the §2.5 trap.
def tracking_quality(tracks, flow)                      -> DataFrame
      # distribution of (follow()-chosen displacement - flow-predicted displacement);
      # plus split/merge flags where a predicted mask overlaps two candidate labels.
def front_strength_series(track, G_by_time, two_F_by_time,
                          matched_pixels=True)          -> DataFrame
      # matched_pixels=True evaluates on the ADVECTED pixel set, not "front at t" vs
      # "front at t+dt" -- front-mean G over a changing pixel set is not material (Q14).
```
`figures.py`: one function per figure, `fig01_maps(...)` ... `fig10_term_budget(...)`, plus
`figV1..figV6`, each writing a PNG to `dev/frontogenesis/figs/`. `fig02` draws the M1 baseline at
**0.981 with its band [0.970, 0.994]** (the V3 real-velocity slope, `form='discrete'`), with V3b's
systematic band 0.954-1.003 beside it (decided 2026-09-30, M1-Q4).

---

## 5. Test strategy

`dev/frontogenesis/py/tests/`, pytest, **all offline** (no network) except one explicitly
marked `@pytest.mark.network` smoke test of the OSN pull.

| Test | Guards |
|---|---|
| `test_operators.py` | gradient of an analytic field; `F` vs the repo's `frontogenesis_tendency` unfiltered; factor-of-two convention |
| `test_masking.py` | halo width; the two `halo_mask` defects; `True`=retained; `ocean_mask == isfinite(Theta)`; the tile-edge margin covers the crop-test rim (`m0_qa_checks.check_edge_rim`; M0 task 5) |
| `test_semilag.py` | zero-velocity identity; uniform-flow translation; interpolation order |
| `test_coarsegrain.py` | `tau` -> 0 as `L` -> 0; Germano consistency |
| `test_stats.py` | estimators on synthetic data with known slope |
| `test_validate.py` | the four gates run and report |
| `test_nan_finding.py` | **`fronts_from_gradb2` on a NaN-containing field** — no such test exists today (planning §10) |

---

## 6. Milestones

Each becomes one execution prompt doc.

### M0 — Access and reconnaissance  *(planning Phase 0a)*

**Goal:** prove we can read the data and settle every open empirical question before
writing physics.

Tasks:
1. Environment: `dbof` importable (`pip install -e . --no-deps` + `xgcm>=0.10`,
   `scikit-fmm`, `s3fs`, `ujson`, `xmitgcm`, `zarr`, `dask`). Fall back to a py3.13 env if
   py3.14 wheels are missing.
2. `osn_tiles.tile_spec()`, `load_grid()`, `load_hour()` for **one** timestamp.
3. Answer, empirically, and record in the log (**five** questions — `drF` is not among them;
   the OSN gridfile carries it as a 0-d scalar, `drF = 1.0`, confirmed in M0 task 2 and
   cross-checked in M2 from the chunk grid). **All five answered 2026-09-28 (M0 task 3 log):**
   - Is OSN land stored as **0 or NaN**? — **NaN**, cell-for-cell equal to `hFacC`/`hFacW`/`hFacS`.
   - Is `W[k_l=0] ~ 0`? — **No: `W(0) = dEta/dt`** (linear free surface); planning §2.2 rewritten.
   - `dxC`/`dyC` at 37N — **1.71 / 1.85 km** (tile 1.68-2.07); face 10 is rotated (§3.1 attrs).
   - `grad CS`, `grad SN` terms vs strain — **identically zero**; `u tan(phi)/a` 0.1% median, <= 0.7% p99.
   - Advection scheme — **OS7MP (`tempAdvScheme = 7`)**, `diffKhT = 0`; scale-dependent `kappa_num` in planning §2.3.
4. **Write `tile330_grid.zarr` (§3.1)**, including `hFacW`, `hFacS`, the 0-d `drF`/`Z`/`Zl` and
   the orientation attrs. The grid is static — one pull. M1's V2 and V6 tests need it, so it
   belongs here rather than in M2.
5. **Pull two consecutive timestamps**, not one. M1's gate 3 has a variant driven by real LLC
   velocities, which needs a midpoint velocity and therefore two hours.
6. QA plot of one snapshot: `Theta`, `G`, and the land mask from `hFacC`. **No halo yet** — the
   halo is M1. Land is NaN (task 3), so there is no coastal gradient ribbon to see; the plot
   should instead show the stencil's own NaN rim along the coast (`G` undefined 1 cell from
   land, the Jacobian 2 cells — not "~3") and confirm `isfinite(Theta) == (hFacC > 0)`. Also
   show the invalid rim that `_tile_indexer` leaves on **all four** tile edges (the staggered
   dims take the same slice as the centred ones and xgcm pads with 0, so the rim is finite:
   1 cell for `G`, 1 / 2 cells for the Jacobian on the low / high edges). **Done 2026-09-28:**
   `figs/m0_qa_tile330_20120702T00.png`, `py/m0_qa_plot.py`, `py/m0_qa_checks.py`.
   *(Corrected 2026-09-28, M0 task 3; measured values M0 task 5.)*
5. Confirm the three §2 traps on real data: comodo attrs survive `process_llc4320_grid`
   (or are restored), `halo_mask.py:75` is not reached, and the two timestamp formats are
   converted correctly.

**Acceptance:** two consecutive hours load end to end from both OSN stores;
`tile330_grid.zarr` written; all five questions answered in the log with numbers; QA plot written
and the coastline inspected (a stencil NaN rim is expected; a gradient ribbon is not, since land
is NaN — if one appears, something upstream has filled NaN with 0).
**M0 closed 2026-09-28** (task-5 log entry: all five criteria PASS; all four §2 traps confirmed
on real data).

### M1 — Operators and validation  *(planning Phase 0b)*  — **HARD GATE**

**Goal:** operators that are known correct before any science.

Tasks: `masking.py`, `operators.py`, `semilag.py`, `coarsegrain.py`, `validate.py`, plus their
tests. **`masking.py` is M1's, so `tile330_masks.nc` (§3.5) is written here**, using the static
grid from M0 — not in M2. `analysis_mask` includes the **tile-edge margin** (`edge_cells`, §4.2)
— the edge rim is finite and the land halo does not remove it (added 2026-09-28, M0 task 5). `coarsegrain.py` is also M1's: V-gate closure of `tau` is part of
validating the operators, not part of the budget run.

**Acceptance — all four must pass:**
1. Cartesian deformation reproduces `exp(2 alpha t)` to < 1% **at the reference front width
   8 dx** (decided 2026-09-30, M1-Q5: 0.78% over 8 h there; 1.57 / 3.09 / 4.43 / 7.86% at 6 / 4 /
   3 / 2 dx, the centred-stencil truncation `G` and `F` share; the semi-Lagrangian step alone is
   < 0.36% at every width).
2. Native-metric test reproduces analytic gradients to < 1%.
3. **`test_discrete_null` gives slope = 1 +/- 0.05 on front pixels.** If it fails, co-locate
   the operators or raise the scheme order until it passes. *Do not proceed on a failure* —
   C-grid interpolation attenuation alone can bias the slope 0.7-1.4 (planning §6); M0 task 5
   measured the interpolated Jacobian trace at **0.80x** the flux-form divergence (corr 0.97),
   so a correction of that size is expected, and `G` must be `b_x^2 + b_y^2` from the same
   `grad_b` as `F` (§1.1) or the slope starts a further 0.91x off (corrected 2026-09-28, M0
   task 5). *(Corrected 2026-09-30, M1 task 7: PASS at 1.004 [0.995, 1.017] strain / 0.981
   [0.970, 0.994] llc with `form='discrete'` and the cubic departure velocity (task 6). The
   attenuation is invisible to this null by construction, and V3b (task 6b, a flux-form truth)
   measured its effect on the slope at **−0.006 ± 0.025** — it does not bias the slope.)*
4. `test_interpolation_bias` quantifies the uniform-flow bias; it becomes a permanent error
   bar on every later slope. *(Decided 2026-09-30, M1-Q6: **0.28-1.0% of `G` per hour at order
   3**; order 5 as an M3 sensitivity. V3b's recorded bias 0.975 [0.954, 1.003] is the
   model-advection systematic band around the 0.981 baseline.)*
5. **All six PNGs (V1-V6: four gates plus two supporting) written to `figs/`** (seven with V3b,
   task 6b). Lauren asked for these
   decisions to be visible rather than asserted; they are acceptance criteria, not extras.
6. Tests pass: `test_operators.py`, `test_masking.py`, `test_semilag.py`, `test_coarsegrain.py`,
   `test_validate.py`, `test_nan_finding.py` (prompt 2 criterion 6).
7. Regression: `operators.frontogenesis(form='chain')` is bit-for-bit `frontogenesis_tendency`
   (reworded 2026-09-30, M1-Q7; the default `form='discrete'` is the science product).

**M1 closed 2026-09-30** (prompt 2 task-7 log entry: all seven criteria PASS — V1 0.78% at 8 dx,
V2 0.077%, V3 1.004 / 0.981 with `form='discrete'` and the cubic departure velocity, V4 0.28-1.0%
of `G`/h recorded, V3b 0.975 [0.954, 1.003] recorded as the model-advection systematic, seven PNGs
in `git status`, suite 84 passed + 3 strict xfails that document `fronts` bugs, the chain-form
oracle bit-for-bit). The operator change that passed gate 3 — the discretely consistent `F` and
the cubic departure velocity — is the discretisation finding for the writeup.

### M2 — Data pull  *(planning Phase 1)*

Tasks: `pull_series` (resumable), both OSN stores, 72 hours
**2012-07-02 00:00 -> 2012-07-04 23:00 UTC**, write the §3.2 zarr. Then
`vertical.load_chunk_levels` for `k = 0..2` + the three flux fields -> §3.3 zarr, **including
`drF` for `k = 0..2` from the chunk 3-D grid** (read directly from the chunk `grid.zarr`, which
carries `drF/Z/Zl/Zu/Zp1`; `process_llc4320_3d_grid` is a column filter on a grid Dataset, not
the source — corrected 2026-10-04, M2 task 6), which `vertical.py`
needs; OSN carries only the `k = 0` scalar (`drF = 1.0`, already in §3.1 — corrected
2026-09-28). Cross-check `drF[0] = 1.0 m` here. The chunk `W` must include **`k_l = 1`** (the
cell-base velocity `vertical_term` takes; §4.6), not just `k_l = 0`.

`tile330_grid.zarr` came from M0; `tile330_masks.nc` comes from M1. M2 writes neither.

**Acceptance:** 72 timesteps present, no gaps; schema matches §3.2-§3.3; re-running is a no-op;
`KPPhbl` present; `drF` captured; missing chunk hours listed explicitly.

**M2 closed 2026-10-04** (prompt 3 task-6 log entry: all seven criteria PASS — 72/72 hours in
both stores, no gaps, `verify_series` and `verify_chunk_series` OK re-run fresh (§3.2/§3.3
schema, land-NaN == `hFac` in every hour and level, `niter` steps 144, `KPPhbl` present); both
re-runs no-ops with every chunk file sha256-identical (834 / 692 files); `drF[0] = 1.0 m`
(`drF[0..2]` 1.0 / 1.14 / 1.30 m); **0 missing chunk hours**; OSN 765 MiB in 27.3 min (22.4 s per
hour median), chunk 973 MiB in 7.03 h (299 s per hour median, 12.6 GB fetched); suite 127 passed +
3 strict xfails, both network smoke tests pass; `float32` on disk, `U` on `i_g` / `V` on `j_g`,
no halo applied).

**Split by dependency.** The OSN half needs nothing from anyone and can start immediately.
The chunk half waits on Lauren's hourly transfer (Q13: all 51 levels, plus `oceQsw` and
`oceFWflx` added to `transfer.variables`); ~~11 of the 72 stores already exist~~ *(corrected
2026-10-04, M2 task 6: the transfer completed 2026-10-01 — all 72 hourly stores exist, the 11 old
ones rewritten with the new variables; M2 task 4)*. Do **not** block
M3's development on the chunk half — `compute_budget` runs without `chunk_ds`, just with a
catch-all residual.

### M3 — Field-level budget  *(planning Phase 2)*  — **HARD GATE**

Tasks: `vertical.py` (physics), `budget.py`, `stats.py`; run the `L_cells` sweep; compute the
**measured** vertical and surface-flux terms from the §3.3 chunk product.

Front pixels here are selected as `G` above a stated percentile at the **midpoint time** inside
`mask_analysis` — **no labelling, no `tile_find`**. That stays in M4, and keeping it out means the
budget is not entangled with thresholding/thinning choices. Bootstrap over **contiguous spatial
blocks and hours** at this milestone (front *features* only exist from M4).

**Carried from M1 (decided 2026-09-30, M1-Q1 / Q2 / Q4 / Q6):** run `operators.frontogenesis`
with **both** `form='discrete'` (primary) and `form='chain'`, the difference a stated systematic;
the baseline is **0.981 [0.970, 0.994]** (V3 llc) and the model-advection systematic band from V3b
is **0.954-1.003 (0.975 ± 0.025)** — **no upward correction** for the Jacobian attenuation;
report the slope per front width and subtract the advection-numerics shortfall (−2% at 2 dx, −4%
at 1.5 dx, −11% at 1 dx) before any diffusion is inferred on the sharpest fronts; `order = 3`
default with **order 5 as a sensitivity**; V4's bar 0.28-1.0% of `G`/h (order 3).

Figures produced here: **1, 2, 2b, 3, 3b, 4, 5, 6, 7, 10.** They are built as the data lands;
M5 consolidates, captions and wires up one-command regeneration.

**Acceptance — closure, not a slope:**
```
measured - 2F - subfilter - vertical - surface_flux  ~  numerical + interior KPP
```
to a stated tolerance, with semi-Lagrangian and Eulerian estimates agreeing, and with
`vertical` and `surface_flux` **measured** from the chunk store rather than assumed. **No
efficiency number is quoted before this passes.** If closure fails, that is the result
(planning §12).

### M4 — Fronts and tracking  *(planning Phase 3)*

Tasks: per-hour front finding on the tile; **flow-informed** `track_top_n(n=10, flow=...)`;
per-front strength series vs `integral(2F dt)` on the **advected pixel set**;
`tracking_quality` diagnostic.

**Acceptance (strengthened by Q14):**

1. 10 tracks spanning >= 12 h each.
2. **Flow-informed scoring in use** — mask advection feeding `score_candidate`'s weights.
3. `tracking_quality` reported: the distribution of `follow()`-chosen minus flow-predicted
   displacement, with split/merge flags. If these disagree often, the track is not following
   the fluid and Phase 3 is not interpretable — better to find that out here.
4. **Reconciliation on Lagrangian-matched pixels:** per-front measured `d(front-mean G)/dt`
   equals the advected-pixel-set average of the Phase-2 per-pixel `DG/Dt` to a stated
   tolerance. Comparing "front at `t`" to "front at `t+dt`" does **not** satisfy this — a mean
   over a changing pixel set is not material.
5. **Figures 8 and 9 plus the tracking-quality diagnostic** written. Bootstrap here is over
   **frontal features and hours**.

### M5 — Figures and report  *(planning Phase 5)*

Tasks: **consolidate** `figures.py` (the figures themselves were produced in M1/M3/M4);
standardise captions so each states its `L` and its mask; wire up one-command regeneration; write
`frontogenesis_report.md` and a reproducibility `README`.

**Acceptance:** every figure regenerable from the stores by one command; every caption states `L`
and mask; the report states the closure tolerance, the null-test baseline, and the §12
null-result criteria explicitly, with limitations unhedged.

### M6 — Depth  *(planning Phase 4, deferred)*

Full-depth budget against `CHUNKS/monterey_bay`. Out of scope until M3 passes.

---

## 7. Critical path

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
       gate          gate
```
M2 depends on **M0 only** — it pulls raw fields and needs no physics, no masks and no operators
(masks are M1's, the static grid is M0's). So **M1 and M2 can genuinely run in parallel.**
Everything after M3 is blocked on closure.

---

## 8. Pitfall checklist

Distilled from the adversarial review. Re-read before each milestone.

- [ ] Compared **`2F`**, not `F`, against the measured tendency.
- [ ] Interpolated **`b`**, not `G`, at the departure point, at order >= 3.
- [ ] Evaluated `F` and selected fronts at the **midpoint time**, not an endpoint.
- [ ] Same filter applied to `b`, `U` **and** `V`.
- [ ] Land halo applied **before** any differencing, not after.
- [ ] Used `calculate_fields.buoyancy_of_field` (JMD95), **not** `utils/physical_calculations.buoyancy_of_field` (legacy: g in km/s^2, rho_ref=1025), and not TEOS-10.
- [ ] Bootstrapped over **features**, not pixels.
- [ ] Quoted every slope **against the M1 discrete-null baseline**, never against 1.
- [ ] Did not call a slope < 1 "diabatic damping" without Figure 2b separating numerical
      diffusion from air-sea forcing.
- [ ] Did not claim the residual is purely diabatic (it is not — planning §2.3).
- [ ] `ensure_comodo_attrs` run after `process_llc4320_grid`, before `set_xgcm_grid`.
- [ ] `halo_mask.py:75` early-return not silently hit; output shape asserted.
- [ ] Timestamps converted between `dbof`'s `'%Y-%m-%d %H:%M:%S'` and `front_tracking`'s
      `'%Y-%m-%dT%H_%M_%S'`.
- [ ] Corner-located strain/vorticity interpolated to centres before use.
- [ ] `nan_policy='omit'` passed to `colocate_fronts_with_properties` (default is
      `'propagate'`, and our fields are NaN over land/halo).
- [ ] Advected the **boolean mask**, never the integer label field.
- [ ] Reconciled Phase 3 against Phase 2 on the **advected pixel set**, not on labels at each
      endpoint.
- [ ] `vertical` and `surface_flux` terms actually present in the budget — if `chunk_ds` is
      missing, `closure_report` says so rather than quietly reverting to a catch-all residual.
- [ ] Loaded only `k = 0..2` from the chunk store; it holds all 51 levels.
- [ ] `vertical_term` built from the chunk `W(k_l=1)`, not from `delta` and not from the OSN
      `W(k_l=0)` — which is `dEta/dt`, a free-surface signal, not a flux (planning §2.2).
- [ ] Remembered that on face 10 `i`/`U`/`dxC` are meridional (`i` southward) and `j`/`V`/`dyC`
      zonal (`CS=0, SN=-1`); any "zonal/meridional" label on a native axis checked against §3.1.
- [ ] `oceTAUX`/`oceTAUY` re-masked with `hFacW`/`hFacS` before any stress derivative.
- [ ] `kappa_num` quoted at the scale of the feature (planning §2.3), not as one number.
- [ ] Every validation test wrote its PNG (and it shows in `git status` — `figs/.gitignore`
      un-ignores `*.png`).
- [ ] §2's API line numbers re-verified 2026-09-28 at `938bce1` (M0 task 5); redo if the Q15
      merges land.
- [ ] Output dims asserted after **every** dbof operator call (`('face', 'j', 'i')` or the
      staggered pair): a centred mask x staggered field, or swapped `calculate_jacobian` args,
      silently broadcasts to 4-D on numpy-backed inputs (OOM) and raises only on dask
      (M0 tasks 4-5).
- [ ] Tile-edge margin (`edge_cells`) applied on **all four** edges — the rim is finite, not
      NaN, and neither the land halo nor the offshore cut removes it (M0 task 5).
- [ ] `G` formed from the same `b_x, b_y` as `F` (`operators.gradb2 = b_x^2 + b_y^2`), never
      from `calculate_grad_squared_tracer` (0.911x; M0 task 5).
