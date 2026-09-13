# Frontogenesis — Coding / Execution Document

**Companion to:** `frontogenesis_planning.md` (physics, decisions, figures). This document
is the *implementation* contract: modules, signatures, data schemas, milestones, and
acceptance criteria.

**How to use this document.** Each milestone in §6 (M0-M5) is intended to become **one
execution prompt doc**. A milestone is not "done" until its acceptance criteria pass; M1
and M3 are hard gates — do not proceed past them on a failure, because everything after
inherits the error silently.

**Created:** 2026-09-12. **Blocking prerequisite:** Q11 (branch strategy) — see planning §10.

---

## 1. Conventions

Fix these once. Most of the failure modes in planning §2.3 and §11 are convention slips.

### 1.1 Physical

| Quantity | Symbol in code | Units | Notes |
|---|---|---|---|
| buoyancy | `b` | m s^-2 | `b = +g sigma0/rho0` via `calculate_fields.buoyancy_of_field` (**JMD95**, not TEOS-10). `g=9.81`, `rho0=1000.0`. **Increases with density** — the negative of textbook `b`. Harmless (`G`,`F` quadratic; alignment enters as `cos 2theta`). Do not 'fix' it. |
| front strength | `G` | s^-4 | `G = |grad_h b|^2`; the repo's `gradb2` |
| frontogenesis tendency | `F` | s^-5 | `F = -(u_x b_x^2 + (u_y+v_x) b_x b_y + v_y b_y^2)` |
| measured tendency | `DGDt` | s^-5 | `D_h G / Dt` |

**The factor of two.** `F = (1/2) DG/Dt`. Every comparison, axis label and regression uses
**`2F` vs `DGDt`**. Name the predicted variable `two_F` in code so it cannot be confused.

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
  The **same** filter is applied to `b`, `U` and `V` (planning §5.4).
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
**Do not guess these.** Note especially the three marked ***TRAP***.

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
      #     Always _ensure_comodo_attrs() before set_xgcm_grid(). ***
```

### 2.2 Tile indexing and xgcm (`dbof`)

```python
# dbof/tiles/tile_mapping.py
def rect_ij_to_tile(i_rect: int, j_rect: int) -> TileInfo   # L112  NOTE ARG ORDER (i, j)
@dataclass(frozen=True) class TileInfo:                     # L47
    tile_idx, tile_j_rect, tile_i_rect, rect_j_slice, rect_i_slice,
    face_idx, j_face_slice, i_face_slice
# ours: rect_ij_to_tile(13320, 9720) -> face_idx=10, j 0:720, i 2880:3600

# dbof/llc4320_ingestion/grid.py
COMODO_COORD_META = {                                       # L9
    'j':   {'axis': 'Y'},
    'j_g': {'axis': 'Y', 'c_grid_axis_shift': -0.5},
    'i':   {'axis': 'X'},
    'i_g': {'axis': 'X', 'c_grid_axis_shift': -0.5},
}
def set_xgcm_grid(ds_grid, use_connections: bool = True)    # L44 -> xgcm.Grid; we pass False

# dbof/tiles/tile_utils.py  -- both are short private helpers; copy rather than import if preferred
def _ensure_comodo_attrs(ds) -> xr.Dataset                  # L334
def _tile_indexer(ds, tile) -> dict                         # L372
      # {'j','j_g'} -> tile.j_face_slice ; {'i','i_g'} -> tile.i_face_slice, for dims present.
      # Staggered dims take the SAME slice -> the high-edge derivative rim is invalid.
```

### 2.3 Physics operators (`dbof`)

```python
# dbof/utils/native_gradient.py                -- ALL return geographic-basis quantities
def calculate_native_gradient_tracer(ds_value, ds_grid, grid)   # L126 -> (zonal, merid)
def calculate_jacobian(u_x, v_y, ds_merge, grid)                # L53  -> (du_dx, du_dy, dv_dx, dv_dy)
      # *** TRAP: args named u_x, v_y but they ARE U and V (the raw staggered fields). ***
def calculate_grad_squared_tracer(ds_value, ds_grid, grid)      # L224 -> |grad s|^2 at centres
def calculate_native_strain_vorticity(u_x, v_y, ds_grid, grid)  # L301 -> dict, NOT a tuple:
      #   'strain_normal_center', 'divergence_center'   on (j, i)
      #   'vorticity_corner',     'strain_shear_corner' on (j_g, i_g)  <- interp to centres!
def rotate_vector_to_geographic(u_x, v_y, ds_merge, grid, *, interpolate=True)   # L13

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
vars   : XC, YC, dxC, dyC, dxG, dyG, rA, rAz, CS, SN, hFacC, Depth
attrs  : face_index=10, j_face_start=0, i_face_start=2880, rect_i=13320, rect_j=9720,
         source='OSN', git_commit, created
```
Comodo attrs (`axis`, `c_grid_axis_shift=-0.5`) must be present on `i`,`i_g`,`j`,`j_g`.

### 3.2 `tile330_raw_20120702T00_72h.zarr` — the Phase-1 product

```
dims   : (time: 72, j: 720, i: 720) + i_g, j_g
vars   : Theta(time,j,i), Salt(time,j,i), U(time,j,i_g), V(time,j_g,i),
         W(time,j,i), Eta(time,j,i),
         KPPhbl(time,j,i), oceTAUX(time,j,i_g), oceTAUY(time,j_g,i)
coords : time (datetime64), XC, YC
attrs  : iterations (list), endpoint, stores=['llc_surf','llc_wind'], git_commit
```
`KPPhbl`/`oceTAU*` come from the **second** OSN store (`llc_wind`), which covers our window.
Heat fluxes are in neither store — the diabatic term is inferred as a residual.

### 3.3 `tile330_derived_L{L}.zarr` — per filter scale

```
vars : b, G, two_F, DGDt_semilag, DGDt_euler, subfilter, residual,
       delta, sigma_n, sigma_s, sigma_mag, theta_align
```

### 3.4 `tile330_masks.nc`

```
vars : mask_ocean, mask_halo, mask_offshore, mask_analysis, coast_distance_km
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

### 4.2 `py/masking.py`

```python
def ocean_mask(grid_ds):                          -> np.ndarray  # bool, True=ocean, from hFacC
def halo_mask(grid_ds, halo_km=13.0):             -> np.ndarray  # bool, True=retained
def coast_distance_km(grid_ds):                   -> np.ndarray  # float, km to nearest land
def offshore_mask(grid_ds, min_km=100.0):         -> np.ndarray  # bool
def analysis_mask(grid_ds, halo_cells=7,
                  min_km=100.0):                  -> np.ndarray  # halo & offshore & ocean
```
`halo_mask` wraps `llc_native_grid_halo_mask` and **must handle two known defects**
(planning §5.5): the 2-D early return when a face is entirely land, and a `k`-carrying
`hFacC` that makes the mask 4-D and breaks `skfmm`. Collapse `k` first; assert output shape.

**Halo is specified in km, not cells** — `generate_halo_land_mask(ds_grid, target_km_res,
...)` takes `target_km_res` and uses it directly as `halo_km`. Our 7-cell requirement
(3 Jacobian+interp stencil + 4 widest filter half-width) is **~13 km** at the tile's ~1.9 km
spacing. Confirm the spacing in M0 and set `halo_km` from it rather than hard-coding.

### 4.3 `py/operators.py` — the single shared operator

This module is the reason the study is trustworthy: **both sides of the comparison go
through it** (planning §5.1).

```python
def buoyancy(ds):                                 -> xr.DataArray  # wraps calculate_fields.buoyancy_of_field
def lowpass(field, L_cells):                      -> same type     # L_cells=0 -> identity
def grad_b(b, grid_ds, grid):                     -> (b_x, b_y)    # via calculate_native_gradient_tracer (geographic)
def gradb2(b, grid_ds, grid):                     -> G
def jacobian(U, V, grid_ds, grid):                -> (u_x, u_y, v_x, v_y)
def frontogenesis(b, U, V, grid_ds, grid):        -> F             # inputs ALREADY filtered
def strain_divergence(U, V, grid_ds, grid):       -> (delta, sigma_n, sigma_s, sigma_mag)
      # calculate_native_strain_vorticity returns a DICT with sigma_s and vorticity on
      # CORNERS (j_g, i_g) -- interpolate to centres before combining. See §2.3.
def strain_alignment(b_x, b_y, sigma_n, sigma_s): -> theta         # radians, for Figure 4
```
`frontogenesis` takes **pre-filtered** inputs — filtering happens once, at the call site, so
it cannot silently differ between the two sides.

### 4.4 `py/semilag.py`

```python
def centre_velocities(U, V, grid_ds, grid):       -> (u_c, v_c)   # interp to cell centres
def departure_index(u_c, v_c, grid_ds, dt=3600.0,
                    n_iter=3):                    -> (di, dj)     # index-space displacement
def interp_to_departure(field, di, dj, order=3):  -> same shape   # cubic+ REQUIRED
def measured_DGDt(b_t, b_tp1, u_mid, v_mid, grid_ds, grid,
                  dt=3600.0, order=3):            -> DGDt
def eulerian_DGDt(G_t, G_tp1, u_mid, v_mid, grid_ds, grid,
                  dt=3600.0):                     -> DGDt         # independent cross-check
```
**`measured_DGDt` interpolates `b`, then differentiates** — it does *not* interpolate `G`.
Interpolating `G` bilinearly biases it negative at maxima by 25-80% of the signal, i.e. it
fabricates frontogenesis (planning §5.3). `order >= 3` is not optional.

### 4.5 `py/coarsegrain.py`

```python
def subfilter_flux(b, U, V, L_cells, grid_ds, grid):   -> (tau_x, tau_y)  # mean(ub) - ubar bbar
def subfilter_term(b_bar, tau_x, tau_y, grid_ds, grid):-> term            # -grad(bbar).grad(div tau)
```
Without this the filter sweep is uninterpretable (planning §5.4).

### 4.6 `py/budget.py`

```python
def compute_budget(raw_ds, grid_ds, grid, masks, L_cells,
                   dt=3600.0, with_vertical=False):     -> xr.Dataset
def closure_report(budget_ds, mask):                    -> dict
```
`compute_budget` returns `measured, two_F, subfilter, vertical, residual` and is the object
on which the Phase-2 exit criterion is evaluated.

### 4.7 `py/stats.py`

```python
def slope_ols(x, y);  def slope_tls(x, y);  def slope_bisector(x, y)
def ratio_estimator(x, y):                              -> sum(y)/sum(x)
def binned_conditional_mean(x, y, bins, split_sign=True)-> DataFrame
def feature_bootstrap(x, y, labels, estimator, n=1000)  -> (lo, hi)
```
**Bootstrap over frontal features and hours, never over pixels** (planning §11). Every slope
is reported relative to the M1 discrete-null baseline.

### 4.8 `py/validate.py` — the four gates

```python
def test_cartesian_deformation(alpha=1e-5, ...)   -> dict   # G ~ exp(2 alpha t), continuum
def test_native_metric(...)                       -> dict   # analytic f(XC,YC), known gradients
def test_discrete_null(velocities='strain', ...)  -> dict   # MUST give slope = 1 +/- 0.05
def test_interpolation_bias(...)                  -> dict   # uniform zero-strain flow; true DGDt = 0
```

### 4.9 `py/tracking.py`, `py/figures.py`

```python
def find_fronts_series(derived_zarr, mask, config='D')  -> dict[time -> label array]
def track_top_n(labels_by_time, times, n=10)            -> list[Track]
      # `times` MUST be '%Y-%m-%dT%H_%M_%S' (underscores) for front_tracking.parse_time,
      # NOT dbof's '%Y-%m-%d %H:%M:%S'. Convert here; see the §2.5 trap.
def front_strength_series(track, G_by_time, two_F_by_time) -> DataFrame
```
`figures.py`: one function per figure, `fig01_maps(...)` ... `fig10_validation(...)`,
each writing a PNG to `dev/frontogenesis/figs/`.

---

## 5. Test strategy

`dev/frontogenesis/py/tests/`, pytest, **all offline** (no network) except one explicitly
marked `@pytest.mark.network` smoke test of the OSN pull.

| Test | Guards |
|---|---|
| `test_operators.py` | gradient of an analytic field; `F` vs the repo's `frontogenesis_tendency` unfiltered; factor-of-two convention |
| `test_masking.py` | halo width; the two `halo_mask` defects; `True`=retained |
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
1. Environment: `dbof` importable (`pip install -e . --no-deps` + `xgcm<0.10`,
   `scikit-fmm`, `s3fs`, `ujson`, `xmitgcm`, `zarr`, `dask`). Fall back to a py3.13 env if
   py3.14 wheels are missing.
2. `osn_tiles.tile_spec()`, `load_grid()`, `load_hour()` for **one** timestamp.
3. Answer, empirically, and record in the log:
   - Is OSN land stored as **0 or NaN**?
   - Is `W[k_l=0] ~ 0`? (validates planning §2.2)
   - Top-cell thickness from `drF`.
   - `dxC`/`dyC` at 37N — the actual native spacing.
   - Magnitude of `grad CS`, `grad SN` terms vs strain (expect < 0.5%).
   - The model's tracer advection scheme, and an estimate of `kappa_num`.
4. QA plot of one snapshot: `Theta`, `G`, land mask, halo rim.
5. Confirm the three §2 traps on real data: comodo attrs survive `process_llc4320_grid`
   (or are restored), `halo_mask.py:75` is not reached, and the two timestamp formats are
   converted correctly.

**Acceptance:** one hour loads end-to-end; all six questions answered in the log; QA plot
shows a clean coastline with no gradient ribbon.

### M1 — Operators and validation  *(planning Phase 0b)*  — **HARD GATE**

**Goal:** operators that are known correct before any science.

Tasks: `masking.py`, `operators.py`, `semilag.py`, `validate.py`, plus their tests.

**Acceptance — all four must pass:**
1. Cartesian deformation reproduces `exp(2 alpha t)` to < 1%.
2. Native-metric test reproduces analytic gradients to < 1%.
3. **`test_discrete_null` gives slope = 1 +/- 0.05 on front pixels.** If it fails, co-locate
   the operators or raise the scheme order until it passes. *Do not proceed on a failure* —
   C-grid interpolation attenuation alone can bias the slope 0.7-1.4 (planning §6).
4. `test_interpolation_bias` quantifies the uniform-flow bias; it becomes a permanent error
   bar on every later slope.

### M2 — Data pull  *(planning Phase 1)*

Tasks: `pull_series` (resumable), both OSN stores, 72 hours
**2012-07-02 00:00 -> 2012-07-04 23:00 UTC**, write §3.1-3.2 zarrs, write `tile330_masks.nc`.

**Acceptance:** 72 timesteps present, no gaps; schema matches §3; re-running is a no-op;
masks match the M0 QA plot.

### M3 — Field-level budget  *(planning Phase 2)*  — **HARD GATE**

Tasks: `coarsegrain.py`, `budget.py`, `stats.py`; run the `L_cells` sweep; bound the
finite-top-cell vertical term using `CHUNKS/monterey_bay` (11 snapshots inside the window).

**Acceptance — closure, not a slope:**
```
measured - 2F - subfilter - vertical - numerical  ~  0
```
to a stated tolerance, with semi-Lagrangian and Eulerian estimates agreeing. **No efficiency
number is quoted before this passes.** If closure fails, that is the result (planning §12).

### M4 — Fronts and tracking  *(planning Phase 3)*

Tasks: per-hour front finding on the tile, `track_top_n(n=10)`, per-front strength series vs
`integral(2F dt)`.

**Acceptance:** 10 tracks spanning >= 12 h each; per-front measured and predicted series;
Phase-3 front-mean `G` reconciles with the Phase-2 per-pixel result on the same pixels.

### M5 — Figures and report  *(planning Phase 5)*

Tasks: `figures.py` (Figures 1-10, incl. **2b**, the numerical-vs-diabatic discriminator);
`frontogenesis_report.md`.

**Acceptance:** every figure regenerable from the zarrs by one command; the report states
the closure tolerance, the null-test baseline, and the §12 null-result criteria explicitly.

### M6 — Depth  *(planning Phase 4, deferred)*

Full-depth budget against `CHUNKS/monterey_bay`. Out of scope until M3 passes.

---

## 7. Critical path

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
       gate          gate
```
M2 depends on M0 only (data pull needs no physics), so **M1 and M2 can run in parallel** if
convenient. Everything after M3 is blocked on closure.

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
- [ ] `_ensure_comodo_attrs` run after `process_llc4320_grid`, before `set_xgcm_grid`.
- [ ] `halo_mask.py:75` early-return not silently hit; output shape asserted.
- [ ] Timestamps converted between `dbof`'s `'%Y-%m-%d %H:%M:%S'` and `front_tracking`'s
      `'%Y-%m-%dT%H_%M_%S'`.
- [ ] Corner-located strain/vorticity interpolated to centres before use.
- [ ] `nan_policy='omit'` passed to `colocate_fronts_with_properties` (default is
      `'propagate'`, and our fields are NaN over land/halo).
