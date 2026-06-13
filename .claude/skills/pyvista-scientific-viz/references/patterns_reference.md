# PyVista patterns reference

Deeper guidance the SKILL.md points to. Read the relevant section when the task
calls for it; you don't need all of this in context at once.

## Contents
1. Choosing a dataset type from your data
2. Building datasets from NumPy
3. Common filters (the analysis verbs)
4. Vectors, glyphs, and streamlines
5. Volumes and isosurfaces
6. Multi-view / comparison layouts
7. Animation
8. Colormap selection
9. Headless / CI notes

---

## 1. Choosing a dataset type from your data

| You have | Use | PyVista entry point |
|---|---|---|
| Scattered (x, y, z) points, maybe with values | Point cloud | `pv.PolyData(points)` |
| A surface mesh (triangles/quads) | Surface | `pv.PolyData(points, faces)` or load a file |
| Values on a regular 3D grid | Uniform grid | `pv.ImageData(dimensions=...)` |
| Values on a curvilinear/logically-rectangular grid | Structured grid | `pv.StructuredGrid(x, y, z)` |
| Arbitrary cells (FE meshes, mixed element types) | Unstructured grid | `pv.UnstructuredGrid(...)` or `meshio` |
| Tabular columns you want to plot in 3D | Build points, then as above | `pv.PolyData(np.c_[x, y, z])` |

Rule of thumb: prefer `ImageData` when your data is genuinely on a regular
lattice -- it is the most memory-efficient and many filters are faster on it.

## 2. Building datasets from NumPy

```python
import numpy as np
import pyvista as pv

# Point cloud with a per-point scalar.
pts = np.random.default_rng(0).random((500, 3))
cloud = pv.PolyData(pts)
cloud["height"] = pts[:, 2]

# Regular volume: dimensions are POINTS per axis, not cells.
grid = pv.ImageData(dimensions=(64, 64, 64), spacing=(1, 1, 1), origin=(0, 0, 0))
grid["field"] = scalar_array.ravel(order="F")  # VTK expects Fortran order
```

The Fortran-order `.ravel(order="F")` for `ImageData` is the single most common
silent bug -- C-order will "work" but transpose your field.

## 3. Common filters (the analysis verbs)

All return a new dataset; chain them, inspect intermediates.

```python
contours   = grid.contour(isosurfaces=[0.2, 0.5, 0.8], scalars="field")
sliced     = grid.slice_orthogonal()              # 3 axis-aligned planes
slc        = grid.slice(normal="z", origin=(0, 0, 5))
clipped    = mesh.clip(normal="x")
thresh     = grid.threshold(value=(0.3, 0.7), scalars="field")
warped     = surface.warp_by_scalar("elevation", factor=2.0)
stredec    = mesh.decimate(0.5)                    # reduce triangle count
sampled    = target.sample(source)                # interpolate fields across meshes
```

## 4. Vectors, glyphs, and streamlines

```python
mesh["vectors"] = vec_array               # (N, 3)
arrows = mesh.glyph(orient="vectors", scale="magnitude", factor=0.3)
pl.add_mesh(arrows)

stream = grid.streamlines(
    "vectors", source_radius=10, n_points=200, max_time=100.0
)
pl.add_mesh(stream.tube(radius=0.2))
```

Always downsample glyphs (`mesh.glyph(..., tolerance=0.05)` or decimate first)
on large meshes -- one arrow per cell turns into an unreadable hairball.

## 5. Volumes and isosurfaces

Two honest ways to show a 3D scalar field:

```python
# (a) Volume rendering -- good for continuous fields, set opacity transfer fn.
pl.add_volume(grid, scalars="field", cmap="viridis", opacity="sigmoid")

# (b) Isosurfaces -- good when specific level sets are meaningful.
iso = grid.contour([0.3, 0.6], scalars="field")
add_scalar_field(pl, iso, "field", label="Field", units="a.u.")
```

Prefer isosurfaces when the *levels* carry meaning (e.g. a density threshold);
prefer volume rendering when the whole gradient matters. Don't volume-render
just because it looks impressive -- it is easy to hide structure in opacity.

## 6. Multi-view / comparison layouts

```python
pl = new_plotter(shape=(1, 2))
pl.subplot(0, 0); add_scalar_field(pl, a, "field", label="Run A", units="K")
pl.subplot(0, 1); add_scalar_field(pl, b, "field", label="Run B", units="K")
pl.link_views()                       # share camera across panels
save_figure(pl, "_static/figs/ab.png")
```

For fair A/B comparison, set a shared `clim` on both so colors mean the same
thing in each panel.

## 7. Animation

```python
pl = new_plotter()
pl.open_gif("_static/figs/rotate.gif")     # or open_movie("out.mp4")
add_scalar_field(pl, mesh, "field", label="Field", units="a.u.")
for angle in range(0, 360, 3):
    pl.camera.azimuth = angle
    pl.write_frame()
pl.close()
```

## 8. Colormap selection

* Sequential, ordered data -> `viridis`, `cividis` (cividis is the safest for
  color-vision deficiency).
* Diverging around a meaningful midpoint (anomalies, +/- ) -> `RdBu_r`,
  `coolwarm`, with a symmetric `clim` centered on the midpoint.
* Categorical/labels -> a qualitative map with explicit `n_colors`, annotated.
* Avoid `jet`/rainbow: it invents banding and misranks magnitudes.

## 9. Headless / CI notes

* Call `ensure_display()` (in `pv_helpers`) once before rendering on a headless
  Linux box. It starts Xvfb only when needed.
* Set `pv.OFF_SCREEN = True` globally in test/CI modules, or always pass
  `off_screen=True`.
* For pixel-stable figures across machines, capture the camera triple returned
  by `save_figure` and pass it back as `cpos` -- the isometric default can
  differ slightly with bounds.
* `export_html` / volume rendering need the `trame` extras:
  `pip install 'pyvista[jupyter]'`.
