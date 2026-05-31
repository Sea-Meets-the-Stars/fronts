"""Reusable helpers for reproducible, publication-quality PyVista figures.

Design goals
------------
* Off-screen, deterministic rendering suitable for CI and headless servers.
* Perceptually-uniform, colorblind-safe defaults; honest scalar bars with units.
* First-class reStructuredText (Sphinx) output so figures drop straight into docs.

Nothing here is magic: every function is a thin, inspectable wrapper over the
normal PyVista API. Reach past these helpers whenever you need finer control --
they return the underlying ``Plotter``/actor objects so you can keep going.

Typical use
-----------
>>> import pyvista as pv
>>> from pv_helpers import new_plotter, add_scalar_field, save_with_rst
>>> mesh = pv.Wavelet()
>>> pl = new_plotter()
>>> add_scalar_field(pl, mesh, "RTData", label="Intensity", units="a.u.")
>>> rst = save_with_rst(
...     pl, "_static/figs/wavelet.png",
...     caption="Wavelet test field, isometric view.",
...     alt="3D wavelet scalar field",
... )
>>> print(rst)            # paste into your .rst, or write it to a file
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import pyvista as pv

# Perceptually-uniform sequential default. Override per-call when a diverging
# or categorical map is more honest for the data (see references/patterns.md).
DEFAULT_CMAP = "viridis"
DEFAULT_DIVERGING_CMAP = "RdBu_r"

# A camera position is either the "iso"/"xy"/... string PyVista accepts, or the
# explicit [(pos), (focal_point), (up)] triple. Capture and reuse the triple to
# make a figure pixel-reproducible across runs.
CameraPosition = Union[str, Sequence[Sequence[float]]]


def ensure_display() -> None:
    """Start a virtual framebuffer if rendering headless on Linux.

    Off-screen VTK still needs an OpenGL context. On a headless Linux box
    without one, call this once at process start. No-op on macOS/Windows or
    when a display is already present. Requires the system ``Xvfb`` binary.
    """
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except OSError as exc:  # Xvfb not installed -> surface a clear hint.
            raise RuntimeError(
                "Headless rendering needs Xvfb. Install it "
                "(e.g. `apt-get install -y libgl1-mesa-glx xvfb`) "
                "or run with an active display."
            ) from exc


def scientific_theme(
    *,
    background: str = "white",
    cmap: str = DEFAULT_CMAP,
    font_size: int = 14,
    label_size: int = 12,
    transparent_background: bool = False,
) -> "pv.themes.Theme":
    """Return a DocumentTheme tuned for figures that go into papers and docs.

    White background, anti-aliasing on, no edge clutter, sane fonts. Apply it
    per-plotter via ``new_plotter(theme=...)`` rather than mutating the global
    theme, so the skill never leaves global side effects behind.
    """
    theme = pv.themes.DocumentTheme()
    theme.background = background
    theme.cmap = cmap
    theme.anti_aliasing = "ssaa"  # supersampled: cleanest edges for stills.
    theme.transparent_background = transparent_background
    theme.font.family = "arial"
    theme.font.size = font_size
    theme.font.label_size = label_size
    theme.show_edges = False
    theme.nan_color = "lightgray"
    return theme


def new_plotter(
    *,
    off_screen: bool = True,
    window_size: Sequence[int] = (1600, 1200),
    theme: Optional["pv.themes.Theme"] = None,
    **plotter_kwargs: Any,
) -> "pv.Plotter":
    """Create a Plotter with the scientific theme and headless-safe defaults.

    Defaults to ``off_screen=True`` because the common case here is generating
    a file, not opening a window. Pass ``off_screen=False`` for interactive
    exploration. Extra kwargs flow through to ``pv.Plotter``.
    """
    ensure_display()
    return pv.Plotter(
        off_screen=off_screen,
        window_size=list(window_size),
        theme=theme or scientific_theme(),
        **plotter_kwargs,
    )


def add_scalar_field(
    plotter: "pv.Plotter",
    mesh: Any,
    scalars: Union[str, Sequence[float]],
    *,
    label: str = "",
    units: str = "",
    cmap: str = DEFAULT_CMAP,
    clim: Optional[Sequence[float]] = None,
    n_labels: int = 5,
    show_edges: bool = False,
    smooth_shading: bool = True,
    fmt: str = "%.3g",
    **mesh_kwargs: Any,
) -> Any:
    """Add a mesh colored by a scalar field with an honest, labelled scalar bar.

    The scalar bar title becomes ``"label [units]"`` when units are given --
    unlabeled axes are the most common way a 3D figure misleads. Returns the
    actor so you can tweak it afterwards.
    """
    title = f"{label} [{units}]" if units else label
    scalar_bar_args = dict(
        title=title,
        n_labels=n_labels,
        fmt=fmt,
        title_font_size=16,
        label_font_size=12,
        shadow=False,
    )
    return plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        smooth_shading=smooth_shading,
        scalar_bar_args=scalar_bar_args,
        **mesh_kwargs,
    )


def save_figure(
    plotter: "pv.Plotter",
    path: Union[str, Path],
    *,
    cpos: Optional[CameraPosition] = None,
    scale: int = 2,
    transparent_background: Optional[bool] = None,
    close: bool = True,
) -> CameraPosition:
    """Render off-screen to a PNG and return the camera position used.

    ``scale`` supersamples (2 = render at 2x then downsample) for crisp output
    without changing layout. Set ``cpos`` to a captured triple to reproduce an
    exact view; otherwise an isometric default is used. The returned camera
    position can be fed back in next time to lock the framing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if cpos is not None:
        plotter.camera_position = cpos
    else:
        plotter.view_isometric()

    plotter.screenshot(
        str(path),
        scale=scale,
        transparent_background=transparent_background,
    )
    used_cpos = plotter.camera_position
    if close:
        plotter.close()
    return used_cpos


def rst_figure(
    image_uri: Union[str, Path],
    *,
    caption: str = "",
    alt: str = "",
    width: str = "80%",
    align: str = "center",
) -> str:
    """Return a reStructuredText ``.. figure::`` block for an image.

    ``image_uri`` should be the path as Sphinx will resolve it (e.g. an
    absolute-from-source ``/_static/figs/foo.png`` or a path relative to the
    .rst file). The caption is the figure legend; ``alt`` is accessibility text.
    """
    lines = [f".. figure:: {image_uri}"]
    if alt:
        lines.append(f"   :alt: {alt}")
    if width:
        lines.append(f"   :width: {width}")
    if align:
        lines.append(f"   :align: {align}")
    block = "\n".join(lines) + "\n"
    if caption:
        block += "\n" + textwrap.indent(caption.strip(), "   ") + "\n"
    return block


def export_interactive(
    plotter: "pv.Plotter",
    html_path: Union[str, Path],
    *,
    close: bool = False,
) -> Path:
    """Write a self-contained interactive HTML version of the current scene.

    Useful alongside the static PNG: readers get a rotatable view. Requires the
    ``trame`` extras (``pip install 'pyvista[jupyter]'``). Returns the path.
    """
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(str(html_path))
    if close:
        plotter.close()
    return html_path


def rst_raw_html(html_uri: Union[str, Path]) -> str:
    """Return an ``.. raw:: html`` block that inlines an exported scene file.

    Pair with :func:`export_interactive` to embed a rotatable view directly in
    a Sphinx page. The ``:file:`` path is resolved relative to the .rst source.
    """
    return f".. raw:: html\n   :file: {html_uri}\n"


def save_with_rst(
    plotter: "pv.Plotter",
    png_path: Union[str, Path],
    *,
    caption: str = "",
    alt: str = "",
    width: str = "80%",
    align: str = "center",
    image_uri: Optional[Union[str, Path]] = None,
    cpos: Optional[CameraPosition] = None,
    scale: int = 2,
    rst_path: Optional[Union[str, Path]] = None,
    interactive_html: Optional[Union[str, Path]] = None,
) -> str:
    """Render a PNG (optionally an interactive HTML too) and return rst for it.

    This is the one-call path for "make the figure and give me something to
    paste into my docs". If ``rst_path`` is set, the rst is also written there.
    ``image_uri`` overrides how the path appears in the directive (default: the
    PNG's own path); set it to the Sphinx-resolved location if they differ.

    Note: pass ``interactive_html`` BEFORE saving closes the plotter -- this
    function handles ordering by exporting the HTML first.
    """
    png_path = Path(png_path)
    blocks = []

    if interactive_html is not None:
        export_interactive(plotter, interactive_html, close=False)
        blocks.append(rst_raw_html(interactive_html))

    save_figure(plotter, png_path, cpos=cpos, scale=scale, close=True)
    blocks.append(
        rst_figure(
            image_uri if image_uri is not None else png_path,
            caption=caption,
            alt=alt,
            width=width,
            align=align,
        )
    )

    rst = "\n".join(blocks)
    if rst_path is not None:
        rst_path = Path(rst_path)
        rst_path.parent.mkdir(parents=True, exist_ok=True)
        rst_path.write_text(rst)
    return rst
