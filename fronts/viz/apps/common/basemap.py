"""The Pacific-centred global map.

Layers, bottom to top:

1. the selected field, from the display pyramid, datashaded;
2. land in gray, taken from the model's own NaN mask, reduced the same
   way the field is so the two agree at every zoom level;
3. optional coastlines from cartopy, when its Natural Earth data is
   available locally;
4. optional binary fronts;
5. lat/lon gridlines and labels.

Land comes from the data rather than from an external coastline dataset,
which means it always agrees with the model grid exactly.  Cartopy
coastlines are a nicety layered on top, and their absence (no network on
first use, for instance) degrades to a still-correct map.
"""

from __future__ import annotations

import logging

import holoviews as hv
import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.common import pyramid

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Datashader is an optimisation here, not a requirement
# --------------------------------------------------------------------------
# Datashader exists to re-aggregate an enormous raster on every zoom.  The
# display pyramid has already done the reduction, so a pyramid level is a
# few million cells at most -- a size Bokeh can send to the browser
# directly.  Datashading it is nicer (it re-aggregates as you zoom instead
# of resampling client-side), but it is not what makes the map correct.
#
# So the import is optional.  Datashader pulls in numba, which tracks
# NumPy releases with a lag; when numba does not yet support the installed
# NumPy the import raises and, without this, the whole page dies on a
# transitive dependency it does not strictly need.
try:
    from holoviews.operation.datashader import rasterize as _rasterize_op
    HAVE_DATASHADER = True
    DATASHADER_ERROR = ""
except Exception as _exc:                                  # noqa: BLE001
    _rasterize_op = None
    HAVE_DATASHADER = False
    DATASHADER_ERROR = f"{type(_exc).__name__}: {_exc}"
    log.warning(
        "datashader unavailable (%s) -- the map falls back to sending "
        "pyramid levels directly, capped at %d cells.  See "
        "docs/viz/apps/WIRING.md.",
        DATASHADER_ERROR, 1_500_000,
    )

#: Cell budget for a single map layer when datashading is unavailable.
#: A level over this is dropped to a coarser one rather than shipped whole.
MAX_DIRECT_CELLS = 1_500_000


def _rasterize(element):
    """Datashade an element, or pass it through when datashader is absent."""
    return _rasterize_op(element) if HAVE_DATASHADER else element


def _affordable_width(width: int, extent=None) -> int:
    """The finest pyramid width we are willing to send without datashader.

    Heights are derived from width, so cells scale as ``width**2``; this
    steps down through ``config.PYRAMID_WIDTHS`` until the raster fits the
    budget.  With datashader present the requested width is used as-is.

    The budget is deliberately counted over the **whole** raster, not the
    cropped window, even though only the window is sent.  Counting the
    window instead would let every zoomed-in map jump to the finest level
    -- which is more correct in principle and much heavier in practice, on
    a map that redraws on every navigation.  Detail on demand belongs
    behind a button, not on the interactive path; the static region figure
    on Field Characteristics is where it lives.

    *extent* is accepted and ignored, so callers need not care.
    """
    if HAVE_DATASHADER:
        return width

    lat0, lat1 = config.PYRAMID_LAT_RANGE

    def cells(w):
        return w * max(int(round(w * (lat1 - lat0) / 360.0)), 2)

    if cells(width) <= MAX_DIRECT_CELLS:
        return width
    affordable = [w for w in config.PYRAMID_WIDTHS
                  if cells(w) <= MAX_DIRECT_CELLS]
    return max(affordable) if affordable else min(config.PYRAMID_WIDTHS)


def _visible_fraction(extent) -> float:
    """Fraction of the raster's area a zoom window covers."""
    if extent is None:
        return 1.0
    (lon0, lon1), (lat0, lat1) = extent
    lat_lo, lat_hi = config.PYRAMID_LAT_RANGE
    flon = min(abs(lon1 - lon0) / 360.0, 1.0)
    flat = min(abs(lat1 - lat0) / (lat_hi - lat_lo), 1.0)
    return max(flon * flat, 1e-6)


def width_for_extent(extent) -> int:
    """The finest pyramid width whose *visible* cells fit the budget.

    Zooming used to change only the axis limits, so a region kept the
    resolution of the global view no matter how far in you went.  The
    pyramid exists precisely so that does not have to happen: a window
    covering a hundredth of the globe can carry a ten-times finer raster
    for the same number of cells on screen.
    """
    lat0, lat1 = config.PYRAMID_LAT_RANGE
    frac = _visible_fraction(extent)

    def visible_cells(w):
        h = max(int(round(w * (lat1 - lat0) / 360.0)), 2)
        return w * h * frac

    ok = [w for w in config.PYRAMID_WIDTHS
          if visible_cells(w) <= MAX_DIRECT_CELLS]
    return max(ok) if ok else min(config.PYRAMID_WIDTHS)


def bokeh_cmap(name: str, n: int = 256) -> list[str]:
    """Resolve a field-style colormap name to hex colours Bokeh can use.

    ``fronts.viz.field_styles`` names colormaps for matplotlib and PyVista,
    including cmocean ones (``dense``, ``thermal``, ``haline``) that Bokeh
    has never heard of.  Resolving them here is what lets the tile map use
    the same colours as the curtains rather than a hardcoded viridis.
    """
    import matplotlib.colors as mcolors

    from fronts.viz import field_styles

    cmap = field_styles.resolve_cmap(name)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _crop(lon, lat, arr, extent):
    """Slice a raster down to the zoom window, with a margin.

    Only the visible part is sent to the browser, which is what makes a
    finer level affordable.  A window that straddles the 0/360 seam is not
    contiguous in this array, so it is left uncropped rather than rolled.
    """
    if extent is None:
        return lon, lat, arr

    (lon0, lon1), (lat0, lat1) = extent
    if lon0 < lon[0] or lon1 > lon[-1]:            # crosses the seam
        return lon, lat, arr

    mlon = 0.05 * (lon1 - lon0)
    mlat = 0.05 * (lat1 - lat0)
    ix = np.searchsorted(lon, [lon0 - mlon, lon1 + mlon])
    iy = np.searchsorted(lat, [lat0 - mlat, lat1 + mlat])

    xs = slice(max(int(ix[0]) - 1, 0), int(ix[1]) + 1)
    ys = slice(max(int(iy[0]) - 1, 0), int(iy[1]) + 1)
    if lon[xs].size < 2 or lat[ys].size < 2:
        return lon, lat, arr
    return lon[xs], lat[ys], arr[ys, xs]

# Longitude ticks for a Pacific-centred 0..360 axis, labelled in E/W.
_LON_TICKS = [
    (0, "0"), (60, "60E"), (120, "120E"), (180, "180"),
    (240, "120W"), (300, "60W"), (360, "0"),
]
_LAT_TICKS = [(v, f"{abs(v)}{'N' if v > 0 else 'S' if v < 0 else ''}")
              for v in (-60, -30, 0, 30, 60)]

_FIELD_CMAPS = {
    # Greyscale for the same reason as the tile map: the fronts overlay is
    # drawn on top of this, and a coloured base fights with it.
    "gradb2": "gray",
    "relative_vorticity": "RdBu_r",
    "divergence": "RdBu_r",
    "okubo_weiss": "RdBu_r",
    "strain_n": "RdBu_r",
    "strain_s": "RdBu_r",
    "Eta": "RdBu_r",
    "strain_mag": "viridis",
    "coriolis_f": "RdBu_r",
    "SSTK": "inferno",
}

#: Positive-definite fields the map draws on a log axis.
#:
#: Kept alongside ``_FIELD_CMAPS`` and ``_DIVERGING`` because the map has
#: its own display path -- these are keyed on the *root* name, and a
#: DEPTH channel is matched by stripping its suffix, so gradb2 and
#: gradb2_mld are drawn identically and the two views of one field can be
#: compared directly.
_LOG_FIELDS = {"gradb2", "gradtheta2", "gradsalt2", "gradrho2"}
_DIVERGING = {
    "relative_vorticity", "divergence", "okubo_weiss",
    "strain_n", "strain_s", "Eta", "coriolis_f",
}


def _root(name: str) -> str:
    """A channel name with any DEPTH suffix removed."""
    from fronts.viz import field_styles
    return field_styles.strip_depth_suffix(name)


def field_display(arr: np.ndarray, name: str):
    """Transform a field for display, returning ``(values, clim, label)``."""
    if _root(name) in _LOG_FIELDS:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.log10(np.where(arr > 0, arr, np.nan))
        label = f"log10({name})"
    else:
        out = arr
        label = name

    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out, (0.0, 1.0), label

    lo, hi = np.nanpercentile(finite, [2, 98])
    if _root(name) in _DIVERGING:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return out, (float(lo), float(hi)), label


def _image(lon, lat, arr, label, group):
    """A HoloViews Image on the regular display raster."""
    return hv.Image(
        (lon, lat, arr), kdims=["lon", "lat"], vdims=[label], group=group
    )


def field_layer(provider, date, name, width=None,
                *, tools=("box_select",), extent=None):
    """The field raster, datashaded.

    ``tools`` has to be attached to a concrete element, not to the
    enclosing Overlay -- an Overlay-level ``active_tools`` naming a tool
    nothing added just logs "could not be found" and the box-select never
    appears.
    """
    if width is None:
        width = width_for_extent(extent)
    lon, lat, arr = pyramid.level(provider, date, name,
                                  _affordable_width(width, extent))
    lon, lat, arr = _crop(lon, lat, arr, extent)
    values, clim, label = field_display(arr, name)
    img = _image(lon, lat, values, label, "Field")
    # Through the depth suffix as well, so gradb2 at depth gets the same
    # greyscale base the surface one does rather than the default.
    cmap = _FIELD_CMAPS.get(name, _FIELD_CMAPS.get(_root(name), "viridis"))
    return _rasterize(img).opts(
        cmap=cmap, clim=clim, colorbar=True, colorbar_position="bottom",
        tools=list(tools),
    )


def land_layer(provider, date, width=None, *, extent=None):
    """Land in gray, from the model's own mask."""
    if width is None:
        width = width_for_extent(extent)
    lon, lat, arr = pyramid.land_level(provider, date,
                                       _affordable_width(width, extent))
    lon, lat, arr = _crop(lon, lat, arr, extent)
    masked = np.where(arr, 1.0, np.nan)
    img = _image(lon, lat, masked, "land", "Land")
    return _rasterize(img).opts(
        cmap=["#b0b0b0"], clim=(0, 1), colorbar=False,
    )


def fronts_layer(provider, date, width=None, *, extent=None):
    """Binary fronts, drawn on top of the field."""
    if width is None:
        width = width_for_extent(extent)
    # 'any', unlike land: a front is one cell wide, so a majority rule
    # would erase it at every level of the pyramid.
    lon, lat, arr = pyramid.level(provider, date, "__fronts__",
                                  _affordable_width(width, extent),
                                  reduce="any")
    lon, lat, arr = _crop(lon, lat, arr, extent)
    masked = np.where(arr > 0, 1.0, np.nan)
    img = _image(lon, lat, masked, "front", "Fronts")
    return _rasterize(img).opts(
        cmap=["#00e5ff"], clim=(0, 1), colorbar=False,
    )


def coastline_layer():
    """Cartopy coastlines, if the Natural Earth data is available.

    Returns ``None`` when it is not -- the map still reads correctly,
    because land already comes from the model mask.
    """
    try:
        import cartopy.io.shapereader as shpreader
        path = shpreader.natural_earth(
            resolution="110m", category="physical", name="coastline"
        )
    except Exception:
        return None

    try:
        segments = []
        for geom in shpreader.Reader(path).geometries():
            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                xy = np.asarray(line.coords)
                if xy.size:
                    xy[:, 0] %= 360.0
                    # Break the path where it wraps, so no line streaks
                    # across the whole map.
                    brk = np.nonzero(np.abs(np.diff(xy[:, 0])) > 180)[0]
                    for part in np.split(xy, brk + 1):
                        if len(part) > 1:
                            segments.append(part)
        if not segments:
            return None
        return hv.Path(segments).opts(color="black", line_width=0.5)
    except Exception:
        return None


def global_map(
    provider,
    date: str,
    field: str,
    *,
    show_fronts: bool = False,
    width: int | None = None,
    height: int = 480,
    title: str = "",
    tools=("box_select",),
    active_tools=("box_select",),
    extent=None,
):
    """Assemble the full Pacific-centred map.

    ``extent`` is the ``((lon0, lon1), (lat0, lat1))`` the map will be
    shown at.  Every layer picks its pyramid level from it and is cropped
    to it, so zooming in genuinely buys resolution instead of just
    enlarging the same pixels.

    Returns a HoloViews ``Overlay`` on a 0..360 longitude axis.
    """
    layers = [field_layer(provider, date, field, tools=tools, extent=extent),
              land_layer(provider, date, extent=extent)]

    coast = coastline_layer()
    if coast is not None:
        layers.append(coast)

    if show_fronts:
        layers.append(fronts_layer(provider, date, extent=extent))

    xlim, ylim = extent if extent else ((0, 360), config.PYRAMID_LAT_RANGE)

    # width=None -> fill the container.  The pages give the maps the full
    # page width now; a fixed width would leave them at their old size
    # with whitespace beside them.  HoloViews spells this responsive=True
    # (the Panel-style sizing_mode is not a valid Overlay option).
    sizing = {"width": width} if width else {"responsive": True}
    return hv.Overlay(layers).opts(
        hv.opts.Overlay(
            height=height, title=title,
            xlabel="longitude", ylabel="latitude",
            xticks=_LON_TICKS, yticks=_LAT_TICKS,
            show_grid=True, active_tools=list(active_tools),
            xlim=tuple(xlim), ylim=tuple(ylim), **sizing,
        )
    )
