"""The Pacific-centred global map.

Layers, bottom to top:

1. the selected field, from the display pyramid, datashaded;
2. land in gray, taken from the model's own NaN mask;
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


def _affordable_width(width: int) -> int:
    """The finest pyramid width we are willing to send without datashader.

    Heights are derived from width, so cells scale as ``width**2``; this
    steps down through ``config.PYRAMID_WIDTHS`` until the raster fits the
    budget.  With datashader present the requested width is used as-is.
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

# Longitude ticks for a Pacific-centred 0..360 axis, labelled in E/W.
_LON_TICKS = [
    (0, "0"), (60, "60E"), (120, "120E"), (180, "180"),
    (240, "120W"), (300, "60W"), (360, "0"),
]
_LAT_TICKS = [(v, f"{abs(v)}{'N' if v > 0 else 'S' if v < 0 else ''}")
              for v in (-60, -30, 0, 30, 60)]

_FIELD_CMAPS = {
    "gradb2": "magma",
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

_LOG_FIELDS = {"gradb2", "gradtheta2"}
_DIVERGING = {
    "relative_vorticity", "divergence", "okubo_weiss",
    "strain_n", "strain_s", "Eta", "coriolis_f",
}


def field_display(arr: np.ndarray, name: str):
    """Transform a field for display, returning ``(values, clim, label)``."""
    if name in _LOG_FIELDS:
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
    if name in _DIVERGING:
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


def field_layer(provider, date, name, width=config.PYRAMID_WIDTHS[1],
                *, tools=("box_select",)):
    """The field raster, datashaded.

    ``tools`` has to be attached to a concrete element, not to the
    enclosing Overlay -- an Overlay-level ``active_tools`` naming a tool
    nothing added just logs "could not be found" and the box-select never
    appears.
    """
    lon, lat, arr = pyramid.level(provider, date, name,
                                  _affordable_width(width))
    values, clim, label = field_display(arr, name)
    img = _image(lon, lat, values, label, "Field")
    cmap = _FIELD_CMAPS.get(name, "viridis")
    return _rasterize(img).opts(
        cmap=cmap, clim=clim, colorbar=True, colorbar_position="bottom",
        tools=list(tools),
    )


def land_layer(provider, date, width=config.PYRAMID_WIDTHS[1]):
    """Land in gray, from the model's own mask."""
    lon, lat, arr = pyramid.level(provider, date, "__land__",
                                  _affordable_width(width), reduce="any")
    masked = np.where(arr > 0, 1.0, np.nan)
    img = _image(lon, lat, masked, "land", "Land")
    return _rasterize(img).opts(
        cmap=["#b0b0b0"], clim=(0, 1), colorbar=False,
    )


def fronts_layer(provider, date, width=config.PYRAMID_WIDTHS[2]):
    """Binary fronts, drawn on top of the field."""
    lon, lat, arr = pyramid.level(provider, date, "__fronts__",
                                  _affordable_width(width), reduce="any")
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
    width: int = 760,
    height: int = 430,
    title: str = "",
    tools=("box_select",),
    active_tools=("box_select",),
):
    """Assemble the full Pacific-centred map.

    Returns a HoloViews ``Overlay`` on a 0..360 longitude axis.
    """
    layers = [field_layer(provider, date, field, tools=tools),
              land_layer(provider, date)]

    coast = coastline_layer()
    if coast is not None:
        layers.append(coast)

    if show_fronts:
        layers.append(fronts_layer(provider, date))

    return hv.Overlay(layers).opts(
        hv.opts.Overlay(
            width=width, height=height, title=title,
            xlabel="longitude", ylabel="latitude",
            xticks=_LON_TICKS, yticks=_LAT_TICKS,
            show_grid=True, active_tools=list(active_tools),
            xlim=(0, 360), ylim=config.PYRAMID_LAT_RANGE,
        )
    )
