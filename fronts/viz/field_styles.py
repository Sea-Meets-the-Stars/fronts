"""
Display styles for coloring 3-D front scenes by a secondary field.

The preprocessing repo's ``dbof/tiles/field_registry.py`` defines *what* can
be computed into a tile NetCDF (raw physical values, no display policy).
This module is its display-side counterpart: for each tile variable name
(``out_name`` in the preprocessing registry, e.g. ``"Ri"``, ``"vorticity"``)
it declares *how* the field should be transformed and colored when used as
the color scalar in ``fronts_viz_3d``.

The two registries are deliberately decoupled -- they are matched only by
the variable name stored in the tile NetCDF, which is self-describing
(``units`` / ``long_name`` attrs).  Unknown variables fall back to a linear
transform with percentile color limits (see :func:`get_style`).

Transforms
----------
``log10``
    ``log10(clip(x, *clip))`` with non-positive values -> NaN.  For
    positive-definite fields spanning decades (Ri, N2, shear, |grad b|^2).
``symlog``
    Signed pseudo-log: ``sign(x) * log10(1 + |x| / linthresh)``.  For signed
    fields spanning decades (Okubo-Weiss, frontogenesis, Ertel PV).
``linear``
    Optionally clipped passthrough.  For bounded or narrow-range fields
    (vorticity, divergence, Turner angle).

A ``center`` of 0.0 marks diverging fields: the default color limits are
made symmetric about 0 so the colormap midpoint sits at zero.
"""

# stdlib
from __future__ import annotations
import dataclasses as _dc
from dataclasses import dataclass

# numerical
import numpy as np


# Neutral gray used for NaN cells (land, clipped, undefined) on colored
# surfaces -- keeps the geometry hole-free while flagging missing data.
NAN_COLOR = "#9e9e9e"


@dataclass(frozen=True)
class FieldStyle:
    """Display policy for one tile variable used as a color scalar.

    Attributes
    ----------
    transform : str
        ``'log10'``, ``'symlog'``, or ``'linear'``.
    clip : tuple of (float, float) or None
        Clip range applied in *raw* units before the transform.
    cmap : str
        Default colormap (PyVista/matplotlib name).
    title : str
        Scalar-bar title (post-transform units).
    center : float or None
        If set (typically 0.0), default clim is symmetric about it.
    linthresh : float
        Linear threshold for ``symlog`` (ignored otherwise).
    clim : tuple of (float, float) or None
        Pinned post-transform color limits.  None -> percentile-based.
    scale : float
        Divisor applied to the values in the ``linear`` transform (after any
        ``clip``), so the displayed numbers are ``value / scale`` -- e.g.
        ``scale=1e-7`` shows the field in units of 1e-7.  Ignored by the
        ``log10`` / ``symlog`` transforms.
    """
    transform: str = "linear"
    clip: tuple[float, float] | None = None
    cmap: str = "viridis"
    title: str = ""
    center: float | None = None
    linthresh: float = 1e-12
    clim: tuple[float, float] | None = None
    scale: float = 1.0


# Keyed by the tile NetCDF variable name (= ``out_name`` in the preprocessing
# repo's ``dbof.tiles.field_registry.TILE_PROPERTIES``).  Add a row when you
# add a property over there; pre-rename names live in LEGACY_VAR_NAMES below.
FIELD_STYLES: dict[str, FieldStyle] = {
    "sigma0": FieldStyle(
        transform="linear", cmap="dense",
        title="sigma0 [kg/m^3]",
    ),
    "Theta": FieldStyle(
        transform="linear", cmap="thermal",
        title="Theta [degC]",
    ),
    "Salt": FieldStyle(
        transform="linear", cmap="haline",
        title="Salt [psu]",
    ),
    # Ri and R_ib share this style deliberately: they are the same
    # quantity expressed two ways, and separate bars make them
    # incomparable at a glance.  Linear, not log: the interesting range is
    # around the O(1) instability threshold, and a log axis pushes it into
    # the middle of the bar where it reads as nothing in particular.
    "Ri": FieldStyle(
        transform="linear", clip=(0.0, 5.0), cmap="RdYlBu",
        # Ri < 0.25 is shear-unstable, so the clim keeps that in the warm
        # end rather than compressed against the edge.
        clim=(0.0, 2.0), title="Ri",
    ),
    # Same object with a different label, so a change to one is a change
    # to both -- see RI_STYLE below.
    "R_ib": None,        # filled in after the table, from "Ri"
    # Vertical velocity is signed, so it gets cmocean's diverging
    # ``balance`` with the midpoint pinned at zero -- up and down have to
    # read as opposite, not as two ends of a sequential ramp.
    "W": FieldStyle(
        transform="linear", cmap="balance",
        title="W [m/s]", center=0.0,
    ),
    "N2": FieldStyle(
        transform="linear", clip=(0.0, 1e-3), cmap="viridis",
        title="N^2 [s^-2]",
    ),
    "vertical_shear": FieldStyle(
        transform="linear", clip=(0.0, 1e-1), cmap="viridis",
        title="vertical shear [s^-1]",
    ),
    "relative_vorticity": FieldStyle(
        transform="linear", cmap="RdBu_r",
        title="zeta [1/s]", center=0.0,
    ),
    "divergence": FieldStyle(
        transform="linear", cmap="RdBu_r",
        title="div [1/s]", center=0.0,
    ),
    "strain_mag": FieldStyle(
        transform="linear", clip=(0.0, 1e-3), cmap="viridis",
        title="strain [s^-1]",
    ),
    # The registry also exposes the signed normal/shear strain components;
    # unlike the magnitude these change sign, so they get a diverging map.
    "strain_n": FieldStyle(
        transform="linear", cmap="RdBu_r",
        title="normal strain [1/s]", center=0.0,
    ),
    "strain_s": FieldStyle(
        transform="linear", cmap="RdBu_r",
        title="shear strain [1/s]", center=0.0,
    ),
    "okubo_weiss": FieldStyle(
        transform="linear", cmap="RdBu_r",
        title="OW [s^-2]", center=0.0,
    ),
    "Ro": FieldStyle(
        transform="linear", clip=(-10.0, 10.0), cmap="RdBu_r",
        title="Ro", center=0.0,
    ),
    # ``rossby_number`` is a separate registry channel that writes the same
    # quantity under its long name; display it identically to ``Ro``.
    "rossby_number": FieldStyle(
        transform="linear", clip=(-10.0, 10.0), cmap="RdBu_r",
        title="Ro", center=0.0,
    ),
    "Fr": FieldStyle(
        transform="log10", clip=(1e-3, 1e1), cmap="viridis",
        title="log10(Fr)",
    ),
    "Bu": FieldStyle(
        transform="log10", clip=(1e-3, 1e3), cmap="viridis",
        title="log10(Bu)",
    ),
    "frontogenesis_tendency": FieldStyle(
        transform="linear", scale=1e-7, cmap="RdBu_r",
        title="Fs [s^-5]", center=0.0,
    ),
    "frontogenesis_geo": FieldStyle(
        transform="linear", scale=1e-7, cmap="RdBu_r",
        title="Fs geo [s^-5]", center=0.0,
    ),
    "frontogenesis_ageo": FieldStyle(
        transform="linear", scale=1e-7, cmap="RdBu_r",
        title="Fs ageo [s^-5]", center=0.0,
    ),
    "ertel_pv": FieldStyle(
        transform="linear", scale=1e-7, cmap="RdBu_r",
        title="q / 1e-7 [s^-3]", center=0.0,
    ),
    "turner_angle": FieldStyle(
        transform="linear", cmap="twilight_shifted",
        title="Tu [deg]", center=0.0,
    ),
    # Squared buoyancy gradient: strictly positive and spanning many orders
    # of magnitude, so linear limits put every front in the top percent of
    # the bar and the rest of the field at zero.  Same treatment as the
    # other squared-gradient channels below.
    # No clip: values <= 0 already become NaN before the log, and the 2/98
    # percentiles set the limits from the data, which is safer than pinning
    # a range across a field that spans many orders of magnitude.
    # Same bar as the other squared gradients, so the four are comparable.
    #
    # Note the global maps keep gradb2 in greyscale -- see
    # ``basemap._FIELD_CMAPS``.  That is a different job: there the field
    # is a backdrop for the coloured front overlay, and a coloured base
    # competes with it.  Here it is the subject.
    "gradb2": FieldStyle(
        transform="log10", cmap="magma",
        title="log10(|grad b|^2 [s^-4])",
    ),
    "gradrho2": FieldStyle(
        transform="log10", cmap="magma",
        title="log10(|grad rho|^2 [kg^2 m^-8])",
    ),
    "gradtheta2": FieldStyle(
        transform="log10", cmap="magma",
        title="log10(|grad Theta|^2 [degC^2 m^-2])",
    ),
    "gradsalt2": FieldStyle(
        transform="log10", cmap="magma",
        title="log10(|grad S|^2 [psu^2 m^-2])",
    ),
    "wB": FieldStyle(
        # Signed vertical buoyancy flux (down/up-gradient).  cmocean's
        # ``balance`` rather than RdBu_r: it is the oceanographic
        # diverging map and is perceptually uniform either side of zero.
        #
        # The transform stays symlog, which is a separate question from
        # the colormap: wB changes sign *and* spans decades, and a linear
        # scale puts everything except the extremes at the midpoint.
        transform="symlog", cmap="balance",
        title="symlog(wB [m^2 s^-3])", center=0.0, linthresh=1e-4,
    ),
}


#: Variable names written by the *old* preprocessing ``tiles_field`` branch,
#: mapped to their current out-names.  Tiles already on disk keep their old
#: variable name inside the NetCDF, so resolve through this before falling
#: back to the default style.
#: R_ib shares Ri's display policy exactly; only the label differs.  Set
#: here rather than duplicated above so the two cannot drift apart.
FIELD_STYLES["R_ib"] = _dc.replace(FIELD_STYLES["Ri"], title="R_ib")


LEGACY_VAR_NAMES: dict[str, str] = {
    "vorticity": "relative_vorticity",
    "strain":    "strain_mag",
    "Fs":        "frontogenesis_tendency",
}


def resolve_cmap(name: str):
    """Resolve a style's colormap name to something matplotlib accepts.

    ``FieldStyle.cmap`` carries names for matplotlib *and* PyVista, and
    the cmocean ones (``dense``, ``thermal``, ``haline``) are not
    matplotlib names -- passing them straight to ``pcolormesh`` raises
    ``ValueError: 'dense' is not a valid value for cmap``.  That made
    every matplotlib panel fail for a density-styled field, which is
    exactly the default on the Tiles page.

    Returns a ``Colormap`` object, which every matplotlib entry point
    accepts in place of a name.
    """
    import matplotlib as mpl

    for candidate in (name, f"cmo.{name}"):
        try:
            return mpl.colormaps[candidate]
        except (KeyError, AttributeError):
            continue
    try:
        import cmocean
        return getattr(cmocean.cm, name)
    except Exception:                                       # noqa: BLE001
        return mpl.colormaps["viridis"]


#: Suffixes the DEPTH pipeline appends to a channel name.
#:
#: Longest first, because ``mld`` is a prefix of ``mld_mean``: stripping
#: ``_mld`` from ``N2_mld_mean`` leaves ``N2_mean``, which is registered
#: nowhere and silently falls back to a linear style.
DEPTH_SUFFIXES = ("_mld_mean", "_z25m", "_mld", "_sfc")


def strip_depth_suffix(var_name: str) -> str:
    """``gradb2_mld`` -> ``gradb2``.  Anything else is returned unchanged."""
    for suffix in DEPTH_SUFFIXES:
        if var_name.endswith(suffix):
            return var_name[: -len(suffix)]
    return var_name


def get_style(var_name: str) -> FieldStyle:
    """Look up the style for a tile variable, with a safe linear fallback.

    Parameters
    ----------
    var_name : str
        Tile NetCDF variable name.

    Returns
    -------
    FieldStyle
        The registered style, or a default linear style (titled with the
        variable name) when the field is unknown.
    """
    style = FIELD_STYLES.get(var_name)
    if style is None and var_name in LEGACY_VAR_NAMES:
        # A tile written by the old branch: same field, older variable name.
        style = FIELD_STYLES.get(LEGACY_VAR_NAMES[var_name])
    if style is None:
        # A DEPTH channel: gradb2_mld is gradb2, and should be drawn the
        # same way -- log10, same colours -- so the surface and depth
        # views of one field are directly comparable.  Without this every
        # depth field fell back to a linear percentile style.
        root = strip_depth_suffix(var_name)
        if root != var_name:
            style = FIELD_STYLES.get(root)
    if style is None:
        import logging
        logging.getLogger(__name__).warning(
            "No FIELD_STYLES entry for %r -- falling back to a linear "
            "transform with percentile color limits.  Add an entry to "
            "fronts/viz/field_styles.py to control its display.", var_name,
        )
        return FieldStyle(title=var_name)
    return style


def apply_transform(
    values: np.ndarray,
    style: FieldStyle,
    *,
    clip_override: tuple[float, float] | None = None,
    transform_override: str | None = None,
) -> np.ndarray:
    """Transform raw field values into display values per a style.

    Non-finite inputs stay NaN.  For ``log10``, values <= 0 become NaN
    (e.g. negative Ri from unstable stratification) and the remainder are
    clipped into ``clip`` before the log.  For ``symlog`` and ``linear``,
    clipping (when configured) is a plain ``np.clip``.

    Parameters
    ----------
    values : numpy.ndarray
        Raw field values (any shape).
    style : FieldStyle
        Display policy (see :data:`FIELD_STYLES`).
    clip_override, transform_override : optional
        CLI-level overrides for the style's ``clip`` / ``transform``.

    Returns
    -------
    numpy.ndarray
        float64 array of display values, NaN where undefined.
    """
    transform = transform_override or style.transform
    clip = clip_override if clip_override is not None else style.clip

    out = np.asarray(values, dtype=np.float64).copy()

    if transform == "log10":
        out[~np.isfinite(out)] = np.nan
        out[out <= 0] = np.nan
        if clip is not None:
            out = np.clip(out, clip[0], clip[1])
        return np.log10(out)

    if transform == "symlog":
        if clip is not None:
            out = np.clip(out, clip[0], clip[1])
        lt = float(style.linthresh)
        return np.sign(out) * np.log10(1.0 + np.abs(out) / lt)

    if transform == "linear":
        if clip is not None:
            out = np.clip(out, clip[0], clip[1])
        if style.scale != 1.0:
            out = out / style.scale
        return out

    raise ValueError(
        f"Unknown transform {transform!r}; expected 'log10', 'symlog', "
        "or 'linear'."
    )


def default_clim(
    display_values: np.ndarray,
    style: FieldStyle,
    *,
    clip_override: tuple[float, float] | None = None,
    transform_override: str | None = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[float, float]:
    """Default post-transform color limits for a field.

    Order of precedence: the style's pinned ``clim`` (only when neither
    override is given -- a pinned clim is calibrated to the style's own
    display space, so a CLI transform/clip override invalidates it);
    symmetric limits about ``style.center`` (when set) using the larger
    percentile excursion; plain 2/98 percentiles otherwise.

    Parameters
    ----------
    display_values : numpy.ndarray
        Transformed (display-space) values; NaNs ignored.
    style : FieldStyle
        Display policy.
    clip_override, transform_override : optional
        CLI-level overrides for the style's ``clip`` / ``transform``.
        Pass the same values given to :func:`apply_transform`.
    percentile_low, percentile_high : float, optional
        Percentile bounds (default 2/98).

    Returns
    -------
    tuple of (float, float)
    """
    style_clim_ok = (
        style.clim is not None
        and clip_override is None
        and transform_override is None
    )
    if style_clim_ok:
        return tuple(style.clim)
    lo = float(np.nanpercentile(display_values, percentile_low))
    hi = float(np.nanpercentile(display_values, percentile_high))
    if style.center is not None:
        span = max(abs(lo - style.center), abs(hi - style.center))
        return (style.center - span, style.center + span)
    return (lo, hi)
