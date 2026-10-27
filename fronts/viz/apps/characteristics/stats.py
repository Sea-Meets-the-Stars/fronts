"""Exact region statistics for page 1.

Everything here runs on the **native** grid at full resolution.  The
display pyramid is not involved: the map is a picture, this is the
arithmetic.

Two sample sets come out of one region:

``all``
    every finite grid cell inside the box;
``fronts``
    only the cells the labelled-front mask marks, which is the pixel-level
    counterpart of the colocated table.

Both are fed to the same three panel builders from the preprocessing repo,
so the two columns are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fronts.viz.apps.common import cache
from fronts.viz.apps.common.selection import BBox, bbox_mask


@dataclass
class RegionSamples:
    """Flat sample arrays for one region and one field.

    Attributes
    ----------
    values : numpy.ndarray
        The selected field.
    zeta_f, sigma_f : numpy.ndarray
        Normalised vorticity and strain, ready for the joint PDFs.
    n_cells : int
        How many grid cells the box selected before filtering.
    missing : tuple of str
        Roles that could not be resolved to a channel, so the joint PDFs
        cannot be drawn.  Empty when everything is present.
    missing_level : str
        Set when the roles in *missing* do exist in the store but not at
        the selected depth level -- a different thing to go and fix, so it
        is said differently.
    unavailable : str
        Why this column could not be built at all, or ``""`` when it was.
        Used for the fronts column while the front products are still
        being generated.
    """

    values: np.ndarray
    zeta_f: np.ndarray
    sigma_f: np.ndarray
    n_cells: int
    missing: tuple[str, ...] = ()
    missing_level: str = ""
    unavailable: str = ""

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def has_kinematics(self) -> bool:
        return not self.missing and self.zeta_f.size > 0


#: Below this latitude the Coriolis normalisation blows up, so f-normalised
#: quantities are dropped.  Matches the convention noted in ``dbof.plotting.pdfs``.
EQUATORIAL_CUTOFF_DEG = 2.0


def extract(
    provider,
    date: str,
    field: str,
    box: BBox,
    *,
    fronts_only: bool = False,
    resolve=None,
    level: str = "",
) -> RegionSamples:
    """Pull the sample arrays for one region.

    Parameters
    ----------
    provider : DataProvider
    date, field : str
    box : BBox
        The region, from the map's box-select.
    fronts_only : bool
        Restrict to labelled-front cells.
    resolve : callable, optional
        Maps a base field name to the channel name in the store.  The
        Depth page passes one that appends the depth suffix, so the
        vorticity and strain used by the joint PDFs come from the *same*
        depth level as the selected field rather than the surface.
        Default is the identity.
    level : str
        The depth level being examined, named in the message when a
        kinematic role exists in the store but not at that level.

    Returns
    -------
    RegionSamples
    """
    if resolve is None:
        def resolve(name):
            return name

    XC, YC = provider.coords(date)
    sel = bbox_mask(XC, YC, box)
    n_cells = int(sel.sum())

    # Ice-covered cells are excluded from every sample set, for the same
    # reason they are dropped from the map: they are a different regime,
    # and they dominate the distributions where they occur.
    ice = provider.ice_mask(date)
    if ice is not None:
        sel = sel & ~ice

    if fronts_only:
        # Always the *surface* fronts, whatever depth is being examined:
        # a front is a surface feature, and the question this column asks
        # is what the field looks like underneath one.  The provider points
        # its front lookups at the surface store for exactly this reason.
        sel = sel & (provider.labels(date) > 0)

    values = np.asarray(provider.field(date, resolve(field)))[sel]
    lat = np.asarray(YC)[sel]

    # Coriolis is a function of latitude alone so it has no depth variant,
    # and needs no special-casing: resolve falls back to the bare channel
    # when no suffixed one exists.  Vorticity and strain do have variants
    # and must match the level of the field being examined -- but a store
    # can hold a role at some levels and not others, and asking for the
    # missing one raised straight out of here.  That took down all six
    # panels, including the PDF of a field that had loaded perfectly well.
    # A role that cannot be resolved *at this level* is missing, not fatal.
    roles = provider.resolve_channels(date)
    channels, missing, missing_level = {}, [], ""
    for role, root in roles.items():
        if root is None:
            missing.append(role)
            continue
        try:
            channels[role] = resolve(root)
        except KeyError:
            missing.append(role)
            missing_level = level

    if missing:
        good = np.isfinite(values)
        return RegionSamples(
            values=values[good],
            zeta_f=np.empty(0), sigma_f=np.empty(0),
            n_cells=n_cells, missing=tuple(missing),
            missing_level=missing_level,
        )

    zeta = np.asarray(provider.field(date, channels["vorticity"]))[sel]
    sigma = np.asarray(provider.field(date, channels["strain"]))[sel]
    f0 = np.asarray(provider.field(date, channels["coriolis"]))[sel]

    # Away from the equator, and finite everywhere it matters.
    ok = (
        np.isfinite(values) & np.isfinite(zeta) & np.isfinite(sigma)
        & np.isfinite(f0) & (np.abs(lat) > EQUATORIAL_CUTOFF_DEG)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        zeta_f = zeta[ok] / f0[ok]
        sigma_f = sigma[ok] / np.abs(f0[ok])

    return RegionSamples(
        values=values[ok],
        zeta_f=zeta_f,
        sigma_f=sigma_f,
        n_cells=n_cells,
    )


def extract_both(provider, date, field, box, *, resolve=None, tag="") -> dict:
    """Both columns' samples, cached on everything that changes the answer.

    ``tag`` distinguishes cache entries that differ only by the resolver --
    the depth level, in practice.
    """
    key = cache.make_key("samples-v3-ice", provider.mode, date, field, tag,
                         box.key())
    hit = cache.get(key)
    if hit is not None:
        return hit

    out = {"all": extract(provider, date, field, box, fronts_only=False,
                          resolve=resolve, level=tag)}

    # The left column describes grid cells and owes nothing to the front
    # detection.  Only the right column needs the labels, so a build_v5
    # step that has not run yet costs that column and no more.
    try:
        out["fronts"] = extract(provider, date, field, box, fronts_only=True,
                                resolve=resolve, level=tag)
    except Exception as exc:                            # noqa: BLE001
        out["fronts"] = RegionSamples(
            values=np.empty(0), zeta_f=np.empty(0), sigma_f=np.empty(0),
            n_cells=0,
            unavailable=f"surface fronts unavailable for {date}: {exc}",
        )

    cache.put(key, out)
    return out


def cost_estimate(provider, date: str, box: BBox) -> int:
    """How many native cells a box selects, for the pre-flight message."""
    XC, YC = provider.coords(date)
    return int(bbox_mask(XC, YC, box).sum())
