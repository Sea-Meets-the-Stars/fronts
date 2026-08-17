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
    """

    values: np.ndarray
    zeta_f: np.ndarray
    sigma_f: np.ndarray
    n_cells: int
    missing: tuple[str, ...] = ()

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

    if fronts_only:
        sel = sel & (provider.labels(date) > 0)

    values = np.asarray(provider.field(date, resolve(field)))[sel]
    lat = np.asarray(YC)[sel]

    roles = provider.resolve_channels(date)
    missing = tuple(r for r, c in roles.items() if c is None)

    if missing:
        good = np.isfinite(values)
        return RegionSamples(
            values=values[good],
            zeta_f=np.empty(0), sigma_f=np.empty(0),
            n_cells=n_cells, missing=missing,
        )

    # Coriolis is a function of latitude alone, so it has no depth
    # variant; vorticity and strain do and must match the selected level.
    zeta = np.asarray(provider.field(date, resolve(roles["vorticity"])))[sel]
    sigma = np.asarray(provider.field(date, resolve(roles["strain"])))[sel]
    f0 = np.asarray(provider.field(date, roles["coriolis"]))[sel]

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
    key = cache.make_key("samples-v2", provider.mode, date, field, tag,
                         box.key())
    hit = cache.get(key)
    if hit is not None:
        return hit

    out = {
        "all": extract(provider, date, field, box, fronts_only=False,
                       resolve=resolve),
        "fronts": extract(provider, date, field, box, fronts_only=True,
                          resolve=resolve),
    }
    cache.put(key, out)
    return out


def cost_estimate(provider, date: str, box: BBox) -> int:
    """How many native cells a box selects, for the pre-flight message."""
    XC, YC = provider.coords(date)
    return int(bbox_mask(XC, YC, box).sum())
