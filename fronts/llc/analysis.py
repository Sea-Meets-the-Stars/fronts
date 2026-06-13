"""
Stratification diagnostics for LLC4320 density/temperature columns.

This module is the canonical home for column-wise (1-D profile) and
field-wise (3-D ``(k, j, i)``) helpers that turn a potential-density or
potential-temperature volume into derived stratification fields such as
the mixed-layer depth (MLD) using a configurable threshold.

The scalar versions ported here (`mixed_layer_depth`) replace the private
`_mixed_layer_depth` / `_pycnocline_depth` helpers that previously lived in
`dev/mld/plot_top_N_density_profiles.py`; the field-wise vectorised
twin (`mixed_layer_depth_field`) is new and intended for use by the
3-D fronts visualisation script (`fronts/scripts/fronts_viz_3d.py`).

Sections
--------
* MLD definitions and threshold constants
* Scalar (single-profile) helpers
* Field (3-D volume) helpers
"""

# stdlib
from __future__ import annotations

# numerical
import numpy as np


# ---------------------------------------------------------------------------
# MLD definitions and threshold constants
# ---------------------------------------------------------------------------

# Default delta_sigma0 threshold used for the "pycnocline depth" criterion in
# Bodner-style analyses (kg m^-3 above the reference-depth density).  The
# 3-D-fronts viz script uses this default; existing callers in
# dev/mld/plot_top_N_density_profiles.py pass the older 0.03 (mixed layer
# proper) or 0.125 (pycnocline) values explicitly.
DEFAULT_DELTA_SIGMA0 = 0.125  # kg m^-3
DEFAULT_REFERENCE_DEPTH_M = 10.0  # metres; "10 m below the surface"


# ---------------------------------------------------------------------------
# Scalar (single-profile) helpers
# ---------------------------------------------------------------------------

def reference_k(Z: np.ndarray, reference_depth_m: float = DEFAULT_REFERENCE_DEPTH_M) -> int:
    """Index of the LLC level whose depth is closest to ``reference_depth_m``.

    Parameters
    ----------
    Z : numpy.ndarray
        1-D depth array, length ``K``, in metres.  Convention: negative
        downward (matches LLC4320); only ``|Z|`` is used here.
    reference_depth_m : float
        Target depth in metres (positive).

    Returns
    -------
    int
        Index ``k`` minimising ``abs(|Z[k]| - reference_depth_m)``.
    """
    # abs(Z) handles either sign convention; argmin is exact, not interpolated.
    return int(np.abs(np.abs(Z) - float(reference_depth_m)).argmin())


def mixed_layer_depth(
    sigma0_profile: np.ndarray,
    Z: np.ndarray,
    delta_sigma0: float = DEFAULT_DELTA_SIGMA0,
    reference_depth_m: float = DEFAULT_REFERENCE_DEPTH_M,
) -> float | None:
    """Mixed-layer depth from a single ``sigma0(z)`` profile.

    The mixed layer is defined as the depth interval over which sigma0 has
    not yet exceeded its reference-depth value by more than
    ``delta_sigma0``.  This helper returns the depth of the *deepest* level
    still inside that mixed layer (i.e. the deepest level where
    ``sigma0(z) - sigma0(z = -reference_depth_m) <= delta_sigma0``).

    The scalar logic is a direct port of ``_mixed_layer_depth`` /
    ``_pycnocline_depth`` from
    ``dev/mld/plot_top_N_density_profiles.py`` -- the difference is that
    the threshold is now a parameter instead of a module constant.

    Parameters
    ----------
    sigma0_profile : numpy.ndarray
        1-D potential-density column, length ``K``, in kg m^-3.
    Z : numpy.ndarray
        1-D depth array, length ``K``, in metres.  Convention: negative
        downward (matches LLC4320).
    delta_sigma0 : float, optional
        Threshold density jump above the reference-depth density, in
        kg m^-3.  Defaults to the 0.125 kg m^-3 pycnocline-depth criterion;
        pass 0.03 for the conventional "mixed-layer" criterion.
    reference_depth_m : float, optional
        Reference depth in metres (positive).  Defaults to 10 m.

    Returns
    -------
    float or None
        Depth of the mixed-layer base in metres (negative downward), or
        ``None`` if the profile is empty or its value at the reference
        depth is non-finite.
    """
    if sigma0_profile.size == 0:
        return None
    k_ref = reference_k(Z, reference_depth_m)
    surface = float(sigma0_profile[k_ref])
    if not np.isfinite(surface):
        return None
    # delta is positive inside the mixed layer and grows with depth in a
    # stably stratified column; well-mixed water sits at delta <= threshold.
    delta = sigma0_profile - surface
    well_mixed = np.where(delta <= delta_sigma0)[0]
    if well_mixed.size == 0:
        return None
    return float(Z[well_mixed].min())


# ---------------------------------------------------------------------------
# Field (3-D volume) helpers
# ---------------------------------------------------------------------------

def mixed_layer_depth_field(
    sigma0: np.ndarray,
    Z: np.ndarray,
    delta_sigma0: float = DEFAULT_DELTA_SIGMA0,
    reference_depth_m: float = DEFAULT_REFERENCE_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised mixed-layer depth across a 3-D sigma0 array.

    Same definition as :func:`mixed_layer_depth` but applied to every
    ``(j, i)`` column of a 3-D sigma0 array in one pass.  Returns both the
    depth and the corresponding LLC level index so downstream code can clip
    a volume to a depth a few levels below the deepest MLD without doing a
    second search.

    Parameters
    ----------
    sigma0 : numpy.ndarray
        Potential-density array of shape ``(K, J, I)``, in kg m^-3.
    Z : numpy.ndarray
        1-D depth array of length ``K``, in metres (negative downward).
    delta_sigma0 : float, optional
        Threshold density jump above the reference-depth density, kg m^-3.
    reference_depth_m : float, optional
        Reference depth in metres (positive).

    Returns
    -------
    z_mld : numpy.ndarray
        ``(J, I)`` float array of mixed-layer depths in metres (negative
        downward).  NaN where the column's reference-depth density is
        non-finite or no well-mixed level exists.
    k_mld : numpy.ndarray
        ``(J, I)`` int32 array of the LLC level index corresponding to
        ``z_mld``.  ``-1`` where ``z_mld`` is NaN.
    """
    if sigma0.ndim != 3:
        raise ValueError(
            f"mixed_layer_depth_field expects sigma0 of shape (K, J, I); "
            f"got shape {sigma0.shape}."
        )
    K, J, I = sigma0.shape
    if Z.shape != (K,):
        raise ValueError(
            f"Z must be 1-D of length K={K}; got shape {Z.shape}."
        )

    k_ref = reference_k(Z, reference_depth_m)
    surface = sigma0[k_ref]  # (J, I)
    # Broadcast subtraction; NaN propagates through delta and silently
    # disqualifies the column from being marked well-mixed (NaN comparisons
    # return False in numpy).
    delta = sigma0 - surface[None, :, :]
    well_mixed = delta <= delta_sigma0  # (K, J, I), False where NaN

    # For each column, find the deepest k where well_mixed is True.  We
    # broadcast a (K, 1, 1) index array and mask invalid positions to -1
    # so that the per-column max gives the deepest valid index (or -1 if
    # none).
    k_indices = np.arange(K, dtype=np.int32)[:, None, None]
    k_well = np.where(well_mixed, k_indices, np.int32(-1))
    k_mld = k_well.max(axis=0).astype(np.int32)  # (J, I)

    valid = (k_mld >= 0) & np.isfinite(surface)
    # np.take is safer than fancy indexing when k_mld has -1 entries; we
    # mask the result afterwards via `valid`.
    z_mld = np.where(valid, Z[np.clip(k_mld, 0, K - 1)], np.nan).astype(np.float64)
    k_mld_out = np.where(valid, k_mld, np.int32(-1))
    return z_mld, k_mld_out
