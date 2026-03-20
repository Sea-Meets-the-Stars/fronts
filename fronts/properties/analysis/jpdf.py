"""
fronts.properties.jpdf
======================
Joint probability distribution function (JPDF) of surface vorticity and
strain, and conditional means of arbitrary scalar fields in vorticity–strain
space.

Based on the diagnostic framework of
    Balwada et al. (2021) "Vertical Fluxes Conditioned on Vorticity and Strain
    Reveal Submesoscale Ventilation", J. Phys. Oceanogr., 51, 2883–2902.
    https://doi.org/10.1175/JPO-D-21-0016.1

Mathematical background (Appendix B of Balwada et al.)
-------------------------------------------------------
Let F(x,y), ζ(x,y), σ(x,y) be scalar fields over spatial domain A.

JPDF (Eq. B3)
    P(ζ, σ) = F̃(ζ, σ; F=1) / A

where F̃ is the fraction of domain area with (ζ', σ') in [ζ, ζ+dζ)×[σ, σ+dσ),
normalised so that ∬_R P(ζ,σ) dζ dσ = 1.

Conditional mean (Eq. B5)
    F̄^{ζσ}(ζ, σ) = Σ_{(i,j)∈B} F_{ij}  /  |B|

where B = {(i,j) ∈ A | ζ'_{ij} ∈ [ζ, ζ+dζ), σ'_{ij} ∈ [σ, σ+dσ)}.
The Dirac deltas in Eq. B5 become bin membership in the discrete
implementation: for each (ζ, σ) bin, average F over all pixels falling in it.

Flow regions (Balwada et al. section 2b)
    AVD  ζ < 0  and  σ < |ζ|   anticyclonic vorticity dominated
    CVD  ζ > 0  and  σ < |ζ|   cyclonic vorticity dominated
    SD   σ ≥ |ζ|                strain dominated  (associated with fronts)
The boundaries are the lines σ = |ζ|.

Typical normalisation
---------------------
Following Balwada et al., normalise before calling:
    vorticity / f   →  ζ/f   (Rossby number, signed, x-axis of JPDF)
    strain    / |f| →  σ/|f| (always ≥ 0, y-axis of JPDF)

Equatorial singularity
----------------------
f = 2Ω sin(lat) → 0 at the equator, so ζ/f and σ/|f| diverge there.
Use the helper :func:`normalise_by_coriolis` which safely masks the
equatorial band before you call the JPDF functions::

    ro, snorm = normalise_by_coriolis(vorticity, strain, coriolis_f,
                                      min_abs_lat=5.0, lat=lat_global)

Pixels within ``min_abs_lat`` degrees of the equator (or wherever |f| is
below the equivalent threshold) are set to NaN and automatically excluded
from the histogram.  Balwada et al. work in the Southern Ocean and never
encounter this; for global LLC4320 analysis the equatorial band must be
excluded.

Multi-snapshot averaging
------------------------
When averaging over multiple snapshots (Balwada et al. use ~30 snapshots
separated by 10 days), use JPDFAccumulator.  It accumulates raw bin counts
and field sums separately across calls, then divides at finalize() to
recover time-mean statistics with correctly propagated denominators::

    acc = JPDFAccumulator(n_vort_bins=100, n_strain_bins=100,
                          vort_range=(-6, 6), strain_range=(0, 6))
    for vort_t, strain_t, w_t, gradb2_t in snapshots:
        acc.add(vort_t, strain_t, fields={'w': w_t, 'gradb2': gradb2_t})
    jpdf, cond_means = acc.finalize()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_valid(*arrays: np.ndarray) -> Tuple[Tuple[np.ndarray, ...], int]:
    """Flatten and co-mask NaN from multiple co-registered arrays.

    Returns (tuple_of_1d_valid_arrays, n_valid).
    """
    flat = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    valid = np.ones(flat[0].shape, dtype=bool)
    for f in flat:
        valid &= np.isfinite(f)
    return tuple(f[valid] for f in flat), int(valid.sum())


def _auto_range(
    data: np.ndarray,
    clip_pct: float,
    symmetric: bool = False,
) -> Tuple[float, float]:
    """Percentile-based bin range, optionally forced symmetric around zero."""
    lo = float(np.nanpercentile(data, 100.0 - clip_pct))
    hi = float(np.nanpercentile(data, clip_pct))
    if symmetric:
        bound = max(abs(lo), abs(hi))
        return (-bound, bound)
    # Strain is non-negative: floor at 0
    return (max(0.0, lo), hi)


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


# ---------------------------------------------------------------------------
# Public normalisation helper
# ---------------------------------------------------------------------------

def normalise_by_coriolis(
    vorticity:   np.ndarray,
    strain:      np.ndarray,
    coriolis_f:  np.ndarray,
    min_abs_lat: float = 5.0,
    lat:         Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ζ/f and σ/|f|, masking the equatorial band where f → 0.

    f = 2Ω sin(lat) vanishes at the equator, so ζ/f and σ/|f| diverge
    there and would dominate the JPDF axis range for a global domain.
    This function sets pixels in the equatorial band to NaN so they are
    automatically excluded from the JPDF histogram.

    Two guards are applied:
    1. **Latitude mask** (if ``lat`` is provided): pixels with
       ``|lat| < min_abs_lat`` degrees are masked.
    2. **Coriolis magnitude mask**: pixels where ``|f| < f_eq`` are
       masked, where ``f_eq = 2Ω sin(min_abs_lat°)``.  This catches any
       remaining near-zero f values regardless of latitude.

    Parameters
    ----------
    vorticity : ndarray
        Raw surface relative vorticity ζ (s⁻¹), any shape.
    strain : ndarray
        Strain magnitude σ (s⁻¹), same shape as ``vorticity``.
    coriolis_f : ndarray
        Coriolis parameter f (s⁻¹), same shape.  Must not be all-zero.
    min_abs_lat : float
        Minimum absolute latitude (degrees) to include.  Pixels within
        this distance of the equator are set to NaN.  Default 5°.
    lat : ndarray, optional
        Latitude array (degrees), same shape as ``vorticity``.  When
        provided, the latitude mask is applied in addition to the
        Coriolis-magnitude mask.  Pass ``lat_global`` directly.

    Returns
    -------
    ro : ndarray
        ζ / f  (Rossby number, signed).  NaN in equatorial band.
    snorm : ndarray
        σ / |f|  (normalised strain, ≥ 0).  NaN in equatorial band.

    Examples
    --------
    >>> ro, snorm = normalise_by_coriolis(
    ...     property_arrays['relative_vorticity'],
    ...     property_arrays['strain_mag'],
    ...     property_arrays['coriolis_f'],
    ...     min_abs_lat=5.0, lat=lat_global,
    ... )
    >>> jpdf = compute_jpdf(ro, snorm)
    """
    OMEGA = 7.2921e-5   # Earth's rotation rate (rad s⁻¹)
    f_min = 2.0 * OMEGA * np.sin(np.deg2rad(min_abs_lat))  # > 0

    vort_a  = np.asarray(vorticity,  dtype=np.float64)
    strain_a = np.asarray(strain,    dtype=np.float64)
    f_a     = np.asarray(coriolis_f, dtype=np.float64)

    # Mask pixels where |f| is too small (includes exactly-zero at equator)
    f_safe = np.where(np.abs(f_a) >= f_min, f_a, np.nan)

    # Optional additional latitude mask
    if lat is not None:
        lat_a  = np.asarray(lat, dtype=np.float64)
        f_safe = np.where(np.abs(lat_a) >= min_abs_lat, f_safe, np.nan)

    ro    = vort_a  / f_safe
    snorm = strain_a / np.abs(f_safe)

    n_masked = np.isnan(f_safe).sum() - np.isnan(f_a).sum()
    if n_masked > 0:
        print(f"normalise_by_coriolis: masked {n_masked:,} equatorial pixels "
              f"(|lat| < {min_abs_lat}° or |f| < {f_min:.2e} s⁻¹)")

    return ro, snorm


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class JPDFResult:
    """
    Container for the vorticity–strain JPDF  P(ζ, σ).

    Axes convention (matching Balwada et al. figures)
    -------------------------------------------------
    - First axis  (rows)    : vorticity  ζ  (x-axis in JPDF plots)
    - Second axis (columns) : strain     σ  (y-axis in JPDF plots)

    So ``pdf[i, j]`` is the probability density at
    vorticity ∈ [vort_edges[i], vort_edges[i+1]) and
    strain   ∈ [strain_edges[j], strain_edges[j+1]).

    To plot with pcolormesh using the standard Balwada orientation
    (ζ on x, σ on y), transpose the pdf::

        ax.pcolormesh(vort_centers, strain_centers, result.pdf.T,
                      norm=LogNorm(), cmap='Reds')
    """

    pdf: np.ndarray
    """Shape (n_vort_bins, n_strain_bins).  Units: 1 / (vort_unit × strain_unit).
    Normalised so that ∬ pdf dζ dσ = 1 (discrete: sum(pdf * dζ * dσ) = 1)."""

    count: np.ndarray
    """Shape (n_vort_bins, n_strain_bins).  Raw pixel count per bin."""

    vort_edges:    np.ndarray  # (n_vort_bins + 1,)
    strain_edges:  np.ndarray  # (n_strain_bins + 1,)
    vort_centers:  np.ndarray  # (n_vort_bins,)
    strain_centers: np.ndarray  # (n_strain_bins,)

    n_samples:   int    # total valid pixels used
    n_snapshots: int    # number of snapshots accumulated (1 for single call)

    def region_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Boolean masks for the three Balwada et al. flow regions on the JPDF grid.

        Returns
        -------
        avd, cvd, sd : ndarray, shape (n_vort_bins, n_strain_bins)
            AVD — anticyclonic vorticity dominated  (ζ < 0, σ < |ζ|)
            CVD — cyclonic vorticity dominated      (ζ > 0, σ < |ζ|)
            SD  — strain dominated                  (σ ≥ |ζ|)
        """
        zc, sc = np.meshgrid(self.vort_centers, self.strain_centers, indexing='ij')
        avd = (zc < 0) & (sc < np.abs(zc))
        cvd = (zc > 0) & (sc < np.abs(zc))
        sd  = sc >= np.abs(zc)
        return avd, cvd, sd

    def region_fractions(self) -> Dict[str, float]:
        """
        Return the fraction of total probability mass in each flow region.

        Uses the JPDF (not raw counts) so the fractions sum to 1 when
        integrated over the full bin range.
        """
        avd_m, cvd_m, sd_m = self.region_masks()
        dz = np.diff(self.vort_edges)
        ds = np.diff(self.strain_edges)
        dA = np.outer(dz, ds)
        total = (self.pdf * dA).sum()
        return {
            'AVD': float((self.pdf * dA * avd_m).sum() / total),
            'CVD': float((self.pdf * dA * cvd_m).sum() / total),
            'SD':  float((self.pdf * dA * sd_m).sum()  / total),
        }


@dataclass
class ConditionalMeanResult:
    """
    Container for  F̄^{ζσ}(ζ, σ) — the conditional mean of F given ζ and σ.

    Axes convention: same as JPDFResult (first axis = vorticity, second = strain).
    Plot with ``result.mean.T`` on axes (ζ, σ).

    Bins with fewer than ``min_count`` samples are set to NaN.
    """

    mean: np.ndarray
    """Shape (n_vort_bins, n_strain_bins).  NaN where count < min_count."""

    count: np.ndarray
    """Shape (n_vort_bins, n_strain_bins).  Number of samples per bin."""

    vort_edges:     np.ndarray
    strain_edges:   np.ndarray
    vort_centers:   np.ndarray
    strain_centers: np.ndarray

    field_name:  str
    n_samples:   int
    n_snapshots: int
    min_count:   int


# ---------------------------------------------------------------------------
# Single-snapshot functions
# ---------------------------------------------------------------------------

def compute_jpdf(
    vorticity:    Union[np.ndarray, Sequence],
    strain:       Union[np.ndarray, Sequence],
    n_vort_bins:  int = 100,
    n_strain_bins: int = 100,
    vort_range:   Optional[Tuple[float, float]] = None,
    strain_range: Optional[Tuple[float, float]] = None,
    clip_pct:     float = 99.5,
) -> JPDFResult:
    """
    Compute the vorticity–strain JPDF  P(ζ, σ)  (Eq. 3 / B3, Balwada et al.).

    Parameters
    ----------
    vorticity : array-like
        Surface relative vorticity — 2-D spatial map or 1-D vector.
        NaN/Inf pixels are excluded.  Typically pre-normalised by f
        (Rossby number ζ/f) to match Balwada et al. axis convention.
    strain : array-like
        Strain magnitude — same shape as ``vorticity``.  Must be ≥ 0.
        Typically pre-normalised by |f| (σ/|f|).
    n_vort_bins, n_strain_bins : int
        Number of bins along each axis.
    vort_range, strain_range : (float, float), optional
        Explicit bin-edge ranges.  If None, auto-computed from ``clip_pct``
        percentile of the data; vorticity range is forced symmetric.
    clip_pct : float
        Percentile used for auto-ranging (ignored when explicit ranges given).
        For vorticity, the range is ±P{clip_pct}th.
        For strain, the range is [0, P{clip_pct}th].

    Returns
    -------
    JPDFResult
        ``result.pdf`` is normalised so that the discrete integral
        sum(pdf * dζ * dσ) equals 1.

    Examples
    --------
    >>> # With Rossby-number normalisation
    >>> ro   = vorticity_map / coriolis_f_map
    >>> snorm = strain_map   / np.abs(coriolis_f_map)
    >>> jpdf = compute_jpdf(ro, snorm, n_vort_bins=100, n_strain_bins=80)
    >>> print(jpdf.region_fractions())
    >>>
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import LogNorm
    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(jpdf.vort_edges, jpdf.strain_edges, jpdf.pdf.T,
    ...               norm=LogNorm(), cmap='Reds')
    """
    (vort_v, strain_v), n_valid = _flatten_valid(vorticity, strain)

    if n_valid == 0:
        raise ValueError('No valid (non-NaN/Inf) pixels found.')

    if vort_range is None:
        vort_range = _auto_range(vort_v, clip_pct, symmetric=True)
    if strain_range is None:
        strain_range = _auto_range(strain_v, clip_pct, symmetric=False)

    vort_edges   = np.linspace(vort_range[0],   vort_range[1],   n_vort_bins + 1)
    strain_edges = np.linspace(strain_range[0], strain_range[1], n_strain_bins + 1)

    # histogram2d with density=True normalises so sum(count * dz * ds) = 1
    pdf, _, _ = np.histogram2d(
        vort_v, strain_v,
        bins=[vort_edges, strain_edges],
        density=True,
    )

    count, _, _ = np.histogram2d(
        vort_v, strain_v,
        bins=[vort_edges, strain_edges],
        density=False,
    )

    return JPDFResult(
        pdf=pdf.astype(np.float64),
        count=count.astype(np.int64),
        vort_edges=vort_edges,
        strain_edges=strain_edges,
        vort_centers=_bin_centers(vort_edges),
        strain_centers=_bin_centers(strain_edges),
        n_samples=n_valid,
        n_snapshots=1,
    )


def conditional_mean(
    F:            Union[np.ndarray, Sequence],
    vorticity:    Union[np.ndarray, Sequence],
    strain:       Union[np.ndarray, Sequence],
    field_name:   str = 'F',
    vort_edges:   Optional[np.ndarray] = None,
    strain_edges: Optional[np.ndarray] = None,
    n_vort_bins:  int = 100,
    n_strain_bins: int = 100,
    vort_range:   Optional[Tuple[float, float]] = None,
    strain_range: Optional[Tuple[float, float]] = None,
    clip_pct:     float = 99.5,
    min_count:    int = 5,
) -> ConditionalMeanResult:
    """
    Compute  F̄^{ζσ}(ζ, σ)  — conditional mean of F given ζ and σ  (Eq. B5).

    For each (ζ, σ) bin, average F over all spatial points whose vorticity
    and strain fall in that bin.  This is the discrete implementation of the
    Dirac-delta integrals in Eq. B5 of Balwada et al.

    Parameters
    ----------
    F : array-like
        Scalar field to average — same shape as ``vorticity``.
        NaN values in F, vorticity, or strain are all excluded.
    vorticity, strain : array-like
        Same as in :func:`compute_jpdf`.
    field_name : str
        Label stored in the result (used for plots / repr).
    vort_edges, strain_edges : ndarray, optional
        Pre-computed bin edges — pass ``jpdf_result.vort_edges`` and
        ``jpdf_result.strain_edges`` to guarantee alignment with a
        previously computed JPDF.  When provided, ``n_vort_bins``,
        ``n_strain_bins``, ``vort_range``, ``strain_range`` are ignored.
    n_vort_bins, n_strain_bins, vort_range, strain_range, clip_pct
        Used only when ``vort_edges``/``strain_edges`` are not provided;
        same semantics as in :func:`compute_jpdf`.
    min_count : int
        Bins with fewer than this many samples are masked to NaN.
        Prevents noisy estimates in sparsely sampled phase-space regions.

    Returns
    -------
    ConditionalMeanResult

    Examples
    --------
    >>> # Share bin edges with a previously computed JPDF
    >>> jpdf  = compute_jpdf(ro, snorm)
    >>> cond_w = conditional_mean(w_map, ro, snorm, field_name='w',
    ...                           vort_edges=jpdf.vort_edges,
    ...                           strain_edges=jpdf.strain_edges)
    >>>
    >>> fig, ax = plt.subplots()
    >>> pm = ax.pcolormesh(cond_w.vort_edges, cond_w.strain_edges,
    ...                    cond_w.mean.T, cmap='RdBu_r')
    """
    (F_v, vort_v, strain_v), n_valid = _flatten_valid(F, vorticity, strain)

    if n_valid == 0:
        raise ValueError('No valid (non-NaN/Inf) pixels found.')

    # Determine bin edges
    if vort_edges is None or strain_edges is None:
        if vort_range is None:
            vort_range = _auto_range(vort_v, clip_pct, symmetric=True)
        if strain_range is None:
            strain_range = _auto_range(strain_v, clip_pct, symmetric=False)
        vort_edges   = np.linspace(vort_range[0],   vort_range[1],   n_vort_bins + 1)
        strain_edges = np.linspace(strain_range[0], strain_range[1], n_strain_bins + 1)

    # Compute binned sum and count separately so they can be accumulated later
    F_sum, _, _, _ = binned_statistic_2d(
        vort_v, strain_v, F_v,
        statistic='sum',
        bins=[vort_edges, strain_edges],
    )
    count, _, _, _ = binned_statistic_2d(
        vort_v, strain_v, F_v,
        statistic='count',
        bins=[vort_edges, strain_edges],
    )

    safe_count = np.where(count >= min_count, count, 1)   # avoid divide-by-zero
    mean = np.where(count >= min_count, F_sum / safe_count, np.nan)

    return ConditionalMeanResult(
        mean=mean.astype(np.float64),
        count=count.astype(np.int64),
        vort_edges=vort_edges,
        strain_edges=strain_edges,
        vort_centers=_bin_centers(vort_edges),
        strain_centers=_bin_centers(strain_edges),
        field_name=field_name,
        n_samples=n_valid,
        n_snapshots=1,
        min_count=min_count,
    )


# ---------------------------------------------------------------------------
# Multi-snapshot accumulator
# ---------------------------------------------------------------------------

class JPDFAccumulator:
    """
    Accumulate JPDF and conditional means over multiple snapshots.

    Balwada et al. (2021) average over ~30 snapshots separated by 10 days.
    This class handles the accumulation correctly:

    - **JPDF**: the time-mean PDF is the arithmetic mean of per-snapshot PDFs,
      giving each snapshot equal statistical weight regardless of the number
      of valid pixels in that snapshot.
    - **Conditional means**: numerator (Σ F) and denominator (Σ count) are
      accumulated separately and divided at finalize, so the result is the
      pooled sample mean weighted by snapshot sample size — equivalent to
      treating all snapshots as a single large dataset.

    Parameters
    ----------
    n_vort_bins, n_strain_bins : int
        Number of bins (fixed for the lifetime of the accumulator).
    vort_range, strain_range : (float, float)
        Bin edge ranges.  Must be set explicitly since they cannot be
        inferred from individual snapshots during accumulation.

    Examples
    --------
    >>> acc = JPDFAccumulator(n_vort_bins=100, n_strain_bins=80,
    ...                       vort_range=(-6, 6), strain_range=(0, 6))
    >>> for ro_t, sn_t, w_t, gradb2_t in snapshots:
    ...     acc.add(ro_t, sn_t, fields={'w': w_t, 'gradb2': gradb2_t})
    >>> jpdf, cond_means = acc.finalize()
    >>> print(jpdf.region_fractions())
    >>> cond_w = cond_means['w']
    """

    def __init__(
        self,
        n_vort_bins:  int,
        n_strain_bins: int,
        vort_range:   Tuple[float, float],
        strain_range: Tuple[float, float],
        min_count:    int = 5,
    ) -> None:
        self.vort_edges   = np.linspace(vort_range[0],   vort_range[1],   n_vort_bins + 1)
        self.strain_edges = np.linspace(strain_range[0], strain_range[1], n_strain_bins + 1)
        self.min_count    = min_count

        shape = (n_vort_bins, n_strain_bins)

        # JPDF accumulation: sum of per-snapshot density arrays
        self._pdf_sum:   np.ndarray = np.zeros(shape, dtype=np.float64)

        # Conditional mean accumulation: sum(F) and count per bin (pooled)
        self._field_sums:   Dict[str, np.ndarray] = {}
        self._field_counts: Dict[str, np.ndarray] = {}

        # Total valid-pixel counts across all snapshots (for metadata)
        self._n_samples_total: int = 0
        self._n_snapshots:     int = 0

    def add(
        self,
        vorticity: Union[np.ndarray, Sequence],
        strain:    Union[np.ndarray, Sequence],
        fields:    Optional[Dict[str, Union[np.ndarray, Sequence]]] = None,
    ) -> None:
        """
        Add one snapshot to the accumulator.

        Parameters
        ----------
        vorticity, strain : array-like
            Vorticity and strain for this snapshot (same shape; NaN excluded).
        fields : dict {name: array-like}, optional
            Scalar fields for conditional means.  Each array must have the
            same shape as ``vorticity``/``strain``.  NaN in any field or in
            vorticity/strain are excluded independently per field.
        """
        # ── JPDF ──────────────────────────────────────────────────────────────
        (vort_v, strain_v), n_valid = _flatten_valid(vorticity, strain)
        if n_valid == 0:
            return

        pdf, _, _ = np.histogram2d(
            vort_v, strain_v,
            bins=[self.vort_edges, self.strain_edges],
            density=True,
        )
        self._pdf_sum += pdf
        self._n_samples_total += n_valid
        self._n_snapshots     += 1

        # ── Conditional means ─────────────────────────────────────────────────
        if fields is None:
            return

        for fname, F_arr in fields.items():
            (F_v, vort_fv, strain_fv), n_f = _flatten_valid(F_arr, vorticity, strain)
            if n_f == 0:
                continue

            F_sum, _, _, _ = binned_statistic_2d(
                vort_fv, strain_fv, F_v,
                statistic='sum',
                bins=[self.vort_edges, self.strain_edges],
            )
            cnt, _, _, _ = binned_statistic_2d(
                vort_fv, strain_fv, F_v,
                statistic='count',
                bins=[self.vort_edges, self.strain_edges],
            )

            if fname not in self._field_sums:
                self._field_sums[fname]   = np.zeros_like(F_sum)
                self._field_counts[fname] = np.zeros_like(cnt)

            self._field_sums[fname]   += F_sum
            self._field_counts[fname] += cnt

    def finalize(self) -> Tuple[JPDFResult, Dict[str, ConditionalMeanResult]]:
        """
        Compute time-mean JPDF and conditional means from accumulated data.

        Returns
        -------
        jpdf : JPDFResult
            Time-mean JPDF (arithmetic mean of per-snapshot PDFs).
        cond_means : dict {field_name: ConditionalMeanResult}
            Pooled conditional mean for each accumulated field.
            Empty dict if no fields were passed to :meth:`add`.
        """
        if self._n_snapshots == 0:
            raise RuntimeError('No snapshots have been added.')

        # Time-mean PDF
        mean_pdf = self._pdf_sum / self._n_snapshots

        # Recover counts from pdf for metadata
        # (approximate: use mean pdf * area * n_samples / n_snapshots)
        dz = np.diff(self.vort_edges)
        ds = np.diff(self.strain_edges)
        dA = np.outer(dz, ds)
        n_per_snap = self._n_samples_total / self._n_snapshots
        approx_count = np.round(mean_pdf * dA * n_per_snap).astype(np.int64)

        jpdf = JPDFResult(
            pdf=mean_pdf,
            count=approx_count,
            vort_edges=self.vort_edges.copy(),
            strain_edges=self.strain_edges.copy(),
            vort_centers=_bin_centers(self.vort_edges),
            strain_centers=_bin_centers(self.strain_edges),
            n_samples=self._n_samples_total,
            n_snapshots=self._n_snapshots,
        )

        # Pooled conditional means
        cond_means: Dict[str, ConditionalMeanResult] = {}
        for fname in self._field_sums:
            total_sum   = self._field_sums[fname]
            total_count = self._field_counts[fname]
            mean = np.where(total_count >= self.min_count,
                            total_sum / total_count, np.nan)
            cond_means[fname] = ConditionalMeanResult(
                mean=mean.astype(np.float64),
                count=total_count.astype(np.int64),
                vort_edges=self.vort_edges.copy(),
                strain_edges=self.strain_edges.copy(),
                vort_centers=_bin_centers(self.vort_edges),
                strain_centers=_bin_centers(self.strain_edges),
                field_name=fname,
                n_samples=self._n_samples_total,
                n_snapshots=self._n_snapshots,
                min_count=self.min_count,
            )

        return jpdf, cond_means
