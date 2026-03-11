"""
fronts.properties.pca
=====================
Principal Component Analysis on oceanographic front properties.

Two entry points are provided for the two natural use-cases:

run_pca(properties)
    Gridded-map PCA.  Inputs are N co-registered 2-D arrays (H × W), one per
    physical property.  Outputs are PC *score maps* (H × W each) that show
    where each mode of co-variability is spatially strongest, plus the usual
    loadings and variance statistics.

run_pca_fronts(df, property_cols)
    Front-colocated PCA.  Input is a DataFrame where each row is a detected
    front and columns contain co-located property means (plus centroid_lat /
    centroid_lon).  Outputs are per-front PC scores (one value per front per
    PC), loadings, and variance statistics.  The companion ``to_map`` method
    on the result bins the per-front scores onto a lat/lon grid using
    ``binned_statistic_2d`` so they can be visualised with pcolormesh.

Shared outputs
--------------
loadings : DataFrame, shape (n_properties, n_components), cols = 'PC1','PC2',…
    Large absolute loading → that property strongly defines the PC.

explained_variance_ratio : ndarray, shape (n_components,)
    Fraction of total variance explained by each PC.

cumulative_variance_ratio : ndarray, shape (n_components,)
    Running sum; useful for choosing how many PCs to retain.

summary() / top_loadings(n)
    Convenience methods for quick interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    """
    Container for PCA results on 2-D property maps.

    See module docstring for a full description of each field.
    """

    # ── Core spatial output ──────────────────────────────────────────────────
    pc_maps: np.ndarray
    """Shape (n_components, H, W).  NaN at masked pixels."""

    # ── Variance statistics ──────────────────────────────────────────────────
    explained_variance_ratio: np.ndarray
    """Fraction of variance explained by each PC."""

    explained_variance: np.ndarray
    """Absolute variance (eigenvalue) of each PC."""

    cumulative_variance_ratio: np.ndarray
    """Cumulative sum of explained_variance_ratio."""

    # ── Loadings ─────────────────────────────────────────────────────────────
    loadings: pd.DataFrame
    """
    Shape (n_properties, n_components), columns = ['PC1','PC2',…].
    Describes which properties define each PC.
    """

    # ── Metadata ─────────────────────────────────────────────────────────────
    property_names: List[str]
    grid_shape: tuple
    n_valid_pixels: int
    n_fit_pixels: int
    """Number of pixels actually used to fit PCA (≤ n_valid_pixels if subsampled)."""
    standardized: bool

    # ── Fitted objects (kept for transform / reconstruct) ────────────────────
    _pca: PCA = field(repr=False)
    _scaler: Optional[StandardScaler] = field(repr=False, default=None)
    _valid_mask: np.ndarray = field(repr=False, default=None)
    """Boolean (H, W) mask — True where all properties were finite."""

    # ── Convenience ──────────────────────────────────────────────────────────

    def reconstruct(
        self,
        pc_indices: Sequence[int] | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct property maps from a subset of PCs.

        Parameters
        ----------
        pc_indices : sequence of int, optional
            0-based PC indices to include (e.g. [0, 1] for PC1+PC2).
            Default: all components.

        Returns
        -------
        dict
            {property_name: 2-D ndarray (H, W)} in the original (un-standardised)
            property units.  NaN where the valid mask is False.
        """
        H, W = self.grid_shape
        n_comp = self._pca.n_components_

        if pc_indices is None:
            pc_indices = list(range(n_comp))

        # Build a score matrix that zeroes out unwanted PCs
        scores_full = np.zeros((self.n_valid_pixels, n_comp), dtype=np.float32)
        scores_full[:, pc_indices] = self.pc_maps[pc_indices][
            :, self._valid_mask
        ].T                                                     # (n_valid, n_sel)

        # Inverse PCA transform
        X_std = self._pca.inverse_transform(scores_full)        # (n_valid, n_props)

        # Inverse standardisation
        if self._scaler is not None:
            X_raw = self._scaler.inverse_transform(X_std)
        else:
            X_raw = X_std

        # Reshape back to 2-D maps
        out = {}
        for i, name in enumerate(self.property_names):
            arr = np.full((H, W), np.nan, dtype=np.float32)
            arr[self._valid_mask] = X_raw[:, i]
            out[name] = arr

        return out

    def summary(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame summarising variance explained per PC.

        Columns: PC, explained_variance, explained_variance_ratio,
                 cumulative_variance_ratio.
        """
        n = len(self.explained_variance_ratio)
        return pd.DataFrame({
            'PC':                        [f'PC{i+1}' for i in range(n)],
            'explained_variance':        self.explained_variance,
            'explained_variance_ratio':  self.explained_variance_ratio,
            'cumulative_variance_ratio': self.cumulative_variance_ratio,
        })

    def top_loadings(self, n: int = 3) -> pd.DataFrame:
        """
        For each PC, list the ``n`` properties with the largest |loading|.

        Useful for quick interpretation: "PC1 is dominated by X and Y."
        """
        rows = []
        for pc_col in self.loadings.columns:
            col = self.loadings[pc_col].abs().sort_values(ascending=False)
            for rank, (prop, val) in enumerate(col.head(n).items(), start=1):
                signed = self.loadings.loc[prop, pc_col]
                rows.append({'PC': pc_col, 'rank': rank,
                              'property': prop, 'loading': signed,
                              '|loading|': val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_pca(
    properties: Dict[str, np.ndarray],
    n_components: Optional[int] = None,
    standardize: bool = True,
    max_fit_pixels: Optional[int] = 1_000_000,
    random_state: int = 42,
    chunk_size: int = 500_000,
) -> PCAResult:
    """
    Run PCA on a set of co-registered 2-D oceanographic property maps.

    Parameters
    ----------
    properties : dict
        Mapping of property name → 2-D ndarray (H, W), all the same shape.
        NaN pixels (land, ice, cloud) are automatically excluded.
    n_components : int, optional
        Number of PCs to retain.  Defaults to min(n_properties, n_valid_pixels).
    standardize : bool
        If True (strongly recommended), z-score each property before PCA so that
        all properties contribute equally regardless of physical units.
    max_fit_pixels : int, optional
        Maximum number of pixels used to *fit* the PCA.  If the valid pixel
        count exceeds this, a random subsample is drawn for fitting and the
        full valid set is then *transformed* in chunks.  Set None to use all
        pixels (may require large memory for global LLC4320 grids).
    random_state : int
        Random seed for subsampling reproducibility.
    chunk_size : int
        Number of pixels processed at once during the transform step.
        Increase for speed, decrease to reduce peak memory.

    Returns
    -------
    PCAResult
        See class docstring for field descriptions.

    Notes
    -----
    Memory guide (float32, global LLC4320 12960×17280 ≈ 224 M pixels):
      - 10 properties × 500 000 fit pixels × 4 bytes ≈ 20 MB   (fine)
      - 10 properties × 224 M  pixels       × 4 bytes ≈ 9  GB  (use max_fit_pixels)
    The transform is done in chunks so peak memory is manageable even for full
    global grids.

    Examples
    --------
    >>> from fronts.properties.pca import run_pca
    >>> result = run_pca(property_arrays, n_components=4)
    >>> print(result.summary())
    >>> pc1_map = result.pc_maps[0]          # shape (H, W)
    >>> print(result.loadings)               # which properties drive each PC
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not properties:
        raise ValueError('properties dict is empty.')

    prop_names = list(properties.keys())
    arrays = [np.asarray(properties[k], dtype=np.float32) for k in prop_names]

    shapes = [a.shape for a in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(
            f'All property arrays must have the same shape. Got: {shapes}')
    if any(a.ndim != 2 for a in arrays):
        raise ValueError('All property arrays must be 2-D.')

    H, W = shapes[0]
    n_props = len(prop_names)

    if n_components is not None and n_components > n_props:
        raise ValueError(
            f'n_components ({n_components}) cannot exceed the number of '
            f'properties ({n_props}).')

    # ── Build valid-pixel mask ────────────────────────────────────────────────
    # A pixel is valid only if ALL properties are finite there.
    valid_mask = np.ones((H, W), dtype=bool)
    for arr in arrays:
        valid_mask &= np.isfinite(arr)

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError('No valid (non-NaN) pixels found across all properties.')

    # Stack into (n_valid, n_props) data matrix
    X_full = np.column_stack([arr[valid_mask] for arr in arrays])  # (n_valid, n_props)

    # ── Standardise ───────────────────────────────────────────────────────────
    scaler = None
    if standardize:
        scaler = StandardScaler()
        # Fit scaler on subsample if large (consistent with PCA subsample below)
        if max_fit_pixels and n_valid > max_fit_pixels:
            rng = np.random.default_rng(random_state)
            fit_idx = rng.choice(n_valid, max_fit_pixels, replace=False)
            scaler.fit(X_full[fit_idx])
        else:
            scaler.fit(X_full)
        X_full = scaler.transform(X_full)   # in-place replacement for memory

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    n_comp = n_components if n_components is not None else min(n_props, n_valid)

    rng = np.random.default_rng(random_state)
    if max_fit_pixels and n_valid > max_fit_pixels:
        fit_idx = rng.choice(n_valid, max_fit_pixels, replace=False)
        X_fit = X_full[fit_idx]
        n_fit = max_fit_pixels
    else:
        X_fit = X_full
        n_fit = n_valid

    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(X_fit)

    # ── Transform full valid set in chunks ────────────────────────────────────
    scores = np.empty((n_valid, n_comp), dtype=np.float32)
    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        scores[start:end] = pca.transform(X_full[start:end])

    # ── Reshape scores back to 2-D maps ──────────────────────────────────────
    pc_maps = np.full((n_comp, H, W), np.nan, dtype=np.float32)
    for i in range(n_comp):
        pc_maps[i][valid_mask] = scores[:, i]

    # ── Loadings DataFrame ────────────────────────────────────────────────────
    # pca.components_ shape: (n_comp, n_props)
    # Transpose so rows = properties, cols = PCs — easier to read
    pc_labels = [f'PC{i+1}' for i in range(n_comp)]
    loadings = pd.DataFrame(
        pca.components_.T,          # (n_props, n_comp)
        index=prop_names,
        columns=pc_labels,
    )

    return PCAResult(
        pc_maps=pc_maps,
        explained_variance_ratio=pca.explained_variance_ratio_,
        explained_variance=pca.explained_variance_,
        cumulative_variance_ratio=np.cumsum(pca.explained_variance_ratio_),
        loadings=loadings,
        property_names=prop_names,
        grid_shape=(H, W),
        n_valid_pixels=n_valid,
        n_fit_pixels=n_fit,
        standardized=standardize,
        _pca=pca,
        _scaler=scaler,
        _valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# Front-colocated PCA
# ---------------------------------------------------------------------------

@dataclass
class FrontPCAResult:
    """
    Container for PCA results on front-colocated property means.

    Each row in ``scores`` corresponds to one front; the PC columns ('PC1',
    'PC2', …) give that front's position in principal component space.

    Use ``to_map(lat, lon)`` to bin scores onto a lat/lon grid for
    spatial visualisation.
    """

    # ── Per-front scores ─────────────────────────────────────────────────────
    scores: pd.DataFrame
    """
    Shape (n_fronts, n_components).  Index matches the input DataFrame index.
    Columns are 'PC1', 'PC2', … .
    """

    # ── Variance statistics ──────────────────────────────────────────────────
    explained_variance_ratio: np.ndarray
    explained_variance: np.ndarray
    cumulative_variance_ratio: np.ndarray

    # ── Loadings ─────────────────────────────────────────────────────────────
    loadings: pd.DataFrame
    """Shape (n_properties, n_components), columns = ['PC1','PC2',…]."""

    # ── Metadata ─────────────────────────────────────────────────────────────
    property_names: List[str]
    n_fronts: int
    """Total fronts with all-finite properties (used for transform)."""
    n_fit_fronts: int
    """Fronts used to *fit* PCA (≤ n_fronts if subsampled)."""
    standardized: bool

    # ── Fitted objects ───────────────────────────────────────────────────────
    _pca: PCA = field(repr=False)
    _scaler: Optional[StandardScaler] = field(repr=False, default=None)

    # ── Spatial binning ───────────────────────────────────────────────────────

    def to_map(
        self,
        lat: Union[pd.Series, np.ndarray],
        lon: Union[pd.Series, np.ndarray],
        lat_bins: Union[int, np.ndarray] = 90,
        lon_bins: Union[int, np.ndarray] = 180,
        statistic: str = 'mean',
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Bin per-front PC scores onto a lat/lon grid.

        Parameters
        ----------
        lat, lon : array-like, length n_fronts
            Centroid latitudes and longitudes for each front in ``scores``.
            Must align with ``scores`` (same index / order).
        lat_bins, lon_bins : int or array
            Number of equal-width bins or explicit bin-edge arrays.
            Default: 90 lat bins (2°) X 180 lon bins (2°).
        statistic : str
            Aggregation applied within each bin ('mean', 'median', 'std', …).
            Passed directly to ``scipy.stats.binned_statistic_2d``.

        Returns
        -------
        dict
            Keys are 'PC1', 'PC2', … .  Each value is a tuple
            ``(grid, lat_edges, lon_edges)`` where ``grid`` is a 2-D masked
            array of shape ``(n_lat_bins, n_lon_bins)`` — NaN where no fronts
            fall in a bin — and the edge arrays allow exact ``pcolormesh``
            calls::

                grid, lat_e, lon_e = result.to_map(lat, lon)['PC1']
                lon_e2d, lat_e2d = np.meshgrid(lon_e, lat_e)
                ax.pcolormesh(lon_e2d, lat_e2d, np.ma.masked_invalid(grid))

        Notes
        -----
        Only fronts present in ``scores`` (i.e., with all-finite properties)
        are included.  ``lat`` / ``lon`` must cover the same set; rows with
        NaN lat/lon are silently dropped.
        """
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)

        # Drop rows where lat or lon is invalid
        valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)
        lat_v = lat_arr[valid]
        lon_v = lon_arr[valid]
        scores_v = self.scores.values[valid]   # (n_valid_fronts, n_comp)

        out = {}
        for i, pc_col in enumerate(self.scores.columns):
            stat, lat_edges, lon_edges, _ = binned_statistic_2d(
                lat_v, lon_v, scores_v[:, i],
                statistic=statistic,
                bins=[lat_bins, lon_bins],
                range=[[-90, 90], [-180, 180]],
            )
            # Replace empty bins (returned as NaN by scipy) with np.nan
            grid = np.where(np.isnan(stat), np.nan, stat)
            out[pc_col] = (grid, lat_edges, lon_edges)

        return out

    # ── Convenience ──────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame of variance explained per PC."""
        n = len(self.explained_variance_ratio)
        return pd.DataFrame({
            'PC':                        [f'PC{i+1}' for i in range(n)],
            'explained_variance':        self.explained_variance,
            'explained_variance_ratio':  self.explained_variance_ratio,
            'cumulative_variance_ratio': self.cumulative_variance_ratio,
        })

    def top_loadings(self, n: int = 3) -> pd.DataFrame:
        """For each PC, list the ``n`` properties with the largest |loading|."""
        rows = []
        for pc_col in self.loadings.columns:
            col = self.loadings[pc_col].abs().sort_values(ascending=False)
            for rank, (prop, val) in enumerate(col.head(n).items(), start=1):
                signed = self.loadings.loc[prop, pc_col]
                rows.append({'PC': pc_col, 'rank': rank,
                              'property': prop, 'loading': signed,
                              '|loading|': val})
        return pd.DataFrame(rows)


def run_pca_fronts(
    df: pd.DataFrame,
    property_cols: List[str],
    n_components: Optional[int] = None,
    standardize: bool = True,
    max_fit_fronts: Optional[int] = None,
    random_state: int = 42,
) -> FrontPCAResult:
    """
    Run PCA on front-colocated property means.

    Parameters
    ----------
    df : DataFrame
        One row per front.  Must contain all columns listed in
        ``property_cols``.  
    property_cols : list of str
        Names of the property columns to include in PCA (e.g.
        ``['gradb2_mean', 'strain_mag_mean', 'okubo_weiss_mean', ...]``).
    n_components : int, optional
        Number of PCs to retain.  Defaults to ``len(property_cols)``.
    standardize : bool
        Z-score each property before PCA.
    max_fit_fronts : int, optional
        If set and the number of valid fronts exceeds this, a random
        subsample is drawn for *fitting*.  The full valid set is always
        *transformed*. 
    random_state : int
        Random seed for subsampling reproducibility.

    Returns
    -------
    FrontPCAResult
        ``result.scores`` is a DataFrame aligned with the valid rows of ``df``
        (NaN-dropped rows are excluded).  Call ``result.to_map(lat, lon)`` to
        produce spatial bin maps.

    Examples
    --------
    >>> prop_cols = ['gradb2_mean', 'strain_mag_mean', 'okubo_weiss_mean',
    ...              'relative_vorticity_mean', 'coriolis_f_mean']
    >>> result = run_pca_fronts(df_enriched, prop_cols, n_components=3)
    >>> print(result.summary())
    >>> print(result.top_loadings(n=2))
    >>>
    >>> # Spatial map of PC1 scores
    >>> grid, lat_e, lon_e = result.to_map(
    ...     df_enriched.loc[result.scores.index, 'centroid_lat'],
    ...     df_enriched.loc[result.scores.index, 'centroid_lon'],
    ... )['PC1']
    >>> lon_e2d, lat_e2d = np.meshgrid(lon_e, lat_e)
    >>> ax.pcolormesh(lon_e2d, lat_e2d, np.ma.masked_invalid(grid))
    """
    # ── Choose variables, remove incomplete rows ─────────────────────────────────
    missing = [c for c in property_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Columns not found in df: {missing}')

    df_valid = df[property_cols].dropna()
    n_valid = len(df_valid)

    if n_valid == 0:
        raise ValueError(
            'No rows with all-finite values found for the requested property columns.')

    n_props = len(property_cols)
    n_comp = n_components if n_components is not None else n_props
    if n_comp > n_props:
        raise ValueError(
            f'n_components ({n_comp}) cannot exceed n_properties ({n_props}).')

    X_full = df_valid.values.astype(np.float32)   # (n_valid, n_props)

    # ── Standardize ───────────────────────────────────────────────────────────
    scaler = None
    if standardize:
        scaler = StandardScaler()
        if max_fit_fronts and n_valid > max_fit_fronts:
            rng = np.random.default_rng(random_state)
            fit_idx = rng.choice(n_valid, max_fit_fronts, replace=False)
            scaler.fit(X_full[fit_idx])
        else:
            scaler.fit(X_full)
        X_full = scaler.transform(X_full)

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    if max_fit_fronts and n_valid > max_fit_fronts:
        rng = np.random.default_rng(random_state)
        fit_idx = rng.choice(n_valid, max_fit_fronts, replace=False)
        X_fit = X_full[fit_idx]
        n_fit = max_fit_fronts
    else:
        X_fit = X_full
        n_fit = n_valid

    pca_model = PCA(n_components=n_comp, random_state=random_state)
    pca_model.fit(X_fit)

    # ── Transform all valid fronts ────────────────────────────────────────────
    scores_arr = pca_model.transform(X_full)   # (n_valid, n_comp)

    pc_labels = [f'PC{i+1}' for i in range(n_comp)]
    scores_df = pd.DataFrame(
        scores_arr,
        index=df_valid.index,
        columns=pc_labels,
        dtype=np.float32,
    )

    # ── Loadings ──────────────────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_model.components_.T,   # (n_props, n_comp)
        index=property_cols,
        columns=pc_labels,
    )

    return FrontPCAResult(
        scores=scores_df,
        explained_variance_ratio=pca_model.explained_variance_ratio_,
        explained_variance=pca_model.explained_variance_,
        cumulative_variance_ratio=np.cumsum(pca_model.explained_variance_ratio_),
        loadings=loadings,
        property_names=property_cols,
        n_fronts=n_valid,
        n_fit_fronts=n_fit,
        standardized=standardize,
        _pca=pca_model,
        _scaler=scaler,
    )
