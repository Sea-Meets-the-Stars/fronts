"""
Co-locate labeled fronts with mapped property fields and compute per-front statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage

PropertySource = Union[np.ndarray, Tuple[str, str], str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SUPPORTED_STATS = frozenset(['mean', 'std', 'median', 'min', 'max', 'count'])

_NDIMAGE_FUNC = {
    'mean':   ndimage.mean,
    'std':    ndimage.standard_deviation,
    'median': ndimage.median,
    'min':    ndimage.minimum,
    'max':    ndimage.maximum,
}

_NANFUNC = {
    'mean':   np.nanmean,
    'std':    np.nanstd,
    'median': np.nanmedian,
    'min':    np.nanmin,
    'max':    np.nanmax,
}


def _load_property_file(source: PropertySource):
    fpath, varname = source
    import xarray as xr
    with xr.open_dataset(fpath) as ds:
        return ds[varname].values.squeeze()


def _dilate_labeled_array(
    labeled_fronts: np.ndarray,
    valid_labels: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """
    Expand each valid front outward by *dilation_radius* pixels.

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Integer label array where 0 is background.
    valid_labels : np.ndarray
        Labels to keep and dilate.
    dilation_radius : int
        Dilation radius in pixels.

    Returns
    -------
    np.ndarray
        Labeled array with valid fronts expanded into nearby background pixels.
    """

    # Mark pixels belonging to fronts we want to keep
    # This allows filtering, e.g. by min_npix
    max_lbl = int(labeled_fronts.max())
    lookup  = np.zeros(max_lbl + 1, dtype=bool)
    lookup[valid_labels] = True
    valid_mask = lookup[labeled_fronts]        

    # For each background pixel, find distance to nearest valid front pixel
    background = ~valid_mask
    dist, nearest_idx = ndimage.distance_transform_edt(
        background, return_indices=True
    )

    # Start from the original array, but remove invalid fronts/background
    dilated = labeled_fronts.copy()
    dilated[background] = 0                      

    # Background pixels within radius inherit nearest valid front label
    expand_mask = background & (dist <= dilation_radius)
    dilated[expand_mask] = labeled_fronts[
        nearest_idx[0][expand_mask],
        nearest_idx[1][expand_mask],
    ]

    return dilated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def colocate_fronts_with_properties(
    labeled_fronts: np.ndarray,
    properties: Dict[str, np.ndarray],
    stats=None,
    percentiles=None,
    min_npix: int = 1,
    nan_policy: str = 'propagate',
    dilation_radius: int = 0,
) -> pd.DataFrame:
    """Co-locate fronts with mapped property fields.
    Parameters
    ----------
    labeled_fronts : np.ndarray
        Integer label array where 0 is background.
    properties : dict
        Property arrays with the same shape as labeled_fronts.
    stats : list, optional
        Statistics to compute for each property.
        Any combination of ``'mean'``, ``'std'``, ``'median'``, ``'min'``, ``'max'``,``'count'``.
    percentiles : sequence, optional
        Percentiles to compute for each property, e.g. ``[10, 25, 75, 90]``
    min_npix : int, optional
        Minimum front size to keep.
    nan_policy : {'propagate', 'omit'}
        How to handle NaNs.
    dilation_radius : int, optional
        Number of pixels to dilate each retained front before computing stats.
        Default is 0 (no dilation).

    Returns
    -------
    pd.DataFrame
        One row per front.  Columns:
        - flabel
        - npix
        - {prop}_{stat}
        - {prop}_p{pct}

    Usage
    --------
    Basic usage with two property arrays: 
        -include percentiles 
        -ignore NaN (e.g. land pixels)
        -dilate fronts by dilation_radius pixels

    >>> labeled = np.load('labeled_fronts_global_20121109T12_00_00.npy')
    >>> df = colocate_fronts_with_properties(
    ...     labeled,
    ...     properties={
    ...         'relative_vorticity': np.load('relative_vorticity.npy'),
    ...         'rossby_number':        np.load('rossby_number.npy'),
    ...     },
    ...     dilation_radius=5,
    ...     stats=['mean', 'std'],
    ...     percentiles=[10, 90],
    ...     min_npix=5,
    ...     nan_policy='omit', 
    ... )

    """

    # Input validation
    # ------------------------------------------------------------------
    if stats is None:
        stats = ['mean', 'std', 'median']
    
    if nan_policy not in ("propagate", "omit"):
        raise ValueError("nan_policy must be 'propagate' or 'omit'")

    for prop_name, prop_arr in properties.items():
        if prop_arr.shape != labeled_fronts.shape:
            raise ValueError(f"{prop_name} shape does not match labeled_fronts")


    # identify front labels and original pixel counts
    # ------------------------------------------------------------------
    all_labels, all_counts = np.unique(labeled_fronts, return_counts=True)

    # define all fronts
    bg_mask  = all_labels > 0
    flabels  = all_labels[bg_mask]
    npix     = all_counts[bg_mask]

    if len(flabels) == 0:
        return pd.DataFrame()

    # define fronts large enough to keep
    keep     = npix >= min_npix
    flabels  = flabels[keep]
    npix     = npix[keep]

    if len(flabels) == 0:
        return pd.DataFrame()


    # Build the label array used for statistics
    # ------------------------------------------------------------------
    if dilation_radius > 0:
        stat_labels = _dilate_labeled_array(labeled_fronts, flabels, dilation_radius)
    else:
        # Restrict to valid labels only (zero out filtered-out fronts)
        if len(flabels) < len(all_labels[bg_mask]):
            max_lbl = int(labeled_fronts.max())
            lut = np.zeros(max_lbl + 1, dtype=labeled_fronts.dtype)
            lut[flabels] = flabels
            stat_labels = lut[labeled_fronts]
        else:
            stat_labels = labeled_fronts


    # Compute statistics
    # ------------------------------------------------------------------
    result = {'flabel': flabels, 'npix':   npix, }

    for prop_name, prop_arr in properties.items():
        prop_float = prop_arr.astype(np.float64)

        for stat in stats:
            col = f'{prop_name}_{stat}'
            if stat == 'count':
                result[col] = npix.copy()
                continue

            if nan_policy == 'omit':
                nan_fn = _NANFUNC[stat]
                values = ndimage.labeled_comprehension(
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                    func=nan_fn,
                    out_dtype=np.float64,
                    default=np.nan,
                )
            else:  # 'propagate'
                values = _NDIMAGE_FUNC[stat](
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                )

            result[col] = np.asarray(values, dtype=np.float64)

        # Percentiles — always use nan-aware path
        if percentiles is not None:
            for pct in percentiles:
                pct_label = f'{int(pct)}' if pct == int(pct) else f'{pct}'
                col = f'{prop_name}_p{pct_label}'

                if nan_policy == 'omit':
                    pct_fn = lambda x, q=pct: np.nanpercentile(x, q)
                else:
                    pct_fn = lambda x, q=pct: np.percentile(x, q)

                values = ndimage.labeled_comprehension(
                    prop_float,
                    labels=stat_labels,
                    index=flabels,
                    func=pct_fn,
                    out_dtype=np.float64,
                    default=np.nan,
                )
                result[col] = np.asarray(values, dtype=np.float64)

    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_and_colocate(
    labeled_fronts_file: str,
    property_files: Dict[str, PropertySource],
    stats: Optional[List[str]] = None,
    percentiles: Optional[Sequence[Union[int, float]]] = None,
    min_npix: int = 1,
    nan_policy: str = 'propagate',
    dilation_radius: int = 0,
) -> pd.DataFrame:
    """Load front and property arrays from files, then co-locate.

    Parameters
    ----------
    labeled_fronts_file : str
        Path to .npy file containing labeled fronts.
    property_files : dict
        Mapping from property name to property source.


    """


    # ---- load -----------------------------------------------------------
    labeled_fronts = np.load(labeled_fronts_file)
    properties = {
        name: _load_property_file(src)
        for name, src in property_files.items()
    }

    return colocate_fronts_with_properties(
        labeled_fronts,
        properties,
        stats=stats,
        percentiles=percentiles,
        min_npix=min_npix,
        nan_policy=nan_policy,
        dilation_radius=dilation_radius,
    )
