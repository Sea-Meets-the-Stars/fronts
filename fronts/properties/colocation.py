"""
Front–Property Co-location Module

Co-locates labeled (grouped) fronts with mapped property fields and computes
per-front summary statistics.  Designed to work directly with the integer
labeled arrays produced by :mod:`fronts.properties.group_labels` and with
2-D numpy arrays that share the same spatial grid.

Typical usage
-------------
>>> labeled = np.load('labeled_fronts_global_20121109T12_00_00.npy')
>>> import xarray as xr
>>> props = {
...     'relative_vorticity': xr.open_dataset(
...         'LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc'
...     )['relative_vorticity'].values.squeeze(),
...     'rossby_number': xr.open_dataset(
...         'LLC4320_2012-11-09T12_00_00_rossby_number_v1.nc'
...     )['rossby_number'].values.squeeze(),
... }
>>> df = colocate_fronts_with_properties(labeled, props,
...         stats=['mean', 'std', 'median'],
...         percentiles=[10, 90],
...         min_npix=5)
>>> print(df.head())
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage

# Type alias: a property source is either
#   - a pre-loaded ndarray, or
#   - a (filepath, variable_name) tuple pointing to a NetCDF file, or
#   - a filepath string pointing to a .npy file
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


def _load_property_file(source: PropertySource) -> np.ndarray:
    """Load a single property array from a file or return it as-is.

    Parameters
    ----------
    source : ndarray, str, or (str, str) tuple
        - **ndarray** — returned unchanged (already loaded by the caller).
        - **str ending in** ``'.npy'`` — loaded with :func:`numpy.load`.
        - **``(filepath, variable_name)`` tuple** — opens a NetCDF / HDF5
          file with :func:`xarray.open_dataset` and extracts *variable_name*.
          A trailing singleton dimension (e.g. time=1) is squeezed away
          automatically so the result is always 2-D.
        - **str ending in** ``'.nc'`` **without a variable name** — raises
          :class:`ValueError` with a helpful message.

    Returns
    -------
    np.ndarray
        2-D float array ready for co-location.

    Examples
    --------
    >>> arr = _load_property_file(
    ...     ('LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc',
    ...      'relative_vorticity')
    ... )
    >>> arr = _load_property_file('front_mask.npy')
    """
    # Already an array — nothing to do
    if isinstance(source, np.ndarray):
        return source

    # (filepath, varname) tuple → NetCDF
    if isinstance(source, (list, tuple)):
        fpath, varname = source
        import xarray as xr
        ds = xr.open_dataset(fpath)
        if varname not in ds:
            raise KeyError(
                f"Variable '{varname}' not found in '{fpath}'. "
                f"Available variables: {list(ds.data_vars)}"
            )
        arr = ds[varname].values.squeeze()
        ds.close()
        return arr

    # Plain string path
    fpath = str(source)
    ext   = os.path.splitext(fpath)[-1].lower()

    if ext == '.npy':
        return np.load(fpath)

    if ext in ('.nc', '.nc4', '.h5', '.hdf5'):
        raise ValueError(
            f"NetCDF/HDF5 file '{fpath}' requires a variable name. "
            f"Pass a tuple instead of a plain string:\n"
            f"  ('{fpath}', '<variable_name>')\n"
            f"You can inspect available variables with:\n"
            f"  import xarray as xr; print(list(xr.open_dataset('{fpath}').data_vars))"
        )

    # Unknown extension — try np.load as a last resort
    return np.load(fpath)


def _dilate_labeled_array(
    labeled_fronts: np.ndarray,
    valid_labels: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """Expand each valid front outward by *dilation_radius* pixels.

    Background pixels (and pixels belonging to filtered-out fronts) that lie
    within *dilation_radius* of a valid front pixel are assigned that front's
    label via nearest-neighbour assignment.  This is mathematically equivalent
    to dilating each binary front mask with a disk of radius *dilation_radius*
    and re-compositing the results, with ties broken by the lower label index
    (scipy EDT convention).

    Parameters
    ----------
    labeled_fronts : np.ndarray
        Original integer label array (0 = background).
    valid_labels : np.ndarray
        1-D array of label integers that should be retained and dilated.
        Labels absent from this array are treated as background.
    dilation_radius : int
        Dilation radius in pixels.  Uses Euclidean distance, so the
        structuring element is a filled disk of this radius.

    Returns
    -------
    dilated : np.ndarray, same shape and dtype as *labeled_fronts*
        Array where background/filtered pixels within *dilation_radius* of a
        valid front have been relabelled to that front's label.

    Notes
    -----
    The implementation uses ``scipy.ndimage.distance_transform_edt`` for an
    O(N) vectorised pass over the full array — no per-front Python loops.
    Memory usage is roughly 3× the array size (distance float64 + two int32
    index arrays).
    """
    # Build a boolean mask of pixels that belong to valid fronts
    max_lbl = int(labeled_fronts.max())
    lookup  = np.zeros(max_lbl + 1, dtype=bool)
    lookup[valid_labels] = True
    valid_mask = lookup[labeled_fronts]          # True = valid front pixel

    # EDT: for each non-front pixel, distance to and index of nearest front px
    background = ~valid_mask
    dist, nearest_idx = ndimage.distance_transform_edt(
        background, return_indices=True
    )

    # Start from the original array; clear filtered-out front pixels
    dilated = labeled_fronts.copy()
    dilated[background] = 0                      # clear all background/invalid

    # Expand: background pixels within radius inherit nearest valid front label
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
    stats: Optional[List[str]] = None,
    percentiles: Optional[Sequence[Union[int, float]]] = None,
    min_npix: int = 1,
    nan_policy: str = 'propagate',
    dilation_radius: int = 0,
) -> pd.DataFrame:
    """Co-locate grouped fronts with mapped property fields.

    For every labeled front in *labeled_fronts*, the function extracts the
    pixel-level values of each entry in *properties* and computes the
    requested summary statistics, returning one row per front.

    Parameters
    ----------
    labeled_fronts : np.ndarray, shape (nrows, ncols), dtype int
        Integer label array where ``0`` is background and each positive
        integer identifies a distinct connected front.  Typically produced by
        :func:`fronts.properties.group_labels.label_fronts`.
    properties : dict[str, np.ndarray]
        Mapping from property name to a 2-D numpy array **with the same shape
        as** *labeled_fronts*.  Values at grid cells that do not belong to any
        front are ignored.  Examples: ``{'vorticity': v_arr, 'Ro': ro_arr}``.
    stats : list of str, optional
        Statistics to compute for every property.  Any combination of
        ``'mean'``, ``'std'``, ``'median'``, ``'min'``, ``'max'``,
        ``'count'``.  Defaults to ``['mean', 'std', 'median']``.
    percentiles : sequence of numbers, optional
        Additional percentile statistics.  E.g. ``[10, 25, 75, 90]`` adds
        columns ``'vorticity_p10'``, ``'vorticity_p25'``, etc.
        Default is ``None`` (no percentile columns).
    min_npix : int, optional
        Fronts with fewer pixels than this threshold are excluded from the
        output.  Default is ``1`` (keep all fronts).
    nan_policy : {'propagate', 'omit'}, optional
        How to handle ``NaN`` values in property arrays at front pixels.

        - ``'propagate'`` *(default)* — uses fast ``scipy.ndimage`` functions;
          any ``NaN`` in a front will propagate to that front's statistic.
        - ``'omit'`` — uses nan-aware numpy functions via
          ``scipy.ndimage.labeled_comprehension``; ``NaN`` pixels are ignored.
          Slightly slower but safer when property arrays contain masked/land
          values.

    dilation_radius : int, optional
        If > 0, each front mask is dilated outward by this many pixels
        (Euclidean disk) **before** computing statistics.  Background pixels
        within the dilated region are included in the per-front statistics,
        allowing characterisation of the environment surrounding each front
        (e.g. the background vorticity or stratification).
        Implemented via ``scipy.ndimage.distance_transform_edt``; equivalent
        to applying ``skimage.morphology.dilation`` with a
        ``skimage.morphology.disk(dilation_radius)`` structuring element to
        each front's binary mask.
        Default is ``0`` (no dilation).

        .. note::
           The ``npix`` column always reflects the **original** (un-dilated)
           pixel count, so ``min_npix`` filtering is always based on the true
           front size.  The dilated pixel count is not separately reported.

    Returns
    -------
    pd.DataFrame
        One row per front that survives the *min_npix* filter.  Columns:

        - ``flabel`` — integer front label
        - ``npix``   — number of pixels in the original (un-dilated) front
        - ``{prop}_{stat}`` — e.g. ``'vorticity_mean'``, ``'Ro_std'``
        - ``{prop}_p{pct}`` — e.g. ``'vorticity_p10'`` (if requested)

        Returns an empty ``DataFrame`` if no fronts are found or all are
        filtered out.

    Raises
    ------
    ValueError
        If a property array's shape does not match *labeled_fronts*, or an
        unsupported statistic name is supplied.

    Examples
    --------
    Basic usage with two property arrays:

    >>> labeled = np.load('labeled_fronts_global_20121109T12_00_00.npy')
    >>> df = colocate_fronts_with_properties(
    ...     labeled,
    ...     properties={
    ...         'vorticity': np.load('vorticity.npy'),
    ...         'Ro':        np.load('Ro.npy'),
    ...     },
    ...     stats=['mean', 'std'],
    ...     min_npix=5,
    ... )

    Include percentiles and ignore NaN (e.g. land pixels):

    >>> df = colocate_fronts_with_properties(
    ...     labeled,
    ...     properties={'SSH': np.load('SSH.npy')},
    ...     stats=['mean', 'median'],
    ...     percentiles=[10, 90],
    ...     nan_policy='omit',
    ... )

    Dilate fronts by 5 pixels to characterise the surrounding environment:

    >>> df = colocate_fronts_with_properties(
    ...     labeled,
    ...     properties={'vorticity': np.load('vorticity.npy')},
    ...     dilation_radius=5,
    ...     nan_policy='omit',
    ... )

    Notes
    -----
    * With ``nan_policy='propagate'`` the implementation uses
      ``scipy.ndimage.mean`` / ``standard_deviation`` / … which operate at
      C speed on the full array without building per-front masks.
    * With ``nan_policy='omit'`` the implementation calls
      ``scipy.ndimage.labeled_comprehension`` with the corresponding
      ``np.nan*`` function; this is still vectorised at the label level but
      involves a Python callback per front.
    * For very large arrays (global LLC4320 grids ~12960×17280) prefer
      ``nan_policy='propagate'`` unless NaNs are expected at front pixels.
    * Dilation with ``dilation_radius > 0`` runs a single ``distance_transform_edt``
      call over the full array (O(N), C-level) regardless of how many fronts
      are present.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if stats is None:
        stats = ['mean', 'std', 'median']

    unknown = set(stats) - _SUPPORTED_STATS
    if unknown:
        raise ValueError(
            f"Unsupported statistic(s): {sorted(unknown)}. "
            f"Supported: {sorted(_SUPPORTED_STATS)}"
        )

    if nan_policy not in ('propagate', 'omit'):
        raise ValueError(
            f"nan_policy must be 'propagate' or 'omit', got '{nan_policy}'"
        )

    if dilation_radius < 0:
        raise ValueError(
            f"dilation_radius must be >= 0, got {dilation_radius}"
        )

    for prop_name, prop_arr in properties.items():
        if prop_arr.shape != labeled_fronts.shape:
            raise ValueError(
                f"Shape mismatch: property '{prop_name}' has shape "
                f"{prop_arr.shape} but labeled_fronts has shape "
                f"{labeled_fronts.shape}."
            )

    # ------------------------------------------------------------------
    # Identify front labels and original pixel counts
    # (min_npix is always applied to the un-dilated array)
    # ------------------------------------------------------------------
    all_labels, all_counts = np.unique(labeled_fronts, return_counts=True)

    bg_mask  = all_labels > 0
    flabels  = all_labels[bg_mask]
    npix     = all_counts[bg_mask]

    if len(flabels) == 0:
        return pd.DataFrame()

    keep     = npix >= min_npix
    flabels  = flabels[keep]
    npix     = npix[keep]

    if len(flabels) == 0:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Build the label array used for statistics
    # (dilated version if requested, original otherwise)
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

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    result: Dict[str, np.ndarray] = {
        'flabel': flabels,
        'npix':   npix,          # always original (un-dilated) pixel count
    }

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

    A thin wrapper around :func:`colocate_fronts_with_properties` that handles
    file I/O.  Both ``.npy`` and NetCDF (``.nc``) property files are supported.

    Parameters
    ----------
    labeled_fronts_file : str
        Path to a ``.npy`` file containing the integer labeled-fronts array.
    property_files : dict[str, PropertySource]
        Mapping from property name to its data source.  Each value may be:

        - A ``(filepath, variable_name)`` **tuple** for a NetCDF file::

              ('LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc',
               'relative_vorticity')

        - A **str** path to a ``.npy`` file::

              '/data/relative_vorticity_20121109T12_00_00.npy'

        - A pre-loaded **ndarray** (skips I/O entirely).

        Full example::

            property_files = {
                'relative_vorticity': (
                    'LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc',
                    'relative_vorticity'),
                'rossby_number': (
                    'LLC4320_2012-11-09T12_00_00_rossby_number_v1.nc',
                    'rossby_number'),
            }

    stats, percentiles, min_npix, nan_policy, dilation_radius
        Forwarded unchanged to :func:`colocate_fronts_with_properties`.

    Returns
    -------
    pd.DataFrame
        See :func:`colocate_fronts_with_properties`.

    Raises
    ------
    FileNotFoundError
        If any of the supplied file paths do not exist.

    Examples
    --------
    >>> df = load_and_colocate(
    ...     'labeled_fronts_global_20121109T12_00_00.npy',
    ...     property_files={
    ...         'relative_vorticity': (
    ...             'LLC4320_2012-11-09T12_00_00_relative_vorticity_v1.nc',
    ...             'relative_vorticity'),
    ...         'rossby_number': (
    ...             'LLC4320_2012-11-09T12_00_00_rossby_number_v1.nc',
    ...             'rossby_number'),
    ...     },
    ...     stats=['mean', 'std', 'median'],
    ...     percentiles=[10, 90],
    ...     min_npix=5,
    ...     nan_policy='omit',
    ...     dilation_radius=3,
    ... )
    >>> df.to_csv('front_properties_20121109T12_00_00.csv', index=False)
    """
    # ---- validate file existence before loading anything ----------------
    missing = []
    if not os.path.isfile(labeled_fronts_file):
        missing.append(f'labeled_fronts: {labeled_fronts_file}')

    for name, src in property_files.items():
        if isinstance(src, np.ndarray):
            continue                         # already loaded
        fpath = src[0] if isinstance(src, (list, tuple)) else src
        if not os.path.isfile(fpath):
            missing.append(f'{name}: {fpath}')

    if missing:
        raise FileNotFoundError(
            "The following files were not found:\n  " + "\n  ".join(missing)
        )

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
