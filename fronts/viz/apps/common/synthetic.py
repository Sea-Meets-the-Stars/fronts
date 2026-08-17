"""A synthetic ocean, so the pages run with no data.

This module fabricates everything the real stores would supply: a global
2-D field set with land, a binary front mask, labelled fronts, the
geometry and colocation tables, and 3-D tiles in the exact NetCDF layout
``dbof.cli.generate_tile`` writes.

Two properties are deliberate, not incidental:

* **The grid is irregular.**  ``YC`` uses Mercator-like spacing and ``XC``
  carries a small per-row warp, mirroring the fact that the real rect grid
  is stitched from rotated faces rather than interpolated.  Code that
  quietly assumes a regular lat/lon axis will misbehave here, which is the
  point.
* **Land is NaN**, as it is in the real LLC output.  The map draws land
  from that mask rather than from an external coastline dataset.

Nothing here is physically meaningful.  It exists so the layout,
interactions and plumbing can be built and reviewed before the data is
wired up.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field as _dcfield
from functools import lru_cache

import numpy as np
import pandas as pd

from fronts.viz.apps import config

_OMEGA = 7.2921e-5


# --------------------------------------------------------------------------
# Random field helpers
# --------------------------------------------------------------------------

def _spectral_field(shape, seed, exponent=-1.8, n_smooth=0):
    """A smooth random field with a power-law spectrum.

    Cheap stand-in for mesoscale ocean structure: red noise in Fourier
    space, so it has coherent eddies rather than pixel grass.
    """
    rng = np.random.default_rng(seed)
    nj, ni = shape
    white = rng.standard_normal((nj, ni))

    kj = np.fft.fftfreq(nj)[:, None]
    ki = np.fft.fftfreq(ni)[None, :]
    k = np.sqrt(kj ** 2 + ki ** 2)
    k[0, 0] = 1.0

    spec = np.fft.fft2(white) * (k ** exponent)
    spec[0, 0] = 0.0
    out = np.real(np.fft.ifft2(spec))

    for _ in range(n_smooth):
        out = 0.25 * (
            np.roll(out, 1, 0) + np.roll(out, -1, 0)
            + np.roll(out, 1, 1) + np.roll(out, -1, 1)
        )

    return (out - out.mean()) / (out.std() + 1e-30)


try:
    from skimage.measure import label as _sk_label            # noqa: F401
    from skimage.morphology import closing, disk, skeletonize  # noqa: F401
    _HAVE_SKIMAGE = True
except ImportError:                                            # pragma: no cover
    _HAVE_SKIMAGE = False


def _label_connected(binary):
    """8-connected labelling, via SciPy so scikit-image stays optional."""
    from scipy import ndimage as ndi
    labels, _ = ndi.label(binary, structure=np.ones((3, 3), dtype=int))
    return labels


def _thin_via_skimage(gradb2, percentile=88.0):
    """Mirror the real pipeline: threshold, close the speckle, skeletonize."""
    finite = np.isfinite(gradb2)
    thresh = np.nanpercentile(gradb2[finite], percentile)
    binary = finite & (gradb2 > thresh)
    binary = closing(binary, disk(2))
    binary &= finite
    return skeletonize(binary)


def _zero_crossings(field, levels):
    """One-pixel-wide contour mask: cells where ``field - level`` changes sign."""
    out = np.zeros(field.shape, dtype=bool)
    for level in levels:
        d = field - level
        right = np.zeros_like(out)
        down = np.zeros_like(out)
        right[:, :-1] = (d[:, :-1] * d[:, 1:]) < 0
        down[:-1, :] = (d[:-1, :] * d[1:, :]) < 0
        out |= right | down
    return out


def _thin_via_contours(buoyancy, gradb2, n_levels=9, gradient_percentile=80.0):
    """Fronts as strong-gradient contour segments, without scikit-image.

    A contour of a smooth field is one pixel wide and connected by
    construction, so no morphological thinning is needed.  But a single
    contour level of a spectral field percolates -- it comes back as one
    curve spanning the domain, which is not a front.

    Masking the contours by gradient magnitude fixes that and is the more
    faithful definition anyway: a front is the stretch of an isopycnal
    where the gradient is strong.  The result is many medium-length
    filaments rather than one monster.
    """
    finite = np.isfinite(buoyancy) & np.isfinite(gradb2)
    if not finite.any():
        return np.zeros(buoyancy.shape, dtype=bool)

    levels = np.nanpercentile(buoyancy[finite], np.linspace(5, 95, n_levels))
    contours = _zero_crossings(np.where(finite, buoyancy, np.nan), levels)

    strong = gradb2 > np.nanpercentile(gradb2[finite], gradient_percentile)
    return contours & strong & finite


def _gradients(a):
    """Centred differences with wrap in x and edge-clamp in y."""
    dy = np.gradient(a, axis=0)
    dx = 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))
    return dy, dx


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------

@dataclass
class SyntheticWorld:
    """A fabricated global dataset for one timestamp."""

    date: str
    shape: tuple[int, int] = config.SYNTH_SHAPE
    seed: int = 20120516

    XC: np.ndarray = _dcfield(init=False, repr=False)
    YC: np.ndarray = _dcfield(init=False, repr=False)
    land: np.ndarray = _dcfield(init=False, repr=False)
    fields: dict = _dcfield(init=False, repr=False, default_factory=dict)
    fronts: np.ndarray = _dcfield(init=False, repr=False)
    labels: np.ndarray = _dcfield(init=False, repr=False)
    _buoyancy: np.ndarray = _dcfield(init=False, repr=False, default=None)

    def __post_init__(self):
        self._build_grid()
        self._build_land()
        self._build_fields()
        self._build_fronts()

    # -- grid ------------------------------------------------------------
    def _build_grid(self):
        nj, ni = self.shape

        # Longitude: nominally regular, with a small per-row warp so the
        # array is honestly 2-D and column-roll assumptions are testable.
        lon1d = np.linspace(-180.0, 180.0, ni, endpoint=False)
        row = np.arange(nj)[:, None]
        warp = 0.35 * np.sin(2 * np.pi * row / nj)
        self.XC = (lon1d[None, :] + warp + 180.0) % 360.0 - 180.0

        # Latitude: Mercator-like, so spacing is emphatically non-uniform.
        lat_max = 80.0
        y_max = np.log(np.tan(np.pi / 4 + np.radians(lat_max) / 2))
        y = np.linspace(-y_max, y_max, nj)
        lat1d = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
        self.YC = np.repeat(lat1d[:, None], ni, axis=1)

    # -- land ------------------------------------------------------------
    def _build_land(self):
        """Blobby continents, from a thresholded smooth random field."""
        blobs = _spectral_field(self.shape, self.seed + 7, exponent=-2.6,
                                n_smooth=4)
        land = blobs > 0.85

        # Polar ice caps -- LLC4320 output is masked there too.
        land |= np.abs(self.YC) > 78.0
        self.land = land

    # -- fields ----------------------------------------------------------
    def _build_fields(self):
        nj, ni = self.shape
        f0 = 2 * _OMEGA * np.sin(np.radians(self.YC))

        # A streamfunction, and the flow that goes with it.
        psi = _spectral_field(self.shape, self.seed, exponent=-2.2, n_smooth=1)
        psi *= (1.0 - 0.6 * np.exp(-(self.YC / 12.0) ** 2))   # weak equator

        psi_y, psi_x = _gradients(psi)
        u, v = -psi_y, psi_x

        u_y, u_x = _gradients(u)
        v_y, v_x = _gradients(v)

        vort = v_x - u_y
        strain_n = u_x - v_y
        strain_s = v_x + u_y
        strain = np.hypot(strain_n, strain_s)
        div = u_x + v_y

        # Scale so zeta/f and sigma/|f| concentrate near the origin with
        # tails reaching the usual -7..7 / 0..7 axes, rather than filling
        # the plane uniformly.
        scale = 0.8 * np.abs(f0) / (np.std(vort) + 1e-30)
        vort, strain_n, strain_s, strain, div = (
            a * scale for a in (vort, strain_n, strain_s, strain, div)
        )

        # A buoyancy field with sharp filaments, and its squared gradient.
        buoy = _spectral_field(self.shape, self.seed + 3, exponent=-1.5)
        buoy = np.tanh(2.2 * buoy)
        self._buoyancy = np.where(self.land, np.nan, buoy)
        b_y, b_x = _gradients(buoy)
        gradb2 = b_y ** 2 + b_x ** 2
        gradb2 *= 1e-14 / (np.median(gradb2) + 1e-30)

        sst = (
            302.0 - 32.0 * (np.abs(self.YC) / 80.0) ** 1.4
            + 1.6 * _spectral_field(self.shape, self.seed + 11, exponent=-2.0)
        )

        raw = {
            "gradb2": gradb2,
            "relative_vorticity": vort,
            "strain_mag": strain,
            "strain_n": strain_n,
            "strain_s": strain_s,
            "divergence": div,
            "coriolis_f": f0,
            "SSTK": sst,
            "Eta": 0.7 * psi / (np.std(psi) + 1e-30),
            "okubo_weiss": strain ** 2 - vort ** 2,
        }

        self.fields = {
            name: np.where(self.land, np.nan, arr).astype(np.float32)
            for name, arr in raw.items()
        }

    # -- fronts ----------------------------------------------------------
    def _build_fronts(self):
        """Build thin, connected, labelled fronts.

        Two routes to the same shape of answer.  With scikit-image
        available we mirror the real pipeline: threshold ``gradb2``, close
        the speckle into filaments, thin to a skeleton.  Without it we take
        contours of the buoyancy field instead, which are thin by
        construction.

        Either way the curtain code needs fronts with a real main axis to
        walk along, so tiny fragments are dropped at the end.
        """
        binary = (
            _thin_via_skimage(self.fields["gradb2"]) if _HAVE_SKIMAGE
            else _thin_via_contours(self._buoyancy, self.fields["gradb2"])
        )

        labels = _label_connected(binary)

        # Drop the specks: keep pieces long enough to have a main axis.
        counts = np.bincount(labels.ravel())
        keep = np.zeros(counts.size, dtype=bool)
        keep[counts >= 8] = True
        keep[0] = False
        binary = keep[labels]

        self.fronts = binary.astype(np.uint8)
        self.labels = _label_connected(binary).astype(np.int32)

    # -- depth variants --------------------------------------------------
    def depth_variant(self, base: str, suffix: str) -> np.ndarray:
        """A surface field modulated to stand in for a depth level.

        The synthetic world is 2-D.  Rather than fabricate a whole depth
        dimension, each level applies a deterministic, level-specific
        transform to the surface field: enough for the Depth page to show
        four visibly different maps and for its selectors and statistics to
        be exercised, with no pretence of physics.
        """
        arr = self.fields[base]
        if suffix == "sfc":
            return arr

        # Weaker and smoother with depth; the mixed-layer mean is smoother
        # still.  Deterministic, so the page is reproducible.
        gain, smooth = {
            "z25m": (0.72, 1),
            "mld": (0.45, 2),
            "mld_mean": (0.55, 3),
        }.get(suffix, (1.0, 0))

        import warnings

        out = np.array(arr, dtype=np.float32)
        for _ in range(smooth):
            stack = np.stack([
                np.roll(out, 1, 0), np.roll(out, -1, 0),
                np.roll(out, 1, 1), np.roll(out, -1, 1),
                out,
            ])
            with warnings.catch_warnings():
                # An all-land neighbourhood is all-NaN; nanmean warns and
                # returns NaN, which is the answer we want.
                warnings.simplefilter("ignore", RuntimeWarning)
                out = np.nanmean(stack, axis=0)

        mean = np.nanmean(arr)
        out = mean + gain * (out - mean)
        return np.where(self.land, np.nan, out).astype(np.float32)

    # -- tables ----------------------------------------------------------
    def geometry_table(self) -> pd.DataFrame:
        """One row per labelled front, matching the geometry parquet.

        Uses ``scipy.ndimage`` rather than ``skimage.measure.regionprops``
        so the synthetic provider carries no scikit-image dependency.
        """
        from scipy import ndimage as ndi

        lab = self.labels
        n = int(lab.max())
        if n == 0:
            return pd.DataFrame()

        index = np.arange(1, n + 1)
        boxes = ndi.find_objects(lab)
        counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]

        rows = []
        for k, (label, box) in enumerate(zip(index, boxes)):
            if box is None:
                continue
            js, iss = box
            sub = lab[js, iss] == label
            jj_local, ii_local = np.nonzero(sub)
            jj = int(round(jj_local.mean())) + js.start
            ii = int(round(ii_local.mean())) + iss.start

            # Orientation from the second moments of the pixel cloud,
            # matching regionprops' convention (radians, from the j axis).
            dj = jj_local - jj_local.mean()
            di = ii_local - ii_local.mean()
            orientation = 0.5 * np.arctan2(
                2.0 * float((dj * di).mean()),
                float((di * di).mean()) - float((dj * dj).mean()),
            )

            rows.append(
                {
                    "label": int(label),
                    "npix": int(counts[k]),
                    "centroid_lat": float(self.YC[jj, ii]),
                    "centroid_lon": float(self.XC[jj, ii]),
                    "centroid_j": jj,
                    "centroid_i": ii,
                    "bbox_j0": js.start, "bbox_i0": iss.start,
                    "bbox_j1": js.stop, "bbox_i1": iss.stop,
                    "length_km": float(counts[k]) * 2.0,
                    "orientation": float(orientation),
                }
            )
        return pd.DataFrame(rows)

    def colocation_table(self) -> pd.DataFrame:
        """Per-front property medians, matching the colocation parquet."""
        lab = self.labels
        keep = lab > 0
        flat_lab = lab[keep]
        order = np.argsort(flat_lab, kind="stable")
        sorted_lab = flat_lab[order]
        uniq, starts = np.unique(sorted_lab, return_index=True)
        splits = np.split(order, starts[1:])

        out = {"flabel": uniq, "npix": np.array([s.size for s in splits])}

        # Match the real colocation table's column scheme: '{field}_{stat}'
        # for mean/std/median and '{field}_p{pct}' for percentiles.  The
        # percentiles here are the ones run_v5_100_timesteps.yaml asks for
        # -- 25, 75, 90.  Note there is no p95.
        reducers = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "p25": lambda a: np.nanpercentile(a, 25),
            "p75": lambda a: np.nanpercentile(a, 75),
            "p90": lambda a: np.nanpercentile(a, 90),
        }

        for name in sorted(self.fields):
            vals = self.fields[name][keep]
            for stat, fn in reducers.items():
                with np.errstate(invalid="ignore"):
                    out[f"{name}_{stat}"] = np.array([
                        fn(vals[s]) if s.size else np.nan for s in splits
                    ])
        return pd.DataFrame(out)

    # -- 3-D tiles -------------------------------------------------------
    def tile_slices(self, tile_idx: int) -> tuple[slice, slice]:
        """Rect-grid slices for a synthetic tile index."""
        n = config.SYNTH_TILE_SIZE
        nj, ni = self.shape
        n_i = max(ni // n, 1)
        tj, ti = divmod(int(tile_idx) % max((nj // n) * n_i, 1), n_i)
        return slice(tj * n, (tj + 1) * n), slice(ti * n, (ti + 1) * n)

    def tile_dataset(self, tile_idx: int, prop: str):
        """A 3-D tile in the layout ``dbof.cli.generate_tile`` writes.

        Returns an :class:`xarray.Dataset` with one 3-D variable on
        ``(k, j, i)``, plus ``XC``/``YC``/``Z`` and the provenance attrs
        the tile loaders require.
        """
        import xarray as xr

        n = config.SYNTH_TILE_SIZE
        nz = config.SYNTH_NZ
        js, iss = self.tile_slices(tile_idx)

        XC = self.XC[js, iss]
        YC = self.YC[js, iss]

        # Depth: stretched, like LLC levels.
        Z = -np.cumsum(np.linspace(1.0, 14.0, nz)) * 1.6

        # A tilted front: the isopycnals outcrop across the tile, following
        # the same horizontal structure the surface fields have.
        surf = _spectral_field((n, n), self.seed + tile_idx, exponent=-2.4,
                               n_smooth=2)
        across = np.tanh(1.8 * surf)

        kk = np.arange(nz)[:, None, None]
        pyc = 12.0 + 9.0 * across[None, :, :]          # pycnocline level
        sigma0 = 25.4 + 2.1 / (1.0 + np.exp(-(kk - pyc) / 3.0))
        sigma0 += 0.05 * _spectral_field((n, n), self.seed + 99,
                                         exponent=-2.0)[None, :, :]

        land_tile = self.land[js, iss]
        sigma0 = np.where(land_tile[None, :, :], np.nan, sigma0)

        if prop in ("density", "sigma0"):
            var_name, data, units, long_name = (
                "sigma0", sigma0, "kg/m^3", "potential density anomaly",
            )
        else:
            # Something Ri-like: small in the pycnocline, large below.
            shear = np.exp(-((kk - pyc) / 4.0) ** 2)
            data = 0.05 + 60.0 * (1.0 - shear) + 0.4 * np.abs(
                _spectral_field((n, n), self.seed + 5, exponent=-1.6)
            )[None, :, :]
            data = np.where(land_tile[None, :, :], np.nan, data)
            var_name, units, long_name = prop, "1", f"synthetic {prop}"

        ds = xr.Dataset(
            {var_name: (("k", "j", "i"), data.astype(np.float32),
                        {"units": units, "long_name": long_name})},
            coords={
                "Z": ("k", Z),
                "XC": (("j", "i"), XC),
                "YC": (("j", "i"), YC),
            },
            attrs={
                "tile_index": int(tile_idx),
                "face_index": 0,
                "rect_i_start": int(iss.start),
                "rect_j_start": int(js.start),
                "timestamp": self.date,
                "synthetic": 1,
            },
        )
        return ds


def _stable_seed(date: str) -> int:
    """A seed that is the same in every process.

    ``hash()`` on a string is salted per interpreter (PYTHONHASHSEED), so
    using it here would give a different ocean on every run -- fronts
    appearing and disappearing between restarts, and cached pyramid levels
    silently disagreeing with freshly computed ones.  CRC32 is stable.
    """
    return zlib.crc32(date.encode()) & 0x7FFFFFFF


@lru_cache(maxsize=4)
def get_world(date: str, shape: tuple = config.SYNTH_SHAPE) -> SyntheticWorld:
    """Build (and cache) the synthetic world for a date.

    Deterministic: the same date always gives the same ocean.
    """
    return SyntheticWorld(date=date, shape=shape, seed=_stable_seed(date))
