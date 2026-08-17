"""Synthetic evolving chunks for the Evolution page.

A **chunk** is one spatial box — the same 720×720 extent as a tile — saved
at many consecutive timesteps, rather than the whole globe saved at a few.
Real chunks live at
``s3://dbof/LLC4320_RAW/CHUNKS/{chunk}/YYYYMMDD_HHMMSS.zarr``.

What this module fabricates is a chunk whose front genuinely *evolves*: it
drifts, lengthens and sharpens over the window. That matters more than it
sounds — a movie of a static field with noise on top looks like a movie,
but nothing in it can be read, and the time-series panels would be flat.
So the evolution here is deliberate and monotonic enough to see:

* the front **drifts** across the box at a constant rate;
* it **lengthens** through the middle of the window and shortens after;
* it **rotates** slowly, so the orientation series has structure;
* the colour field **sharpens** as the front tightens.

Nothing here is physical. It exists so the player, the time series and the
moving cursor can be built and judged before real chunks exist.
"""

from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np

from fronts.viz.apps import config
from fronts.viz.apps.common import synthetic


def timesteps(chunk: str, n: int = config.EVOLUTION_N_STEPS) -> list[str]:
    """``n`` consecutive hourly timestamps for a chunk."""
    date, time = config.EVOLUTION_START.split("T")
    y, m, d = (int(p) for p in date.split("-"))
    hour0 = int(time.split("_")[0])

    out = []
    for step in range(n):
        hour = hour0 + step
        day = d + hour // 24
        out.append(f"{y:04d}-{m:02d}-{day:02d}T{hour % 24:02d}_00_00")
    return out


def _seed(chunk: str) -> int:
    return zlib.crc32(chunk.encode()) & 0x7FFFFFFF


class EvolvingChunk:
    """One chunk across its whole time window."""

    def __init__(self, chunk: str, n_steps: int = config.EVOLUTION_N_STEPS):
        self.chunk = chunk
        self.n_steps = n_steps
        self.times = timesteps(chunk, n_steps)
        self.seed = _seed(chunk)
        self.size = config.SYNTH_TILE_SIZE
        self.nz = config.SYNTH_NZ

        # One spatial pattern for the whole window; time acts on it.
        self._base = synthetic._spectral_field(
            (self.size, self.size), self.seed, exponent=-2.3, n_smooth=2)
        self._pert = synthetic._spectral_field(
            (self.size, self.size), self.seed + 17, exponent=-1.9)

        # Coordinates: a small box, regular enough at this scale.
        lat0, lon0 = 36.0, -125.0
        self.YC = np.repeat(
            np.linspace(lat0, lat0 + 6.0, self.size)[:, None], self.size, 1)
        self.XC = np.repeat(
            np.linspace(lon0, lon0 + 7.5, self.size)[None, :], self.size, 0)

        self.Z = -np.cumsum(np.linspace(1.0, 14.0, self.nz)) * 1.6

    # -- time-varying structure ------------------------------------------

    def _phase(self, step: int) -> dict:
        """How far through the window we are, and what that does."""
        t = step / max(self.n_steps - 1, 1)
        return {
            "t": t,
            "drift": 26.0 * t,                        # pixels of translation
            "rotate": np.deg2rad(35.0 * t),           # slow turn
            # Lengthen to mid-window, then relax -- so the length series is
            # not monotonic and the moving cursor has something to point at.
            "extent": 0.55 + 0.45 * np.sin(np.pi * t),
            "sharpen": 0.7 + 0.9 * t,
        }

    def _pattern(self, step: int) -> np.ndarray:
        """The chunk's scalar structure at one step."""
        ph = self._phase(step)
        n = self.size

        jj, ii = np.mgrid[0:n, 0:n].astype(float)
        cj = ci = n / 2.0

        # Rotate and translate about the box centre.
        c, s = np.cos(ph["rotate"]), np.sin(ph["rotate"])
        dj, di = jj - cj, ii - ci - ph["drift"]
        rj = c * dj - s * di + cj
        ri = s * dj + c * di + ci

        src = np.clip(np.stack([rj, ri]).astype(int), 0, n - 1)
        base = self._base[src[0], src[1]]

        # A tightening filament across the box, modulated by the pattern.
        across = (jj - cj) / (n * 0.5)
        ridge = np.exp(-((across * 4.0 / ph["extent"]) ** 2) * ph["sharpen"])
        return np.tanh(2.0 * base) * (0.35 + ridge) \
            + 0.12 * ph["t"] * self._pert

    # -- products --------------------------------------------------------

    @lru_cache(maxsize=64)
    def labels(self, step: int) -> np.ndarray:
        """Labelled fronts at one step.

        The strongest filament keeps **label 1** at every step, so a front
        can be followed through the movie. Real chunks will need genuine
        tracking; this is the stand-in that lets the page be built.
        """
        pattern = self._pattern(step)
        gy, gx = np.gradient(pattern)
        grad = gy ** 2 + gx ** 2

        binary = synthetic._zero_crossings(
            pattern, np.percentile(pattern, [35, 50, 65]))
        binary &= grad > np.percentile(grad, 78)

        lab = synthetic._label_connected(binary)
        counts = np.bincount(lab.ravel())
        if len(counts) < 2:
            return lab.astype(np.int32)

        # Relabel so the largest piece is 1 and the rest follow by size.
        order = np.argsort(counts[1:])[::-1] + 1
        remap = np.zeros(counts.size, dtype=np.int32)
        for new, old in enumerate(order, start=1):
            remap[old] = new if counts[old] >= 12 else 0
        return remap[lab]

    def field(self, step: int, name: str) -> np.ndarray:
        """A 3-D field for one step, shaped ``(k, j, i)``."""
        pattern = self._pattern(step)
        ph = self._phase(step)
        kk = np.arange(self.nz)[:, None, None]

        pyc = 12.0 + 9.0 * pattern[None, :, :]
        sigma0 = 25.4 + 2.1 / (1.0 + np.exp(-(kk - pyc) / 3.0))

        if name in ("density", "sigma0"):
            return sigma0.astype(np.float32)

        shear = np.exp(-((kk - pyc) / 4.0) ** 2)
        data = 0.05 + 60.0 * (1.0 - shear) * (1.0 / ph["sharpen"])
        data = data + 0.3 * np.abs(pattern)[None, :, :]
        return data.astype(np.float32)

    def dataset(self, step: int, prop: str):
        """One step as a tile-shaped :class:`xarray.Dataset`.

        Same layout ``generate_tile`` writes, so everything downstream --
        the pipeline, the curtain builders, the 3-D scene -- treats a chunk
        step exactly like a tile.
        """
        import xarray as xr

        var = "sigma0" if prop in ("density", "sigma0") else prop
        data = self.field(step, prop)

        return xr.Dataset(
            {var: (("k", "j", "i"), data,
                   {"units": "1", "long_name": f"synthetic {var}"})},
            coords={"Z": ("k", self.Z),
                    "XC": (("j", "i"), self.XC),
                    "YC": (("j", "i"), self.YC)},
            attrs={
                "tile_index": 0,
                "face_index": 0,
                "rect_i_start": 0,
                "rect_j_start": 0,
                "timestamp": self.times[step],
                "chunk": self.chunk,
                "step": int(step),
                "synthetic": 1,
            },
        )


@lru_cache(maxsize=4)
def get_chunk(chunk: str, n_steps: int = config.EVOLUTION_N_STEPS
              ) -> EvolvingChunk:
    """Build (and cache) a synthetic chunk.  Deterministic per name."""
    return EvolvingChunk(chunk, n_steps)
