""" Shared fixtures for dev/frontogenesis/py/tests (coding doc §5).

Everything runs offline.  Tests marked ``needs_grid`` use the M0 stores
from ``dev/frontogenesis/data/`` and are skipped, not failed, when the
stores are absent; the synthetic-field tests never touch the disk.
"""

import sys
from pathlib import Path

import pytest

# the modules under test live one level up, alongside osn_tiles.py
PY_DIR = Path(__file__).resolve().parents[1]
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import osn_tiles as ot  # noqa: E402

GRID_PATH = ot.DATA_DIR / 'tile330_grid.zarr'
RAW_PATH = ot.DATA_DIR / 'tile330_raw_20120702T00_2h.zarr'
MASKS_PATH = ot.DATA_DIR / 'tile330_masks.nc'


@pytest.fixture(scope='session')
def grid_ds():
    """The M0 static grid, in memory, with the length-1 ``face`` dim the
    dbof operators expect."""
    if not GRID_PATH.exists():
        pytest.skip(f'{GRID_PATH} not on disk (M0 task 4)')
    return ot.open_grid(GRID_PATH, with_face=True)


@pytest.fixture(scope='session')
def raw_ds():
    """M0's two hours, in memory, stored layout ``(time, j, i)``."""
    import xarray as xr
    if not RAW_PATH.exists():
        pytest.skip(f'{RAW_PATH} not on disk (M0 task 4)')
    return xr.open_zarr(RAW_PATH).load()


@pytest.fixture(scope='session')
def masks_ds():
    """``tile330_masks.nc`` (M1 task 1), masks as bool on ``(j, i)``."""
    import masking as mk
    if not MASKS_PATH.exists():
        pytest.skip(f'{MASKS_PATH} not on disk (M1 task 1)')
    return mk.open_masks(MASKS_PATH)
