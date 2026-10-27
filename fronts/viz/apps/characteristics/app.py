"""Backwards-compatible alias.

The page was split into a shared assembly (:mod:`.page`) plus two thin
entry points when the Depth page was added.  This module keeps the old
import path working.
"""

from fronts.viz.apps.characteristics.page import (  # noqa: F401
    DEPTH, SURFACE, CharacteristicsPage, Mode,
)
from fronts.viz.apps.characteristics.surface import page  # noqa: F401
