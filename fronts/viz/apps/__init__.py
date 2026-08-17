"""Browser pages for exploring LLC4320 fronts.

Three Panel apps served from one process:

``characteristics``
    Statistics of one field over a lat/lon box drawn on a global map.
``tiles``
    One front inside a 720x720 tile: a 3-D scene plus five 2-D figures.
``evolution``
    Placeholder; specification pending.

Importing this package pulls in the web stack (panel, holoviews,
datashader).  The figure library under ``fronts.viz`` does not import it,
so batch scripts stay free of that dependency.
"""

__all__ = ["config"]
