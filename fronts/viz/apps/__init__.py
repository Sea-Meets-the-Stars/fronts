"""Browser pages for exploring LLC4320 fronts.

Five Panel apps served from one process:

``surface`` / ``depth``
    Statistics of one field over a lat/lon box drawn on a global map, at
    the surface or at four depth levels.  One assembly, two entry points.
``bivariate``
    Every front coloured by two fields at once.
``tiles``
    One front inside a 720x720 tile: a 3-D scene plus five 2-D figures,
    in up to three field columns.
``evolution``
    One front played through 24 consecutive hours of a saved chunk.

Importing this package pulls in the web stack (panel, holoviews,
datashader).  The figure library under ``fronts.viz`` does not import it,
so batch scripts stay free of that dependency.
"""

__all__ = ["config"]
