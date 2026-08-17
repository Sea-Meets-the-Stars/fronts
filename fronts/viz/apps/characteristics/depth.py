"""Field Characteristics at Depth — page 2.

Identical to the Surface page apart from the DEPTH LEVEL selector and the
restriction to the timestamps that have full 3-D data.
"""

from fronts.viz.apps.characteristics import page as _page


def page(provider=None):
    """Entry point used by ``serve.py``."""
    return _page.page(_page.DEPTH, provider=provider)
