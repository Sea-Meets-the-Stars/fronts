"""Field Characteristics at Depth — page 2.

Identical to the Surface page apart from the DEPTH LEVEL selector and the
restriction to the timestamps that have full 3-D data.
"""

from fronts.viz.apps.characteristics import page as _page
from fronts.viz.apps.common import sources


def page(provider=None):
    """Entry point used by ``serve.py``.

    Explicitly a DEPTH provider.  The depth fields live in their own S3
    prefix and their channels carry a suffix, so a SURF provider here
    would find only bare surface names -- and would show them without
    complaint, under a depth selector that did nothing.
    """
    if provider is None:
        provider = sources.get_provider("DEPTH")
    return _page.page(_page.DEPTH, provider=provider)
