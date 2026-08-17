"""Field Characteristics at the Surface — page 1."""

from fronts.viz.apps.characteristics import page as _page


def page(provider=None):
    """Entry point used by ``serve.py``."""
    return _page.page(_page.SURFACE, provider=provider)
