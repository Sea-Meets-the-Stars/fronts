"""Rendering helpers shared by the pages.

**Matplotlib is not thread-safe.**  The pages compute off the event loop so
the UI stays responsive, and two panels rendering at once corrupts
matplotlib's global state -- in practice it surfaces as a mathtext parse
error deep inside a tick formatter, which looks like a bug in the figure
code and is not.  Every figure build goes through :data:`MPL_LOCK`.

Figures are also built with the object-oriented API rather than
``pyplot``, so they never enter pyplot's global registry and cannot leak
between sessions.  ``fronts.viz.curtains`` uses pyplot internally, which
is fine -- the lock is what makes it safe, not the constructor.
"""

from __future__ import annotations

import threading

from matplotlib.figure import Figure

#: Held for the duration of any figure build, anywhere in the app.
MPL_LOCK = threading.RLock()


def new_figure(figsize, dpi=110):
    """A standalone figure and axis, outside pyplot's registry."""
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.subplots()
    return fig, ax
