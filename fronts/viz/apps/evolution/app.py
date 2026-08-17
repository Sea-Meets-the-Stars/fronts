"""Page 3 -- evolution.

Placeholder.  The specification is pending; this page exists so the route
and the shared layer are in place, and so the navigation does not have a
hole in it.

When the spec lands, this follows the same shape as the other two: a
``param.Parameterized`` in ``common/state.py`` holding the selection, a
builder module for the figures, and a thin view here.
"""

from __future__ import annotations

import panel as pn

from fronts.viz.apps.common import widgets
from fronts.viz.apps.common.state import PageState


def page(provider=None):
    """Entry point used by ``serve.py``."""
    state = PageState(provider=provider)

    body = pn.Column(
        pn.pane.Markdown("### Evolution", margin=(4, 10, 0, 10)),
        pn.pane.Alert(
            "**Not specified yet.**  This page will follow the same shape as "
            "the other two — global map, pick a region, get plots — with the "
            "plots showing how fronts evolve across timestamps.\n\n"
            "The shared layer it will use is already in place: "
            "`common/sources.py` for data, `common/state.py` for the "
            "selection, `common/basemap.py` for the map, and "
            "`common/selection.py` for regions.",
            alert_type="secondary",
            margin=(6, 10),
        ),
        pn.pane.Markdown(
            f"<small>Provider: **{state.provider.mode}** · "
            f"dates available: {', '.join(state.provider.dates())}</small>",
            margin=(0, 10),
        ),
        sizing_mode="stretch_width",
    )

    note = widgets.banner(state.provider)
    return pn.Column(*([note] if note else []), body,
                     sizing_mode="stretch_width")
