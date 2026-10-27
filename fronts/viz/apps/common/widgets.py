"""Small pieces of chrome shared by the pages."""

from __future__ import annotations

import panel as pn


def degraded_notice() -> pn.pane.Alert | None:
    """Say so when an optional accelerator is missing, rather than silently.

    Neither of these breaks the map -- the pyramid already does the
    downsampling and the model's NaN mask already draws land -- but a
    coarser map with no explanation looks like a bug.
    """
    from fronts.viz.apps.common import basemap

    notes = []
    if not basemap.HAVE_DATASHADER:
        notes.append(
            "**Datashader unavailable**, so the map is drawn from a coarser "
            "pyramid level instead of being re-aggregated on zoom "
            f"(`{basemap.DATASHADER_ERROR}`). Statistics are unaffected — "
            "they never use the pyramid. Usually a numba/NumPy version "
            "clash: `conda install 'numpy<2.4'`."
        )
    if basemap.coastline_layer() is None:
        notes.append(
            "**Cartopy coastlines unavailable** (no local Natural Earth "
            "data). Land still draws correctly from the model's own mask."
        )
    if not notes:
        return None
    return pn.pane.Alert("  \n\n".join(notes), alert_type="info",
                         margin=(0, 10, 8, 10))


def banner(provider) -> pn.pane.Alert | None:
    """A loud notice when the numbers on screen are fabricated."""
    if not provider.synthetic:
        return None
    return pn.pane.Alert(
        "**Synthetic data.** Every field, front and tile on this page is "
        "fabricated so the layout and interactions can be reviewed before "
        "the real stores are wired up. Nothing here is physically "
        "meaningful. See `docs/viz/apps/WIRING.md` to switch to real data.",
        alert_type="warning",
        margin=(0, 10, 8, 10),
    )


def status(text: str = "", **kwargs) -> pn.pane.Markdown:
    """A one-line status readout."""
    kwargs.setdefault("margin", (0, 10))
    kwargs.setdefault("styles", {"font-size": "0.85em", "color": "#555"})
    return pn.pane.Markdown(text, **kwargs)


def error_card(exc: Exception, title: str = "Not available") -> pn.pane.Alert:
    """Render an exception as something a person can act on."""
    body = str(exc).replace("\n", "  \n")
    return pn.pane.Alert(f"**{title}**  \n{body}", alert_type="danger",
                         margin=(6, 10))


def section(title: str) -> pn.pane.Markdown:
    return pn.pane.Markdown(f"#### {title}", margin=(4, 10, 0, 10))
