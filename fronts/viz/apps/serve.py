"""Server entry point for the front visualisation pages.

One process, five routes::

    python -m fronts.viz.apps.serve --show

    /                 index
    /surface          field statistics at the surface
    /depth            the same, at four depth levels
    /bivariate        fronts coloured by two fields at once
    /tiles            one front in 3-D and cross-section
    /evolution        placeholder

The data source comes from ``FRONTS_APP_DATA``: ``synthetic`` (the
default, which needs no data at all) or ``s3``.  See
``docs/viz/apps/WIRING.md``.
"""

from __future__ import annotations

import argparse
import os

import panel as pn

from fronts.viz.apps.common import sources

TITLE = "Fronts — LLC4320"

PAGES = {
    "surface": "Field Characteristics — Surface",
    "depth": "Field Characteristics — Depth",
    "bivariate": "Bivariate maps",
    "tiles": "Tiles — one front in 3-D",
    "evolution": "Evolution",
}


def ensure_display() -> None:
    """PyVista >= 0.44 dropped ``start_xvfb``; any non-empty DISPLAY skips it.

    Matches step 1 of ``docs/viz/fronts_viz_3d_runbook.md``.  OSMesa ignores
    DISPLAY entirely, so this is safe on a headless server.
    """
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = "dummy"


def _nav(active: str) -> pn.pane.HTML:
    links = "".join(
        f'<a href="/{slug}" style="'
        f'{"font-weight:600;text-decoration:underline;" if slug == active else ""}'
        f'margin-right:1.4em;color:#1f4e5f">{title}</a>'
        for slug, title in PAGES.items()
    )
    return pn.pane.HTML(
        f"<div style='padding:4px 10px;font-size:0.9em'>"
        f"<a href='/' style='margin-right:1.4em;color:#1f4e5f'>Index</a>"
        f"{links}</div>",
        sizing_mode="stretch_width",
    )


def _template(slug: str | None, body) -> pn.template.BaseTemplate:
    name = PAGES.get(slug or "", "")
    return pn.template.FastListTemplate(
        title=f"{TITLE}{' — ' + name if name else ''}",
        main=[_nav(slug or ""), body],
        header_background="#1f4e5f",
        main_max_width="none",
    )


def index():
    provider = sources.get_provider()
    rows = "\n".join(f"- [{title}](/{slug})" for slug, title in PAGES.items())
    warning = (
        "\n\n> **Synthetic data.** Nothing on these pages is physically "
        "meaningful. See `docs/viz/apps/WIRING.md` to switch to real data."
        if provider.synthetic else ""
    )
    return _template(None, pn.pane.Markdown(
        f"## Front visualisation\n\n{rows}\n\n---\n\n"
        f"Data provider: **{provider.mode}**{warning}",
        margin=(10, 20),
    ))


def surface():
    from fronts.viz.apps.characteristics import surface as mod
    return _template("surface", mod.page())


def depth():
    from fronts.viz.apps.characteristics import depth as mod
    return _template("depth", mod.page())


def bivariate():
    from fronts.viz.apps.bivariate import app
    return _template("bivariate", app.page())


def tiles():
    ensure_display()
    from fronts.viz.apps.tiles import app
    return _template("tiles", app.page())


def evolution():
    from fronts.viz.apps.evolution import app
    return _template("evolution", app.page())


ROUTES = {
    "/": index,
    "/surface": surface,
    "/depth": depth,
    "/bivariate": bivariate,
    "/tiles": tiles,
    "/evolution": evolution,
    # The Surface page was called "characteristics" before the Depth page
    # existed; keep the old URL working.
    "/characteristics": surface,
}


def _local_origins(port: int) -> list[str]:
    """Both spellings of the loopback address, for the given port.

    Bokeh compares the browser's Origin header literally, so a server told
    to allow ``localhost:5006`` refuses ``127.0.0.1:5006`` -- the page
    shell loads and the websocket carrying every widget is rejected, which
    looks exactly like a blank page.  They are the same machine; allow both.
    """
    return [f"localhost:{port}", f"127.0.0.1:{port}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5006)
    ap.add_argument("--show", action="store_true",
                    help="Open a browser on start.")
    ap.add_argument("--address", default="localhost")
    ap.add_argument("--allow-websocket-origin", action="append", default=None)
    ap.add_argument("--data", choices=("synthetic", "s3"), default=None,
                    help="Override FRONTS_APP_DATA for this run.")
    args = ap.parse_args(argv)

    if args.data:
        os.environ["FRONTS_APP_DATA"] = args.data
        sources.get_provider.cache_clear()

    ensure_display()
    pn.extension("vtk", notifications=True)

    origins = args.allow_websocket_origin or _local_origins(args.port)

    provider = sources.get_provider()
    print(f"[fronts-viz] provider={provider.mode} "
          f"synthetic={provider.synthetic}")
    print(f"[fronts-viz] http://{args.address}:{args.port}/")

    pn.serve(
        ROUTES,
        port=args.port,
        address=args.address,
        show=args.show,
        websocket_origin=origins,
        title=TITLE,
    )


if __name__ == "__main__":
    main()
