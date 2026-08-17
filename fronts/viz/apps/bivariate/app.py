"""Bivariate maps — fronts coloured by two fields at once.

A full-width map of every labelled front, coloured by an ``n x n``
bivariate scheme over two per-front statistics. The plotting lives in
:mod:`fronts.viz.bivariate`, which is a plain matplotlib module usable from
a notebook; this page is the controls around it.

**Mode** switches the whole page between SURFACE and DEPTH, which decides
the dates offered, whether a depth level applies, and therefore which
channels the statistics come from.
"""

from __future__ import annotations

import asyncio

import panel as pn
import param

from fronts.viz import bivariate as BV
from fronts.viz.apps import config
from fronts.viz.apps.characteristics import front_props as FP
from fronts.viz.apps.common import pyramid, sources, widgets
from fronts.viz.apps.common.state import PageState

MODES = ("Surface", "Depth")


class BivariateState(PageState):
    """Mode, a date, two fields, a statistic, and the section count."""

    mode = param.Selector(objects=list(MODES), default="Surface",
                          doc="Surface or depth-resolved channels.")
    depth_level = param.Selector(objects=["Surface"], default="Surface",
                                 doc="Depth level (depth mode only).")
    field_a = param.Selector(objects=[], doc="Drives lightness.")
    field_b = param.Selector(objects=[], doc="Drives hue.")
    stat = param.Selector(objects=list(config.FRONT_STATS),
                          default=config.DEFAULT_FRONT_STAT,
                          doc="Per-front statistic for both fields.")
    sections = param.Integer(default=2, bounds=(2, 6),
                             doc="Colour divisions per field.")
    spatial_binning = param.Boolean(True, doc="Aggregate into spatial bins.")
    bin_degrees = param.Number(default=2.0, bounds=(0.5, 10.0),
                               doc="Spatial bin size, degrees.")

    def __init__(self, provider=None, **params):
        super().__init__(provider=provider, **params)
        self.refresh_dates()
        self.refresh_fields()

    @param.depends("mode", watch=True)
    def refresh_dates(self):
        """Depth mode is limited to the timestamps with 3-D data."""
        dates = (self.provider.dates_3d() if self.mode == "Depth"
                 else self.provider.dates())
        if dates:
            self.param.date.objects = dates
            if self.date not in dates:
                self.date = dates[0]

        levels = (self.provider.depth_levels(self.date)
                  if self.mode == "Depth" else ["Surface"])
        self.param.depth_level.objects = levels
        if self.depth_level not in levels:
            self.depth_level = levels[0]

    @param.depends("date", watch=True)
    def refresh_fields(self):
        """Field lists come from the colocation table's actual columns."""
        try:
            table = FP.merged_table(self.provider, self.date)
        except Exception:                                   # noqa: BLE001
            return
        names = FP.available_fields(table)
        if not names:
            return

        self.param.field_a.objects = names
        self.param.field_b.objects = names
        if self.field_a not in names:
            self.field_a = _prefer(names, "gradb2")
        if self.field_b not in names:
            self.field_b = _prefer(names, "turner_angle", "relative_vorticity")

        stats = [s for s in config.FRONT_STATS
                 if any(c.endswith(f"_{s}") for c in table.columns)]
        if stats:
            self.param.stat.objects = stats
            if self.stat not in stats:
                self.stat = stats[0]

    def resolve(self, field: str) -> str:
        if self.mode != "Depth":
            return field
        return self.provider.channel(field, self.depth_level)


def _prefer(names, *wanted):
    for w in wanted:
        if w in names:
            return w
    return names[0]


class BivariatePage:
    """Assembles the bivariate page."""

    def __init__(self, provider=None):
        self.state = BivariateState(provider=provider)
        self._fig = pn.pane.Matplotlib(tight=False, format="png", dpi=110,
                                       sizing_mode="stretch_width",
                                       min_height=520)
        self._status = widgets.status()
        self._token = 0

        s = self.state
        self.w_mode = pn.widgets.RadioButtonGroup.from_param(
            s.param.mode, button_type="primary", width=180)
        self.w_date = pn.widgets.Select.from_param(s.param.date, width=185)
        self.w_depth = pn.widgets.Select.from_param(s.param.depth_level,
                                                    width=185)
        self.w_a = pn.widgets.Select.from_param(s.param.field_a,
                                                name="Field A (lightness)",
                                                width=185)
        self.w_b = pn.widgets.Select.from_param(s.param.field_b,
                                                name="Field B (hue)", width=185)
        self.w_stat = pn.widgets.Select.from_param(s.param.stat, width=120)
        self.w_sections = pn.widgets.IntSlider.from_param(
            s.param.sections, name="Sections per field", width=180)
        self.w_binning = pn.widgets.Checkbox.from_param(
            s.param.spatial_binning, name="Spatial binning")
        self.w_deg = pn.widgets.FloatSlider.from_param(
            s.param.bin_degrees, name="Bin size [deg]", width=160)

        s.param.watch(lambda *_: self.schedule(),
                      ["mode", "date", "depth_level", "field_a", "field_b",
                       "stat", "sections", "spatial_binning", "bin_degrees"])
        self.schedule()

    # -- rendering -------------------------------------------------------

    def schedule(self):
        self._token += 1
        token = self._token
        self._fig.loading = True
        self._status.object = "building bivariate map…"

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._build(token)
        else:
            pn.state.execute(
                lambda: asyncio.ensure_future(self._build_async(token)))

    async def _build_async(self, token):
        await asyncio.to_thread(self._build, token)

    def _build(self, token):
        s = self.state
        try:
            table = FP.merged_table(s.provider, s.date)
            if table.empty:
                raise ValueError("no colocation table for this date")

            col_a = FP.stat_column(table, s.resolve(s.field_a), s.stat) \
                or FP.stat_column(table, s.field_a, s.stat)
            col_b = FP.stat_column(table, s.resolve(s.field_b), s.stat) \
                or FP.stat_column(table, s.field_b, s.stat)
            if col_a is None or col_b is None:
                missing = s.field_a if col_a is None else s.field_b
                raise ValueError(
                    f"no colocated statistic for {missing!r} at this level")

            land = self._land_raster(s)
            fig, scheme = BV.figure_bivariate(
                table,
                table[col_a].to_numpy(dtype=float),
                table[col_b].to_numpy(dtype=float),
                n=s.sections,
                name_a=col_a, name_b=col_b,
                spatial_bin_deg=s.bin_degrees if s.spatial_binning else None,
                land_from=land,
                title=f"{col_a}  ×  {col_b}   —   {s.date}",
            )
        except Exception as exc:                            # noqa: BLE001
            if token == self._token:
                self._fig.object = None
                self._fig.loading = False
                self._status.object = f"**Unavailable:** {exc}"
            return

        if token != self._token:
            return

        self._fig.object = fig
        self._fig.loading = False

        natural = [n for n in (col_a, col_b)
                   if n.rsplit("_", 1)[0] in BV.NATURAL_SPLITS]
        note = (f" · split at 0 for {', '.join(natural)}" if natural else
                " · quantile edges")
        self._status.object = (
            f"**{len(table):,}** fronts · {s.sections}×{s.sections} sections"
            f"{note}"
        )

    def _land_raster(self, s):
        """Land in gray under the fronts, from the model's own mask."""
        try:
            lon, lat, arr = pyramid.level(s.provider, s.date, "__land__",
                                          config.PYRAMID_WIDTHS[0],
                                          reduce="any", pacific=False)
            return lon, lat, arr
        except Exception:                                   # noqa: BLE001
            return None

    # -- layout ----------------------------------------------------------

    def view(self):
        controls = pn.Column(
            pn.Row(pn.pane.Markdown("**Mode**", margin=(8, 5, 0, 10)),
                   self.w_mode, self.w_date, self.w_depth,
                   margin=(0, 10)),
            pn.Row(self.w_a, self.w_b, self.w_stat, self.w_sections,
                   self.w_binning, self.w_deg, margin=(0, 10)),
            sizing_mode="stretch_width",
        )

        body = pn.Column(
            pn.pane.Markdown("### Bivariate maps", margin=(4, 10, 0, 10)),
            pn.pane.Markdown(
                "<small>Every labelled front, coloured by two fields at "
                "once: **field A sets lightness**, **field B sets hue**. "
                "Bin edges are quantiles, except where a field has a "
                "physically meaningful split (Turner angle, vorticity — "
                "divided at zero).</small>",
                margin=(0, 10)),
            controls,
            self._status,
            self._fig,
            sizing_mode="stretch_width",
        )

        notes = [n for n in (widgets.banner(self.state.provider),
                             widgets.degraded_notice()) if n]
        return pn.Column(*notes, body, sizing_mode="stretch_width")


def page(provider=None):
    """Entry point used by ``serve.py``."""
    return BivariatePage(provider=provider).view()
