"""Page state, as Param classes.

Every page keeps its whole selection -- date, field, region, front -- in a
``param.Parameterized``, with the data loading and computation as methods.
The Panel layout is a thin view over that.

The point is testability: the state machine can be driven and asserted
without a browser, a server, or a display.  ``fronts/tests/test_viz_apps.py``
does exactly that.
"""

from __future__ import annotations

import param

from fronts.viz.apps import config
from fronts.viz.apps.common import regions as regions_mod
from fronts.viz.apps.common import sources
from fronts.viz.apps.common.selection import BBox


class PageState(param.Parameterized):
    """What every page has: a provider and a date."""

    date = param.Selector(objects=list(config.DATES), default=config.DEFAULT_DATE,
                          doc="Timestamp being viewed.")

    def __init__(self, provider=None, **params):
        self._provider = provider or sources.get_provider()
        params.setdefault("date", self._provider.dates()[0])
        self.param.date.objects = self._provider.dates()
        super().__init__(**params)

    @property
    def provider(self):
        return self._provider

    @property
    def synthetic(self) -> bool:
        return self._provider.synthetic


class CharacteristicsState(PageState):
    """A field, a region, optionally a depth level, and a front statistic.

    Shared by the Surface and Depth pages; ``depth_mode`` decides whether
    the depth level is offered and whether the date list is restricted to
    the timestamps with 3-D data.
    """

    field = param.Selector(objects=[], doc="Field shown on the map.")
    show_fronts = param.Boolean(False, doc="Overlay the binary front mask.")
    box = param.Parameter(default=BBox.globe(), doc="Selected region.")
    depth_level = param.Selector(objects=["Surface"], default="Surface",
                                 doc="Depth level (Depth page only).")
    front_stat = param.Selector(objects=list(config.FRONT_STATS),
                                default=config.DEFAULT_FRONT_STAT,
                                doc="Per-front statistic for panels (e)/(f).")

    def __init__(self, provider=None, depth_mode=False, **params):
        self._depth_mode = bool(depth_mode)
        super().__init__(provider=provider, **params)

        if self._depth_mode:
            dates = self.provider.dates_3d()
            if dates:
                self.param.date.objects = dates
                if self.date not in dates:
                    self.date = dates[0]

        self.refresh_fields()
        self.refresh_depth_levels()

    @property
    def depth_mode(self) -> bool:
        return self._depth_mode

    def refresh_fields(self):
        """Repopulate the field list from the store, keeping the selection.

        The list is of *base* field names in both modes -- the depth suffix
        is applied when a channel name is needed, not here, so switching
        depth level does not reset the field.
        """
        names = self.provider.field_names(self.date)
        self.param.field.objects = names
        if self.field not in names:
            preferred = next(
                (n for n in ("gradb2", "relative_vorticity") if n in names),
                names[0] if names else None,
            )
            self.field = preferred

    def refresh_depth_levels(self):
        levels = self.provider.depth_levels(self.date)
        self.param.depth_level.objects = levels
        if self.depth_level not in levels:
            self.depth_level = levels[0]

    def refresh_front_stats(self, table=None):
        """Offer only the statistics the colocation table actually has."""
        if table is not None and not getattr(table, "empty", True):
            suffixes = tuple(f"_{s}" for s in config.FRONT_STATS)
            found = {c.rsplit("_", 1)[1] for c in table.columns
                     if c.endswith(suffixes)}
            stats = [s for s in config.FRONT_STATS if s in found]
        else:
            stats = self.provider.front_stats(self.date)
        if stats and list(self.param.front_stat.objects) != stats:
            self.param.front_stat.objects = stats
            if self.front_stat not in stats:
                self.front_stat = stats[0]

    @param.depends("date", watch=True)
    def _on_date(self):
        self.refresh_fields()
        self.refresh_depth_levels()

    # -- derived ---------------------------------------------------------

    def set_bounds(self, bounds):
        """Take a HoloViews ``BoundsXY`` tuple, in 0..360 map coordinates."""
        if bounds is None:
            self.box = BBox.globe()
            return
        x0, y0, x1, y1 = bounds
        self.box = BBox.from_bounds(
            (((x0 + 180) % 360) - 180, y0, ((x1 + 180) % 360) - 180, y1)
        )

    def reset_region(self):
        self.box = BBox.globe()

    def region_label(self) -> str:
        return self.box.label()

    def missing_roles(self) -> tuple[str, ...]:
        roles = self.provider.resolve_channels(self.date)
        return tuple(r for r, c in roles.items() if c is None)


class TilesState(PageState):
    """A region, up to three 3-D fields, and a front label.

    Restricted to the timestamps with full 3-D data.  Figure building is
    explicitly triggered, not reactive -- see ``dirty``.
    """

    #: How many fields can be compared side by side.
    MAX_FIELDS = 3

    region = param.Selector(objects=regions_mod.names(),
                            default=regions_mod.names()[0],
                            doc="Named region, i.e. one LLC tile.")
    fields = param.ListSelector(objects=list(config.TILE_FIELDS_3D),
                                default=["Ri"],
                                doc="Up to three 3-D fields, compared "
                                    "side by side.")
    field = param.Selector(objects=list(config.TILE_FIELDS_3D), default="Ri",
                           doc="Field shown on the tile map.")
    show_fronts = param.Boolean(True, doc="Overlay the labelled fronts.")
    front_label = param.Integer(default=0, bounds=(0, None),
                                doc="Selected front, 0 = none.")
    n_offsets = param.Integer(default=3, bounds=(1, 8),
                              doc="Offset rows per side in the offsets figure.")
    perp_half_width = param.Integer(default=30, bounds=(5, 120),
                                    doc="Half-width of the cross-front transect.")
    dirty = param.Boolean(True, doc="Settings changed since the last build.")

    def __init__(self, provider=None, **params):
        super().__init__(provider=provider, **params)
        dates = self.provider.dates_3d()
        if dates:
            self.param.date.objects = dates
            if self.date not in dates:
                self.date = dates[0]

    @param.depends("region", "date", "fields", "front_label", "n_offsets",
                   "perp_half_width", watch=True)
    def _mark_dirty(self):
        """Any change to the inputs stales the figures.

        Rebuilding costs a tile generation (~15 s) per new field, so the
        user presses Regenerate rather than paying it on every nudge of a
        dropdown.
        """
        self.dirty = True

    @param.depends("fields", watch=True)
    def _cap_fields(self):
        """Keep the selection within the comparable-columns limit."""
        if len(self.fields) > self.MAX_FIELDS:
            self.fields = self.fields[: self.MAX_FIELDS]
        elif not self.fields:
            self.fields = [self.field]
        if self.field not in self.fields:
            self.field = self.fields[0]

    @property
    def region_obj(self) -> regions_mod.Region:
        return regions_mod.by_name(self.region)

    def tile_index(self) -> int:
        """The tile this region resolves to.

        In synthetic mode there is no LLC face geometry, so a stable
        pseudo-index is used instead.
        """
        r = self.region_obj
        if self.synthetic:
            return regions_mod.synthetic_tile_idx(r)
        if r.tile_idx is None:
            raise ValueError(
                f"Region {r.name!r} has no resolved tile index yet.  Run "
                "regions.resolve_all() against the real grid and record the "
                "result in regions.REGIONS."
            )
        return r.tile_idx

    def select_front(self, label) -> bool:
        """Set the selected front.  Returns True when it changed."""
        label = int(label or 0)
        if label == self.front_label:
            return False
        self.front_label = max(label, 0)
        return True

    @param.depends("region", watch=True)
    def _on_region(self):
        self.front_label = 0
