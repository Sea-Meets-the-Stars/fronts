"""Page state, as Param classes.

Every page keeps its whole selection -- date, field, region, front -- in a
``param.Parameterized``, with the data loading and computation as methods.
The Panel layout is a thin view over that.

The point is testability: the state machine can be driven and asserted
without a browser, a server, or a display.  ``fronts/tests/test_viz_apps.py``
does exactly that.
"""

from __future__ import annotations

from functools import lru_cache

import param

from fronts.viz.apps import config
from fronts.viz.apps.common import regions as regions_mod
from fronts.viz.apps.common import sources
from fronts.viz.apps.common.selection import BBox


@lru_cache(maxsize=32)
def _resolved_tile(provider, date: str, key: str) -> int:
    """Region -> tile index, searched once per region and remembered.

    The grid does not change between dates, but the provider supplies the
    coordinates, so the date is carried along for the lookup.
    """
    return regions_mod.tile_index_for(provider, date,
                                      regions_mod.BY_KEY[key])


class PageState(param.Parameterized):
    """What every page has: a provider and a date."""

    date = param.Selector(objects=list(config.DATES), default=config.DEFAULT_DATE,
                          doc="Timestamp being viewed.")

    def __init__(self, provider=None, **params):
        self._provider = provider or sources.get_provider()
        dates = self.offered_dates()
        params.setdefault("date", dates[0])
        self.param.date.objects = dates
        super().__init__(**params)

    def offered_dates(self) -> list[str]:
        """Dates this page offers: what the store has, narrowed by config.

        The store listing on its own is not the answer -- a folder can
        hold more timestamps than the build covers, and every page was
        offering all of them because this read ``provider.dates()``
        directly.  ``config.DATES`` is the allow-list; the intersection
        keeps a configured date from appearing before it exists.
        """
        available = self._provider.dates()
        allowed = list(config.DATES)
        if not allowed:
            return available
        return [d for d in available if d in allowed] or available

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
    dirty = param.Boolean(True, doc="Settings changed since the last build.")

    #: Changing any of these stales the figures rather than rebuilding.
    #: Statistics run on the native grid, so picking a field, then a
    #: depth, then a region would pay for the whole computation three
    #: times over for two results nobody looked at.
    REBUILD_TRIGGERS = ("date", "field", "depth_level", "box", "show_fronts",
                        "front_stat")

    @param.depends(*REBUILD_TRIGGERS, watch=True)
    def _mark_dirty(self):
        self.dirty = True

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
    #: Field shown on the tile map.  Density by default: the isopycnal
    #: control is the next thing the user touches, and the map is where
    #: they read off which sigma values the volume actually contains.
    field = param.Selector(objects=list(config.TILE_FIELDS_3D),
                           default=config.TILE_GEOMETRY_FIELD,
                           doc="Field shown on the tile map.")
    show_fronts = param.Boolean(True, doc="Overlay the labelled fronts.")
    front_label = param.Integer(default=0, bounds=(0, None),
                                doc="Selected front, 0 = none.")
    n_offsets = param.Integer(default=3, bounds=(1, 8),
                              doc="Offset rows per side in the offsets figure.")
    perp_half_width = param.Integer(default=30, bounds=(5, 120),
                                    doc="Half-width of the cross-front transect.")
    #: Isopycnal for the depth map.  ``0`` means "use this volume's
    #: median", resolved once a scene exists -- a fixed default would
    #: often name a surface that is nowhere in the tile, and the map
    #: would come back entirely gray.
    sigma = param.Number(default=0.0, bounds=(0.0, 40.0),
                         doc="Isopycnal for the depth map, kg/m^3.")

    #: Field for the region map that sits beside the isopycnal depth --
    #: chosen at the same step, before the per-column fields.
    region_field = param.Selector(objects=list(config.TILE_FIELDS_3D),
                                  default=config.TILE_GEOMETRY_FIELD,
                                  doc="Field for the region overview map.")

    #: Depth for the inset's second row.  0 keeps it at the surface, which
    #: makes the second row a duplicate -- so it defaults deeper.
    inset_depth = param.Number(default=-50.0, bounds=(-6000.0, 0.0),
                               doc="Depth of the inset's second row, m.")
    dirty = param.Boolean(True, doc="Settings changed since the last build.")

    #: Where along the front axis the cross-front transect is cut.
    #: ``-1`` means "pick the field extremum", which is what the page did
    #: before the plan view could be clicked.
    perp_index = param.Integer(default=-1, bounds=(-1, None),
                               doc="Axis vertex for the transect, -1 = auto.")

    #: Locations for the vertical profiles, as (j, i) in the crop frame.
    #: Cleared whenever the front changes: the same pixel on a different
    #: front is a different place, so keeping them would quietly plot the
    #: wrong column.
    profile_points = param.List(default=[], item_type=tuple,
                                doc="Up to MAX_PROFILES (j, i) locations.")
    MAX_PROFILES = 5

    #: What a click on the plan view does.
    pick_mode = param.Selector(objects=["perpendicular", "profiles"],
                               default="perpendicular",
                               doc="What clicking the plan view sets.")

    #: Stage 2 is stale.  Stage 1 (`dirty`) implies this; this never
    #: implies stage 1 -- one arrow, so two buttons stay comprehensible.
    sections_dirty = param.Boolean(True,
                                   doc="Sections stale since the last build.")

    @param.depends("perp_index", "profile_points", "perp_half_width",
                   "n_offsets", watch=True)
    def _stale_sections(self):
        self.sections_dirty = True

    @param.depends("front_label", watch=True)
    def _clear_profiles(self):
        self.profile_points = []
        self.perp_index = -1

    def add_profile_point(self, j: int, i: int) -> bool:
        """Record a profile location.  False when already at the limit."""
        if len(self.profile_points) >= self.MAX_PROFILES:
            return False
        self.profile_points = self.profile_points + [(int(j), int(i))]
        return True

    def clear_profile_points(self):
        self.profile_points = []

    def __init__(self, provider=None, **params):
        super().__init__(provider=provider, **params)
        dates = self.usable_dates()
        if dates:
            self.param.date.objects = dates
            if self.date not in dates:
                self.date = dates[0]

    def usable_dates(self) -> list[str]:
        """Dates this page can actually show something for.

        A tile needs 3-D raw data *and* a label map, and build_v5 has only
        run for some dates.  Offering a date with no fronts gives a page
        that loads a tile and then says there is nothing on it, so those
        are left out -- unless none qualify, in which case the 3-D dates
        are offered and the page explains what is missing.
        """
        dates_3d = self.provider.dates_3d()
        with_fronts = self.provider.dates_with_fronts(dates_3d)
        return with_fronts or dates_3d

    @param.depends("region", "date", "fields", "front_label", "n_offsets",
                   "perp_half_width", "sigma", watch=True)
    def _mark_dirty(self):
        """Any change to the inputs stales the figures.

        Rebuilding costs a tile generation (~15 s) per new field, so the
        user presses Regenerate rather than paying it on every nudge of a
        dropdown.
        """
        self.dirty = True

    @param.depends("fields", watch=True)
    def _cap_fields(self):
        """Keep the selection within the comparable-columns limit.

        Over the limit, the *oldest* entries go.  Truncating from the end
        would drop what the user just picked and keep whatever was there
        by default, which reads as the page ignoring the click.
        """
        if len(self.fields) > self.MAX_FIELDS:
            self.fields = self.fields[-self.MAX_FIELDS:]
        elif not self.fields:
            self.fields = [config.TILE_GEOMETRY_FIELD]
        # The map field is deliberately *not* forced into `fields`: the map
        # is for orientation and reading the density range, the columns are
        # the comparison.  Tying them meant choosing density on the map
        # silently replaced a figure column.

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
        if r.tile_idx is not None:
            return r.tile_idx
        return _resolved_tile(self.provider, self.date, r.key)

    def tile_index_of(self, region) -> int:
        """The tile index for any region, not just the selected one."""
        if region.tile_idx is not None:
            return region.tile_idx
        return _resolved_tile(self.provider, self.date, region.key)

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
