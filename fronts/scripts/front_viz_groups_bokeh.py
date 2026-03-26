#!/usr/bin/env python
"""
Bokeh-based interactive viewer for front properties from the groups table.

Displays a background oceanographic field with labeled fronts overlaid.
Hovering over a front pixel shows per-front properties in a tooltip.

Usage
-----
    python front_viz_groups_bokeh.py 2012-11-09T12_00_00 \\
        --field gradb2 --bbox 100 200 500 600

    python front_viz_groups_bokeh.py 2012-11-09T12_00_00 \\
        --field gradb2 --latlon_bbox 30.0 -140.0 45.0 -120.0
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, HoverTool, CustomJS,
    MultiSelect, Select, Div,
)
from bokeh.layouts import column, row
from bokeh.io import output_file

from fronts.llc import io as llc_io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GROUP_FRONTS_DIR = os.path.join(
    os.environ.get('OS_OGCM', ''), 'LLC', 'Fronts', 'group_fronts', 'v1'
)


def _safe_timestamp(timestamp):
    """Convert 2012-11-09T12_00_00 -> 20121109T12_00_00 (remove date dashes)."""
    # Split at T, remove dashes from the date part only
    parts = timestamp.split('T', 1)
    date_part = parts[0].replace('-', '')
    return f"{date_part}T{parts[1]}" if len(parts) > 1 else date_part


def _labeled_fronts_path(timestamp, version='1', config_lbl='A'):
    safe_ts = _safe_timestamp(timestamp)
    fname = f'labeled_fronts_global_{safe_ts}_v{version}_bin_{config_lbl}.npy'
    return os.path.join(_GROUP_FRONTS_DIR, fname)


def _properties_path(timestamp, version='1', config_lbl='A'):
    safe_ts = _safe_timestamp(timestamp)
    fname = f'front_properties_{safe_ts}_v{version}_bin_{config_lbl}.parquet'
    return os.path.join(_GROUP_FRONTS_DIR, fname)


# ---------------------------------------------------------------------------
# Lat/lon -> pixel conversion (from front_property_viewer.py)
# ---------------------------------------------------------------------------

def latlon_to_pixel_bbox(lat0, lon0, lat1, lon1):
    """Convert a lat/lon bounding box to pixel (x, y) indices."""
    print("Loading LLC coords for lat/lon -> pixel conversion...")
    coord_ds = llc_io.load_coords()
    lat = coord_ds.lat.values
    lon = coord_ds.lon.values
    lon = ((lon + 180) % 360) - 180

    def nearest(lat_val, lon_val):
        dist = (lat - lat_val) ** 2 + (lon - lon_val) ** 2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx  # (row, col) = (y, x)

    r0, c0 = nearest(lat0, lon0)
    r1, c1 = nearest(lat1, lon1)
    x0, y0 = int(min(c0, c1)), int(min(r0, r1))
    x1, y1 = int(max(c0, c1)), int(max(r0, r1))
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_background_field(timestamp, field, bbox, version='1'):
    """Load a background field cropped to bbox."""
    fname = llc_io.derived_filename(timestamp, field, version=version)
    print(f"Loading {field} from: {fname}")
    ds = xr.open_dataset(fname)
    var = field if field in ds else list(ds.data_vars)[0]
    x0, y0, x1, y1 = bbox
    data = ds[var].values[y0:y1, x0:x1]
    ds.close()
    return data


def load_labeled_fronts(timestamp, bbox, version='1', config_lbl='A'):
    """Load labeled fronts array cropped to bbox."""
    fpath = _labeled_fronts_path(timestamp, version, config_lbl)
    print(f"Loading labeled fronts from: {fpath}")
    arr = np.load(fpath, mmap_mode='r')
    x0, y0, x1, y1 = bbox
    return np.array(arr[y0:y1, x0:x1])


def load_properties(timestamp, version='1', config_lbl='A'):
    """Load the front properties parquet table."""
    fpath = _properties_path(timestamp, version, config_lbl)
    print(f"Loading properties from: {fpath}")
    return pd.read_parquet(fpath)


# ---------------------------------------------------------------------------
# Image building
# ---------------------------------------------------------------------------

def field_to_rgba(data, cmap='grey', percentile=98):
    """Convert a 2D field to an RGBA uint8 image.

    Parameters
    ----------
    data : np.ndarray
        2D field array.
    cmap : str
        'grey' for inverted grayscale, 'divergent' for blue-white-red.
    percentile : float
        Percentile for clipping.
    """
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return np.zeros((*data.shape, 4), dtype=np.uint8)

    if cmap == 'divergent':
        vmax = np.percentile(np.abs(valid), percentile)
        vmin = -vmax
    else:
        vmin = np.percentile(valid, 100 - percentile)
        vmax = np.percentile(valid, percentile)

    # Normalize to [0, 1]; NaN stays NaN (handled below)
    if vmax == vmin:
        norm = np.zeros_like(data)
    else:
        norm = (data - vmin) / (vmax - vmin)
    nan_mask = ~np.isfinite(norm)
    norm = np.nan_to_num(norm, nan=0.0)
    norm = np.clip(norm, 0, 1)

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    if cmap == 'divergent':
        # Blue-white-red (seismic-like)
        # 0.0 -> blue (58, 76, 139)
        # 0.5 -> white (245, 245, 245)
        # 1.0 -> red (139, 58, 58)
        r = np.where(norm < 0.5,
                     58 + (245 - 58) * (norm / 0.5),
                     245 + (139 - 245) * ((norm - 0.5) / 0.5))
        g = np.where(norm < 0.5,
                     76 + (245 - 76) * (norm / 0.5),
                     245 + (58 - 245) * ((norm - 0.5) / 0.5))
        b = np.where(norm < 0.5,
                     139 + (245 - 139) * (norm / 0.5),
                     245 + (58 - 245) * ((norm - 0.5) / 0.5))
        rgba[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
        rgba[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
        rgba[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
    else:
        # Inverted grayscale: white (high norm=0) -> black (low norm=1)
        grey = (255 * (1 - norm)).astype(np.uint8)
        rgba[:, :, 0] = grey
        rgba[:, :, 1] = grey
        rgba[:, :, 2] = grey

    rgba[:, :, 3] = 255
    # NaN pixels -> transparent
    rgba[nan_mask, :] = 0

    return rgba


def fronts_to_rgba(labeled, color='red', alpha=160):
    """Convert labeled fronts to an RGBA overlay.

    Parameters
    ----------
    labeled : np.ndarray
        Integer labeled fronts (0 = background).
    color : str
        'red' or 'yellow'.
    alpha : int
        Alpha value for front pixels (0-255).
    """
    rgba = np.zeros((*labeled.shape, 4), dtype=np.uint8)
    mask = labeled > 0
    if color == 'yellow':
        rgba[mask, 0] = 255
        rgba[mask, 1] = 255
        rgba[mask, 2] = 0
    else:
        rgba[mask, 0] = 255
        rgba[mask, 1] = 0
        rgba[mask, 2] = 0
    rgba[mask, 3] = alpha
    return rgba


def rgba_to_bokeh_image(rgba):
    """Convert (H, W, 4) uint8 RGBA to Bokeh-compatible uint32 image."""
    h, w = rgba.shape[:2]
    img = np.zeros((h, w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((h, w, 4))
    view[:, :, :] = rgba
    return img


# ---------------------------------------------------------------------------
# Properties lookup dict for JS
# ---------------------------------------------------------------------------

# Columns to exclude from the selectable property list
_EXCLUDE_PREFIXES = (
    'gradeta2', 'gradrho2', 'gradsalt2', 'gradtheta2', 'rossby_number',
    'strain_n', 'strain_s',
)

# Un-normalized columns that have _over_f counterparts — exclude the raw versions.
# We match exactly these prefixes + stat suffixes so that e.g. strain_mag_std (no
# _over_f counterpart) is still excluded along with its normalised siblings.
_EXCLUDE_RAW_PREFIXES = ('strain_mag', 'relative_vorticity', 'divergence')

# Fields whose mean/median/p10/p25/p75/p90 should be divided by |coriolis_f_median|
_NORMALIZE_BY_F = ('strain_mag', 'strain_n', 'strain_s',
                    'relative_vorticity', 'divergence')
_NORMALIZE_SUFFIXES = ('_mean', '_median', '_p10', '_p25', '_p75', '_p90')


def _add_normalized_columns(df):
    """Add *_over_f columns for strain/vorticity/divergence normalised by |f|."""
    if 'coriolis_f_median' not in df.columns:
        return df
    abs_f = df['coriolis_f_median'].abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        for prefix in _NORMALIZE_BY_F:
            for suffix in _NORMALIZE_SUFFIXES:
                src = f'{prefix}{suffix}'
                dst = f'{prefix}{suffix}_over_f'
                if src in df.columns:
                    df[dst] = df[src] / abs_f
    return df


def filter_property_columns(columns):
    """Remove excluded columns and return the filtered list."""
    result = []
    for c in columns:
        if c == 'flabel':
            continue
        if any(c.startswith(ex) for ex in _EXCLUDE_PREFIXES):
            continue
        # Exclude raw (un-normalized) strain_mag/relative_vorticity/divergence
        # but keep their *_over_f counterparts
        if any(c.startswith(p) for p in _EXCLUDE_RAW_PREFIXES) and not c.endswith('_over_f'):
            continue
        result.append(c)
    return result


def build_properties_lookup(properties_df, labeled_crop, selected_cols):
    """Build a JSON-serialisable dict: flabel -> {col: value, ...}.

    Only includes labels present in the cropped region.
    """
    labels_in_view = np.unique(labeled_crop)
    labels_in_view = labels_in_view[labels_in_view > 0]

    # Subset to labels in view
    df = properties_df[properties_df['flabel'].isin(labels_in_view)].copy()
    df = df.set_index('flabel')

    # Add normalised columns
    df = _add_normalized_columns(df)

    lookup = {}
    cols_to_include = [c for c in selected_cols if c in df.columns]
    cols_to_include = ['flabel'] + [c for c in cols_to_include if c != 'flabel']

    for flabel, row_data in df.iterrows():
        entry = {'flabel': int(flabel)}
        for c in cols_to_include:
            if c == 'flabel':
                continue
            val = row_data[c]
            if pd.isna(val):
                entry[c] = 'NaN'
            elif isinstance(val, (np.floating, float)):
                entry[c] = f'{val:.4g}'
            else:
                entry[c] = str(val)
        lookup[int(flabel)] = entry

    return lookup


# ---------------------------------------------------------------------------
# Main Bokeh visualization
# ---------------------------------------------------------------------------

def create_visualization(timestamp, field, bbox, config_lbl='B', version='1'):
    """Build and display the Bokeh visualization."""

    # Load data
    bg_data = load_background_field(timestamp, field, bbox, version)
    labeled = load_labeled_fronts(timestamp, bbox, version, config_lbl)
    props_df = load_properties(timestamp, version, config_lbl)

    ny, nx = bg_data.shape
    print(f"Region shape: {ny} x {nx}")

    # Add normalised columns to the DataFrame so we can enumerate them
    props_df = _add_normalized_columns(props_df.copy())

    # Build the selectable property list (filtered)
    all_prop_cols = filter_property_columns(props_df.columns)

    # Default display columns
    default_cols = ['gradb2_median', 'strain_mag_median_over_f']
    default_cols = [c for c in default_cols if c in all_prop_cols]

    # Build images
    bg_grey_rgba = field_to_rgba(bg_data, cmap='grey')
    bg_div_rgba = field_to_rgba(bg_data, cmap='divergent')
    fronts_red_rgba = fronts_to_rgba(labeled, color='red')
    fronts_yellow_rgba = fronts_to_rgba(labeled, color='yellow')

    bg_grey_img = rgba_to_bokeh_image(bg_grey_rgba)
    bg_div_img = rgba_to_bokeh_image(bg_div_rgba)
    fronts_red_img = rgba_to_bokeh_image(fronts_red_rgba)
    fronts_yellow_img = rgba_to_bokeh_image(fronts_yellow_rgba)

    # Build properties lookup (include all selectable columns)
    lookup = build_properties_lookup(props_df, labeled, all_prop_cols)

    # Store labeled array as flat list for JS access
    # (labeled is int64; convert to int32 for JS)
    labeled_flat = labeled.astype(np.int32).ravel().tolist()

    # --- Bokeh figure ---
    x0, y0, x1, y1 = bbox
    p = figure(
        title=f"{field} | {timestamp} | bbox=({x0},{y0},{x1},{y1})",
        width=900, height=int(900 * ny / nx) if nx > 0 else 700,
        x_range=(0, nx), y_range=(0, ny),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
    )
    p.axis.visible = False
    p.grid.visible = False

    # Background image source
    bg_source = ColumnDataSource(data=dict(
        image=[bg_grey_img],
        image_grey=[bg_grey_img],
        image_div=[bg_div_img],
    ))
    p.image_rgba(image='image', x=0, y=0, dw=nx, dh=ny, source=bg_source)

    # Fronts overlay source
    fronts_source = ColumnDataSource(data=dict(
        image=[fronts_red_img],
        image_red=[fronts_red_img],
        image_yellow=[fronts_yellow_img],
    ))
    p.image_rgba(image='image', x=0, y=0, dw=nx, dh=ny, source=fronts_source)

    # --- Tooltip div (updated via JS on mouse move) ---
    tooltip_div = Div(
        text="<b>Hover over a front to see properties</b>",
        width=350, height=400,
        styles={'font-size': '13px', 'padding': '8px',
                'border': '1px solid #ccc', 'background': '#f9f9f9',
                'overflow-y': 'auto'},
    )

    # --- MultiSelect for properties ---
    prop_options = sorted(all_prop_cols)
    multi_select = MultiSelect(
        title="Properties to display:",
        value=default_cols,
        options=[(c, c) for c in prop_options],
        size=min(20, len(prop_options)),
        width=350,
    )

    # --- Colormap toggle ---
    cmap_select = Select(
        title="Colormap:",
        value="Greys",
        options=["Greys", "Divergent (seismic)"],
        width=350,
    )

    # Colormap toggle JS callback
    cmap_cb = CustomJS(args=dict(
        bg_source=bg_source,
        fronts_source=fronts_source,
    ), code="""
        const cmap = cb_obj.value;
        if (cmap === "Greys") {
            bg_source.data['image'] = bg_source.data['image_grey'];
            fronts_source.data['image'] = fronts_source.data['image_red'];
        } else {
            bg_source.data['image'] = bg_source.data['image_div'];
            fronts_source.data['image'] = fronts_source.data['image_yellow'];
        }
        bg_source.change.emit();
        fronts_source.change.emit();
    """)
    cmap_select.js_on_change('value', cmap_cb)

    # --- Hover callback using CustomJS on mouse move ---
    # We embed the labeled array and properties lookup in JS
    hover_cb = CustomJS(args=dict(
        tooltip_div=tooltip_div,
        multi_select=multi_select,
        p=p,
    ), code="""
        // Embedded data
        const labeled_flat = cb_obj.labeled_flat;
        const props_lookup = cb_obj.props_lookup;
        const nx = cb_obj.nx;
        const ny = cb_obj.ny;

        // Get mouse position in data coordinates
        const geom = cb_data.geometry;
        const ix = Math.floor(geom.x);
        const iy = Math.floor(geom.y);

        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) {
            tooltip_div.text = "<i>Outside region</i>";
            return;
        }

        const flabel = labeled_flat[iy * nx + ix];

        if (flabel === 0) {
            tooltip_div.text = "<i>No front at (" + ix + ", " + iy + ")</i>";
            return;
        }

        const entry = props_lookup[flabel.toString()];
        if (!entry) {
            tooltip_div.text = "<b>flabel: " + flabel + "</b><br><i>No properties available</i>";
            return;
        }

        const selected_props = multi_select.value;
        let html = "<b>flabel: " + flabel + "</b><br>";
        html += "<b>pixel: (" + ix + ", " + iy + ")</b><br><hr>";
        for (const prop of selected_props) {
            if (entry[prop] !== undefined) {
                html += "<b>" + prop + ":</b> " + entry[prop] + "<br>";
            }
        }
        tooltip_div.text = html;
    """)

    # We need to attach the large data to the callback object.
    # Bokeh CustomJS args can handle dicts/lists.
    # However, for large arrays we'll use a different approach:
    # embed them as properties on the callback itself.
    # Actually, let's use a hidden ColumnDataSource approach or
    # pass them through a Div with JSON.

    # Better approach: use a HoverTool with a custom callback
    # that reads from embedded data sources.

    # We'll embed the labeled array and lookup into a JS-accessible source.
    # For the labeled array, since it can be very large, we'll use a
    # ColumnDataSource with a single column.
    label_source = ColumnDataSource(data=dict(
        labels=labeled_flat,
    ))

    # For the properties lookup, serialize to JSON and embed via Div
    props_json = json.dumps(lookup)

    # Rebuild the hover callback with proper data access
    hover_cb = CustomJS(args=dict(
        tooltip_div=tooltip_div,
        multi_select=multi_select,
        label_source=label_source,
    ), code=f"""
        const props_lookup = {props_json};
        const nx = {nx};
        const ny = {ny};
        const labeled = label_source.data['labels'];

        const geom = cb_data.geometry;
        const ix = Math.floor(geom.x);
        const iy = Math.floor(geom.y);

        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) {{
            tooltip_div.text = "<i>Outside region</i>";
            return;
        }}

        const flabel = labeled[iy * nx + ix];

        if (flabel === 0) {{
            tooltip_div.text = "<i>No front at (" + ix + ", " + iy + ")</i>";
            return;
        }}

        const entry = props_lookup[flabel.toString()];
        if (!entry) {{
            tooltip_div.text = "<b>flabel: " + flabel + "</b><br><i>No properties available</i>";
            return;
        }}

        const selected_props = multi_select.value;
        let html = "<b>flabel: " + flabel + "</b><br>";
        html += "<b>pixel: (" + ix + ", " + iy + ")</b><br><hr>";
        for (const prop of selected_props) {{
            if (entry[prop] !== undefined) {{
                html += "<b>" + prop + ":</b> " + entry[prop] + "<br>";
            }}
        }}
        tooltip_div.text = html;
    """)

    # Attach as a HoverTool
    hover_tool = HoverTool(callback=hover_cb, tooltips=None)
    p.add_tools(hover_tool)

    # --- Layout ---
    controls = column(
        Div(text="<h3>Front Property Viewer</h3>"),
        cmap_select,
        multi_select,
        Div(text="<hr>"),
        tooltip_div,
    )
    layout = row(p, controls)

    output_file(f"front_viz_{field}_{timestamp}.html",
                title=f"Front Viz: {field} | {timestamp}")
    show(layout)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Bokeh viewer for front properties from groups table',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('timestamp', type=str,
                   help="LLC timestamp, e.g. '2012-11-09T12_00_00'")
    p.add_argument('--field', type=str, default='gradb2',
                   help='Background field to display')
    p.add_argument('--config_lbl', type=str, default='A',
                   help='Config label for fronts (e.g. A, B)')
    p.add_argument('--version', type=str, default='1',
                   help='Data version string')

    bbox_group = p.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument('--bbox', type=int, nargs=4,
                            metavar=('X0', 'Y0', 'X1', 'Y1'),
                            help='Pixel bounding box')
    bbox_group.add_argument('--latlon_bbox', type=float, nargs=4,
                            metavar=('LAT0', 'LON0', 'LAT1', 'LON1'),
                            help='Lat/lon bounding box (converted to pixels)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.bbox is not None:
        bbox = tuple(args.bbox)
    else:
        lat0, lon0, lat1, lon1 = args.latlon_bbox
        bbox = latlon_to_pixel_bbox(lat0, lon0, lat1, lon1)
        print(f"Lat/lon bbox converted to pixel bbox: {bbox}")

    create_visualization(
        timestamp=args.timestamp,
        field=args.field,
        bbox=bbox,
        config_lbl=args.config_lbl,
        version=args.version,
    )


if __name__ == '__main__':
    main()
