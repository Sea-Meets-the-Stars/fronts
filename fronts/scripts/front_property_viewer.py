#!/usr/bin/env python
"""
PyQt6 + pyqtgraph GUI for viewing front properties in a regional bounding box.

Displays four panels in a 2x2 grid:
  - Panel 0: gradb2 field with binary fronts overlaid
  - Panels 1-3: user-specified derived property fields

All panels are linked: pan/zoom in one panel updates all others.

Usage
-----
    python front_property_viewer.py <timestamp> \\
        --fields vorticity strain_rate OW \\
        --bbox 100 200 500 600 \\
        [--config_lbl A] [--version 1]

    python front_property_viewer.py <timestamp> \\
        --fields vorticity strain_rate OW \\
        --latlon_bbox 30.0 -140.0 45.0 -120.0 \\
        [--config_lbl A] [--version 1]
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QStatusBar,
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from fronts.llc import io as llc_io
from fronts.finding import io as finding_io
from fronts.scripts.viz_utils import (
    make_colormap, compute_levels, make_fronts_rgba, make_nan_rgba,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latlon_to_pixel_bbox(lat0, lon0, lat1, lon1):
    """Convert a lat/lon bounding box to pixel (x,y) indices.

    Uses the LLC coordinate file loaded via :func:`fronts.llc.io.load_coords`.

    Parameters
    ----------
    lat0, lon0 : float
        Lower-left corner (degrees).
    lat1, lon1 : float
        Upper-right corner (degrees).

    Returns
    -------
    tuple[int, int, int, int]
        (x0, y0, x1, y1) pixel indices into the LLC global grid.
    """
    print("Loading LLC coords for lat/lon -> pixel conversion...")
    coord_ds = llc_io.load_coords()
    lat = coord_ds.lat.values  # shape (ny, nx)
    lon = coord_ds.lon.values

    # Wrap lon to [-180, 180] to match convention
    lon = ((lon + 180) % 360) - 180

    # Distance metric to find the nearest pixel for each corner
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
# Main Window
# ---------------------------------------------------------------------------

class FrontPropertyViewer(QMainWindow):
    """Four-panel viewer: gradb2+fronts and three derived property fields."""

    # Grid layout positions for the 4 panels: (row, data_col, cbar_col)
    _PANEL_GRID = [(0, 0, 1), (0, 2, 3), (1, 0, 1), (1, 2, 3)]

    def __init__(self, timestamp, bbox, fields, config_lbl='A', version='1'):
        """
        Parameters
        ----------
        timestamp : str
            LLC timestamp, e.g. '2012-11-09T12_00_00'.
        bbox : tuple[int,int,int,int]
            Pixel bounding box (x0, y0, x1, y1).
        fields : list[str]
            Three field names for panels 1-3.
        config_lbl : str
            Config label for the binary fronts file (e.g. 'A').
        version : str
            Data version string (e.g. '1').
        """
        super().__init__()
        self.timestamp = timestamp
        self.bbox = bbox          # (x0, y0, x1, y1)
        self.fields = fields      # list of 3 field names
        self.config_lbl = config_lbl
        self.version = version

        # Panel titles: panel 0 always 'gradb2'
        self.panel_titles = ['gradb2'] + list(fields)

        # Fields that use a divergent (blue-white-red) colormap centered on 0
        self._DIVERGENT_FIELDS = {'okubo_weiss', 'vorticity', 'divergence', 'frontogenesis_tendency',
            'relative_vorticity'}

        # Data arrays (loaded later)
        self.panel_data = [None, None, None, None]
        self.fronts_data = None

        # pyqtgraph items per panel
        self.image_items = [None, None, None, None]
        self.cbar_items = [None, None, None, None]
        self.nan_image_items = [None, None, None, None]
        self.fronts_image_items = [None, None, None, None]
        self.panels = []   # list of PlotItem

        self._init_ui()
        self._load_all_data()
        self._render_all_panels()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle('Front Property Viewer')
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Control bar ---
        ctrl = QHBoxLayout()

        self.fronts_checkbox = QCheckBox('Show Fronts (red)')
        self.fronts_checkbox.setChecked(True)
        self.fronts_checkbox.stateChanged.connect(self._toggle_fronts)
        ctrl.addWidget(self.fronts_checkbox)

        ctrl.addSpacing(20)

        contrast_label = QLabel('Contrast:')
        ctrl.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(99)
        self.contrast_slider.setValue(95)
        self.contrast_slider.setMaximumWidth(200)
        self.contrast_slider.valueChanged.connect(self._update_contrast)
        ctrl.addWidget(self.contrast_slider)

        self.contrast_label = QLabel('95%')
        ctrl.addWidget(self.contrast_label)

        ctrl.addSpacing(20)

        reset_btn = QPushButton('Reset View')
        reset_btn.clicked.connect(self._reset_view)
        ctrl.addWidget(reset_btn)

        ctrl.addStretch()

        info = QLabel(
            f'Timestamp: {self.timestamp}  |  '
            f'Config: {self.config_lbl}  |  '
            f'Version: {self.version}  |  '
            f'BBox: {self.bbox}'
        )
        ctrl.addWidget(info)

        main_layout.addLayout(ctrl)

        # --- 2x2 panel grid ---
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')

        # Fixed widths/heights for axis areas so every panel reserves the
        # SAME space for axis decorations. Without this, hiding the y-axis
        # (or even just hiding its tick values) on right-column panels lets
        # those panels expand into the freed space and render larger than
        # the left-column panels.
        Y_AXIS_WIDTH = 60
        X_AXIS_HEIGHT = 30

        for i, (row, dcol, _cbarcol) in enumerate(self._PANEL_GRID):
            panel = self.graphics_widget.addPlot(row=row, col=dcol)
            panel.showGrid(x=False, y=False)
            panel.setLabel('bottom', 'i (pixel)')
            panel.setTitle(self.panel_titles[i])

            left_axis = panel.getAxis('left')
            if dcol == 0:
                panel.setLabel('left', 'j (pixel)')
            else:
                left_axis.setStyle(showValues=False)
            left_axis.setWidth(Y_AXIS_WIDTH)
            panel.getAxis('bottom').setHeight(X_AXIS_HEIGHT)

            # Aspect-lock every panel so the data ratio is the same in all
            # four panels; linking still keeps pan/zoom in sync.
            panel.setAspectLocked(True)

            self.panels.append(panel)

        # Link all panel axes to panel 0 so pan/zoom stays in sync
        for panel in self.panels[1:]:
            panel.setXLink(self.panels[0])
            panel.setYLink(self.panels[0])

        # Equal stretch for the two data columns so panels are the same size.
        # Colorbar columns get a fixed width so differing tick-label widths
        # cannot make left/right data columns unequal.
        layout = self.graphics_widget.ci.layout
        layout.setColumnStretchFactor(0, 1)  # left data panel
        layout.setColumnStretchFactor(1, 0)  # left colorbar
        layout.setColumnStretchFactor(2, 1)  # right data panel
        layout.setColumnStretchFactor(3, 0)  # right colorbar
        cbar_width = 80  # fixed pixel width for colorbar columns
        layout.setColumnFixedWidth(1, cbar_width)
        layout.setColumnFixedWidth(3, cbar_width)
        layout.setRowStretchFactor(0, 1)
        layout.setRowStretchFactor(1, 1)

        main_layout.addWidget(self.graphics_widget, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Loading data...')

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all_data(self):
        x0, y0, x1, y1 = self.bbox

        # --- gradb2 ---
        try:
            fname = llc_io.derived_filename(self.timestamp, 'gradb2',
                                            version=self.version)
            print(f"Loading gradb2 from: {fname}")
            ds = xr.open_dataset(fname)
            var = 'gradb2' if 'gradb2' in ds else list(ds.data_vars)[0]
            self.panel_data[0] = ds[var].values[y0:y1, x0:x1]
            print(f"  gradb2 shape: {self.panel_data[0].shape}")
        except Exception as e:
            print(f"ERROR loading gradb2: {e}")
            self.status_bar.showMessage(f'ERROR loading gradb2: {e}')

        # --- binary fronts ---
        try:
            fronts_full = finding_io.load_binary_fronts(
                self.timestamp, self.config_lbl, self.version)
            self.fronts_data = fronts_full[y0:y1, x0:x1]
            print(f"  fronts shape: {self.fronts_data.shape}, "
                  f"front pixels: {np.sum(self.fronts_data > 0)}")
        except Exception as e:
            print(f"ERROR loading fronts: {e}")
            self.status_bar.showMessage(f'ERROR loading fronts: {e}')

        # --- three property fields ---
        for i, field in enumerate(self.fields):
            try:
                fname = llc_io.derived_filename(self.timestamp, field,
                                                version=self.version)
                print(f"Loading {field} from: {fname}")
                ds = xr.open_dataset(fname)
                var = field if field in ds else list(ds.data_vars)[0]
                self.panel_data[i + 1] = ds[var].values[y0:y1, x0:x1]
                print(f"  {field} shape: {self.panel_data[i+1].shape}")
            except Exception as e:
                print(f"ERROR loading {field}: {e}")
                self.status_bar.showMessage(f'ERROR loading {field}: {e}')

        self.status_bar.showMessage(
            f'Loaded | timestamp: {self.timestamp} | '
            f'config: {self.config_lbl} | bbox: {self.bbox}'
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_all_panels(self):
        for i in range(4):
            self._render_panel(i)
        # Auto-range panel 0 only; linked axes propagate to the rest
        self.panels[0].autoRange()

    def _render_panel(self, idx):
        """Render data into panel *idx*, adding colorbar and overlays."""
        data = self.panel_data[idx]
        panel = self.panels[idx]
        row, dcol, cbarcol = self._PANEL_GRID[idx]

        # Clear existing items
        panel.clear()
        if self.cbar_items[idx] is not None:
            try:
                self.graphics_widget.removeItem(self.cbar_items[idx])
            except Exception:
                pass
            self.cbar_items[idx] = None

        if data is None:
            panel.setTitle(f'{self.panel_titles[idx]}  [NO DATA]')
            return

        percentile = self.contrast_slider.value()
        divergent = self.panel_titles[idx] in self._DIVERGENT_FIELDS
        colormap = make_colormap(divergent=divergent)
        vmin, vmax = compute_levels(data, percentile, divergent=divergent)
        print(f"vmin: {vmin}, vmax: {vmax}")

        # Main image
        img_item = pg.ImageItem()
        img_item.setImage(data.T)
        img_item.setColorMap(colormap)
        img_item.setLevels([vmin, vmax])
        panel.addItem(img_item)
        self.image_items[idx] = img_item

        # Colorbar
        cbar = pg.ColorBarItem(
            values=(vmin, vmax),
            colorMap=colormap,
            label=self.panel_titles[idx],
            interactive=False,
            width=15,
        )
        cbar.setImageItem(img_item)
        self.graphics_widget.addItem(cbar, row=row, col=cbarcol)
        self.cbar_items[idx] = cbar

        # NaN overlay (dark green)
        nan_rgba = make_nan_rgba(data)
        if nan_rgba is not None:
            nan_item = pg.ImageItem()
            nan_item.setImage(nan_rgba.transpose(1, 0, 2))
            panel.addItem(nan_item)
            self.nan_image_items[idx] = nan_item

        # Fronts overlay (all panels)
        if self.fronts_data is not None:
            fronts_rgba = make_fronts_rgba(self.fronts_data, divergent=divergent)
            fronts_item = pg.ImageItem()
            fronts_item.setImage(fronts_rgba.transpose(1, 0, 2))
            self.fronts_image_items[idx] = fronts_item
            if self.fronts_checkbox.isChecked():
                panel.addItem(fronts_item)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _update_contrast(self, value):
        self.contrast_label.setText(f'{value}%')
        for idx in range(4):
            data = self.panel_data[idx]
            if data is None or self.image_items[idx] is None:
                continue
            divergent = self.panel_titles[idx] in self._DIVERGENT_FIELDS
            vmin, vmax = compute_levels(data, value, divergent=divergent)
            self.image_items[idx].setLevels([vmin, vmax])
            if self.cbar_items[idx] is not None:
                self.cbar_items[idx].setLevels(values=(vmin, vmax))

    def _toggle_fronts(self, state):
        for idx, fronts_item in enumerate(self.fronts_image_items):
            if fronts_item is None:
                continue
            panel = self.panels[idx]
            if state == Qt.CheckState.Checked.value:
                if fronts_item.scene() is None:
                    panel.addItem(fronts_item)
            else:
                panel.removeItem(fronts_item)

    def _reset_view(self):
        self.panels[0].autoRange()
        self.status_bar.showMessage('View reset', 2000)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(
        description='View front properties in a regional bounding box',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('timestamp', type=str, nargs='?', default=None,
                   help="LLC timestamp, e.g. '2012-11-09T12_00_00'")
    p.add_argument('--fields', type=str, nargs=3, required=False,
                   metavar=('F1', 'F2', 'F3'),
                   help='Three derived field names for panels 1-3')
    p.add_argument('--config_lbl', type=str, default='B',
                   help='Config label for binary fronts file (e.g. A)')
    p.add_argument('--version', type=str, default='1',
                   help='Data version string')

    # Bounding box: pixel or lat/lon (one required at runtime)
    bbox_group = p.add_mutually_exclusive_group(required=False)
    bbox_group.add_argument('--bbox', type=int, nargs=4,
                            metavar=('X0', 'Y0', 'X1', 'Y1'),
                            help='Pixel bounding box')
    bbox_group.add_argument('--latlon_bbox', type=float, nargs=4,
                            metavar=('LAT0', 'LON0', 'LAT1', 'LON1'),
                            help='Lat/lon bounding box (converted to pixels)')
    return p.parse_args()


def main(args):
    # Runtime validation (argparse required=True would break -h)
    if args.timestamp is None:
        print('error: timestamp is required', file=sys.stderr)
        sys.exit(2)
    if args.fields is None:
        print('error: --fields is required', file=sys.stderr)
        sys.exit(2)
    if args.bbox is None and args.latlon_bbox is None:
        print('error: one of --bbox or --latlon_bbox is required', file=sys.stderr)
        sys.exit(2)

    if args.bbox is not None:
        bbox = tuple(args.bbox)
    else:
        lat0, lon0, lat1, lon1 = args.latlon_bbox
        bbox = latlon_to_pixel_bbox(lat0, lon0, lat1, lon1)
        print(f"Lat/lon bbox converted to pixel bbox: {bbox}")

    app = QApplication(sys.argv)

    viewer = FrontPropertyViewer(
        timestamp=args.timestamp,
        bbox=bbox,
        fields=args.fields,
        config_lbl=args.config_lbl,
        version=args.version,
    )
    viewer.show()
    sys.exit(app.exec())

# fronts_property_viewer 2012-11-09T12_00_00 --fields mld strain_mag relative_vorticity --bbox 16200 4900 16800 5700 --version 2 --config_lbl D