#!/usr/bin/env python
"""
PyQt6 + pyqtgraph GUI for visualizing global Divb2 data with detected fronts.

This application loads:
1. NetCDF file containing Divb2 data
2. .npy file containing front detection mask (1=front, 0=no front)

And displays them in an interactive viewer with pan/zoom capabilities.

Usage:
    python global_divb2_viewer.py <divb2_file> <fronts_file> [--downsample N]

Example:
    python global_divb2_viewer.py LLC4320_2012-11-09T12_00_00_divb2.nc fronts.npy --downsample 5
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QFileDialog, QStatusBar
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class Divb2Viewer(QMainWindow):
    """Main window for Divb2 front visualization."""

    def __init__(self, divb2_file=None, fronts_file=None, downsample=1):
        super().__init__()

        self.divb2_file = divb2_file
        self.fronts_file = fronts_file
        self.downsample = downsample

        # Data storage
        self.divb2_data = None
        self.fronts_data = None
        self.ds = None

        # Plot items
        self.divb2_image = None
        self.nan_image = None  # Green overlay for NaN values
        self.fronts_image = None
        self.colorbar = None  # Colorbar for divb2 values

        self.init_ui()

        # Load data if files provided
        if divb2_file and fronts_file:
            self.load_data(divb2_file, fronts_file, downsample)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Global Divb2 Front Viewer')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_layout = QHBoxLayout()

        # Load buttons
        self.load_divb2_btn = QPushButton('Load Divb2 NetCDF')
        self.load_divb2_btn.clicked.connect(self.load_divb2_dialog)
        control_layout.addWidget(self.load_divb2_btn)

        self.load_fronts_btn = QPushButton('Load Fronts .npy')
        self.load_fronts_btn.clicked.connect(self.load_fronts_dialog)
        control_layout.addWidget(self.load_fronts_btn)

        control_layout.addSpacing(20)

        # Show/hide fronts checkbox
        self.show_fronts_checkbox = QCheckBox('Show Fronts')
        self.show_fronts_checkbox.setChecked(True)
        self.show_fronts_checkbox.stateChanged.connect(self.toggle_fronts)
        control_layout.addWidget(self.show_fronts_checkbox)

        control_layout.addSpacing(20)

        # Log scale toggle checkbox
        self.log_scale_checkbox = QCheckBox('Log₁₀ Scale')
        self.log_scale_checkbox.setChecked(True)  # Default to log scale
        self.log_scale_checkbox.stateChanged.connect(self.toggle_log_scale)
        control_layout.addWidget(self.log_scale_checkbox)

        control_layout.addSpacing(20)

        # Reset view button
        self.reset_view_btn = QPushButton('Reset View')
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)

        # Adjust limits to current view button
        self.adjust_to_view_btn = QPushButton('Adjust Limits to View')
        self.adjust_to_view_btn.clicked.connect(self.adjust_limits_to_view)
        control_layout.addWidget(self.adjust_to_view_btn)

        control_layout.addSpacing(20)

        # Contrast controls
        contrast_label = QLabel('Contrast:')
        control_layout.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(99)
        self.contrast_slider.setValue(95)
        self.contrast_slider.setMaximumWidth(200)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        control_layout.addWidget(self.contrast_slider)

        self.contrast_value_label = QLabel('95%')
        control_layout.addWidget(self.contrast_value_label)

        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Create pyqtgraph graphics layout with plot and colorbar
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')

        # Create plot item
        self.plot_widget = self.graphics_widget.addPlot(row=0, col=0)
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.showGrid(x=False, y=False)

        # Set labels
        self.plot_widget.setLabel('left', 'j (grid index)')
        self.plot_widget.setLabel('bottom', 'i (grid index)')
        self.plot_widget.setTitle('Divb2 Field with Fronts')

        main_layout.addWidget(self.graphics_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready - Load Divb2 and Fronts files to begin')

    def load_divb2_dialog(self):
        """Open file dialog to load Divb2 NetCDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Divb2 NetCDF File',
            '',
            'NetCDF Files (*.nc);;All Files (*)'
        )

        if file_path:
            self.divb2_file = file_path
            if self.fronts_file:
                self.load_data(self.divb2_file, self.fronts_file, self.downsample)
            else:
                self.status_bar.showMessage(f'Loaded Divb2 file: {Path(file_path).name} - Now load Fronts file')

    def load_fronts_dialog(self):
        """Open file dialog to load fronts .npy file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Fronts .npy File',
            '',
            'NumPy Files (*.npy);;All Files (*)'
        )

        if file_path:
            self.fronts_file = file_path
            if self.divb2_file:
                self.load_data(self.divb2_file, self.fronts_file, self.downsample)
            else:
                self.status_bar.showMessage(f'Loaded Fronts file: {Path(file_path).name} - Now load Divb2 file')

    def load_data(self, divb2_file, fronts_file, downsample=1):
        """
        Load Divb2 and fronts data from files.

        Parameters:
            divb2_file (str): Path to NetCDF file with Divb2 data
            fronts_file (str): Path to .npy file with front mask
            downsample (int): Downsampling factor
        """
        try:
            self.status_bar.showMessage('Loading data...')

            # Load Divb2 from NetCDF
            print(f"Loading Divb2 from: {divb2_file}")
            self.ds = xr.open_dataset(divb2_file)

            if 'Divb2' not in self.ds:
                self.status_bar.showMessage('ERROR: Divb2 variable not found in NetCDF file')
                return

            self.divb2_data = self.ds.Divb2.values

            # Apply downsampling
            if downsample > 1:
                print(f"Downsampling by factor of {downsample}")
                self.divb2_data = self.divb2_data[::downsample, ::downsample]

            print(f"Divb2 shape: {self.divb2_data.shape}")
            print(f"Divb2 range: [{np.nanmin(self.divb2_data):.2e}, {np.nanmax(self.divb2_data):.2e}]")

            # Load fronts from .npy
            print(f"Loading fronts from: {fronts_file}")
            self.fronts_data = np.load(fronts_file)

            # Apply downsampling to fronts
            if downsample > 1:
                self.fronts_data = self.fronts_data[::downsample, ::downsample]

            print(f"Fronts shape: {self.fronts_data.shape}")
            print(f"Number of front pixels: {np.sum(self.fronts_data == 1)}")

            # Check dimensions match
            if self.divb2_data.shape != self.fronts_data.shape:
                self.status_bar.showMessage(
                    f'WARNING: Shape mismatch - Divb2: {self.divb2_data.shape}, Fronts: {self.fronts_data.shape}'
                )
                print(f"WARNING: Shape mismatch!")

            # Plot the data
            self.plot_data()

            self.status_bar.showMessage(
                f'Loaded: {Path(divb2_file).name} | '
                f'Shape: {self.divb2_data.shape} | '
                f'Fronts: {np.sum(self.fronts_data == 1)} pixels'
            )

        except Exception as e:
            error_msg = f'Error loading data: {str(e)}'
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()

    def plot_data(self):
        """Plot Divb2 data and fronts."""
        if self.divb2_data is None:
            return

        # Clear previous plots
        self.plot_widget.clear()

        # Prepare Divb2 data for display (log or linear scale)
        if self.log_scale_checkbox.isChecked():
            plot_data = np.log10(np.abs(self.divb2_data))
            plot_data[np.isinf(plot_data)] = np.nan
        else:
            plot_data = self.divb2_data.copy()

        # Track NaN positions for green overlay
        nan_mask = np.isnan(plot_data)

        # Get contrast percentile (only from valid data)
        percentile = self.contrast_slider.value()
        vmin = np.nanpercentile(plot_data, 100 - percentile)
        vmax = np.nanpercentile(plot_data, percentile)

        scale_type = 'log₁₀ scale' if self.log_scale_checkbox.isChecked() else 'linear scale'
        print(f"Display range: [{vmin:.2e}, {vmax:.2e}] ({scale_type})")

        # Create grayscale image for Divb2
        # pyqtgraph ImageItem expects data in (rows, cols) = (y, x)
        self.divb2_image = pg.ImageItem()
        self.divb2_image.setImage(plot_data.T)  # Transpose for correct orientation

        # Set grayscale colormap (inverted: white to black)
        # Higher values = darker (inverted grayscale)
        colors = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.ubyte)
        colormap = pg.ColorMap(pos=np.array([0.0, 1.0]), color=colors)
        self.divb2_image.setColorMap(colormap)

        # Set levels (contrast)
        self.divb2_image.setLevels([vmin, vmax])

        self.plot_widget.addItem(self.divb2_image)

        # Add colorbar for divb2 values
        if self.colorbar is not None:
            # Remove old colorbar if it exists
            self.graphics_widget.removeItem(self.colorbar)

        # Create colorbar with appropriate label
        label = 'log₁₀(Divb2)' if self.log_scale_checkbox.isChecked() else 'Divb2'
        self.colorbar = pg.ColorBarItem(
            values=(vmin, vmax),
            colorMap=colormap,
            label=label,
            interactive=False,
            width=15
        )
        self.colorbar.setImageItem(self.divb2_image)

        # Add colorbar to the right of the plot
        self.graphics_widget.addItem(self.colorbar, row=0, col=1)

        # Create green overlay for NaN values (land/missing data)
        if np.any(nan_mask):
            # Create RGBA image for NaN overlay
            nan_rgba = np.zeros((*self.divb2_data.shape, 4), dtype=np.ubyte)

            # Set dark green color for land
            nan_rgba[:, :, 0] = 0    # Red channel
            nan_rgba[:, :, 1] = 100  # Green channel (darker green)
            nan_rgba[:, :, 2] = 0    # Blue channel

            # Set alpha: opaque where NaN, transparent elsewhere
            nan_rgba[:, :, 3] = nan_mask.astype(np.ubyte) * 255  # Fully opaque for NaN

            # Create ImageItem for NaN overlay
            self.nan_image = pg.ImageItem()
            self.nan_image.setImage(nan_rgba.transpose(1, 0, 2))  # Transpose to match orientation

            self.plot_widget.addItem(self.nan_image)

        # Create fronts overlay as semi-transparent red image
        if self.fronts_data is not None:
            # Create RGBA image for fronts overlay
            # Shape: (height, width, 4) for RGBA
            fronts_rgba = np.zeros((*self.fronts_data.shape, 4), dtype=np.ubyte)

            # Set red channel where fronts exist
            fronts_rgba[:, :, 0] = 255  # Red channel
            fronts_rgba[:, :, 1] = 0    # Green channel
            fronts_rgba[:, :, 2] = 0    # Blue channel

            # Set alpha channel: transparent where no front, semi-transparent where front
            # Alpha value of 60 gives nice visibility without overwhelming the background
            fronts_rgba[:, :, 3] = (self.fronts_data > 0).astype(np.ubyte) * 120

            # Create ImageItem for fronts overlay
            self.fronts_image = pg.ImageItem()
            self.fronts_image.setImage(fronts_rgba.transpose(1, 0, 2))  # Transpose to match orientation

            # Add to plot
            if self.show_fronts_checkbox.isChecked():
                self.plot_widget.addItem(self.fronts_image)

        # Auto-range to fit data
        self.plot_widget.autoRange()

    def update_contrast(self, value):
        """Update the contrast/levels of the Divb2 image."""
        self.contrast_value_label.setText(f'{value}%')

        if self.divb2_data is not None and self.divb2_image is not None:
            # Recalculate levels (log or linear)
            if self.log_scale_checkbox.isChecked():
                plot_data = np.log10(np.abs(self.divb2_data))
                plot_data[np.isinf(plot_data)] = np.nan
            else:
                plot_data = self.divb2_data.copy()

            vmin = np.nanpercentile(plot_data, 100 - value)
            vmax = np.nanpercentile(plot_data, value)

            self.divb2_image.setLevels([vmin, vmax])

            # Update colorbar to reflect new limits
            if self.colorbar is not None:
                self.colorbar.setLevels(values=(vmin, vmax))

    def toggle_fronts(self, state):
        """Show or hide the fronts overlay."""
        if self.fronts_image is not None:
            if state == Qt.CheckState.Checked.value:
                # Add fronts to plot if not already there
                if self.fronts_image.scene() is None:
                    self.plot_widget.addItem(self.fronts_image)
            else:
                # Remove fronts from plot
                self.plot_widget.removeItem(self.fronts_image)

    def toggle_log_scale(self, state):
        """Toggle between log and linear scale."""
        if self.divb2_data is not None:
            # Replot with new scale
            self.plot_data()

    def reset_view(self):
        """Reset the view to show all data."""
        self.plot_widget.autoRange()
        self.status_bar.showMessage('View reset to full extent', 2000)

    def adjust_limits_to_view(self):
        """Adjust limits based on data visible in current view."""
        if self.divb2_data is None or self.divb2_image is None:
            return

        # Get current view range
        view_range = self.plot_widget.viewRange()
        x_range = view_range[0]  # [xmin, xmax]
        y_range = view_range[1]  # [ymin, ymax]

        # Convert to array indices (accounting for transpose)
        x_min = max(0, int(np.floor(x_range[0])))
        x_max = min(self.divb2_data.shape[1], int(np.ceil(x_range[1])))
        y_min = max(0, int(np.floor(y_range[0])))
        y_max = min(self.divb2_data.shape[0], int(np.ceil(y_range[1])))

        # Extract visible data
        visible_data = self.divb2_data[y_min:y_max, x_min:x_max]

        # Apply transform (log or linear)
        if self.log_scale_checkbox.isChecked():
            visible_plot_data = np.log10(np.abs(visible_data))
            visible_plot_data[np.isinf(visible_plot_data)] = np.nan
            scale_label = 'log₁₀ scale'
        else:
            visible_plot_data = visible_data.copy()
            scale_label = 'linear scale'

        # Calculate new levels from visible data
        percentile = self.contrast_slider.value()
        vmin = np.nanpercentile(visible_plot_data, 100 - percentile)
        vmax = np.nanpercentile(visible_plot_data, percentile)

        # Update image levels
        self.divb2_image.setLevels([vmin, vmax])

        # Update colorbar to reflect new limits
        if self.colorbar is not None:
            self.colorbar.setLevels(values=(vmin, vmax))

        self.status_bar.showMessage(
            f'Limits adjusted to view: [{vmin:.2e}, {vmax:.2e}] ({scale_label})',
            3000
        )
        print(f"Adjusted limits to view range: [{vmin:.2e}, {vmax:.2e}] ({scale_label})")


def main():
    """Main function to run the GUI application."""
    parser = argparse.ArgumentParser(
        description='Interactive viewer for global Divb2 data with fronts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('divb2_file', type=str, nargs='?', default=None,
                        help='Path to Divb2 NetCDF file')
    parser.add_argument('fronts_file', type=str, nargs='?', default=None,
                        help='Path to fronts .npy file (1=front, 0=no front)')
    parser.add_argument('--downsample', '-d', type=int, default=1,
                        help='Downsample factor for faster display (1 = no downsampling)')

    args = parser.parse_args()

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show main window
    viewer = Divb2Viewer(
        divb2_file=args.divb2_file,
        fronts_file=args.fronts_file,
        downsample=args.downsample
    )
    viewer.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
