"""
Flux regression: J (or R) across the irradiation disk, with Monte Carlo
uncertainty propagation.

Fits a spatial plane (2D, 3D, or individual disk) to flux monitor measurements
and predicts J or 40/39 R values at unknown disk positions. The fit is weighted
by 1/σ²; uncertainty is propagated via Monte Carlo perturbation of both hole
coordinates and measured values, and scaled by √MSWD when the monitors scatter
about the fit beyond the Wendt & Carl limit. Physical constants default to
the suite-wide calibration (arar_math.DEFAULT_FC_NAME).

Layout of this file, top to bottom:
  Monte Carlo threads, disk coordinates, the plot widget, the disk widget
  FluxRegressionCalculator: UI, disk management, regression, prediction and
    .jflux export, method comparison (cross-validation), dialogs (model
    summary, hole stacking, disk design import), results export, sessions
"""

# ── Imports and configuration ────────────────────────────────────────────────

import sys
import os
import ast
import json
import math
import re
import numpy as np
import pandas as pd
import warnings

# shared math/constants live one folder up, next to the launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arar_math import (
    LAMBDA_DEFAULT, STD_AGE_DEFAULT, STD_AGE_ERR, fsig,
    FC_PRESETS, paired_lambda, force_light, DEFAULT_FC_INDEX, DEFAULT_FC_NAME,
    fc_ingrowth, mswd_critical, mswd_classify, mono_font,
)
import data_folders as dfold

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QLabel, QLineEdit, QCheckBox,
    QGroupBox, QFormLayout, QProgressBar, QMessageBox, QHeaderView,
    QDoubleSpinBox, QTextEdit, QScrollArea, QFrame, QAction, QDialog, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt5.QtGui import QDoubleValidator, QFont, QDrag, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Physical constants from the shared table, defaulting to the suite-wide
# calibration (arar_math.DEFAULT_FC_NAME: Phillips et al. 2022 + Min λ)
DECAY_CONSTANT       = LAMBDA_DEFAULT       # yr⁻¹
FC_AGE_DEFAULT       = STD_AGE_DEFAULT      # Ma
FC_AGE_ERROR_DEFAULT = STD_AGE_ERR          # 1σ Ma

MODE_J = 'J'   # regress on J values
MODE_R = 'R'   # regress on 40/39 ratios

MONTE_CARLO_ITERATIONS = 15000

# Hole coordinates are in cm (the 21-hole disk's outer holes are 4.7 mm apart).
# Coordinate perturbation: 0.015 cm = 150 μm base, scaled 0.1–3.0× per iteration
# → 15–450 μm range (RMS ≈ 0.26 mm).  A monitor value is the mean of n crystals
# lying anywhere on the floor of a hole of diameter d; a point placed at random
# on a disc has a standard deviation of d/4 in each direction, so the mean of n
# crystals sits within about d/(4√n) of the hole centre: 0.2–0.3 mm for ~10
# crystals in the 2.7–4.0 mm monitor holes of the 13- and 21-hole disks.
BASE_COORDINATE_UNCERTAINTY = 0.015
COORDINATE_SCALE_MIN = 0.1
COORDINATE_SCALE_MAX = 3.0

COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown']


def _wls(X, y, sigma):
    """Least-squares coefficients of y on X, weighted by 1/σ².  Falls back to
    unweighted least squares when any σ is not positive."""
    sigma = np.asarray(sigma, float)
    if sigma.shape != np.shape(y) or not np.all(sigma > 0):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    w = 1.0 / sigma
    return np.linalg.lstsq(X * w[:, None], np.asarray(y, float) * w, rcond=None)[0]


def lack_of_fit(X, y, sigma):
    """(MSWD, dof, scale) of the weighted fit of the monitors y on X.

    MSWD = Σ((y − Xb)/σ)² / (n − k).  scale = √MSWD when the MSWD exceeds the
    Wendt & Carl (1991) limit 1 + 2√(2/dof), else 1.  The prediction errors
    are multiplied by scale, so scatter of the monitors about the fitted
    surface that their stated errors do not explain is carried into σJ, the
    same rule the other tools apply to a weighted mean.  (None, dof, 1.0)
    when there are no degrees of freedom or a σ is not positive."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    n, k = X.shape
    dof = n - k
    if dof <= 0 or not np.all(sigma > 0):
        return None, dof, 1.0
    b = _wls(X, y, sigma)
    mswd = float((((y - X @ b) / sigma) ** 2).sum()) / dof
    scale = math.sqrt(mswd) if mswd > mswd_critical(dof + 1) else 1.0
    return mswd, dof, scale

# ── Monte Carlo computation threads ──────────────────────────────────────────

class MonteCarloThread(QThread):
    """Fits the regression plane to perturbed copies of the input data.

    Each iteration applies random coordinate noise (15–450 μm) and Gaussian
    perturbation of the measured values before refitting. The distribution of
    resulting coefficients propagates uncertainty from both positioning and
    measurement errors.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, coords_3d, values, errors, regression_type='2d', complex_3d=False):
        super().__init__()
        self.coords_3d = coords_3d
        self.values = values
        self.errors = errors
        self.regression_type = regression_type
        self.complex_3d = complex_3d

    def run(self):
        results = []
        n_points = len(self.coords_3d)

        for i in range(MONTE_CARLO_ITERATIONS):
            coord_perturbation = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(
                COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
            coords_perturbed   = self.coords_3d + np.random.normal(0, coord_perturbation, (n_points, 3))
            values_perturbed   = np.random.normal(self.values, self.errors)

            try:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    coeffs = self._fit_regression_model(coords_perturbed, values_perturbed)
                if coeffs is not None:
                    results.append(coeffs)
            except Exception:
                continue

            if i % 100 == 0:
                self.progress.emit(int(100 * i / MONTE_CARLO_ITERATIONS))

        self.finished.emit(np.array(results))

    def _fit_regression_model(self, coords_perturbed, values_perturbed):
        """Build the design matrix and return least-squares coefficients."""
        n_points = len(coords_perturbed)

        if self.regression_type == '2d':
            X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], np.ones(n_points)])
        elif self.regression_type == '3d' and not self.complex_3d:
            X = np.column_stack([
                coords_perturbed[:, 0], coords_perturbed[:, 1],
                coords_perturbed[:, 2], np.ones(n_points)
            ])
        else:
            x, y, z = coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2]
            X = np.column_stack([x, y, z, np.ones(n_points), x*z, y*z, x*y, x*y*z])

        return _wls(X, values_perturbed, self.errors)

class IndividualDiskMonteCarloThread(QThread):
    """Runs independent Monte Carlo planar regression for each disk.

    Each disk uses only its own standard holes, giving completely isolated
    per-disk coefficient estimates. The same dual perturbation scheme is
    applied as in the global thread.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, disk_data_list, mode=MODE_J):
        super().__init__()
        self.disk_data_list = disk_data_list
        self.mode = mode

    def run(self):
        all_disk_results = []
        total_disks = len(self.disk_data_list)

        for disk_idx, disk_data in enumerate(self.disk_data_list):
            disk_results = self._process_single_disk(disk_data, disk_idx, total_disks)
            all_disk_results.append(np.array(disk_results))

        self.finished.emit(all_disk_results)

    def _process_single_disk(self, disk_data, disk_idx, total_disks):
        standard_data = disk_data['standard_data']
        coords_2d = np.array([[d['coordinates_2d'][0], d['coordinates_2d'][1]] for d in standard_data])

        if self.mode == MODE_R:
            values = np.array([d['r_value'] for d in standard_data])
            errors = np.array([d['r_error'] for d in standard_data])
        else:
            values = np.array([d['j_value'] for d in standard_data])
            errors = np.array([d['j_error'] for d in standard_data])

        disk_results = []
        n_points = len(coords_2d)

        for i in range(MONTE_CARLO_ITERATIONS):
            coord_perturbation = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(
                COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
            coords_perturbed = coords_2d + np.random.normal(0, coord_perturbation, (n_points, 2))
            values_perturbed = np.random.normal(values, errors)

            try:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], np.ones(n_points)])
                    coeffs = _wls(X, values_perturbed, errors)
                disk_results.append(coeffs)
            except Exception:
                continue

            total_progress = ((disk_idx * MONTE_CARLO_ITERATIONS) + i) / (total_disks * MONTE_CARLO_ITERATIONS)
            if i % 100 == 0:
                self.progress.emit(int(100 * total_progress))

        return disk_results

# ── Disk coordinate definitions ──────────────────────────────────────────────

class DiskCoordinates:
    """Static class containing standard disk designs and their hole coordinates."""

    @staticmethod
    def get_disk_designs():
        """Return dictionary of available disk designs with coordinates."""
        return {
            "2.4mm (7 holes)": {
                "diameter": 2.4,
                "coordinates": [
                    (0.64084, 0.37), (0, -0.74), (-0.64084, 0.37),
                    (0, 0.74), (0.64084, -0.37), (-0.64084, -0.37), (0, 0)
                ]
            },
            "2.5mm (13 holes)": {
                "diameter": 2.5,
                "coordinates": [
                    (0.53536, 0.92723), (1.07072, 0.00000), (0.53536, -0.92723),
                    (-0.53536, -0.92723), (-1.07072, 0.00000), (-0.53536, 0.92723),
                    (0.00000, 0.81656), (0.70716, 0.40828), (0.70716, -0.40828),
                    (0.00000, -0.81656), (-0.70716, -0.40828), (-0.70716, 0.40828),
                    (0.00000, 0.00000)
                ]
            },
            "2.4mm (21 holes)": {
                "diameter": 2.4,
                "coordinates": [
                    (0, 0.913696), (0.456848, 0.79128), (0.79128, 0.456848),
                    (0.913696, 0), (0.79128, -0.456848), (0.456848, -0.79128),
                    (0, -0.913696), (-0.456848, -0.79128), (-0.79128, -0.456848),
                    (-0.913696, 0), (-0.79128, 0.456848), (-0.456848, 0.79128),
                    (0, 0.487648), (0.344736, 0.344736), (0.487648, 0),
                    (0.344736, -0.344736), (0, -0.487648), (-0.344736, -0.344736),
                    (-0.487648, 0), (-0.344736, 0.344736), (0, 0)
                ]
            }
        }

# ── Plotting and visualization ───────────────────────────────────────────────

class SimplePlotWidget(QWidget):
    """Custom plotting widget for 2D and 3D regression visualization."""

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(16, 14))
        self.canvas = FigureCanvas(self.figure)
        self._setup_toolbar_and_navigation()
        self._setup_layout()

    def _setup_toolbar_and_navigation(self):
        """Initialize toolbar and scroll navigation."""
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _setup_layout(self):
        """Setup widget layout."""
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def _on_scroll(self, event):
        """Handle mouse wheel zoom events."""
        if not event.inaxes:
            return

        xlim = event.inaxes.get_xlim()
        ylim = event.inaxes.get_ylim()

        zoom_factor = 1.1 if event.step < 0 else 1/1.1

        xdata, ydata = event.xdata, event.ydata

        # Calculate new limits
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        new_xlim = [xdata - new_x_range/2, xdata + new_x_range/2]
        new_ylim = [ydata - new_y_range/2, ydata + new_y_range/2]

        event.inaxes.set_xlim(new_xlim)
        event.inaxes.set_ylim(new_ylim)

        # Handle 3D plots
        if hasattr(event.inaxes, 'get_zlim'):
            zlim = event.inaxes.get_zlim()
            z_range = zlim[1] - zlim[0]
            new_z_range = z_range * zoom_factor
            z_center = (zlim[0] + zlim[1]) / 2
            event.inaxes.set_zlim([z_center - new_z_range/2, z_center + new_z_range/2])

        self.canvas.draw()

    def plot_2d_regression(self, all_disk_data, mode=MODE_J):
        """Plot 2D regression - Shows original values for standards, predicted for unknowns."""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        all_vals = []
        legend_handles, legend_labels = [], []

        for plot_idx, disk_data in enumerate(all_disk_data):
            if not self._should_plot_disk(disk_data):
                continue
            self._plot_disk_2d_points(ax, disk_data, plot_idx, all_vals, mode)
            self._add_disk_to_legend(legend_handles, legend_labels, disk_data, plot_idx)

        self.plot_2d_fitted_plane(ax, all_disk_data)
        self._finalize_2d_plot(ax, all_vals, legend_handles, legend_labels, all_disk_data, mode)

    def plot_individual_disk_regressions(self, all_disk_data, mode=MODE_J):
        """Plot individual planar regressions for each disk with error bars."""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        legend_handles, legend_labels = [], []
        all_vals = []
        for plot_idx, disk_data in enumerate(all_disk_data):
            if not self._should_plot_disk(disk_data):
                continue
            vals = self._plot_individual_disk(ax, disk_data, plot_idx, mode)
            all_vals.extend(vals)
            self._add_disk_to_legend(legend_handles, legend_labels, disk_data, plot_idx)
        self._finalize_individual_plot(ax, legend_handles, legend_labels, all_vals, mode)

    def plot_3d_regression(self, all_disk_data):
        """Plot 3D regression with improved visualization and uncertainty shown as point size."""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        all_j_values, all_uncertainties = self._collect_3d_plot_data(all_disk_data)

        if not all_j_values:
            return

        # Plot points and setup visualization
        self._plot_3d_points(ax, all_disk_data, all_j_values, all_uncertainties)
        self._setup_3d_visualization(ax, all_j_values, all_uncertainties)

    def _should_plot_disk(self, disk_data):
        """Check if disk should be included in plot."""
        return disk_data.get('coordinates') and disk_data.get('enabled', True)

    def _plot_disk_2d_points(self, ax, disk_data, plot_idx, all_vals, mode=MODE_J):
        """Plot points for a single disk in 2D regression."""
        x_coords = [coord[0] for coord in disk_data['coordinates']]
        y_coords = [coord[1] for coord in disk_data['coordinates']]
        regression_values = disk_data.get('regression_values', [])
        regression_errors = disk_data.get('regression_errors', [])
        all_data = disk_data.get('all_data', [])
        color = COLORS[plot_idx % len(COLORS)]

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if i >= len(all_data):
                continue
            hole_data = all_data[i]
            point_info = self._get_point_plotting_info(
                hole_data, regression_values, regression_errors, i, color, mode)
            if point_info is None:
                continue
            val, err, marker_style, alpha, size, edge_color, edge_width = point_info
            all_vals.append(val)
            ax.scatter(x, y, val, c=color, s=size, alpha=alpha,
                      edgecolors=edge_color, linewidth=edge_width, marker=marker_style)
            self._add_error_bars(ax, x, y, val, err)
            self._add_hole_label(ax, x, y, val, i + 1)

    def _get_point_plotting_info(self, hole_data, regression_values, regression_errors, index, color, mode=MODE_J):
        """Get plotting information for a single point.
        In J mode: standards plot j_value, unknowns plot regression_values (J).
        In R mode: standards plot r_value, unknowns plot regression_values (R).
        regression_values always contains the primary output quantity (J or R).
        """
        is_standard = hole_data.get('is_standard', False)
        output_sigma = getattr(self, 'output_sigma', 1)

        if is_standard:
            if mode == MODE_R and hole_data.get('r_value') is not None:
                val   = hole_data['r_value']
                err   = hole_data.get('r_error', 0) * output_sigma
                return val, err, 'o', 0.9, 180, 'black', 2
            elif mode == MODE_J and hole_data.get('j_value') is not None:
                val   = hole_data['j_value']
                err   = hole_data.get('j_error', 0) * output_sigma
                return val, err, 'o', 0.9, 180, 'black', 2
            else:
                return None
        elif index < len(regression_values) and regression_values[index] is not None:
            val = regression_values[index]
            err = (regression_errors[index] if index < len(regression_errors) else 0) * output_sigma
            return val, err, 's', 0.7, 120, color, 1
        else:
            return None

    def _add_error_bars(self, ax, x, y, j_val, j_error):
        """Add error bars to a point."""
        if j_error <= 0:
            return

        # Vertical error bar
        ax.plot([x, x], [y, y], [j_val - j_error, j_val + j_error],
               color='black', linewidth=1.5, alpha=0.7)

        # Error bar caps
        cap_size = 0.02
        ax.plot([x-cap_size, x+cap_size], [y, y], [j_val - j_error, j_val - j_error],
               color='black', linewidth=1.5, alpha=0.7)
        ax.plot([x-cap_size, x+cap_size], [y, y], [j_val + j_error, j_val + j_error],
               color='black', linewidth=1.5, alpha=0.7)

    def _add_hole_label(self, ax, x, y, j_val, hole_number):
        """Add hole number label to a point."""
        ax.text(x, y, j_val + j_val*0.002, str(hole_number), fontsize=10,
               ha='center', va='bottom', weight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       alpha=0.8, edgecolor='none'))

    def _add_disk_to_legend(self, legend_handles, legend_labels, disk_data, plot_idx):
        """Add disk entry to legend."""
        color = COLORS[plot_idx % len(COLORS)]
        disk_number = disk_data.get('disk_number', plot_idx + 1)
        disk_label = disk_data.get('disk_label', '').strip()

        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markeredgecolor='black',
                                       markersize=10, markeredgewidth=2))
        label_str = f'Disk {disk_number} [{disk_label}]' if disk_label else f'Disk {disk_number}'
        legend_labels.append(label_str)

    def plot_2d_fitted_plane(self, ax, all_disk_data):
        """Plot fitted plane through J values."""
        coeffs = self._get_regression_coefficients(all_disk_data)
        if coeffs is None or len(coeffs) < 3:
            return

        all_x, all_y = self._collect_all_coordinates(all_disk_data)
        if not all_x or not all_y:
            return

        # Create mesh for plane
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_padding = (x_max - x_min) * 0.3
        y_padding = (y_max - y_min) * 0.3

        x_range = np.linspace(x_min - x_padding, x_max + x_padding, 20)
        y_range = np.linspace(y_min - y_padding, y_max + y_padding, 20)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

        Z_mesh = coeffs[0] * X_mesh + coeffs[1] * Y_mesh + coeffs[2]

        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.4, color='lightblue')
        ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, alpha=0.6, color='blue', linewidth=1)

    def _get_regression_coefficients(self, all_disk_data):
        """Get regression coefficients from first enabled disk."""
        for disk_data in all_disk_data:
            if disk_data.get('enabled', True) and 'coefficients' in disk_data and disk_data['coefficients'] is not None:
                return disk_data['coefficients']
        return None

    def _collect_all_coordinates(self, all_disk_data):
        """Collect all x,y coordinates from enabled disks."""
        all_x, all_y = [], []
        for disk_data in all_disk_data:
            if disk_data.get('enabled', True) and 'coordinates' in disk_data:
                x_coords = [coord[0] for coord in disk_data['coordinates']]
                y_coords = [coord[1] for coord in disk_data['coordinates']]
                all_x.extend(x_coords)
                all_y.extend(y_coords)
        return all_x, all_y

    def _plot_individual_disk(self, ax, disk_data, plot_idx, mode=MODE_J):
        """Plot individual disk regression plane and points."""
        x_coords = [coord[0] for coord in disk_data['coordinates']]
        y_coords = [coord[1] for coord in disk_data['coordinates']]
        individual_coeffs = disk_data.get('individual_plane_coeffs')
        coefficient_errors = disk_data.get('coefficient_errors', [0, 0, 0])
        all_data = disk_data.get('all_data', [])

        if individual_coeffs is None:
            return []

        color = COLORS[plot_idx % len(COLORS)]
        output_sigma = getattr(self, 'output_sigma', 1)
        vals = []

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if i >= len(all_data):
                continue
            hole_data = all_data[i]
            is_standard = hole_data.get('is_standard', False)

            if is_standard:
                if mode == MODE_R and hole_data.get('r_value') is not None:
                    val   = hole_data['r_value']
                    err   = hole_data.get('r_error', 0) * output_sigma
                elif hole_data.get('j_value') is not None:
                    val   = hole_data['j_value']
                    err   = hole_data.get('j_error', 0) * output_sigma
                else:
                    continue
                ax.scatter(x, y, val, c=color, s=180, alpha=0.9,
                          edgecolors='black', linewidth=2)
            else:
                # Predicted value using this disk's coefficients
                val   = individual_coeffs[0] * x + individual_coeffs[1] * y + individual_coeffs[2]
                err   = np.sqrt((x * coefficient_errors[0])**2 +
                                (y * coefficient_errors[1])**2 +
                                coefficient_errors[2]**2) * output_sigma
                ax.scatter(x, y, val, c=color, s=120, alpha=0.7,
                          edgecolors=color, linewidth=1, marker='s')

            vals.append(val)
            self._add_error_bars(ax, x, y, val, err)
            self._add_hole_label(ax, x, y, val, i + 1)

        self._plot_disk_plane(ax, x_coords, y_coords, individual_coeffs, color)
        return vals

    def _plot_disk_plane(self, ax, x_coords, y_coords, coeffs, color):
        """Plot regression plane for individual disk."""
        x_range = np.array(x_coords)
        y_range = np.array(y_coords)
        x_min, x_max = x_range.min() - 0.2, x_range.max() + 0.2
        y_min, y_max = y_range.min() - 0.2, y_range.max() + 0.2

        x_mesh = np.linspace(x_min, x_max, 15)
        y_mesh = np.linspace(y_min, y_max, 15)
        X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)

        Z_mesh = coeffs[0] * X_mesh + coeffs[1] * Y_mesh + coeffs[2]

        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.3, color=color)
        ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, alpha=0.6, color=color, linewidth=1)

    def _collect_3d_plot_data(self, all_disk_data):
        """Collect all J values and uncertainties for 3D plot scaling."""
        all_j_values = []
        all_uncertainties = []

        mode = getattr(self, 'regression_mode', MODE_J)

        for disk_data in all_disk_data:
            if not disk_data.get('enabled', True):
                continue

            all_data = disk_data.get('all_data', [])
            x_coords = [coord[0] for coord in disk_data['coordinates']]
            y_coords = [coord[1] for coord in disk_data['coordinates']]
            disk_height = disk_data['height']

            for i, hole_data in enumerate(all_data):
                if hole_data.get('is_standard'):
                    if mode == MODE_R and hole_data.get('r_value') is not None:
                        all_j_values.append(hole_data['r_value'])
                        all_uncertainties.append(hole_data.get('r_error', 0))
                    elif hole_data.get('j_value') is not None:
                        all_j_values.append(hole_data['j_value'])
                        all_uncertainties.append(hole_data.get('j_error', 0))
                else:
                    # Calculate predicted values for unknown holes
                    if i < len(x_coords) and i < len(y_coords):
                        x, y = x_coords[i], y_coords[i]
                        if hasattr(self, 'predict_j_value'):
                            j_pred, j_error = self.predict_j_value(x, y, disk_height)
                            if j_pred is not None:
                                all_j_values.append(j_pred)
                                all_uncertainties.append(j_error if j_error is not None else 0)

        return all_j_values, all_uncertainties

    def _plot_3d_points(self, ax, all_disk_data, all_j_values, all_uncertainties):
        """Plot all points for 3D regression visualization."""
        if not all_j_values:
            return

        unc_min, unc_max = min(all_uncertainties), max(all_uncertainties)
        unc_range = unc_max - unc_min if unc_max > unc_min else 1e-10

        cmap = plt.cm.turbo

        for disk_idx, disk_data in enumerate(all_disk_data):
            if not disk_data.get('coordinates') or not disk_data.get('enabled', True):
                continue

            self._plot_3d_disk_points(ax, disk_data, disk_idx, all_j_values, unc_min, unc_range, cmap)

    def _plot_3d_disk_points(self, ax, disk_data, disk_idx, all_j_values, unc_min, unc_range, cmap):
        """Plot points for a single disk in 3D visualization."""
        x_coords = [coord[0] for coord in disk_data['coordinates']]
        y_coords = [coord[1] for coord in disk_data['coordinates']]
        all_data = disk_data.get('all_data', [])
        disk_height = disk_data['height']

        mode = getattr(self, 'regression_mode', MODE_J)

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if i >= len(all_data):
                continue

            hole_data = all_data[i]
            is_standard = hole_data.get('is_standard', False)

            if is_standard:
                if mode == MODE_R and hole_data.get('r_value') is not None:
                    j_val    = hole_data['r_value']
                    j_error  = hole_data.get('r_error', 0)
                elif hole_data.get('j_value') is not None:
                    j_val    = hole_data['j_value']
                    j_error  = hole_data.get('j_error', 0)
                else:
                    continue
                marker_style = 'o'
            else:
                j_pred, j_error = self.predict_j_value(x, y, disk_height)
                if j_pred is None:
                    continue
                j_val = j_pred
                marker_style = 's'

            # Calculate color and size
            j_percentile = np.sum(np.array(all_j_values) <= j_val) / len(all_j_values)
            heat_color = cmap(j_percentile)

            # Scale point size based on uncertainty
            if unc_range > 0:
                uncertainty_factor = (j_error - unc_min) / unc_range
            else:
                uncertainty_factor = 0
            point_size = 120 + int(120 * uncertainty_factor)

            # Plot point
            alpha = 0.9 if is_standard else 0.7
            ax.scatter(x, y, disk_height, c=[heat_color], s=point_size, alpha=alpha,
                     edgecolors='black', linewidth=2.5 if is_standard else 2, marker=marker_style)

            # Add hole number label
            label_height = disk_height + max(0.5, disk_height * 0.05)
            ax.text(x, y, label_height, str(i+1), fontsize=9,
                   ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           alpha=0.9, edgecolor='gray', linewidth=0.5))

    def _setup_3d_visualization(self, ax, all_j_values, all_uncertainties):
        """Setup 3D plot visualization elements."""
        mode = getattr(self, 'regression_mode', MODE_J)
        qty  = 'R (40/39)' if mode == MODE_R else 'J'

        # Setup colorbar
        v_min, v_max = min(all_j_values), max(all_j_values)
        cmap = plt.cm.turbo
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(all_j_values)
        sm.set_clim(v_min, v_max)
        cbar = self.figure.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.2)
        cbar.set_label(f'Original {qty} Value', rotation=270, labelpad=15, fontsize=11, weight='bold')
        cbar.ax.tick_params(labelsize=9)

        # Setup axis labels and title
        ax.set_xlabel('X Coordinate (cm)', fontsize=11, labelpad=10)
        ax.set_ylabel('Y Coordinate (cm)', fontsize=11, labelpad=10)
        ax.set_zlabel('Height (mm)', fontsize=11, labelpad=10)
        ax.set_title(f'Multi-Disk 3D Regression\n(Colors = {qty} values, Point size = Uncertainty)', fontsize=12)

        # Setup legends
        self._setup_3d_legends(ax, all_uncertainties)

        ax.view_init(elev=20, azim=45)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)

    def _setup_3d_legends(self, ax, all_uncertainties):
        """Setup legends for 3D plot."""
        mode = getattr(self, 'regression_mode', MODE_J)
        qty  = 'R' if mode == MODE_R else 'J'

        # Marker type legend
        marker_legend_elements = [
            plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                       markersize=8, markeredgecolor='black', markerfacecolor='lightblue',
                       label=f'Standards (original {qty})'),
            plt.Line2D([0], [0], marker='s', color='gray', linestyle='None',
                       markersize=8, markeredgecolor='black', markerfacecolor='lightcoral',
                       label=f'Unknowns (predicted {qty})')
        ]

        marker_legend = ax.legend(handles=marker_legend_elements,
                                 bbox_to_anchor=(0.1, -0.12), loc='upper center',
                                 fontsize=9, markerscale=1.0, framealpha=0.95,
                                 title='Point Types', title_fontsize=10)
        marker_legend.get_frame().set_facecolor('white')
        marker_legend.get_frame().set_edgecolor('gray')

        # Uncertainty legend
        unc_min, unc_max = min(all_uncertainties), max(all_uncertainties)
        unc_range = unc_max - unc_min

        if unc_range > 0:
            uncertainty_elements = [
                plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                           markersize=6, markerfacecolor='gray', alpha=0.7,
                           label=f'Small = Low uncertainty (±{unc_min:.1e})'),
                plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                           markersize=12, markerfacecolor='gray', alpha=0.7,
                           label=f'Large = High uncertainty (±{unc_max:.1e})')
            ]

            uncertainty_legend = ax.legend(handles=uncertainty_elements,
                                         bbox_to_anchor=(0.7, -0.12), loc='upper center',
                                         fontsize=9, markerscale=1.0, framealpha=0.95,
                                         title='Point Size Legend', title_fontsize=10)
            uncertainty_legend.get_frame().set_facecolor('white')
            uncertainty_legend.get_frame().set_edgecolor('gray')

            ax.add_artist(marker_legend)

    def _finalize_2d_plot(self, ax, all_vals, legend_handles, legend_labels, all_disk_data, mode=MODE_J):
        """Finalize 2D plot with labels, legend, and formatting."""
        qty = "R (40/39)" if mode == MODE_R else "J Value"
        ax.set_xlabel('X Coordinate (cm)', labelpad=10)
        ax.set_ylabel('Y Coordinate (cm)', labelpad=10)
        ax.set_zlabel(f'{qty} (Measured for Standards, Predicted for Unknowns)', labelpad=10)
        ax.set_title(f'{qty} Planar Regression with Error Bars\n'
                     f'(Circles = Standards [measured], Squares = Unknowns [predicted])')

        if len(all_disk_data) > 1 and legend_handles:
            ax.legend(legend_handles, legend_labels,
                     bbox_to_anchor=(0.02, 0.98), loc='upper left',
                     fontsize=10, markerscale=1.0, framealpha=0.95)

        if all_vals:
            v_min, v_max = min(all_vals), max(all_vals)
            if v_max > v_min:
                ax.text(0.02, 0.98, 0.02,
                       f'{qty} Range: {v_min:.4e} to {v_max:.4e}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                       verticalalignment='top')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas.draw()

    def _finalize_individual_plot(self, ax, legend_handles, legend_labels, all_vals, mode=MODE_J):
        """Finalize individual disk regression plot with proper axis scaling."""
        qty = "R (40/39)" if mode == MODE_R else "J Value"
        ax.set_xlabel('X Coordinate (cm)', labelpad=10)
        ax.set_ylabel('Y Coordinate (cm)', labelpad=10)
        ax.set_zlabel(qty, labelpad=10)
        ax.set_title(f'Individual Disk Planar Regressions: {qty}\n(Each disk fitted independently)')

        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                     bbox_to_anchor=(0.02, 0.98), loc='upper left',
                     fontsize=10, markerscale=1.0, framealpha=0.95)

        if all_vals:
            v_min, v_max = min(all_vals), max(all_vals)
            v_range = v_max - v_min
            z_padding = v_range * 0.1 if v_range > 0 else abs(v_min) * 0.05
            ax.set_zlim([v_min - z_padding, v_max + z_padding])
            ax.view_init(elev=15, azim=30)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas.draw()

# ── Disk configuration widget ────────────────────────────────────────────────

class DiskWidget(QWidget):
    """Widget for configuring individual disks with drag-and-drop reordering."""

    def __init__(self, disk_number, parent_calculator):
        super().__init__()
        self.disk_number = disk_number
        self.parent_calculator = parent_calculator
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for disk configuration."""
        layout = QVBoxLayout()

        # Configuration section
        config_group = QGroupBox(f"Disk {self.disk_number} Configuration")
        config_layout = QFormLayout()

        # Control buttons
        control_layout = QHBoxLayout()
        self._setup_control_buttons(control_layout)
        layout.addLayout(control_layout)

        # Disk selection and height
        self._setup_disk_configuration(config_layout)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Data table
        self.create_data_table()
        layout.addWidget(self.table_widget)

        self.setLayout(layout)
        self.load_disk_coordinates(preserve_data=False)

    def _setup_control_buttons(self, layout):
        """Setup control buttons for drag and enable/disable."""
        self.drag_button = QPushButton("⇅ Drag to Reorder")
        self.drag_button.setMaximumWidth(120)
        self.drag_button.setStyleSheet("QPushButton { background-color: lightgray; }")
        self.drag_button.mousePressEvent = self.start_drag
        layout.addWidget(self.drag_button)

        self.enable_checkbox = QCheckBox("Include in Regression")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.setStyleSheet("QCheckBox { font-weight: bold; color: darkgreen; }")
        layout.addWidget(self.enable_checkbox)
        layout.addStretch()

        self.remove_btn = QPushButton("✕ Remove Disk")
        self.remove_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #c0392b; color: #888; }"
        )
        self.remove_btn.clicked.connect(lambda: self.parent_calculator.remove_specific_disk(self))
        layout.addWidget(self.remove_btn)

    def _setup_disk_configuration(self, layout):
        """Setup disk design selection and height configuration."""
        self.disk_combo = QComboBox()
        self.disk_combo.addItems(DiskCoordinates.get_disk_designs().keys())
        self.disk_combo.currentTextChanged.connect(lambda _: self.load_disk_coordinates(preserve_data=False))
        layout.addRow("Disk Design:", self.disk_combo)

        # Short label used when exporting (e.g. "E"); defaults to a letter by position.
        self.disk_label_edit = QLineEdit(chr(ord('A') + (self.disk_number - 1) % 26))
        self.disk_label_edit.setToolTip(
            "Short disk label used in the exported flux config (e.g. 'E').\n"
            "Match this to the disk_id in your .arar files so samples line up.")
        layout.addRow("Disk label:", self.disk_label_edit)

        self.height_widget = QWidget()
        height_layout = QFormLayout()
        self.height_spinbox = QDoubleSpinBox()
        self.height_spinbox.setRange(-100, 100)
        self.height_spinbox.setValue(0.0 if self.disk_number == 1 else (self.disk_number - 1) * 2.0)
        self.height_spinbox.setSuffix(" mm")
        self.height_spinbox.valueChanged.connect(self.update_height_column)
        height_layout.addRow("Height:", self.height_spinbox)
        self.height_widget.setLayout(height_layout)
        layout.addRow(self.height_widget)

    def create_data_table(self):
        """Create the data entry table for samples and standards."""
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(7)
        self._set_data_table_headers()

        font = QFont()
        font.setBold(True)
        self.table_widget.horizontalHeader().setFont(font)
        self.table_widget.setMinimumHeight(400)

        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table_widget.keyPressEvent = self.handle_key_press

    def _set_data_table_headers(self):
        """Set column headers based on current regression mode and input sigma."""
        mode        = getattr(self.parent_calculator, 'regression_mode', MODE_J)
        input_sigma = getattr(self.parent_calculator, 'input_sigma', 1)
        sigma_str   = f"{input_sigma}σ"
        if mode == MODE_R:
            val_col  = "R (40/39)"
            err_col  = f"R Error ({sigma_str})"
        else:
            val_col  = "J"
            err_col  = f"J Error ({sigma_str})"
        self.table_widget.setHorizontalHeaderLabels([
            "Sample Name", "Hole #", "Use in Regression",
            val_col, err_col, "Coordinates (X, Y)", "Height"
        ])

    def _update_input_table_headers(self):
        """Called when regression mode changes, update all disk widgets."""
        pass  # headers set via _set_data_table_headers on each DiskWidget directly

    def handle_key_press(self, event):
        """Handle keyboard events, particularly Ctrl+V for paste."""
        if event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            self.paste_data()
        else:
            QTableWidget.keyPressEvent(self.table_widget, event)

    def paste_data(self):
        """Paste clipboard data into the table."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        lines = text.strip().split('\n')
        current_row = self.table_widget.currentRow()
        current_col = self.table_widget.currentColumn()

        if current_row < 0 or current_col < 0:
            return

        for i, line in enumerate(lines):
            row = current_row + i
            if row >= self.table_widget.rowCount():
                break

            values = line.split('\t')
            if len(values) == 1:
                values = line.split(',')

            for j, value in enumerate(values):
                col = current_col + j
                if col >= self.table_widget.columnCount():
                    continue

                self._paste_cell_value(row, col, value)

    def _paste_cell_value(self, row, col, value):
        """Paste a single cell value with appropriate validation."""
        if col == 0:  # Sample name
            self.table_widget.setItem(row, col, QTableWidgetItem(value.strip()))
        elif col == 3 or col == 4:  # J value or J error
            try:
                float(value.strip())
                self.table_widget.setItem(row, col, QTableWidgetItem(value.strip()))
            except ValueError:
                pass  # Skip invalid numeric values

    def load_disk_coordinates(self, preserve_data=False):
        """Load coordinates for selected disk design."""
        disk_name = self.disk_combo.currentText()
        all_designs = DiskCoordinates.get_disk_designs()
        all_designs.update(getattr(self.parent_calculator, 'custom_disk_designs', {}))
        if disk_name not in all_designs:
            return
        disk_data = all_designs[disk_name]
        coordinates = disk_data["coordinates"]

        existing_data = self._collect_existing_data() if preserve_data else {}

        self.table_widget.setRowCount(len(coordinates))

        for i, (x, y) in enumerate(coordinates):
            self._setup_table_row(i, x, y, existing_data)

    def _collect_existing_data(self):
        """Collect existing table data for preservation."""
        existing_data = {}
        if self.table_widget.rowCount() > 0:
            for i in range(self.table_widget.rowCount()):
                sample_item = self.table_widget.item(i, 0)
                checkbox = self.table_widget.cellWidget(i, 2)
                j_item = self.table_widget.item(i, 3)
                j_error_item = self.table_widget.item(i, 4)

                existing_data[i] = {
                    'sample_name': sample_item.text() if sample_item else "",
                    'checked': checkbox.isChecked() if checkbox else False,
                    'j_value': j_item.text() if j_item else "",
                    'j_error': j_error_item.text() if j_error_item else ""
                }
        return existing_data

    def _setup_table_row(self, i, x, y, existing_data):
        """Setup a single row in the table."""
        # Sample name
        if i in existing_data and existing_data[i].get('sample_name'):
            sample_name = existing_data[i]['sample_name']
        else:
            sample_name = f"Sample_{self.disk_number}_{i+1:02d}"
        if self.table_widget.item(i, 0):
            self.table_widget.item(i, 0).setText(sample_name)
        else:
            self.table_widget.setItem(i, 0, QTableWidgetItem(sample_name))

        # Hole number
        if not self.table_widget.item(i, 1):
            self.table_widget.setItem(i, 1, QTableWidgetItem(str(i + 1)))

        # Checkbox
        checkbox = self.table_widget.cellWidget(i, 2)
        if not checkbox:
            checkbox = QCheckBox()
            self.table_widget.setCellWidget(i, 2, checkbox)
        if i in existing_data:
            checkbox.setChecked(existing_data[i]['checked'])
        else:
            disk_name = self.disk_combo.currentText()
            default_standards = {
                '2.5mm (13 holes)': {0, 1, 2, 3, 4, 5, 12},  # holes 1–6, 13
                '2.4mm (7 holes)':  {0, 1, 2, 6},              # holes 1–3, 7
            }
            defaults = default_standards.get(disk_name, set())
            checkbox.setChecked(i in defaults)

        # J value and error
        self._setup_j_value_cells(i, existing_data)

        # Coordinates
        coord_text = f"({x:.5f}, {y:.5f})"
        coord_item = QTableWidgetItem(coord_text)
        coord_item.setFlags(coord_item.flags() & ~Qt.ItemIsEditable)
        if self.table_widget.item(i, 5):
            self.table_widget.item(i, 5).setText(coord_text)
        else:
            self.table_widget.setItem(i, 5, coord_item)

        # Height
        height_item = QTableWidgetItem(f"{self.height_spinbox.value():.2f}")
        height_item.setFlags(height_item.flags() & ~Qt.ItemIsEditable)
        if self.table_widget.item(i, 6):
            self.table_widget.item(i, 6).setText(f"{self.height_spinbox.value():.2f}")
        else:
            self.table_widget.setItem(i, 6, height_item)

    def _setup_j_value_cells(self, i, existing_data):
        """Setup J value and J error cells."""
        val_text = existing_data[i]['j_value'] if i in existing_data else ""
        err_text = existing_data[i]['j_error'] if i in existing_data else ""
        for col, text in [(3, val_text), (4, err_text)]:
            item = self.table_widget.item(i, col)
            if item:
                item.setText(text)
            else:
                self.table_widget.setItem(i, col, QTableWidgetItem(text))

    def update_height_column(self):
        """Update height column when height spinbox changes."""
        height = self.height_spinbox.value()
        for i in range(self.table_widget.rowCount()):
            height_item = QTableWidgetItem(f"{height:.2f}")
            height_item.setFlags(height_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(i, 6, height_item)

    def get_disk_data(self):
        """Get all data for this disk in standardized format."""
        disk_name = self.disk_combo.currentText()
        all_designs = DiskCoordinates.get_disk_designs()
        all_designs.update(getattr(self.parent_calculator, 'custom_disk_designs', {}))
        disk_data = all_designs.get(disk_name, {})
        coordinates = disk_data.get("coordinates", [])
        height = self.height_spinbox.value()
        mode = getattr(self.parent_calculator, 'regression_mode', MODE_J)

        standard_data = []
        all_data = []

        for i in range(self.table_widget.rowCount()):
            hole_data = self._get_hole_data(i, coordinates, height, mode)
            all_data.append(hole_data)

            checkbox = self.table_widget.cellWidget(i, 2)
            primary_key = 'r_value' if mode == MODE_R else 'j_value'
            if checkbox.isChecked() and hole_data[primary_key] is not None:
                standard_data.append(hole_data)

        return {
            'disk_name': disk_name,
            'disk_label': self.disk_label_edit.text().strip() or disk_name,
            'height': height,
            'coordinates': [hd['coordinates_2d'] for hd in all_data],
            'standard_data': standard_data,
            'all_data': all_data,
            'enabled': self.enable_checkbox.isChecked(),
            'regression_mode': mode,
        }

    def _get_hole_data(self, i, coordinates, height, mode=MODE_J):
        """Get data for a single hole, supporting both J and R input modes."""
        checkbox = self.table_widget.cellWidget(i, 2)
        sample_name = self.table_widget.item(i, 0).text()
        stacking_cfg = getattr(self.parent_calculator, 'hole_stacking_config', None)
        x_adj, y_adj = coordinates[i]
        if stacking_cfg is not None:
            key = f'{len(coordinates)}_hole_reference'
            ref_idx = stacking_cfg.get(key)
            if ref_idx is not None and 1 <= ref_idx <= len(coordinates):
                ref_x, ref_y = coordinates[ref_idx - 1]
                x_adj = x_adj - ref_x
                y_adj = y_adj - ref_y
        coord_3d = (x_adj, y_adj, height)
        val_item = self.table_widget.item(i, 3)
        err_item = self.table_widget.item(i, 4)

        hole_data = {
            'sample_name':    sample_name,
            'hole_number':    i + 1,
            'coordinates_3d': coord_3d,
            'coordinates_2d': (x_adj, y_adj),
            'is_standard':    checkbox.isChecked(),
            'j_value':        None,
            'j_error':        None,
            'r_value':        None,
            'r_error':        None,
        }

        if val_item and val_item.text() and err_item and err_item.text():
            try:
                val       = float(val_item.text())
                err_raw   = float(err_item.text())
                input_sigma = getattr(self.parent_calculator, 'input_sigma', 1)
                err       = err_raw / input_sigma  # normalise to 1σ for internal storage
                if mode == MODE_R:
                    hole_data['r_value'] = val
                    hole_data['r_error'] = err  # stored as 1σ
                    try:
                        j   = self.parent_calculator.monitor_ingrowth() / val
                        hole_data['j_value'] = j
                        hole_data['j_error'] = j * (err / val)  # σ_J = J × σ_R / R
                    except Exception:
                        pass
                else:
                    hole_data['j_value'] = val
                    hole_data['j_error'] = err   # stored as 1σ
                    try:
                        r   = self.parent_calculator.monitor_ingrowth() / val
                        hole_data['r_value'] = r
                        hole_data['r_error'] = r * (err / val)  # σ_R = R × σ_J / J
                    except Exception:
                        pass
            except ValueError:
                pass

        return hole_data

    def start_drag(self, event):
        """Start drag operation for reordering."""
        if event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(str(self.disk_number))
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop events for reordering."""
        source_disk_num = int(event.mimeData().text())
        target_disk_num = self.disk_number

        if source_disk_num != target_disk_num:
            self.parent_calculator.reorder_disks(source_disk_num, target_disk_num)
        event.accept()

    def update_disk_number(self, new_number):
        """Update disk number after reordering."""
        self.disk_number = new_number
        for child in self.findChildren(QGroupBox):
            if "Configuration" in child.title():
                child.setTitle(f"Disk {self.disk_number} Configuration")
                break

# ── Flux monitors export ─────────────────────────────────────────────────────

def _num(v):
    """float(v), or None when v is empty or not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_flux_monitors_export(path):
    """The positions of a Flux monitors export (.xlsx) and its calibration line.

    One dict per category of the Results sheet: its reported R and 1σ SE (REML
    or weighted mean, whichever the Flux monitors tool reported) and the J it
    exported.  Raises ValueError when the file has no R results table."""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb['Results'] if 'Results' in wb.sheetnames else wb.worksheets[0]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()
    head = next((i for i, r in enumerate(rows)
                 if r and r[0] == 'Category' and 'Reported R' in r), None)
    if head is None:
        raise ValueError("No table with 'Category' and 'Reported R' columns. Load an "
                         "export of the Flux monitors tool made in 40/39 ratio mode.")
    col = {h: k for k, h in enumerate(rows[head]) if isinstance(h, str)}
    out = []
    for r in rows[head + 1:]:
        if not r or r[0] is None or not str(r[0]).strip():
            break
        get = lambda name: _num(r[col[name]]) if name in col and col[name] < len(r) else None
        s1 = get('Reported SE (1σ)')
        if s1 is None and get('Reported SE (2σ)') is not None:
            s1 = get('Reported SE (2σ)') / 2.0
        out.append({'category': str(r[0]).strip(), 'R': get('Reported R'),
                    'R_sigma': s1, 'J': get('J')})
    calib = next((r[0] for r in rows if r and isinstance(r[0], str)
                  and r[0].startswith('Calibration for J:')), '')
    return out, calib


def split_position_label(label):
    """(disk label, hole number) of a position such as 'C1', 'C-1', 'C 01' or
    'E17'; ('', 13) for a bare number; None when it does not end in a number."""
    m = re.match(r'^\s*(.*?)[\s_\-]*0*(\d+)\s*$', str(label))
    if not m:
        return None
    return m.group(1).strip(), int(m.group(2))


# ── Main application class ───────────────────────────────────────────────────

class FluxRegressionCalculator(QMainWindow):
    """Main window of the Flux regression tool."""

    def monitor_ingrowth(self):
        """e^{λ t_FCs} − 1 for the monitor, from arar_math.fc_ingrowth: on the
        anchored chain R_FCs/A (e^{λ t_A} − 1) for Kuiper, Rivera and Phillips,
        so J = ingrowth / R is the same J every other tool uses."""
        try:
            lam = float(self.lambda_edit.text())
            t_fc = float(self.standard_age_edit.text())
        except Exception:
            lam, t_fc = DECAY_CONSTANT, FC_AGE_DEFAULT
        try:
            t_fc_err = float(self.standard_age_err_edit.text())
        except Exception:
            t_fc_err = None
        return fc_ingrowth(t_fc, lam, t_fc_err)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux regression")
        self.setGeometry(50, 50, 1800, 1000)

        self.input_sigma  = 1   # precision of user-entered errors (1 or 2)
        self.output_sigma = 1   # precision used for all reported errors (1 or 2)

        self.disk_widgets = []
        self.init_ui()

    def init_ui(self):
        """Initialize the main user interface."""
        self.create_menu_bar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create left panel with controls
        left_scroll = self._create_left_panel()

        # Create right panel with plots
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)

        self.plot_widget = SimplePlotWidget()
        self.plot_widget.predict_j_value = self.predict_j_value
        splitter.addWidget(self.plot_widget)

        splitter.setSizes([500, 1100])
        main_layout.addWidget(splitter)

        # Initialize with one disk
        self.add_disk()
        self.update_interface_for_disk_count()

    def _create_left_panel(self):
        """Create the left control panel."""
        left_scroll = QScrollArea()
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(800)

        # Add all sections to left panel
        left_layout.addWidget(self._create_constants_section())
        left_layout.addWidget(self._create_disk_management_section())
        left_layout.addWidget(self._create_calculations_section())
        left_layout.addWidget(self._create_extrapolation_section())

        self.create_results_table()
        left_layout.addWidget(self.results_widget)

        return left_scroll

    def _create_constants_section(self):
        """Create the constants configuration section."""
        constants_group = QGroupBox("Constants && Regression Mode")
        constants_layout = QFormLayout()

        # Calibration preset: fills age, error and the anchored λ below
        self.fc_preset_combo = QComboBox()
        self.fc_preset_combo.addItems([r[0] for r in FC_PRESETS] + ['Custom…'])
        self.fc_preset_combo.setCurrentIndex(DEFAULT_FC_INDEX)
        self.fc_preset_combo.setToolTip(
            'FC monitor calibration. Picking a preset sets the age, its 1σ error\n'
            'and the decay constant that calibration is anchored to\n'
            '(e.g. Phillips 2022 → Min et al. 2000 λ).')
        self.fc_preset_combo.activated.connect(self._on_fc_preset)
        constants_layout.addRow("Calibration:", self.fc_preset_combo)

        # Standard age (with its 1σ error) + lambda on one row
        self.standard_age_edit = QLineEdit(str(FC_AGE_DEFAULT))
        self.standard_age_edit.setValidator(QDoubleValidator())
        self.standard_age_err_edit = QLineEdit(str(FC_AGE_ERROR_DEFAULT))
        self.standard_age_err_edit.setValidator(QDoubleValidator())
        self.standard_age_err_edit.setToolTip(
            '1σ error on the FC monitor age (written to exported .jflux files).')
        self.lambda_edit = QLineEdit(f"{DECAY_CONSTANT:.4e}")
        self.lambda_edit.setValidator(QDoubleValidator())
        for _w in (self.standard_age_edit, self.standard_age_err_edit,
                   self.lambda_edit):
            _w.textEdited.connect(
                lambda _t: self.fc_preset_combo.setCurrentIndex(len(FC_PRESETS)))

        age_lambda_widget = QWidget()
        age_lambda_layout = QHBoxLayout(age_lambda_widget)
        age_lambda_layout.setContentsMargins(0, 0, 0, 0)
        age_lambda_layout.addWidget(self.standard_age_edit)
        age_lambda_layout.addWidget(QLabel("±1σ:"))
        age_lambda_layout.addWidget(self.standard_age_err_edit)
        age_lambda_layout.addWidget(QLabel("λ (yr⁻¹):"))
        age_lambda_layout.addWidget(self.lambda_edit)
        constants_layout.addRow("Age (Ma):", age_lambda_widget)

        ref_label = QLabel(
            f"{DEFAULT_FC_NAME}: FC {STD_AGE_DEFAULT} ± {STD_AGE_ERR} Ma (1σ), "
            "internal errors only")
        ref_label.setStyleSheet("color: gray; font-size: 9px;")
        constants_layout.addRow("", ref_label)

        # Input error precision + output error precision on one row
        self.input_sigma_1_btn = QPushButton("1σ")
        self.input_sigma_2_btn = QPushButton("2σ")
        self.input_sigma_1_btn.setCheckable(True)
        self.input_sigma_2_btn.setCheckable(True)
        self.input_sigma_1_btn.setChecked(True)

        def _set_input_sigma_1():
            self.input_sigma = 1
            self.input_sigma_1_btn.setChecked(True)
            self.input_sigma_2_btn.setChecked(False)
            self._update_input_table_headers()

        def _set_input_sigma_2():
            self.input_sigma = 2
            self.input_sigma_1_btn.setChecked(False)
            self.input_sigma_2_btn.setChecked(True)
            self._update_input_table_headers()

        self.input_sigma_1_btn.clicked.connect(_set_input_sigma_1)
        self.input_sigma_2_btn.clicked.connect(_set_input_sigma_2)

        self.output_sigma_1_btn = QPushButton("1σ")
        self.output_sigma_2_btn = QPushButton("2σ")
        self.output_sigma_1_btn.setCheckable(True)
        self.output_sigma_2_btn.setCheckable(True)
        self.output_sigma_1_btn.setChecked(True)

        def _set_output_sigma_1():
            self.output_sigma = 1
            self.output_sigma_1_btn.setChecked(True)
            self.output_sigma_2_btn.setChecked(False)
            self._update_results_table_headers()

        def _set_output_sigma_2():
            self.output_sigma = 2
            self.output_sigma_1_btn.setChecked(False)
            self.output_sigma_2_btn.setChecked(True)
            self._update_results_table_headers()

        self.output_sigma_1_btn.clicked.connect(_set_output_sigma_1)
        self.output_sigma_2_btn.clicked.connect(_set_output_sigma_2)

        sigma_widget = QWidget()
        sigma_layout = QHBoxLayout(sigma_widget)
        sigma_layout.setContentsMargins(0, 0, 0, 0)
        sigma_layout.addWidget(self.input_sigma_1_btn)
        sigma_layout.addWidget(self.input_sigma_2_btn)
        sigma_layout.addWidget(QLabel("  Reported:"))
        sigma_layout.addWidget(self.output_sigma_1_btn)
        sigma_layout.addWidget(self.output_sigma_2_btn)
        constants_layout.addRow("Input error:", sigma_widget)

        # Regression mode toggle
        mode_label = QLabel("Regression Mode:")
        self.mode_j_radio = QPushButton("J-value mode")
        self.mode_r_radio = QPushButton("R (40/39) mode")
        self.mode_j_radio.setCheckable(True)
        self.mode_r_radio.setCheckable(True)
        self.mode_j_radio.setChecked(True)
        self.regression_mode = MODE_J

        def _set_mode_j():
            self.regression_mode = MODE_J
            self.mode_j_radio.setChecked(True)
            self.mode_r_radio.setChecked(False)
            self._update_input_table_headers()

        def _set_mode_r():
            self.regression_mode = MODE_R
            self.mode_j_radio.setChecked(False)
            self.mode_r_radio.setChecked(True)
            self._update_input_table_headers()

        self.mode_j_radio.clicked.connect(_set_mode_j)
        self.mode_r_radio.clicked.connect(_set_mode_r)

        mode_btn_layout = QHBoxLayout()
        mode_btn_layout.addWidget(self.mode_j_radio)
        mode_btn_layout.addWidget(self.mode_r_radio)
        mode_btn_widget = QWidget()
        mode_btn_widget.setLayout(mode_btn_layout)

        constants_layout.addRow(mode_label, mode_btn_widget)
        constants_group.setLayout(constants_layout)
        return constants_group

    def _on_fc_preset(self, idx):
        if idx >= len(FC_PRESETS):
            return                      # Custom: leave the fields alone
        name, age, err, _ext = FC_PRESETS[idx]
        self.standard_age_edit.setText(str(age))
        self.standard_age_err_edit.setText(str(err))
        pl = paired_lambda(name)
        if pl is not None:
            self.lambda_edit.setText(f"{pl[1]:.4e}")

    def _create_disk_management_section(self):
        """Create the disk management section."""
        disk_management_group = QGroupBox("Disk Management")
        disk_layout = QVBoxLayout()

        # Disk control buttons
        disk_controls = QHBoxLayout()
        self.add_disk_btn = QPushButton("Add Disk")
        self.add_disk_btn.clicked.connect(self.add_disk)
        disk_controls.addWidget(self.add_disk_btn)
        self.import_fm_btn = QPushButton("Import Flux monitors export...")
        self.import_fm_btn.setToolTip(
            "Fill the monitor holes from an .xlsx export of the Flux monitors tool:\n"
            "category C1 goes to the disk labelled C, hole 1, with its reported R and SE.")
        self.import_fm_btn.clicked.connect(lambda: self.import_flux_monitors_export())
        disk_controls.addWidget(self.import_fm_btn)
        disk_layout.addLayout(disk_controls)

        # Irradiation name + flux-config export (for the Plateau & isochron and Age distribution tools)
        irr_row = QHBoxLayout()
        irr_row.addWidget(QLabel("Irradiation:"))
        self.irradiation_edit = QLineEdit("9E11-25")
        self.irradiation_edit.setToolTip(
            "Name of this irradiation. Holes from the SAME irradiation and SAME\n"
            "disk share J covariance; different disks/irradiations are independent.\n"
            "Used to match samples when loading into the Plateau & isochron and Age distribution tools.")
        irr_row.addWidget(self.irradiation_edit)
        self.export_flux_btn = QPushButton("Export Flux Config (.jflux)")
        self.export_flux_btn.setToolTip(
            "Write a per-disk J + cross-hole covariance file (joint Monte Carlo)\n"
            "that the Plateau & isochron and Age distribution tools load to combine same-disk samples\n"
            "with their true correlation. Requires a completed regression.")
        self.export_flux_btn.clicked.connect(self.export_flux_config)
        self.export_flux_btn.setEnabled(False)
        irr_row.addWidget(self.export_flux_btn)
        disk_layout.addLayout(irr_row)

        # Scrollable disk container
        self.scroll_area = QScrollArea()
        self.disk_container = QWidget()
        self.disk_container_layout = QVBoxLayout(self.disk_container)
        self.scroll_area.setWidget(self.disk_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(500)
        disk_layout.addWidget(self.scroll_area)

        disk_management_group.setLayout(disk_layout)
        return disk_management_group

    def _create_calculations_section(self):
        """Create the calculations section."""
        calc_group = QGroupBox("Calculations")
        calc_layout = QVBoxLayout()

        # Regression calculation buttons
        regression_controls = QHBoxLayout()
        self.calculate_2d_btn = QPushButton("Calculate Planar Regression")
        self.calculate_2d_btn.clicked.connect(lambda: self.calculate_regression('2d'))
        self.calculate_3d_btn = QPushButton("Calculate 3D Regression")
        self.calculate_3d_btn.clicked.connect(lambda: self.calculate_regression('3d'))
        self.calculate_complex_3d_btn = QPushButton("Calculate Complex 3D")
        self.calculate_complex_3d_btn.clicked.connect(lambda: self.calculate_regression('3d', complex_3d=True))

        regression_controls.addWidget(self.calculate_2d_btn)
        regression_controls.addWidget(self.calculate_3d_btn)
        regression_controls.addWidget(self.calculate_complex_3d_btn)
        calc_layout.addLayout(regression_controls)

        advanced_controls = QHBoxLayout()
        self.calculate_individual_btn = QPushButton("Individual Disk Regressions")
        self.calculate_individual_btn.clicked.connect(self.calculate_individual_disk_regressions)
        self.compare_methods_btn = QPushButton("Compare Methods")
        self.compare_methods_btn.clicked.connect(self.compare_regression_methods)
        self.compare_methods_btn.setEnabled(False)
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)

        advanced_controls.addWidget(self.calculate_individual_btn)
        advanced_controls.addWidget(self.compare_methods_btn)
        advanced_controls.addWidget(self.export_btn)
        calc_layout.addLayout(advanced_controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        calc_layout.addWidget(self.progress_bar)

        calc_group.setLayout(calc_layout)
        return calc_group

    def _create_extrapolation_section(self):
        """Create the extrapolation section."""
        extrap_group = QGroupBox("3D Extrapolation")
        extrap_layout = QVBoxLayout()

        # Height input
        height_layout = QFormLayout()
        self.extrap_z = QDoubleSpinBox()
        self.extrap_z.setRange(-50, 50)
        self.extrap_z.setDecimals(2)
        self.extrap_z.setValue(0.0)
        self.extrap_z.setSuffix(" mm")
        height_layout.addRow("Prediction Height:", self.extrap_z)
        extrap_layout.addLayout(height_layout)

        # Disk design selection
        disk_extrap_layout = QFormLayout()
        self.extrap_disk_combo = QComboBox()
        self.extrap_disk_combo.addItems(DiskCoordinates.get_disk_designs().keys())
        disk_extrap_layout.addRow("Disk Design:", self.extrap_disk_combo)
        extrap_layout.addLayout(disk_extrap_layout)

        # Prediction button
        self.predict_disk_btn = QPushButton("Predict Entire Disk")
        self.predict_disk_btn.clicked.connect(self.predict_entire_disk)
        self.predict_disk_btn.setEnabled(False)
        extrap_layout.addWidget(self.predict_disk_btn)

        # Results display
        self.extrap_results = QTextEdit()
        self.extrap_results.setMinimumHeight(150)
        self.extrap_results.setPlaceholderText("Extrapolation results will appear here...")
        extrap_layout.addWidget(self.extrap_results)

        extrap_group.setLayout(extrap_layout)
        return extrap_group

    def create_menu_bar(self):
        """Create menu bar with File, Configuration, and Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        save_action = QAction('Save Session', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_session)
        file_menu.addAction(save_action)
        load_action = QAction('Load Session', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_session)
        file_menu.addAction(load_action)
        import_fm_action = QAction('Import Flux monitors export (.xlsx)...', self)
        import_fm_action.triggered.connect(lambda: self.import_flux_monitors_export())
        file_menu.addAction(import_fm_action)

        # Configuration menu
        config_menu = menubar.addMenu('Configuration')
        hole_stack_action = QAction('Configure Hole Stacking', self)
        hole_stack_action.triggered.connect(self.show_hole_stacking_dialog)
        config_menu.addAction(hole_stack_action)

        # Help menu
        help_menu = menubar.addMenu('Help')
        validate_action = QAction('Model summary', self)
        validate_action.triggered.connect(self.show_model_summary)
        help_menu.addAction(validate_action)

        menubar.setNativeMenuBar(False)

    def create_results_table(self):
        """Create the results display table."""
        self.results_widget = QTableWidget()
        self.results_widget.setColumnCount(15)
        self._update_results_table_headers()
        font = QFont()
        font.setBold(True)
        self.results_widget.horizontalHeader().setFont(font)
        self.results_widget.setMinimumHeight(350)
        header = self.results_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

    def _update_results_table_headers(self):
        """Update results table column headers to reflect the current output sigma.

        Measured columns: the monitor's own value (used for monitor holes).
        Plane columns: the fitted surface at every hole (used for unknowns;
        for monitors it is shown for comparison, with the residual)."""
        if not hasattr(self, 'results_widget'):
            return
        sigma_str = f"{self.output_sigma}σ"
        self.results_widget.setHorizontalHeaderLabels([
            "Sample Name", "Disk", "Hole #",
            "Measured J",    f"Measured J Error ({sigma_str})",
            "Measured R",    f"Measured R Error ({sigma_str})",
            "Plane J",       f"Plane J Error ({sigma_str})",  f"Plane J % Error ({sigma_str})",
            "Plane R",       f"Plane R Error ({sigma_str})",  f"Plane R % Error ({sigma_str})",
            "Residual % (measured − plane)", "Type (value used)",
        ])

    # ── Disk management methods ──────────────────────────────────────────────

    def add_disk(self):
        """Add a new disk to the configuration."""
        disk_number = len(self.disk_widgets) + 1
        disk_widget = DiskWidget(disk_number, self)

        # Populate any custom designs that were imported this session
        for name in getattr(self, 'custom_disk_designs', {}):
            if disk_widget.disk_combo.findText(name) == -1:
                disk_widget.disk_combo.addItem(name)

        if self.disk_widgets:
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.disk_container_layout.addWidget(separator)

        self.disk_container_layout.addWidget(disk_widget)
        self.disk_widgets.append(disk_widget)

        self._update_button_states()
        self.update_interface_for_disk_count()

    def remove_disk(self):
        """Remove the last disk from the configuration."""
        if self.disk_widgets:
            disk_widget = self.disk_widgets.pop()

            layout = self.disk_container_layout
            for i in range(layout.count() - 1, -1, -1):
                item = layout.itemAt(i)
                if item and item.widget() == disk_widget:
                    layout.takeAt(i)
                    break
            disk_widget.deleteLater()

            # Remove the separator that preceded this disk
            if layout.count() > 0:
                last_item = layout.itemAt(layout.count() - 1)
                w = last_item.widget() if last_item else None
                if isinstance(w, QFrame) and w.frameShape() == QFrame.HLine:
                    layout.takeAt(layout.count() - 1).widget().deleteLater()

        self._update_button_states()
        self.update_interface_for_disk_count()

    def remove_specific_disk(self, disk_widget):
        """Remove a specific disk widget (called from that disk's own Remove button)."""
        if disk_widget not in self.disk_widgets:
            return

        layout = self.disk_container_layout

        # Find this widget's position in the layout
        disk_layout_idx = -1
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() == disk_widget:
                disk_layout_idx = i
                break

        if disk_layout_idx == -1:
            return

        layout.takeAt(disk_layout_idx)
        self.disk_widgets.remove(disk_widget)
        disk_widget.deleteLater()

        # Remove the adjacent separator.
        # Each non-first disk has a separator immediately before it in the layout.
        # After the disk is gone, that separator sits at disk_layout_idx - 1.
        # If this was the first disk (idx 0), the separator that followed it is now at idx 0.
        sep_idx = disk_layout_idx - 1 if disk_layout_idx > 0 else 0
        if sep_idx < layout.count():
            sep_item = layout.itemAt(sep_idx)
            w = sep_item.widget() if sep_item else None
            if isinstance(w, QFrame) and w.frameShape() == QFrame.HLine:
                layout.takeAt(sep_idx).widget().deleteLater()

        self.renumber_disks()
        self._update_button_states()
        self.update_interface_for_disk_count()

    def _update_button_states(self):
        """Update button enabled states based on disk count."""
        has_multiple_disks = len(self.disk_widgets) > 1
        self.calculate_3d_btn.setEnabled(has_multiple_disks)
        self.calculate_complex_3d_btn.setEnabled(has_multiple_disks)
        self.compare_methods_btn.setEnabled(has_multiple_disks)
        for dw in self.disk_widgets:
            if hasattr(dw, 'remove_btn'):
                dw.remove_btn.setEnabled(has_multiple_disks)

    def _update_input_table_headers(self):
        """Propagate regression mode change to all disk widget table headers."""
        for dw in self.disk_widgets:
            dw._set_data_table_headers()

    def update_interface_for_disk_count(self):
        """Update interface elements based on number of disks."""
        has_multiple_disks = len(self.disk_widgets) > 1

        # Disable (not hide) 3D buttons when only one disk
        self.calculate_3d_btn.setEnabled(has_multiple_disks)
        self.calculate_complex_3d_btn.setEnabled(has_multiple_disks)

        # Update button text
        if has_multiple_disks:
            self.calculate_2d_btn.setText("Calculate 2D Regression")
        else:
            self.calculate_2d_btn.setText("Calculate Planar Regression")

        # Show/hide height controls
        for disk_widget in self.disk_widgets:
            if hasattr(disk_widget, 'height_widget'):
                disk_widget.height_widget.setVisible(has_multiple_disks)

    def reorder_disks(self, source_disk_num, target_disk_num):
        """Reorder disks via drag and drop."""
        try:
            source_widget = None
            target_widget = None

            for widget in self.disk_widgets:
                if widget.disk_number == source_disk_num:
                    source_widget = widget
                elif widget.disk_number == target_disk_num:
                    target_widget = widget

            if source_widget and target_widget:
                self._perform_disk_reorder(source_widget, target_widget)
                self.renumber_disks()

        except Exception as e:
            QMessageBox.warning(self, "Reorder Error", f"Failed to reorder disks: {str(e)}")

    def _perform_disk_reorder(self, source_widget, target_widget):
        """Perform the actual disk widget reordering."""
        layout = self.disk_container_layout
        source_index = -1
        target_index = -1

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() == source_widget:
                source_index = i
            elif item.widget() == target_widget:
                target_index = i

        if source_index >= 0 and target_index >= 0:
            layout.removeWidget(source_widget)
            layout.insertWidget(target_index, source_widget)

            self.disk_widgets.remove(source_widget)
            target_list_index = self.disk_widgets.index(target_widget)
            self.disk_widgets.insert(target_list_index, source_widget)

    def renumber_disks(self):
        """Renumber all disks after reordering."""
        for i, widget in enumerate(self.disk_widgets):
            widget.update_disk_number(i + 1)
            widget.load_disk_coordinates(preserve_data=True)

    # ── Regression calculation methods ───────────────────────────────────────

    def calculate_regression(self, regression_type='2d', complex_3d=False):
        """Calculate regression using Monte Carlo uncertainty propagation."""
        try:
            all_standard_data, all_disk_data = self._collect_regression_data()

            if not self._validate_regression_data(all_disk_data, all_standard_data, complex_3d,
                                                  regression_type):
                return

            # Prepare data arrays, use R or J depending on regression mode
            coords_3d = np.array([data['coordinates_3d'] for data in all_standard_data])
            if self.regression_mode == MODE_R:
                values = np.array([data['r_value'] for data in all_standard_data])
                errors = np.array([data['r_error'] for data in all_standard_data])
            else:
                values = np.array([data['j_value'] for data in all_standard_data])
                errors = np.array([data['j_error'] for data in all_standard_data])

            # Start Monte Carlo calculation
            self._start_monte_carlo_regression(coords_3d, values, errors, regression_type,
                                               complex_3d, all_disk_data)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")
            self.enable_all_buttons()
            self.progress_bar.setVisible(False)

    def _collect_regression_data(self):
        """Collect standard data and disk data for regression."""
        all_standard_data = []
        all_disk_data = []

        for disk_widget in self.disk_widgets:
            disk_data = disk_widget.get_disk_data()
            if disk_data['enabled']:
                all_disk_data.append(disk_data)
                all_standard_data.extend(disk_data['standard_data'])

        return all_standard_data, all_disk_data

    @staticmethod
    def _n_heights(standards):
        """Number of distinct disk heights among the monitors."""
        return len({round(float(s['coordinates_3d'][2]), 6) for s in standards})

    def _validate_regression_data(self, all_disk_data, all_standard_data, complex_3d,
                                  regression_type='2d'):
        """Validate data requirements for regression."""
        if len(all_disk_data) == 0:
            QMessageBox.warning(self, "Error", "No disks are enabled for regression!")
            return False

        if len(all_standard_data) < 3:
            QMessageBox.warning(self, "Error", "Need at least 3 standard holes for regression!")
            return False

        if complex_3d and len(all_standard_data) < 8:
            QMessageBox.warning(self, "Error", "Need at least 8 standard holes for complex 3D regression!")
            return False

        # A 3D fit with every monitor at one height has no information on the
        # vertical term: it would be fitted to the Monte Carlo position shifts.
        if regression_type == '3d' and self._n_heights(all_standard_data) < 2:
            QMessageBox.warning(self, "Error",
                                "3D regression needs monitors on at least two disk heights, "
                                "but all monitors are at the same height.\n\n"
                                "Set each disk's height, or use the planar or "
                                "individual-disk regression.")
            return False

        return True

    def _start_monte_carlo_regression(self, coords_3d, values, errors, regression_type,
                                      complex_3d, all_disk_data):
        """Start Monte Carlo regression calculation thread."""
        self.progress_bar.setVisible(True)
        self._disable_all_buttons()

        self.mc_thread = MonteCarloThread(coords_3d, values, errors, regression_type, complex_3d)
        self.mc_thread.progress.connect(self.progress_bar.setValue)
        self.mc_thread.finished.connect(
            lambda results: self.on_monte_carlo_finished(results, regression_type, complex_3d, all_disk_data)
        )
        self.mc_thread.start()

    def on_monte_carlo_finished(self, mc_results, regression_type, complex_3d, all_enabled_disk_data):
        """Process completed Monte Carlo regression results."""
        try:
            self.progress_bar.setVisible(False)
            self.enable_all_buttons()

            if len(mc_results) == 0:
                QMessageBox.warning(self, "Error", "Monte Carlo simulation failed!")
                return

            # Calculate mean coefficients and uncertainties
            mean_coeffs = np.mean(mc_results, axis=0)
            std_coeffs = np.std(mc_results, axis=0)
            _all_std = [s for dd in all_enabled_disk_data for s in dd['standard_data']]
            std_coeffs = std_coeffs * self._lack_of_fit(
                _all_std, regression_type, complex_3d)[2]

            self.regression_info = {
                'type': regression_type,
                'complex_3d': complex_3d,
                'coefficients': mean_coeffs,
                'coefficient_errors': std_coeffs
            }
            self._last_regression_type = regression_type
            self._last_complex_3d      = complex_3d

            # Process results and update table
            processed_disk_data = self._process_regression_results(all_enabled_disk_data)

            self.export_btn.setEnabled(True)
            self.predict_disk_btn.setEnabled(True)
            self.export_flux_btn.setEnabled(True)

            self.plot_widget.output_sigma    = self.output_sigma
            self.plot_widget.regression_mode = self.regression_mode

            if regression_type == '2d':
                self.plot_widget.plot_2d_regression(processed_disk_data, mode=self.regression_mode)
            else:
                self.plot_widget.plot_3d_regression(processed_disk_data)

            self._show_regression_success_message(regression_type, complex_3d, processed_disk_data)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Post-processing failed: {str(e)}")
            self.progress_bar.setVisible(False)
            self.enable_all_buttons()

    def _process_regression_results(self, all_enabled_disk_data):
        """Process regression results and populate results table."""
        # Prepare disk data with numbering
        all_disk_data_for_table = []
        for disk_idx, disk_widget in enumerate(self.disk_widgets):
            disk_data = disk_widget.get_disk_data()
            if disk_data['enabled']:
                disk_data['disk_number'] = disk_idx + 1
                all_disk_data_for_table.append(disk_data)

        # Setup results table
        total_rows = sum(len(disk_data['all_data']) for disk_data in all_disk_data_for_table)
        self.results_widget.setRowCount(total_rows)

        # Process each disk
        processed_disk_data = []
        row_idx = 0

        for disk_data in all_disk_data_for_table:
            row_idx = self._process_disk_regression_results(disk_data, row_idx)
            processed_disk_data.append(disk_data)

        return processed_disk_data

    def _process_disk_regression_results(self, disk_data, row_idx):
        """Populate the results table for one disk.

        Standards report their measured value and error directly.
        Unknowns use the Monte Carlo prediction mean and standard deviation.
        Both J and R are computed; which is primary depends on regression_mode.
        """
        regression_results = []
        regression_errors  = []
        mode = getattr(self, 'regression_mode', MODE_J)

        const = self.monitor_ingrowth()

        # Coefficients must be set before iterating so the residual calculation can read them
        disk_data['coefficients'] = self.regression_info['coefficients']

        for hole_data in disk_data['all_data']:
            x, y, z = hole_data['coordinates_3d']

            # The Predicted columns always show the regression-plane output at this
            # position: for standards and unknowns alike. For standards the measured
            # value is stored in hole_data and used separately to compute the residual;
            # for unknowns there is no measured value so the residual stays "N/A".
            pred_val, pred_err = self.predict_j_value(x, y, z)
            if pred_val is None:
                j_pred = j_error = r_pred = r_error = None
            else:
                if mode == MODE_R:
                    r_pred  = pred_val
                    r_error = pred_err
                    j_pred  = const / r_pred if r_pred else None
                    j_error = j_pred * (r_error / r_pred) if (j_pred and r_pred) else None
                else:
                    j_pred  = pred_val
                    j_error = pred_err
                    r_pred  = const / j_pred if j_pred else None
                    r_error = r_pred * (j_error / j_pred) if (r_pred and j_pred) else None

            primary_val = r_pred if mode == MODE_R else j_pred
            primary_err = r_error if mode == MODE_R else j_error
            regression_results.append(primary_val)
            regression_errors.append(primary_err)

            self._populate_results_table_row(row_idx, hole_data, disk_data,
                                             j_pred, j_error, r_pred, r_error, mode)
            row_idx += 1

        disk_data['regression_values'] = regression_results
        disk_data['regression_errors']  = regression_errors
        return row_idx

    def _populate_results_table_row(self, row_idx, hole_data, disk_data,
                                    j_pred, j_error, r_pred, r_error, mode):
        """Fill one row of the results table.

        Measured columns show the value entered for the hole (J or R depending
        on mode) and its conversion to the other quantity with the current
        calibration.  A monitor keeps its measured value; the Plane columns show
        the fitted surface at every hole, and the residual compares the two.
        """
        def _fmt(v, fmt=None):
            if v is None:
                return "N/A"
            if fmt is not None:
                return f"{v:{fmt}}"
            # 8 significant figures, fixed-decimal notation, trailing zeros preserved
            if v == 0:
                return "0.00000000"
            mag = int(math.floor(math.log10(abs(v))))
            dp = max(0, 7 - mag)
            return f"{v:.{dp}f}"

        output_sigma = self.output_sigma
        disk_label   = str(disk_data['disk_number'])
        primary_key  = 'r_value' if mode == MODE_R else 'j_value'
        is_std       = hole_data['is_standard'] and hole_data[primary_key] is not None
        row_type     = ("Monitor (measured value)" if is_std
                        else "Unknown (plane value)")

        # All errors are stored internally as 1σ; scale to output_sigma for display.
        # Both J and R are shown: the one entered and its conversion.
        in_j    = hole_data.get('j_value')
        in_j_e  = (hole_data['j_error'] * output_sigma
                   if hole_data.get('j_error') is not None else None)
        in_r    = hole_data.get('r_value')
        in_r_e  = (hole_data['r_error'] * output_sigma
                   if hole_data.get('r_error') is not None else None)

        j_error_out = j_error * output_sigma if j_error is not None else None
        r_error_out = r_error * output_sigma if r_error is not None else None

        # Residual for standards: (measured − plane_prediction) / measured × 100
        residual_str = "N/A"
        if is_std:
            coeffs = disk_data.get('coefficients')
            if coeffs is not None:
                x, y, z    = hole_data['coordinates_3d']
                reg_type   = getattr(self, '_last_regression_type', '2d')
                complex_3d = getattr(self, '_last_complex_3d', False)
                try:
                    if reg_type in ('2d', 'individual'):
                        plane_val = coeffs[0]*x + coeffs[1]*y + coeffs[2]
                    elif not complex_3d:
                        plane_val = coeffs[0]*x + coeffs[1]*y + coeffs[2]*z + coeffs[3]
                    else:
                        plane_val = (coeffs[0]*x + coeffs[1]*y + coeffs[2]*z + coeffs[3]
                                     + coeffs[4]*x*z + coeffs[5]*y*z + coeffs[6]*x*y
                                     + coeffs[7]*x*y*z)
                    measured = in_r if mode == MODE_R else in_j
                    if measured:
                        residual_str = f"{(measured - plane_val) / measured * 100:.3f}"
                except Exception:
                    pass

        pct_j = f"{j_error_out / j_pred * 100:.5f}" if (j_pred and j_error_out) else "N/A"
        pct_r = f"{r_error_out / r_pred * 100:.5f}" if (r_pred and r_error_out) else "N/A"

        self.results_widget.setItem(row_idx, 0,  QTableWidgetItem(hole_data['sample_name']))
        self.results_widget.setItem(row_idx, 1,  QTableWidgetItem(disk_label))
        self.results_widget.setItem(row_idx, 2,  QTableWidgetItem(str(hole_data['hole_number'])))
        self.results_widget.setItem(row_idx, 3,  QTableWidgetItem(_fmt(in_j)))
        self.results_widget.setItem(row_idx, 4,  QTableWidgetItem(fsig(in_j_e,      6) if in_j_e is not None else 'N/A'))
        self.results_widget.setItem(row_idx, 5,  QTableWidgetItem(_fmt(in_r)))
        self.results_widget.setItem(row_idx, 6,  QTableWidgetItem(fsig(in_r_e,      6) if in_r_e is not None else 'N/A'))
        self.results_widget.setItem(row_idx, 7,  QTableWidgetItem(_fmt(j_pred)))
        self.results_widget.setItem(row_idx, 8,  QTableWidgetItem(fsig(j_error_out, 6) if j_error_out is not None else 'N/A'))
        self.results_widget.setItem(row_idx, 9,  QTableWidgetItem(pct_j))
        self.results_widget.setItem(row_idx, 10, QTableWidgetItem(_fmt(r_pred)))
        self.results_widget.setItem(row_idx, 11, QTableWidgetItem(fsig(r_error_out, 6) if r_error_out is not None else 'N/A'))
        self.results_widget.setItem(row_idx, 12, QTableWidgetItem(pct_r))
        self.results_widget.setItem(row_idx, 13, QTableWidgetItem(residual_str))
        self.results_widget.setItem(row_idx, 14, QTableWidgetItem(row_type))

        # Highlight the value that is used: the measured one for a monitor,
        # the plane for an unknown.
        if is_std:
            primary_col = 5 if mode == MODE_R else 3
        else:
            primary_col = 10 if mode == MODE_R else 7
        item = self.results_widget.item(row_idx, primary_col)
        if item:
            item.setBackground(QColor("#e8f4fd") if mode == MODE_R else QColor("#edfced"))

    def _show_regression_success_message(self, regression_type, complex_3d, processed_disk_data):
        """Show success message for regression calculation."""
        regression_label = regression_type.upper()
        if complex_3d:
            regression_label += " Complex"

        enabled_count = len(processed_disk_data)
        total_count = len(self.disk_widgets)
        status_msg = f"{regression_label} regression calculation completed!\n"
        status_msg += f"Used {enabled_count} of {total_count} disks in regression."
        _all_std = [s for dd in processed_disk_data for s in dd['standard_data']]
        status_msg += "\n\n" + self._lack_of_fit_text(
            "Fit", _all_std, regression_type, complex_3d)

        QMessageBox.information(self, "Success", status_msg)

    def calculate_individual_disk_regressions(self):
        """Calculate planar regression for each disk individually using Monte Carlo."""
        try:
            enabled_disks = []
            for disk_idx, disk_widget in enumerate(self.disk_widgets):
                disk_data = disk_widget.get_disk_data()
                if disk_data['enabled'] and len(disk_data['standard_data']) >= 3:
                    disk_data['disk_number'] = disk_idx + 1
                    enabled_disks.append(disk_data)

            if len(enabled_disks) == 0:
                QMessageBox.warning(self, "Error", "No enabled disks with sufficient standards!")
                return

            # Start Monte Carlo calculation
            self.progress_bar.setVisible(True)
            self._disable_all_buttons()

            self.individual_mc_thread = IndividualDiskMonteCarloThread(enabled_disks, mode=self.regression_mode)
            self.individual_mc_thread.progress.connect(self.progress_bar.setValue)
            self.individual_mc_thread.finished.connect(
                lambda results: self.on_individual_monte_carlo_finished(results, enabled_disks)
            )
            self.individual_mc_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Individual disk regression failed: {str(e)}")
            self.progress_bar.setVisible(False)
            self.enable_all_buttons()

    def on_individual_monte_carlo_finished(self, all_disk_results, enabled_disks):
        """Process Monte Carlo results for individual disk regressions."""
        try:
            self.progress_bar.setVisible(False)
            self.enable_all_buttons()

            if len(all_disk_results) == 0:
                QMessageBox.warning(self, "Error", "Monte Carlo simulation failed!")
                return

            # Process each disk's results
            processed_disks = self._process_individual_disk_results(all_disk_results, enabled_disks)

            self.regression_info = {
                'type': 'individual',
                'disk_data': processed_disks
            }
            self._last_regression_type = 'individual'
            self._last_complex_3d      = False

            self.export_btn.setEnabled(True)
            self.predict_disk_btn.setEnabled(True)
            self.export_flux_btn.setEnabled(True)

            self.plot_widget.output_sigma    = self.output_sigma
            self.plot_widget.regression_mode = self.regression_mode
            self.plot_widget.plot_individual_disk_regressions(processed_disks, mode=self.regression_mode)

            fit_lines = "\n".join(
                self._lack_of_fit_text(dd.get('disk_label') or f"Disk {dd.get('disk_number', '?')}",
                                       dd['standard_data'], '2d', False)
                for dd in processed_disks)
            QMessageBox.information(self, "Success",
                                  f"Individual regressions calculated for {len(processed_disks)} disks using Monte Carlo!\n"
                                  f"Each disk calculated independently with {MONTE_CARLO_ITERATIONS:,} iterations.\n\n"
                                  + fit_lines)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Individual disk post-processing failed: {str(e)}")
            self.progress_bar.setVisible(False)
            self.enable_all_buttons()

    def _process_individual_disk_results(self, all_disk_results, enabled_disks):
        """Process results for individual disk regressions."""
        # Setup results table
        total_rows = sum(len(disk_data['all_data']) for disk_data in enabled_disks)
        self.results_widget.setRowCount(total_rows)

        row_idx = 0
        processed_disks = []

        for disk_data, disk_mc_results in zip(enabled_disks, all_disk_results):
            if len(disk_mc_results) == 0:
                continue

            # Calculate coefficients and errors
            mean_coeffs = np.mean(disk_mc_results, axis=0)
            std_coeffs = np.std(disk_mc_results, axis=0) * self._lack_of_fit(
                disk_data['standard_data'], '2d', False)[2]

            disk_data['individual_plane_coeffs'] = mean_coeffs
            disk_data['coefficients'] = mean_coeffs
            disk_data['coefficient_errors'] = std_coeffs

            # Process each hole in this disk
            row_idx = self._process_individual_disk_holes(disk_data, row_idx)
            processed_disks.append(disk_data)

        return processed_disks

    def _process_individual_disk_holes(self, disk_data, row_idx):
        """Process holes for individual disk regression.

        All holes, standards and unknowns, report the regression plane
        prediction at their position. For standards the measured value is
        available in hole_data and used to compute the residual; for unknowns
        the residual is "N/A".
        """
        regression_results = []
        regression_errors  = []
        mode = getattr(self, 'regression_mode', MODE_J)

        const = self.monitor_ingrowth()

        for hole_data in disk_data['all_data']:
            x, y = hole_data['coordinates_2d']

            pred_val, pred_err = self.predict_j_value(
                x, y, disk_data['height'],
                specific_standards=disk_data['standard_data'],
                force_regression_type='2d')
            if pred_val is None:
                j_pred = j_error = r_pred = r_error = None
            else:
                if mode == MODE_R:
                    r_pred  = pred_val
                    r_error = pred_err
                    j_pred  = const / r_pred if r_pred else None
                    j_error = j_pred * (r_error / r_pred) if (j_pred and r_pred) else None
                else:
                    j_pred  = pred_val
                    j_error = pred_err
                    r_pred  = const / j_pred if j_pred else None
                    r_error = r_pred * (j_error / j_pred) if (r_pred and j_pred) else None
            primary_val = r_pred if mode == MODE_R else j_pred
            primary_err = r_error if mode == MODE_R else j_error
            regression_results.append(primary_val)
            regression_errors.append(primary_err)

            self._populate_results_table_row(row_idx, hole_data, disk_data,
                                             j_pred, j_error, r_pred, r_error, mode)
            row_idx += 1

        disk_data['regression_values'] = regression_results
        disk_data['regression_errors']  = regression_errors
        return row_idx

    # ── Prediction and extrapolation methods ─────────────────────────────────

    def predict_j_value(self, x, y, z, n_iterations=15000, specific_standards=None, force_regression_type=None):
        """Predict J or R at (x, y, z) using Monte Carlo uncertainty propagation.

        Returns (mean, 1σ std) of the prediction distribution, or (None, None)
        if insufficient standards are available.
        """
        if force_regression_type is not None:
            regression_type = force_regression_type
            complex_3d = False
        elif hasattr(self, 'regression_info'):
            regression_info = self.regression_info
            regression_type = regression_info.get('type')
            complex_3d = regression_info.get('complex_3d', False)
        else:
            return None, None

        standards = specific_standards if specific_standards is not None else \
            self._get_correct_standards_for_prediction(x, y, z, regression_type)

        if len(standards) < 3:
            return None, None

        predictions = self._run_monte_carlo_prediction(standards, x, y, z, regression_type,
                                                       complex_3d, n_iterations)
        if len(predictions) < 100:
            return None, None

        predictions = np.array(predictions)
        _, _, scale = self._lack_of_fit(standards, regression_type, complex_3d)
        return np.mean(predictions), np.std(predictions) * scale

    def _lack_of_fit(self, standards, regression_type, complex_3d):
        """(MSWD, dof, scale) of the weighted fit through ``standards`` (see
        lack_of_fit), with the design of ``regression_type``."""
        mode = getattr(self, 'regression_mode', MODE_J)
        key = ('r_value', 'r_error') if mode == MODE_R else ('j_value', 'j_error')
        values = np.array([s[key[0]] for s in standards], float)
        errors = np.array([s[key[1]] for s in standards], float)
        if regression_type in ('2d', 'individual'):
            coords = np.array([[s['coordinates_2d'][0], s['coordinates_2d'][1]]
                               for s in standards], float)
        else:
            coords = np.array([s['coordinates_3d'] for s in standards], float)
        X, _ = self._build_prediction_matrices(coords, 0.0, 0.0, 0.0,
                                               regression_type, complex_3d)
        return lack_of_fit(X, values, errors)

    def _lack_of_fit_text(self, label, standards, regression_type, complex_3d):
        mswd, dof, scale = self._lack_of_fit(standards, regression_type, complex_3d)
        if mswd is None:
            return f"{label}: no degrees of freedom for a lack-of-fit test."
        lim = mswd_critical(dof + 1)
        txt = f"{label}: weighted MSWD {mswd:.2f} ({dof} dof, limit {lim:.2f})"
        if scale != 1.0:
            txt += f", σ × {scale:.2f}"
        return txt

    def _run_monte_carlo_prediction(self, standards, x, y, z, regression_type, complex_3d, n_iterations):
        """Monte Carlo prediction at a target position (returns mean, 1σ std)."""
        mode = getattr(self, 'regression_mode', MODE_J)

        if mode == MODE_R:
            values = np.array([s['r_value'] for s in standards])
            errors = np.array([s['r_error'] for s in standards])
        else:
            values = np.array([s['j_value'] for s in standards])
            errors = np.array([s['j_error'] for s in standards])

        if regression_type in ('2d', 'individual'):
            coords_original = np.array([[s['coordinates_2d'][0], s['coordinates_2d'][1]] for s in standards])
        else:
            coords_original = np.array([s['coordinates_3d'] for s in standards])

        predictions = []
        for _ in range(n_iterations):
            try:
                coord_perturbation = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(
                    COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
                coords_perturbed = coords_original + np.random.normal(0, coord_perturbation, coords_original.shape)
                values_perturbed = np.random.normal(values, errors)

                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    X, x_pred = self._build_prediction_matrices(
                        coords_perturbed, x, y, z, regression_type, complex_3d)
                    coeffs = _wls(X, values_perturbed, errors)
                predictions.append(np.dot(x_pred, coeffs))
            except Exception:
                continue

        return predictions

    def _build_prediction_matrices(self, coords_perturbed, x, y, z, regression_type, complex_3d):
        """Build design matrix and prediction vector for regression."""
        if regression_type == '2d' or regression_type == 'individual':
            X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], np.ones(len(coords_perturbed))])
            x_pred = np.array([x, y, 1])
        elif not complex_3d:
            X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2], np.ones(len(coords_perturbed))])
            x_pred = np.array([x, y, z, 1])
        else:
            x_p, y_p, z_p = coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2]
            X = np.column_stack([x_p, y_p, z_p, np.ones(len(coords_perturbed)),
                               x_p*z_p, y_p*z_p, x_p*y_p, x_p*y_p*z_p])
            x_pred = np.array([x, y, z, 1, x*z, y*z, x*y, x*y*z])

        return X, x_pred

    def _get_correct_standards_for_prediction(self, x, y, z, regression_type):
        """Return the standards list appropriate for the given regression type.

        Individual disk regression should pass specific_standards directly to
        predict_j_value; this method is only called for global regressions.
        """
        if regression_type == 'individual':
            return []
        all_standards = []
        for disk_widget in self.disk_widgets:
            disk_data = disk_widget.get_disk_data()
            if disk_data['enabled']:
                all_standards.extend(disk_data['standard_data'])
        return all_standards

    def export_flux_config(self):
        """Export a .jflux config: per-disk J + cross-hole covariance for the
        Plateau & isochron and Age distribution tools.

        A JOINT Monte Carlo: in every iteration the flux model is fit once and
        J is predicted at all holes in the same draw, so the per-disk covariance
        captures the real cross-hole correlation (validated against the
        analytic g_i.C.g_j form).  The fit is the one the results table shows:
        each disk's own standards for an individual-disk regression, the
        standards of every enabled disk for a global 2D / 3D one.  Holes are
        keyed by their hole number; disks by the disk label; the file is keyed
        by the irradiation name.  Monitor holes keep their measured value and
        σ (flagged in 'monitor'); unknowns take the plane.  Only the
        within-disk correlation is stored:
        exact for individual-disk fits, an understatement between disks for a
        global fit (the export message says so).
        """
        if not hasattr(self, 'regression_info'):
            QMessageBox.warning(self, "No regression",
                                "Run a regression first, then export the flux config.")
            return
        reg = self.regression_info
        rtype = reg.get('type')
        complex_3d = reg.get('complex_3d', False)
        mode = getattr(self, 'regression_mode', MODE_J)
        try:
            lam = float(self.lambda_edit.text())
            t_fc = float(self.standard_age_edit.text())
        except Exception:
            lam, t_fc = DECAY_CONSTANT, FC_AGE_DEFAULT
        try:
            t_fc_err = float(self.standard_age_err_edit.text())
        except Exception:
            t_fc_err = FC_AGE_ERROR_DEFAULT
        const = fc_ingrowth(t_fc, lam, t_fc_err)

        irr = self.irradiation_edit.text().strip() or "irradiation"
        n_iter = MONTE_CARLO_ITERATIONS   # same count as the main regression (15000)
        disks_out = []
        self.progress_bar.setVisible(True)

        enabled = [w.get_disk_data() for w in self.disk_widgets]
        enabled = [dd for dd in enabled if dd['enabled']]
        # A global fit ('2d' or '3d') is one model through the standards of
        # every enabled disk, exactly as the results table predicts it; an
        # 'individual' fit uses each disk's own standards.  The exported J
        # must come from the same fit the table shows.
        is_global = rtype not in ('individual',)

        def _std_arrays(standards):
            if rtype in ('2d', 'individual'):
                cs = np.array([[s['coordinates_2d'][0], s['coordinates_2d'][1]]
                               for s in standards])
            else:
                cs = np.array([s['coordinates_3d'] for s in standards])
            key = ('r_value', 'r_error') if mode == MODE_R else ('j_value', 'j_error')
            return (cs, np.array([s[key[0]] for s in standards]),
                    np.array([s[key[1]] for s in standards]))

        def _joint_draws(standards, hole_sets):
            """One fit per iteration through `standards`, predicted at every
            hole of every set in the same draw.  A monitor hole takes its own
            perturbed value from that draw instead of the plane, so it keeps
            its measured value and its correlation with the holes the plane
            predicts from it.  Returns one array (iterations × holes) per set,
            or None when too few draws worked."""
            cs, svals, serrs = _std_arrays(standards)
            std_idx = {id(s): i for i, s in enumerate(standards)}
            mon_cols = [[(j, std_idx[id(h)]) for j, h in enumerate(holes) if id(h) in std_idx]
                        for holes in hole_sets]
            draws = [[] for _ in hole_sets]
            for it in range(n_iter):
                try:
                    cp_pert = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(
                        COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
                    csp = cs + np.random.normal(0, cp_pert, cs.shape)
                    vp = np.random.normal(svals, serrs)
                    with warnings.catch_warnings(), np.errstate(all='ignore'):
                        warnings.simplefilter('ignore')
                        rows = [self._jflux_predict_all(csp, vp, holes, rtype, complex_3d,
                                                        serrs)
                                for holes in hole_sets]
                    if all(r is not None for r in rows):
                        for k, r in enumerate(rows):
                            for j, i in mon_cols[k]:
                                r[j] = vp[i]
                            draws[k].append(r)
                except Exception:
                    pass
                self.progress_bar.setValue(int(100 * (it + 1) / n_iter))
            if len(draws[0]) < 100:
                return None
            return [np.array(d) for d in draws]

        fit_type = '2d' if rtype == 'individual' else rtype
        if is_global:
            all_std = [s for dd in enabled for s in dd['standard_data']]
            per_disk = (_joint_draws(all_std, [dd['all_data'] for dd in enabled])
                        if len(all_std) >= 3 else None) or []
            lof = self._lack_of_fit(all_std, fit_type, complex_3d)
            jobs = [(dd, d, lof) for dd, d in zip(enabled, per_disk)]
        else:
            jobs = []
            for dd in enabled:
                if len(dd['standard_data']) < 3:
                    continue
                res = _joint_draws(dd['standard_data'], [dd['all_data']])
                if res is not None:
                    jobs.append((dd, res[0],
                                 self._lack_of_fit(dd['standard_data'], '2d', False)))

        fit_ids = ({id(s) for s in all_std} if is_global else None)
        for dd, draws, (mswd, dof, scale) in jobs:
            label = dd['disk_label']
            holes = dd['all_data']
            hole_ids = [h['hole_number'] for h in holes]
            ids = fit_ids if is_global else {id(s) for s in dd['standard_data']}
            is_mon = np.array([id(h) in ids for h in holes])
            # draws: iterations x n_holes (plane prediction; the measured value
            # at a monitor hole).  Monitor scatter about the fit beyond the
            # stated errors widens the plane draws about their mean, so σ grows
            # by √MSWD and the correlations are kept; a monitor keeps its own σ.
            if scale != 1.0:
                vmean = draws.mean(axis=0)
                draws = vmean + (draws - vmean) * np.where(is_mon, 1.0, scale)
            if mode == MODE_R:                          # convert R draws -> J draws
                draws = const / draws
            Jmean = draws.mean(axis=0)
            Jcov = np.cov(draws.T)
            # Monitors report their measured value and σ exactly (the draws
            # carry them only to Monte Carlo precision); rescaling the rows
            # and columns keeps the correlations.
            if is_mon.any():
                Jexact = np.array(Jmean, float)
                Sexact = np.sqrt(np.clip(np.diag(Jcov), 0, None))
                for j, h in enumerate(holes):
                    if not is_mon[j]:
                        continue
                    if mode == MODE_R:
                        Jexact[j] = const / h['r_value']
                        Sexact[j] = Jexact[j] * h['r_error'] / h['r_value']
                    else:
                        Jexact[j] = h['j_value']
                        Sexact[j] = h['j_error']
                Sdraw = np.sqrt(np.clip(np.diag(Jcov), 0, None))
                with np.errstate(divide='ignore', invalid='ignore'):
                    fac = np.where(is_mon & (Sdraw > 0), Sexact / Sdraw, 1.0)
                Jcov = Jcov * np.outer(fac, fac)
                Jmean = Jexact
            Jsig = np.sqrt(np.clip(np.diag(Jcov), 0, None))
            denom = np.outer(Jsig, Jsig)
            with np.errstate(divide='ignore', invalid='ignore'):
                Jcorr = np.where(denom > 0, Jcov / denom, 0.0)
            np.fill_diagonal(Jcorr, 1.0)
            # Also store each hole as its monitor ratio R = (e^{λ·t_fc} − 1)/J at
            # this file's own λ / FC.  R is the calibration-independent quantity the
            # Plateau & isochron and Age distribution tools actually use, so writing it here lets them
            # read R directly instead of re-deriving it from J on every load.  J is
            # kept for files without R; R and J stay exact
            # reciprocals through `const`, and σ_R keeps J's relative error.
            with np.errstate(divide='ignore', invalid='ignore'):
                Rmean = np.where(Jmean > 0, const / Jmean, 0.0)
                Rsig  = np.where(Jmean > 0, Rmean * Jsig / Jmean, 0.0)
            disks_out.append(dict(
                disk_id=label,
                disk_design=dd.get('disk_name', ''),
                holes=hole_ids,
                hole_xy=[[float(h['coordinates_2d'][0]), float(h['coordinates_2d'][1])]
                         for h in holes],
                R=[float(v) for v in Rmean],
                R_sigma=[float(v) for v in Rsig],
                J=[float(v) for v in Jmean],
                J_sigma=[float(v) for v in Jsig],
                J_corr=[[round(float(c), 6) for c in r] for r in Jcorr],
                monitor=[bool(m) for m in is_mon],     # measured value, not the plane
                fit_mswd=(None if mswd is None else round(mswd, 4)),
                fit_dof=int(dof),
                sigma_scale=round(float(scale), 6),
            ))
        self.progress_bar.setVisible(False)

        if not disks_out:
            QMessageBox.warning(self, "Nothing to export",
                                "No disk had >=3 standards and a valid joint MC.")
            return
        payload = dict(
            type="jflux_config", version=1, irradiation=irr,
            model=rtype + (" complex" if complex_3d else ""),
            lambdaYr=lam, stdAgeMa=t_fc, stdAgeErrMa=t_fc_err,
            stdAgeErrType="external", n_iter=n_iter, disks=disks_out,
        )
        path, _ = dfold.get_save_file_name(
            self, "Export flux config", f"{irr}.jflux",
            "Flux config (*.jflux);;All files (*)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            note = ("\n\nThis is a global fit through the standards of all disks: "
                    "the J values match the results table, but the file stores the "
                    "J correlation within each disk only, so the Plateau and "
                    "Age distribution tools treat holes on different disks as "
                    "independent. Export an individual-disk regression when "
                    "samples from several disks are to be combined."
                    if is_global and len(disks_out) > 1 else "")
            QMessageBox.information(
                self, "Exported",
                f"Wrote {len(disks_out)} disk(s) for irradiation '{irr}'.\n"
                "Load this in the Plateau & isochron or Age distribution tool to combine same-disk "
                "samples with their true J covariance." + note)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _jflux_predict_all(self, coords_perturbed, values_perturbed, holes,
                           regression_type, complex_3d, errors=None):
        """Fit one model to perturbed standards (weighted by 1/σ² when errors
        are given), predict J/R at every hole."""
        n = len(coords_perturbed)
        if errors is None:
            errors = np.ones(n)
        if regression_type in ('2d', 'individual'):
            X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], np.ones(n)])
            coeffs = _wls(X, values_perturbed, errors)
            P = np.array([[h['coordinates_2d'][0], h['coordinates_2d'][1], 1.0] for h in holes])
            return P @ coeffs
        if not complex_3d:
            X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1],
                                 coords_perturbed[:, 2], np.ones(n)])
            coeffs = _wls(X, values_perturbed, errors)
            P = np.array([[h['coordinates_3d'][0], h['coordinates_3d'][1],
                           h['coordinates_3d'][2], 1.0] for h in holes])
            return P @ coeffs
        xp, yp, zp = coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2]
        X = np.column_stack([xp, yp, zp, np.ones(n), xp*zp, yp*zp, xp*yp, xp*yp*zp])
        coeffs = _wls(X, values_perturbed, errors)
        P = np.array([[h['coordinates_3d'][0], h['coordinates_3d'][1], h['coordinates_3d'][2], 1.0,
                       h['coordinates_3d'][0]*h['coordinates_3d'][2],
                       h['coordinates_3d'][1]*h['coordinates_3d'][2],
                       h['coordinates_3d'][0]*h['coordinates_3d'][1],
                       h['coordinates_3d'][0]*h['coordinates_3d'][1]*h['coordinates_3d'][2]]
                      for h in holes])
        return P @ coeffs

    def predict_entire_disk(self):
        """Predict J and R values for an entire disk at specified height."""
        try:
            disk_name = self.extrap_disk_combo.currentText()
            all_designs = DiskCoordinates.get_disk_designs()
            all_designs.update(getattr(self, 'custom_disk_designs', {}))
            disk_data = all_designs[disk_name]
            coordinates = disk_data["coordinates"]
            stacking_cfg = getattr(self, 'hole_stacking_config', None)
            if stacking_cfg is not None:
                n_holes = len(coordinates)
                key = f'{n_holes}_hole_reference'
                ref_idx = stacking_cfg.get(key)
                if ref_idx is not None and 1 <= ref_idx <= n_holes:
                    ref_x, ref_y = coordinates[ref_idx - 1]
                    coordinates = [(x - ref_x, y - ref_y) for x, y in coordinates]
            height = self.extrap_z.value()
            mode = getattr(self, 'regression_mode', MODE_J)

            const = self.monitor_ingrowth()

            mode_label   = "R (40/39)" if mode == MODE_R else "J"
            output_sigma = self.output_sigma
            sigma_str    = f"{output_sigma}σ"
            results = [f"Predicted {mode_label} values for {disk_name} at height {height:.2f}mm:",
                       f"Errors reported as {sigma_str}",
                       "=" * 65]

            # Which fit predicts the disk.  A global 2D / 3D fit predicts any
            # position.  Individual-disk planes only describe the disks they were
            # fitted to, so the disk(s) fitted at this height are used, each with
            # its own monitors.
            reg_info = getattr(self, 'regression_info', {})
            if reg_info.get('type') == 'individual':
                fitted = reg_info.get('disk_data', [])
                at_height = [dd for dd in fitted if abs(dd['height'] - height) < 1e-6]
                if not at_height:
                    heights = ", ".join(sorted({f"{dd['height']:.2f}" for dd in fitted}))
                    results.append("No individually fitted disk is at this height "
                                   f"(fitted heights: {heights} mm).")
                    results.append("An individual-disk plane only predicts the disk it was "
                                   "fitted to; use a 3D regression to predict between or "
                                   "beyond disks.")
                fits = [(f"Disk {dd.get('disk_number', '?')} [{dd.get('disk_label', '')}], "
                         f"{len(dd['standard_data'])} monitors",
                         dict(specific_standards=dd['standard_data'],
                              force_regression_type='2d'))
                        for dd in at_height]
            else:
                fits = [(None, {})]
                if reg_info.get('type') == '2d':
                    results.append("(planar fit through all enabled disks: height not used)")

            for fit_label, fit_kw in fits:
                if fit_label:
                    results.append(fit_label)
                for i, (x, y) in enumerate(coordinates):
                    pred_val, pred_err = self.predict_j_value(x, y, height, **fit_kw)
                    if pred_val is None:
                        results.append(f"Hole {i+1:2d}: Prediction failed")
                        continue

                    if mode == MODE_R:
                        r_val = pred_val
                        r_err = pred_err  # 1σ from MC
                        j_val = const / r_val if r_val else None
                        j_err = j_val * (r_err / r_val) if (j_val and r_val) else None
                    else:
                        j_val = pred_val
                        j_err = pred_err  # 1σ from MC
                        r_val = const / j_val if j_val else None
                        r_err = r_val * (j_err / j_val) if (r_val and j_val) else None

                    # Scale to output precision for reporting
                    r_err_out = r_err * output_sigma if r_err is not None else None
                    j_err_out = j_err * output_sigma if j_err is not None else None

                    r_str = f"{r_val:.6e}±{r_err_out:.4e}" if r_val else "N/A"
                    j_str = f"{j_val:.6e}±{j_err_out:.4e}" if j_val else "N/A"
                    results.append(f"Hole {i+1:2d}:  R={r_str}   J={j_str}")

            results.append("")
            reg_info = getattr(self, 'regression_info', {})
            regression_type = reg_info.get('type', 'unknown').upper()
            if regression_type == 'INDIVIDUAL':
                regression_type = "Individual Disk Regressions (Monte Carlo)"
            elif reg_info.get('complex_3d'):
                regression_type += " Complex"

            results.append(f"Regression mode: {mode_label}")
            results.append(f"Regression type: {regression_type}")
            results.append(f"Error reporting: {sigma_str}")
            results.append(f"Constants: λ={self.lambda_edit.text()} yr⁻¹,  "
                           f"t_FC={self.standard_age_edit.text()} ± "
                           f"{self.standard_age_err_edit.text()} Ma (1σ)")
            results.append(f"Monte Carlo iterations: {MONTE_CARLO_ITERATIONS}")

            self.extrap_results.setText("\n".join(results))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Disk prediction failed: {str(e)}")

    # ── Method comparison and cross-validation ───────────────────────────────

    def compare_regression_methods(self):
        """Compare regression methods using cross-validation and statistical metrics."""
        try:
            enabled_disks = []
            all_standards = []

            for disk_widget in self.disk_widgets:
                disk_data = disk_widget.get_disk_data()
                if disk_data['enabled'] and len(disk_data['standard_data']) >= 3:
                    enabled_disks.append(disk_data)
                    all_standards.extend(disk_data['standard_data'])

            if len(enabled_disks) < 2:
                QMessageBox.warning(self, "Error", "Need at least 2 enabled disks for method comparison!")
                return

            if len(all_standards) < 8:
                QMessageBox.warning(self, "Error", "Need at least 8 total standards for meaningful comparison!")
                return

            if self._n_heights(all_standards) < 2:
                QMessageBox.warning(self, "Error",
                                    "The 3D models need monitors on at least two disk heights, "
                                    "but all enabled disks are at the same height.\n\n"
                                    "Set each disk's height before comparing methods.")
                return

            # Show progress dialog
            progress_dialog = self._create_comparison_progress_dialog()

            # Run comparisons
            results = self._run_method_comparisons(enabled_disks, all_standards, progress_dialog)

            progress_dialog.close()
            self.show_comparison_results(results, len(all_standards))

        except Exception as e:
            if 'progress_dialog' in locals():
                progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Method comparison failed: {str(e)}")

    def _create_comparison_progress_dialog(self):
        """Create progress dialog for method comparison."""
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Comparing Methods")
        progress_dialog.setModal(True)
        layout = QVBoxLayout()
        label = QLabel("Running cross-validation analysis...")
        progress_bar = QProgressBar()
        layout.addWidget(label)
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)
        progress_dialog.show()
        QApplication.processEvents()
        return progress_dialog

    def _run_method_comparisons(self, enabled_disks, all_standards, progress_dialog):
        """Run all method comparisons and return results."""
        results = {}
        layout = progress_dialog.layout()
        label = layout.itemAt(0).widget()
        progress_bar = layout.itemAt(1).widget()

        # Test Individual Disk Regressions
        progress_bar.setValue(10)
        progress_dialog.repaint()
        QApplication.processEvents()
        label.setText("Testing Individual Disk Regressions...")

        individual_rmse, individual_aic, individual_params, individual_chi2 = self.cross_validate_individual(enabled_disks)
        results['Individual'] = {
            'rmse': individual_rmse,
            'aic': individual_aic,
            'n_params': individual_params,
            'standards_per_param': len(all_standards) / individual_params if individual_params > 0 else 0,
            'chi_squared': individual_chi2
        }

        # Test Simple 3D Regression
        progress_bar.setValue(40)
        progress_dialog.repaint()
        QApplication.processEvents()
        label.setText("Testing Simple 3D Regression...")

        simple_3d_rmse, simple_3d_aic, simple_3d_chi2 = self.cross_validate_3d_regression(all_standards, complex_3d=False)
        results['Simple 3D'] = {
            'rmse': simple_3d_rmse,
            'aic': simple_3d_aic,
            'n_params': 4,
            'standards_per_param': len(all_standards) / 4,
            'chi_squared': simple_3d_chi2
        }

        # Test Complex 3D Regression (if enough standards)
        if len(all_standards) >= 8:
            progress_bar.setValue(70)
            progress_dialog.repaint()
            QApplication.processEvents()
            label.setText("Testing Complex 3D Regression...")

            complex_3d_rmse, complex_3d_aic, complex_3d_chi2 = self.cross_validate_3d_regression(all_standards, complex_3d=True)
            results['Complex 3D'] = {
                'rmse': complex_3d_rmse,
                'aic': complex_3d_aic,
                'n_params': 8,
                'standards_per_param': len(all_standards) / 8,
                'chi_squared': complex_3d_chi2
            }

        progress_bar.setValue(100)
        return results

    def cross_validate_individual(self, enabled_disks):
        """
        Cross-validate individual disk regressions using leave-one-out.

        Statistical Method:
        - For each disk, leave out one standard and predict it from the other standards of that disk only
        - Each disk is treated as an independent 2D planar regression
        - The Monte Carlo approach is the one used for the individual regression
        - Variable coordinate perturbation: 150 μm base × 0.1-3.0 scaling per iteration
        - Calculate RMSE, AIC, and chi-squared for uncertainty realism assessment

        Returns:
        --------
        rmse, aic, total_params, chi_squared_stat : floats
            Cross-validation metrics for model comparison
        """
        try:
            all_residuals = []
            all_predicted_errors = []
            all_measurement_errors = []
            total_params = 0
            log_likelihood = 0

            mode = getattr(self, 'regression_mode', MODE_J)

            for disk_data in enabled_disks:
                standards = disk_data['standard_data']
                if len(standards) < 4:  # Need at least 4 standards for leave-one-out
                    continue

                total_params += 3  # Each disk: 3 parameters (a, b, c)

                for i, test_standard in enumerate(standards):
                    # Leave-one-out: train on all except test_standard
                    train_standards = [s for j, s in enumerate(standards) if j != i]

                    coords_2d = np.array([[s['coordinates_2d'][0], s['coordinates_2d'][1]] for s in train_standards])
                    if mode == MODE_R:
                        values = np.array([s['r_value'] for s in train_standards])
                        errors = np.array([s['r_error'] for s in train_standards])
                    else:
                        values = np.array([s['j_value'] for s in train_standards])
                        errors = np.array([s['j_error'] for s in train_standards])

                    mc_predictions = []
                    for _ in range(MONTE_CARLO_ITERATIONS):
                        coord_perturbation = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
                        coords_perturbed = coords_2d + np.random.normal(0, coord_perturbation, coords_2d.shape)
                        values_perturbed = np.random.normal(values, errors)

                        X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], np.ones(len(train_standards))])
                        try:
                            with warnings.catch_warnings(), np.errstate(all='ignore'):
                                warnings.simplefilter('ignore')
                                coeffs = _wls(X, values_perturbed, errors)
                            test_x, test_y = test_standard['coordinates_2d']
                            predicted = coeffs[0] * test_x + coeffs[1] * test_y + coeffs[2]
                            mc_predictions.append(predicted)
                        except Exception:
                            continue

                    if len(mc_predictions) > 10:
                        predicted_mean = np.mean(mc_predictions)
                        predicted_error = np.std(mc_predictions)
                        actual = test_standard['r_value'] if mode == MODE_R else test_standard['j_value']

                        residual = actual - predicted_mean
                        all_residuals.append(residual)
                        all_predicted_errors.append(predicted_error)

                        measurement_error = test_standard['r_error'] if mode == MODE_R else test_standard['j_error']
                        all_measurement_errors.append(measurement_error)
                        total_error = np.sqrt(predicted_error**2 + measurement_error**2)
                        log_likelihood += -0.5 * np.log(2 * np.pi * total_error**2) - 0.5 * (residual / total_error)**2

            rmse = np.sqrt(np.mean(np.array(all_residuals)**2))
            aic = 2 * total_params - 2 * log_likelihood

            chi_squared_stat = self.calculate_uncertainty_realism(all_residuals, all_predicted_errors, all_measurement_errors)

            return rmse, aic, total_params, chi_squared_stat

        except Exception:
            return float('inf'), float('inf'), 0, float('inf')

    def cross_validate_3d_regression(self, all_standards, complex_3d=False):
        """
        Cross-validate 3D regression using leave-one-out approach.

        Systematically leaves out one standard and predicts it using all
        remaining standards. Tests both Simple 3D and Complex 3D models
        using identical Monte Carlo approach as main regression.

        Parameters:
        -----------
        all_standards : list
            All standard measurements from all enabled disks
        complex_3d : bool
            Use complex 3D model with interaction terms

        Returns:
        --------
        rmse, aic, chi_squared_stat : floats
            Cross-validation metrics for model comparison
        """
        try:
            all_residuals = []
            all_predicted_errors = []
            all_measurement_errors = []
            log_likelihood = 0
            n_params = 8 if complex_3d else 4

            if len(all_standards) <= n_params:
                return float('inf'), float('inf'), float('inf')

            mode = getattr(self, 'regression_mode', MODE_J)

            for i, test_standard in enumerate(all_standards):
                # Leave-one-out: train on all except test_standard
                train_standards = [s for j, s in enumerate(all_standards) if j != i]

                coords_3d = np.array([s['coordinates_3d'] for s in train_standards])
                if mode == MODE_R:
                    values = np.array([s['r_value'] for s in train_standards])
                    errors = np.array([s['r_error'] for s in train_standards])
                else:
                    values = np.array([s['j_value'] for s in train_standards])
                    errors = np.array([s['j_error'] for s in train_standards])

                mc_predictions = []
                for _ in range(MONTE_CARLO_ITERATIONS):
                    coord_perturbation = BASE_COORDINATE_UNCERTAINTY * np.random.uniform(COORDINATE_SCALE_MIN, COORDINATE_SCALE_MAX)
                    coords_perturbed = coords_3d + np.random.normal(0, coord_perturbation, coords_3d.shape)
                    values_perturbed = np.random.normal(values, errors)

                    if complex_3d:
                        x, y, z = coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2]
                        X = np.column_stack([x, y, z, np.ones(len(train_standards)), x*z, y*z, x*y, x*y*z])
                    else:
                        X = np.column_stack([coords_perturbed[:, 0], coords_perturbed[:, 1], coords_perturbed[:, 2], np.ones(len(train_standards))])

                    try:
                        with warnings.catch_warnings(), np.errstate(all='ignore'):
                            warnings.simplefilter('ignore')
                            coeffs = _wls(X, values_perturbed, errors)

                        test_x, test_y, test_z = test_standard['coordinates_3d']
                        if complex_3d:
                            predicted = (coeffs[0] * test_x + coeffs[1] * test_y + coeffs[2] * test_z + coeffs[3] +
                                       coeffs[4] * test_x * test_z + coeffs[5] * test_y * test_z +
                                       coeffs[6] * test_x * test_y + coeffs[7] * test_x * test_y * test_z)
                        else:
                            predicted = coeffs[0] * test_x + coeffs[1] * test_y + coeffs[2] * test_z + coeffs[3]

                        mc_predictions.append(predicted)
                    except Exception:
                        continue

                if len(mc_predictions) > 10:
                    predicted_mean = np.mean(mc_predictions)
                    predicted_error = np.std(mc_predictions)
                    actual = test_standard['r_value'] if mode == MODE_R else test_standard['j_value']

                    residual = actual - predicted_mean
                    all_residuals.append(residual)
                    all_predicted_errors.append(predicted_error)

                    measurement_error = test_standard['r_error'] if mode == MODE_R else test_standard['j_error']
                    all_measurement_errors.append(measurement_error)
                    total_error = np.sqrt(predicted_error**2 + measurement_error**2)
                    log_likelihood += -0.5 * np.log(2 * np.pi * total_error**2) - 0.5 * (residual / total_error)**2

            rmse = np.sqrt(np.mean(np.array(all_residuals)**2))
            aic = 2 * n_params - 2 * log_likelihood

            chi_squared_stat = self.calculate_uncertainty_realism(all_residuals, all_predicted_errors, all_measurement_errors)

            return rmse, aic, chi_squared_stat

        except Exception:
            return float('inf'), float('inf'), float('inf')

    def calculate_uncertainty_realism(self, residuals, predicted_errors, measurement_errors):
        """Calculate chi-squared statistic to assess uncertainty realism."""
        try:
            if len(residuals) != len(predicted_errors) or len(residuals) == 0:
                return float('inf')

            total_errors = [np.sqrt(pred_err**2 + meas_err**2)
                           for pred_err, meas_err in zip(predicted_errors, measurement_errors[:len(predicted_errors)])]

            chi_squared = sum((res / err)**2 for res, err in zip(residuals, total_errors) if err > 0)
            reduced_chi_squared = chi_squared / len(residuals) if len(residuals) > 0 else float('inf')

            return reduced_chi_squared

        except Exception:
            return float('inf')

    def show_comparison_results(self, results, n_standards):
        """Show method comparison results in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Method Comparison Results")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        title = QLabel("<h2>Statistical Method Comparison</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        results_text = QTextEdit()
        results_text.setReadOnly(True)
        results_text.setFont(mono_font(10))

        # Generate comparison table
        table_text = self._generate_comparison_table(results, n_standards)
        results_text.setPlainText(table_text)
        layout.addWidget(results_text)

        # Add buttons
        button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Results")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(table_text))
        button_layout.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_comparison_table(self, results, n_standards):
        """Generate formatted comparison table text."""
        table_text = f"Total Standards: {n_standards}\n\n"
        table_text += "Method Comparison Results:\n"
        table_text += "=" * 95 + "\n"
        table_text += f"{'Method':<15} {'CV RMSE':<12} {'AIC':<10} {'χ²ᵣₑᵈ':<8} {'Params':<8} {'Std/Param':<10} {'Rank':<6}\n"
        table_text += "-" * 95 + "\n"

        sorted_methods = sorted(results.items(), key=lambda x: x[1]['rmse'])

        for rank, (method, data) in enumerate(sorted_methods, 1):
            if data['rmse'] != float('inf'):
                chi2_str = f"{data['chi_squared']:.2f}" if data['chi_squared'] != float('inf') else "N/A"
                table_text += f"{method:<15} {data['rmse']:.4e} {data['aic']:<10.1f} {chi2_str:<8} {data['n_params']:<8} {data['standards_per_param']:<10.1f} {rank:<6}\n"
            else:
                table_text += f"{method:<15} {'Failed':<12} {'N/A':<10} {'N/A':<8} {data['n_params']:<8} {data['standards_per_param']:<10.1f} {'N/A':<6}\n"

        table_text += "\n\nMetric Explanations:\n"
        table_text += "• CV RMSE: Cross-validation Root Mean Square Error (lower = better prediction accuracy)\n"
        table_text += "• AIC: Akaike Information Criterion (lower = better balance of fit vs complexity)\n"
        table_text += (f"• χ²ᵣₑᵈ: Reduced chi-squared of the leave-one-out residuals; within the Wendt & Carl 95% "
                       f"limits for n = {n_standards} ({max(0.0, 1 - 2 * math.sqrt(2 / n_standards)):.2f}–{mswd_critical(n_standards + 1):.2f}) = realistic, "
                       "below = conservative, above = overconfident\n")
        table_text += "• Standards/Parameter: Ratio indicating overfitting risk (≥3 recommended, <2 risky)\n\n"

        # Add recommendation
        best_method = sorted_methods[0][0] if sorted_methods[0][1]['rmse'] != float('inf') else None
        if best_method:
            table_text += self._generate_method_recommendation(best_method, sorted_methods,
                                                               n_standards)
        else:
            table_text += "RECOMMENDATION: Unable to determine - all methods failed\n"

        return table_text

    def _generate_method_recommendation(self, best_method, sorted_methods, n_standards):
        """Build the recommendation block: which method won, why, and a per-method
        read on whether its stated uncertainties look realistic.

        The reduced χ² of n leave-one-out residuals is judged at the Wendt & Carl
        (1991) 95% limits for n degrees of freedom (each left-out monitor is
        predicted without itself), the rule the other tools use for the MSWD."""
        def realism(chi2):
            return mswd_classify(chi2, n_standards + 1)
        recommendation_text = f"RECOMMENDATION: {best_method}\n"
        best_data = sorted_methods[0][1]

        reasons = []
        warnings = []

        if best_data['rmse'] < min([d[1]['rmse'] for d in sorted_methods[1:] if d[1]['rmse'] != float('inf')], default=float('inf')):
            reasons.append("best cross-validation accuracy")

        if best_data['standards_per_param'] >= 3:
            reasons.append("adequate standards-to-parameters ratio")
        elif best_data['standards_per_param'] >= 2:
            warnings.append("borderline standards-to-parameters ratio")
        else:
            warnings.append("WARNING: low standards-to-parameters ratio (high overfitting risk)")

        if best_data['chi_squared'] != float('inf'):
            verdict = realism(best_data['chi_squared'])
            if verdict == 'acceptable':
                reasons.append("realistic uncertainty estimates")
            elif verdict == 'underdispersed':
                warnings.append("possibly conservative uncertainties")
            else:
                warnings.append("possibly overconfident uncertainties")

        if reasons:
            recommendation_text += f"Reasons: {', '.join(reasons)}\n"
        if warnings:
            recommendation_text += f"Notes: {', '.join(warnings)}\n"

        recommendation_text += "\nUncertainty Assessment:\n"
        for method, data in sorted_methods:
            if data['chi_squared'] != float('inf') and data['rmse'] != float('inf'):
                chi2 = data['chi_squared']
                assessment = {
                    'underdispersed': "Conservative (uncertainties larger than the misfit)",
                    'acceptable':     "Realistic uncertainty estimates",
                    'overdispersed':  "Overconfident (uncertainties smaller than the misfit)",
                }.get(realism(chi2), "n/a")
                recommendation_text += f"• {method}: χ²={chi2:.2f} - {assessment}\n"

        return recommendation_text

    # ── Dialog and UI helper methods ─────────────────────────────────────────

    def show_model_summary(self):
        """Show the Monte Carlo configuration and regression model summary."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Model summary")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        title = QLabel("<h2>Model summary</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        validation_text = QTextEdit()
        validation_text.setReadOnly(True)
        validation_text.setFont(mono_font(10))
        validation_text.setPlainText(self.model_summary_text())

        layout.addWidget(validation_text)

        button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Report")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(validation_text.toPlainText()))
        button_layout.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_hole_stacking_dialog(self):
        """Show the hole stacking alignment configuration dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Hole Stacking")
        dialog.setMinimumSize(900, 700)

        layout = QVBoxLayout()

        title_label = QLabel("<h2>Hole Stacking Configuration</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        info_label = QLabel("""
        <p><b>Configure how holes align vertically across different disk types.</b></p>
        <p>When you have multiple disks with different hole patterns, you need to specify which holes
        are vertically aligned. This is crucial for accurate 3D regression calculations.</p>
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        config_widget = QWidget()
        config_layout = QHBoxLayout(config_widget)

        disk_group = QGroupBox("Disk Designs")
        disk_group.setMaximumWidth(400)
        disk_layout = QVBoxLayout(disk_group)

        disk_mgmt_layout = QHBoxLayout()

        disk_mgmt_layout.addWidget(QLabel("View design:"))
        self.disk_selector = QComboBox()
        disk_designs = DiskCoordinates.get_disk_designs()
        for disk_name in disk_designs.keys():
            self.disk_selector.addItem(disk_name)
        disk_mgmt_layout.addWidget(self.disk_selector)

        import_btn = QPushButton("Import Design...")
        import_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; }")
        import_btn.clicked.connect(self._import_disk_design)
        disk_mgmt_layout.addWidget(import_btn)

        disk_layout.addLayout(disk_mgmt_layout)

        self.disk_info_label = QLabel()
        self.disk_info_label.setStyleSheet("font-weight: bold; margin: 5px; padding: 5px; background-color: #ecf0f1;")
        disk_layout.addWidget(self.disk_info_label)

        self.hole_display_table = QTableWidget()
        self.hole_display_table.setColumnCount(3)
        self.hole_display_table.setHorizontalHeaderLabels(["Hole #", "X (cm)", "Y (cm)"])
        self.hole_display_table.setMaximumHeight(300)
        self.hole_display_table.setAlternatingRowColors(True)
        disk_layout.addWidget(self.hole_display_table)

        self.disk_selector.currentTextChanged.connect(self._update_disk_display)

        config_layout.addWidget(disk_group)

        alignment_group = QGroupBox("Hole Alignment Configuration")
        alignment_layout = QVBoxLayout(alignment_group)

        current_label = QLabel("<b>Current Alignment:</b>")
        alignment_layout.addWidget(current_label)

        default_text = QTextEdit()
        default_text.setMaximumHeight(100)
        default_text.setReadOnly(True)
        default_text.setPlainText("""
    actual center holes (0,0 coordinates):
    • 7-hole disk: Hole 7 (center hole at 0,0)
    • 13-hole disk: Hole 13 (center hole at 0,0)
    • 21-hole disk: Hole 21 (center hole at 0,0)

    Current settings use different holes for practical alignment purposes.
        """)
        alignment_layout.addWidget(default_text)

        # Configuration options
        config_label = QLabel("<b>Configure Custom Alignment:</b>")
        alignment_layout.addWidget(config_label)

        # Reference hole selection
        ref_layout = QFormLayout()

        self.ref_7_hole = QComboBox()
        self.ref_7_hole.addItems([f"Hole {i+1}" for i in range(7)])
        self.ref_7_hole.setCurrentIndex(3)  # Hole 4 (index 3) - practical default
        ref_layout.addRow("7-hole reference:", self.ref_7_hole)

        self.ref_13_hole = QComboBox()
        self.ref_13_hole.addItems([f"Hole {i+1}" for i in range(13)])
        self.ref_13_hole.setCurrentIndex(6)  # Hole 7 (index 6) - practical default
        ref_layout.addRow("13-hole reference:", self.ref_13_hole)

        self.ref_21_hole = QComboBox()
        self.ref_21_hole.addItems([f"Hole {i+1}" for i in range(21)])
        self.ref_21_hole.setCurrentIndex(0)  # Hole 1 (index 0) - practical default
        ref_layout.addRow("21-hole reference:", self.ref_21_hole)

        alignment_layout.addLayout(ref_layout)

        # Quick center alignment button
        center_btn = QPushButton("Use True Center Holes")
        center_btn.setStyleSheet("QPushButton { background-color: #f39c12; color: white; font-weight: bold; }")
        center_btn.clicked.connect(self._set_center_holes)
        center_btn.setToolTip("Set references to actual center holes (0,0 coordinates)")
        alignment_layout.addWidget(center_btn)

        # Coordinate display
        coord_label = QLabel("<b>Selected Reference Coordinates:</b>")
        alignment_layout.addWidget(coord_label)

        self.coord_display = QTextEdit()
        self.coord_display.setMaximumHeight(100)
        self.coord_display.setReadOnly(True)
        alignment_layout.addWidget(self.coord_display)

        # Update coordinates when selection changes
        self.ref_7_hole.currentIndexChanged.connect(self._update_reference_coordinates)
        self.ref_13_hole.currentIndexChanged.connect(self._update_reference_coordinates)
        self.ref_21_hole.currentIndexChanged.connect(self._update_reference_coordinates)

        config_layout.addWidget(alignment_group)
        layout.addWidget(config_widget)

        # Action buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_hole_stacking_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        apply_btn = QPushButton("Apply Configuration")
        apply_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }")
        apply_btn.clicked.connect(lambda: self._apply_hole_stacking_config(dialog))
        button_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Initialize displays
        self._update_disk_display()
        self._update_reference_coordinates()

        dialog.exec_()

    def _import_disk_design(self):
        """Import a new disk design from file."""
        try:
            filename, _ = dfold.get_open_file_name(
                self, "Import Disk Design", "",
                "JSON files (*.json);;Text files (*.txt);;All files (*.*)"
            )

            if not filename:
                return

            # Show import dialog
            import_dialog = QDialog(self)
            import_dialog.setWindowTitle("Import Disk Design")
            import_dialog.setMinimumSize(500, 400)

            layout = QVBoxLayout()

            layout.addWidget(QLabel("<b>Import New Disk Design</b>"))
            layout.addWidget(QLabel("Enter disk design data in the format: [(x1,y1), (x2,y2), ...]"))

            # Name input
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Design name:"))
            name_input = QLineEdit()
            name_input.setPlaceholderText("e.g., 2.5mm (15 holes)")
            name_layout.addWidget(name_input)
            layout.addLayout(name_layout)

            # Diameter input
            diam_layout = QHBoxLayout()
            diam_layout.addWidget(QLabel("Diameter (mm):"))
            diam_input = QDoubleSpinBox()
            diam_input.setRange(0.1, 50.0)
            diam_input.setValue(2.4)
            diam_input.setSuffix(" mm")
            diam_layout.addWidget(diam_input)
            diam_layout.addStretch()
            layout.addLayout(diam_layout)

            # Coordinate input
            layout.addWidget(QLabel("Coordinates (cm):"))
            coord_input = QTextEdit()
            coord_input.setPlaceholderText("""Example format:
    [(0.64084, 0.37), (0, -0.74), (-0.64084, 0.37),
    (0, 0.74), (0.64084, -0.37), (-0.64084, -0.37), (0, 0)]

    Or JSON format:
    {"coordinates": [[0.64084, 0.37], [0, -0.74], ...]}""")
            layout.addWidget(coord_input)

            # Try to load file content
            try:
                with open(filename, encoding='utf-8') as f:
                    content = f.read().strip()
                    coord_input.setPlainText(content)
            except OSError:
                pass

            # Buttons
            button_layout = QHBoxLayout()

            parse_btn = QPushButton("Parse && Preview")
            parse_btn.clicked.connect(lambda: self._preview_import(coord_input.toPlainText(), import_dialog))
            button_layout.addWidget(parse_btn)

            button_layout.addStretch()

            import_btn = QPushButton("Import Design")
            import_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }")
            import_btn.clicked.connect(lambda: self._do_import_design(
                name_input.text(), diam_input.value(), coord_input.toPlainText(), import_dialog
            ))
            button_layout.addWidget(import_btn)

            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(import_dialog.reject)
            button_layout.addWidget(cancel_btn)

            layout.addLayout(button_layout)
            import_dialog.setLayout(layout)
            import_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import disk design: {str(e)}")

    def _preview_import(self, coord_text, dialog):
        """Preview imported coordinates."""
        try:
            coords = self._parse_coordinates(coord_text)
            preview_text = f"Parsed {len(coords)} coordinates:\n"
            for i, (x, y) in enumerate(coords[:10]):  # Show first 10
                preview_text += f"Hole {i+1}: ({x:.3f}, {y:.3f})\n"
            if len(coords) > 10:
                preview_text += f"... and {len(coords)-10} more holes"

            QMessageBox.information(dialog, "Preview", preview_text)
        except Exception as e:
            QMessageBox.warning(dialog, "Parse Error", f"Could not parse coordinates: {str(e)}")

    def _parse_coordinates(self, coord_text):
        """Parse coordinate text in various formats."""
        import re

        # Try JSON format first
        try:
            data = json.loads(coord_text)
            if isinstance(data, dict) and 'coordinates' in data:
                return [(float(x), float(y)) for x, y in data['coordinates']]
            elif isinstance(data, list):
                return [(float(x), float(y)) for x, y in data]
        except Exception:
            pass

        # Try Python list format
        try:
            coords = ast.literal_eval(coord_text)
            if isinstance(coords, list):
                return [(float(x), float(y)) for x, y in coords]
        except Exception:
            pass

        # Try comma-separated format
        try:
            # Extract numbers with regex
            numbers = re.findall(r'-?\d+\.?\d*', coord_text)
            if len(numbers) % 2 == 0:
                coords = []
                for i in range(0, len(numbers), 2):
                    coords.append((float(numbers[i]), float(numbers[i+1])))
                return coords
        except Exception:
            pass

        raise ValueError("Could not parse coordinate format")

    def _do_import_design(self, name, diameter, coord_text, dialog):
        """Actually import the disk design."""
        try:
            if not name.strip():
                QMessageBox.warning(dialog, "Error", "Please enter a design name")
                return

            coords = self._parse_coordinates(coord_text)

            if len(coords) < 3:
                QMessageBox.warning(dialog, "Error", "Need at least 3 coordinate pairs")
                return

            # Add to disk designs (for this session)
            if not hasattr(self, 'custom_disk_designs'):
                self.custom_disk_designs = {}

            self.custom_disk_designs[name] = {
                'diameter': diameter,
                'coordinates': coords
            }

            # Update the hole-stacking dialog combo (if open)
            if hasattr(self, 'disk_selector'):
                if self.disk_selector.findText(name) == -1:
                    self.disk_selector.addItem(name)
                self.disk_selector.setCurrentText(name)

            # Update extrapolation combo and all disk widget combos
            if self.extrap_disk_combo.findText(name) == -1:
                self.extrap_disk_combo.addItem(name)
            for dw in self.disk_widgets:
                if dw.disk_combo.findText(name) == -1:
                    dw.disk_combo.addItem(name)

            QMessageBox.information(dialog, "Success",
                                f"Imported '{name}' with {len(coords)} holes")
            dialog.accept()

        except Exception as e:
            QMessageBox.critical(dialog, "Import Error", f"Failed to import: {str(e)}")

    def _set_center_holes(self):
        """Set reference holes to the true center holes (0,0 coordinates)."""
        self.ref_7_hole.setCurrentIndex(6)   # Hole 7 (center)
        self.ref_13_hole.setCurrentIndex(12) # Hole 13 (center)
        self.ref_21_hole.setCurrentIndex(20) # Hole 21 (center)

    def _update_disk_display(self):
        """Update the disk display table based on selected disk."""
        if not hasattr(self, 'disk_selector'):
            return

        disk_name = self.disk_selector.currentText()

        # Check both built-in and custom designs
        disk_designs = DiskCoordinates.get_disk_designs()
        if hasattr(self, 'custom_disk_designs'):
            disk_designs.update(self.custom_disk_designs)

        if disk_name not in disk_designs:
            return

        disk_info = disk_designs[disk_name]
        coordinates = disk_info['coordinates']

        # Update info label
        info_text = f"<b>{disk_name}</b><br>Diameter: {disk_info['diameter']}mm<br>Holes: {len(coordinates)}"
        self.disk_info_label.setText(info_text)

        # Update table
        self.hole_display_table.setRowCount(len(coordinates))

        for i, (x, y) in enumerate(coordinates):
            # Highlight center holes (0,0)
            hole_item = QTableWidgetItem(str(i + 1))
            x_item = QTableWidgetItem(f"{x:.3f}")
            y_item = QTableWidgetItem(f"{y:.3f}")

            if abs(x) < 0.001 and abs(y) < 0.001:  # Center hole
                hole_item.setBackground(QColor("#fff2cc"))
                x_item.setBackground(QColor("#fff2cc"))
                y_item.setBackground(QColor("#fff2cc"))
                hole_item.setToolTip("Center hole (0,0)")

            self.hole_display_table.setItem(i, 0, hole_item)
            self.hole_display_table.setItem(i, 1, x_item)
            self.hole_display_table.setItem(i, 2, y_item)

        self.hole_display_table.resizeColumnsToContents()

    def _update_reference_coordinates(self):
        """Update the reference coordinate display."""
        if not hasattr(self, 'coord_display'):
            return

        # Get designs (including custom)
        disk_designs = DiskCoordinates.get_disk_designs()
        if hasattr(self, 'custom_disk_designs'):
            disk_designs.update(self.custom_disk_designs)

        designs = list(disk_designs.values())

        # Get selected hole indices
        hole_7 = self.ref_7_hole.currentIndex()
        hole_13 = self.ref_13_hole.currentIndex()
        hole_21 = self.ref_21_hole.currentIndex()

        # Get coordinates
        coord_7 = designs[0]['coordinates'][hole_7]  # 7-hole is first
        coord_13 = designs[1]['coordinates'][hole_13]  # 13-hole is second
        coord_21 = designs[2]['coordinates'][hole_21]  # 21-hole is third

        coord_text = f"""Selected reference coordinates:
    7-hole disk (Hole {hole_7+1}): ({coord_7[0]:.3f}, {coord_7[1]:.3f})
    13-hole disk (Hole {hole_13+1}): ({coord_13[0]:.3f}, {coord_13[1]:.3f})
    21-hole disk (Hole {hole_21+1}): ({coord_21[0]:.3f}, {coord_21[1]:.3f})

    These holes will be considered vertically aligned."""

        self.coord_display.setPlainText(coord_text)

    def _reset_hole_stacking_defaults(self):
        """Reset hole stacking to default configuration."""
        self.ref_7_hole.setCurrentIndex(3)   # Hole 4 (practical default)
        self.ref_13_hole.setCurrentIndex(6)  # Hole 7 (practical default)
        self.ref_21_hole.setCurrentIndex(0)  # Hole 1 (practical default)

    def _apply_hole_stacking_config(self, dialog):
        """Apply the hole stacking configuration."""
        # Get current selections
        ref_7 = self.ref_7_hole.currentIndex() + 1
        ref_13 = self.ref_13_hole.currentIndex() + 1
        ref_21 = self.ref_21_hole.currentIndex() + 1

        self.hole_stacking_config = {
            '7_hole_reference': ref_7,
            '13_hole_reference': ref_13,
            '21_hole_reference': ref_21
        }

        # Show confirmation
        QMessageBox.information(dialog, "Configuration Applied",
                            f"Hole stacking configuration updated:\n"
                            f"• 7-hole disk: Hole {ref_7} as reference\n"
                            f"• 13-hole disk: Hole {ref_13} as reference\n"
                            f"• 21-hole disk: Hole {ref_21} as reference\n\n"
                            f"These holes will be considered vertically aligned.")

        dialog.accept()

    # ── Utility and helper methods ───────────────────────────────────────────

    def enable_all_buttons(self):
        """Helper method to re-enable all calculation buttons consistently."""
        self.calculate_2d_btn.setEnabled(True)
        self.calculate_3d_btn.setEnabled(True)
        self.calculate_complex_3d_btn.setEnabled(True)
        self.calculate_individual_btn.setEnabled(True)
        self.compare_methods_btn.setEnabled(len(self.disk_widgets) > 1)

    def _disable_all_buttons(self):
        """Disable all calculation buttons during processing."""
        self.calculate_2d_btn.setEnabled(False)
        self.calculate_3d_btn.setEnabled(False)
        self.calculate_complex_3d_btn.setEnabled(False)
        self.calculate_individual_btn.setEnabled(False)
        self.compare_methods_btn.setEnabled(False)

    def export_results(self):
        """Export results table to CSV with a constants block in adjacent columns."""
        try:
            filename, _ = dfold.get_save_file_name(
                self, "Export Results", "arar_results.csv", "CSV files (*.csv)"
            )

            if filename:
                # --- Main results table ---
                export_data = []
                for row in range(self.results_widget.rowCount()):
                    row_data = []
                    for col in range(self.results_widget.columnCount()):
                        item = self.results_widget.item(row, col)
                        row_data.append(item.text() if item else "")
                    export_data.append(row_data)

                headers = [self.results_widget.horizontalHeaderItem(i).text()
                           for i in range(self.results_widget.columnCount())]
                df = pd.DataFrame(export_data, columns=headers)

                # --- Constants block ---
                try:
                    std_age = self.standard_age_edit.text()
                    lam     = self.lambda_edit.text()
                except Exception:
                    std_age = str(FC_AGE_DEFAULT)
                    lam     = f"{DECAY_CONSTANT:.4e}"

                reg_info = getattr(self, 'regression_info', {})
                reg_type = reg_info.get('type', 'N/A').upper()
                if reg_info.get('complex_3d'):
                    reg_type += ' Complex'
                mode_label = "R (40/39)" if getattr(self, 'regression_mode', MODE_J) == MODE_R else "J"

                constants = [
                    ("Standard Age (Ma)",   std_age),
                    ("lambda (yr-1)",        lam),
                    ("Input error",          f"{self.input_sigma}sigma"),
                    ("Reported error",       f"{self.output_sigma}sigma"),
                    ("Regression mode",      mode_label),
                    ("Regression type",      reg_type),
                    ("MC iterations",        str(MONTE_CARLO_ITERATIONS)),
                    ("Value used",           "monitors: measured; unknowns: plane"),
                ]
                const_df = pd.DataFrame(constants, columns=["Parameter", "Value"])

                # Pad shorter side so both have equal rows
                n_data, n_const = len(df), len(const_df)
                if n_data > n_const:
                    pad = pd.DataFrame([["", ""]] * (n_data - n_const),
                                       columns=["Parameter", "Value"])
                    const_df = pd.concat([const_df, pad], ignore_index=True)
                elif n_const > n_data:
                    pad = pd.DataFrame([[""] * len(df.columns)] * (n_const - n_data),
                                       columns=df.columns)
                    df = pd.concat([df, pad], ignore_index=True)

                # One blank separator column, then the constants block
                sep = pd.DataFrame({"": [""] * len(df)})
                combined = pd.concat([df, sep, const_df], axis=1)
                combined.to_csv(filename, index=False, encoding='utf-8-sig')

                QMessageBox.information(self, "Success", f"Results exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def save_session(self):
        """Save current session to a JSON file."""
        try:
            filename, _ = dfold.get_save_file_name(
                self, "Save Session", "arar_session.json", "JSON files (*.json)"
            )
            if not filename:
                return

            session = {
                'version': 1,
                'constants': {
                    'standard_age':     self.standard_age_edit.text(),
                    'standard_age_err': self.standard_age_err_edit.text(),
                    'lambda':           self.lambda_edit.text(),
                },
                'irradiation':         self.irradiation_edit.text(),
                'input_sigma':         self.input_sigma,
                'output_sigma':        self.output_sigma,
                'regression_mode':     self.regression_mode,
                'hole_stacking_config': getattr(self, 'hole_stacking_config', None),
                'custom_disk_designs': {},
                'disks': [],
            }

            # Serialise custom designs (coordinates are lists of [x, y] already JSON-safe)
            if hasattr(self, 'custom_disk_designs'):
                for name, design in self.custom_disk_designs.items():
                    session['custom_disk_designs'][name] = {
                        'diameter':    design.get('diameter', 0),
                        'coordinates': [list(c) for c in design['coordinates']],
                    }

            # Serialise each disk widget
            for dw in self.disk_widgets:
                holes = []
                for row in range(dw.table_widget.rowCount()):
                    sample_item = dw.table_widget.item(row, 0)
                    checkbox    = dw.table_widget.cellWidget(row, 2)
                    val_item    = dw.table_widget.item(row, 3)
                    err_item    = dw.table_widget.item(row, 4)
                    holes.append({
                        'sample_name': sample_item.text() if sample_item else "",
                        'is_standard': checkbox.isChecked() if checkbox else False,
                        'value':       val_item.text() if val_item else "",
                        'error':       err_item.text() if err_item else "",
                    })
                session['disks'].append({
                    'disk_design': dw.disk_combo.currentText(),
                    'height':      dw.height_spinbox.value(),
                    'enabled':     dw.enable_checkbox.isChecked(),
                    'holes':       holes,
                })

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2)

            QMessageBox.information(self, "Success", f"Session saved to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def import_flux_monitors_export(self, path=None):
        """Fill the monitor holes from a Flux monitors export (.xlsx).

        A category named like 'C1' goes to the disk labelled 'C', hole 1, with
        its reported R and 1σ SE, ticked for the regression.  In J mode J is
        calculated from that R with this tool's FC age and λ, so J and R stay on
        one calibration.  A disk label no disk has yet goes to the first disk
        without values, otherwise to a new disk."""
        if not path:
            path, _ = dfold.get_open_file_name(
                self, "Import Flux monitors export", "", "Excel files (*.xlsx)",
                tool='flux_monitors')
            if not path:
                return
        try:
            rows, calib = read_flux_monitors_export(path)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"{os.path.basename(path)}:\n{e}")
            return

        by_disk, skipped = {}, []
        for r in rows:
            pos = split_position_label(r['category'])
            if pos is None or not (r['R'] or 0) > 0 or not (r['R_sigma'] or 0) > 0:
                skipped.append(r['category'])
                continue
            by_disk.setdefault(pos[0], []).append((pos[1], r))
        if not by_disk:
            QMessageBox.warning(self, "Nothing imported",
                                "No category with a position name (e.g. C1) and a reported R.")
            return

        def label(w):
            return w.disk_label_edit.text().strip().upper()

        def has_values(w):
            t = w.table_widget
            return any(t.item(i, 3) and t.item(i, 3).text().strip() for i in range(t.rowCount()))

        wanted = {d.upper() for d in by_disk if d}
        target, used = {}, set()
        for d in sorted(by_disk, key=lambda d: not any(d and label(w) == d.upper()
                                                       for w in self.disk_widgets)):
            w = next((w for w in self.disk_widgets if d and label(w) == d.upper()), None)
            if w is None:
                w = next((w for w in self.disk_widgets if w not in used
                          and not has_values(w) and label(w) not in wanted), None)
                if w is None:
                    self.add_disk()
                    w = self.disk_widgets[-1]
                if d:
                    w.disk_label_edit.setText(d)
            used.add(w)
            target[d] = w

        c = self.monitor_ingrowth()
        k = self.input_sigma
        lines, recalculated, n = [], False, 0
        for d, items in by_disk.items():
            w = target[d]
            t = w.table_widget
            holes = []
            for hole, r in sorted(items, key=lambda h: h[0]):
                if not 1 <= hole <= t.rowCount():
                    skipped.append(f"{r['category']} (disk {w.disk_label_edit.text()} has a "
                                   f"{t.rowCount()}-hole design; choose its design, then import again)")
                    continue
                R, sR = r['R'], r['R_sigma']
                if self.regression_mode == MODE_R:
                    val, err = R, sR
                else:
                    val, err = c / R, c / R * sR / R
                    if r['J'] and abs(val / r['J'] - 1) > 1e-9:
                        recalculated = True
                i = hole - 1
                t.setItem(i, 0, QTableWidgetItem(r['category']))
                t.setItem(i, 3, QTableWidgetItem(repr(float(val))))
                t.setItem(i, 4, QTableWidgetItem(repr(float(err * k))))
                box = t.cellWidget(i, 2)
                if box is not None:
                    box.setChecked(True)
                holes.append(str(hole))
            if holes:
                n += len(holes)
                lines.append(f"Disk {w.disk_label_edit.text() or w.disk_number}: "
                             f"hole {', '.join(holes)}")

        msg = [f"{n} position(s) from {os.path.basename(path)}:"] + lines
        if self.regression_mode == MODE_R:
            msg.append(f"\nEach hole holds the reported R and its SE (as ±{k}σ), "
                       f"ticked for the regression.")
        else:
            msg.append(f"\nEach hole holds J = (e^(λt) − 1)/R from the reported R, with the "
                       f"FC age and λ set here, and its SE (as ±{k}σ), ticked for the regression.")
            if recalculated:
                msg.append("The J in the export used a different calibration"
                           + (f" ({calib.split(':', 1)[1].strip()})" if calib else "") + ".")
        if skipped:
            msg.append("\nNot imported: " + "; ".join(skipped))
        QMessageBox.information(self, "Flux monitors export imported", "\n".join(msg))

    def load_session(self):
        """Load a previously saved session from a JSON file."""
        try:
            filename, _ = dfold.get_open_file_name(
                self, "Load Session", "", "JSON files (*.json)"
            )
            if not filename:
                return

            with open(filename, encoding='utf-8') as f:
                session = json.load(f)

            if session.get('version') != 1:
                QMessageBox.warning(self, "Warning",
                    "Session file version mismatch, attempting to load anyway.")

            # --- Constants ---
            consts = session.get('constants', {})
            if 'standard_age' in consts:
                self.standard_age_edit.setText(consts['standard_age'])
            if 'standard_age_err' in consts:
                self.standard_age_err_edit.setText(consts['standard_age_err'])
            if 'lambda' in consts:
                self.lambda_edit.setText(consts['lambda'])
            if session.get('irradiation') is not None:
                self.irradiation_edit.setText(str(session['irradiation']))

            # --- Sigma buttons ---
            input_sigma = session.get('input_sigma', 1)
            self.input_sigma = input_sigma
            self.input_sigma_1_btn.setChecked(input_sigma == 1)
            self.input_sigma_2_btn.setChecked(input_sigma == 2)

            output_sigma = session.get('output_sigma', 1)
            self.output_sigma = output_sigma
            self.output_sigma_1_btn.setChecked(output_sigma == 1)
            self.output_sigma_2_btn.setChecked(output_sigma == 2)

            # --- Regression mode ---
            mode = session.get('regression_mode', MODE_J)
            self.regression_mode = mode
            self.mode_j_radio.setChecked(mode == MODE_J)
            self.mode_r_radio.setChecked(mode == MODE_R)
            self._update_input_table_headers()

            # --- Hole stacking config ---
            stacking = session.get('hole_stacking_config')
            if stacking:
                self.hole_stacking_config = stacking
            elif hasattr(self, 'hole_stacking_config'):
                del self.hole_stacking_config

            # --- Custom disk designs ---
            for name, design in session.get('custom_disk_designs', {}).items():
                if not hasattr(self, 'custom_disk_designs'):
                    self.custom_disk_designs = {}
                self.custom_disk_designs[name] = {
                    'diameter':    design.get('diameter', 0),
                    'coordinates': [tuple(c) for c in design['coordinates']],
                }
                # Add to all disk combo boxes if not already present
                for dw in self.disk_widgets:
                    if dw.disk_combo.findText(name) == -1:
                        dw.disk_combo.addItem(name)
                if self.extrap_disk_combo.findText(name) == -1:
                    self.extrap_disk_combo.addItem(name)

            # --- Rebuild disk widgets ---
            while self.disk_widgets:
                self.remove_disk()

            for disk_data in session.get('disks', []):
                self.add_disk()
                dw = self.disk_widgets[-1]

                # Add custom design to this widget's combo if needed
                design_name = disk_data.get('disk_design', '')
                if dw.disk_combo.findText(design_name) == -1:
                    dw.disk_combo.addItem(design_name)

                # Block signals to avoid triggering load_disk_coordinates prematurely
                dw.disk_combo.blockSignals(True)
                dw.disk_combo.setCurrentText(design_name)
                dw.disk_combo.blockSignals(False)

                dw.height_spinbox.setValue(disk_data.get('height', 0.0))
                dw.enable_checkbox.setChecked(disk_data.get('enabled', True))

                # Load coordinates for the selected design (populates the table)
                dw.load_disk_coordinates(preserve_data=False)

                # Fill in hole data
                for row, hole in enumerate(disk_data.get('holes', [])):
                    if row >= dw.table_widget.rowCount():
                        break
                    sample_item = dw.table_widget.item(row, 0)
                    if sample_item:
                        sample_item.setText(hole.get('sample_name', ''))
                    checkbox = dw.table_widget.cellWidget(row, 2)
                    if checkbox:
                        checkbox.setChecked(hole.get('is_standard', False))
                    val_item = dw.table_widget.item(row, 3)
                    if val_item:
                        val_item.setText(hole.get('value', ''))
                    else:
                        dw.table_widget.setItem(row, 3,
                            QTableWidgetItem(hole.get('value', '')))
                    err_item = dw.table_widget.item(row, 4)
                    if err_item:
                        err_item.setText(hole.get('error', ''))
                    else:
                        dw.table_widget.setItem(row, 4,
                            QTableWidgetItem(hole.get('error', '')))

            self._update_button_states()
            self.update_interface_for_disk_count()
            QMessageBox.information(self, "Success", f"Session loaded from {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def model_summary_text(self):
        """Return a summary of the current Monte Carlo configuration and regression models."""
        mode_label = "R (40/39)" if getattr(self, 'regression_mode', MODE_J) == MODE_R else "J"
        input_sigma  = getattr(self, 'input_sigma',  1)
        output_sigma = getattr(self, 'output_sigma', 1)
        lines = [
            "Monte Carlo Configuration",
            "─" * 40,
            f"Iterations:         {MONTE_CARLO_ITERATIONS:,}",
            f"Regression mode:    {mode_label}",
            f"Coordinate noise:   {BASE_COORDINATE_UNCERTAINTY*COORDINATE_SCALE_MIN*1e4:.1f}–"
            f"{BASE_COORDINATE_UNCERTAINTY*COORDINATE_SCALE_MAX*1e4:.0f} μm per iteration",
            "",
            "Error precision",
            "─" * 40,
            f"Input precision:    {input_sigma}σ  (user-entered errors divided by {input_sigma} for internal storage)",
            f"Output precision:   {output_sigma}σ  (internal 1σ values multiplied by {output_sigma} for reporting)",
            "Internal storage:   1σ throughout (MC perturbation, propagation, prediction)",
            "",
            "Regression models",
            "─" * 40,
            "2D planar:   f = ax + by + c                          (3 params, min 3 standards)",
            "Simple 3D:   f = ax + by + cz + d                     (4 params, min 4 standards)",
            "Complex 3D:  f = ax+by+cz+d+e(xz)+f(yz)+g(xy)+h(xyz) (8 params, min 8 standards)",
            "Individual:  independent 2D plane per disk             (3 params per disk)",
            "",
            "Uncertainty propagation",
            "─" * 40,
            "• Coordinate noise applied to all standard holes each iteration",
            "• Measurement uncertainty perturbs each standard's value independently",
            "• Every hole, monitor or unknown, reports the fitted-plane prediction;",
            "  monitors also show their residual (measured − predicted)",
            "",
            "Model comparison metrics",
            "─" * 40,
            "• Leave-one-out cross-validation (CV RMSE)",
            "• Akaike Information Criterion (AIC)",
            "• Reduced χ² of the leave-one-out residuals, judged against the",
            "  Wendt & Carl (1991) 95% limits 1 ± 2√(2/n)",
        ]
        return "\n".join(lines)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Flux regression")
    dfold.use_tool('flux_regression')
    force_light(app)
    window = FluxRegressionCalculator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()