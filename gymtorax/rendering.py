"""Real-time visualization and rendering system for GymTORAX plasma simulations.

This module provides a comprehensive visualization framework for TORAX plasma simulation
data within the Gymnasium environment ecosystem. It combines TORAX native plotting
capabilities with Gymnasium rendering pipeline to support both live visualization
and video recording of simulation episodes.

The system is designed to handle real-time visualization during RL training while
maintaining full compatibility with TORAX plot configurations and styling conventions.
It supports multiple rendering modes and provides optimized performance for video generation.
"""

from __future__ import annotations

import importlib
import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torax._src.plotting import plotruns_lib

from .torax_wrapper import (
    create_figure,
    load_data,
    update_lines,
    validate_plotdata,
)

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def process_plot_config(
    plot_config: str | plotruns_lib.FigureProperties,
) -> plotruns_lib.FigureProperties:
    """Load and validate TORAX plot configuration from string identifier or object.

    This function handles the loading of TORAX plot configurations, which define
    the layout, variables, and styling for plasma simulation visualizations.
    It supports both string-based configuration names (loaded from TORAX
    default configurations) and direct FigureProperties objects.

    Args:
        plot_config (str or plotruns_lib.FigureProperties): Either a string
            identifier for a default TORAX plot configuration (e.g., ``"default"``, ``"simple"``, ...)
            or a pre-configured `FigureProperties` object.

    Returns:
        plotruns_lib.FigureProperties: Validated plot configuration object
            containing axes definitions, layout properties, and styling options.

    Raises:
        TypeError: If `plot_config` is neither a string nor `FigureProperties` instance.
        ImportError: If the specified plot configuration module is not found.
        AttributeError: If the configuration module lacks the `PLOT_CONFIG` attribute.

    Note:
        String-based configurations are loaded from the TORAX package at
        ``torax.plotting.configs.{plot_config}_plot_config.PLOT_CONFIG``.
    """
    if isinstance(plot_config, str):
        try:
            module = importlib.import_module(
                f"torax.plotting.configs.{plot_config}_plot_config"
            )
        except ImportError:
            logger.error(f"""Plot config: {plot_config} not found
                        in `torax.plotting.configs`""")
            return
        try:
            plot_config = getattr(module, "PLOT_CONFIG")
        except AttributeError:
            logger.error(f"""Plot config: {plot_config} does not have a PLOT_CONFIG attribute
                        in `torax.plotting.configs`""")
            return
    elif isinstance(plot_config, plotruns_lib.FigureProperties):
        pass
    else:
        raise TypeError("config_plot must be a string or FigureProperties instance")

    return plot_config


class Plotter:
    """Real-time plotter using TORAX plot style and axis conventions.

    This class provides real-time visualization capabilities for TORAX plasma
    simulation data with support for both spatial profiles and time series plots.

    The plotter can operate in different render modes and supports live
    visualization during simulation runs. It automatically handles figure layout,
    axis scaling, and legend management based on the provided configuration.

    Attributes:
        plot_config (plotruns_lib.FigureProperties): Configuration for plot layout and styling
        fig (matplotlib.figure.Figure): Main figure object
        axes (list[matplotlib.axes.Axes]): List of matplotlib axes objects
        lines (list[matplotlib.lines.Line2D]): List of matplotlib line objects for plotting
    """

    def __init__(
        self,
        plot_config: plotruns_lib.FigureProperties,
        render_mode: str,
    ):
        """Initialize the real-time plotter with configuration and display settings.

        Sets up the plotter with the specified configuration and creates the matplotlib figure structure.

        Automatically configures font scaling based on render mode and plot layout
        to ensure optimal readability in different visualization contexts.

        Args:
            plot_config (plotruns_lib.FigureProperties): Configuration object defining
                the plot layout, variables to display, axis properties, and styling options.
                This determines which plasma variables are plotted and how they are arranged.
            render_mode (str): Rendering mode that affects font scaling and
                backend selection. Supported values are ``"human"`` for interactive display,
                ``"rgb_array"`` for video recording.

        Note:
            The figure and axes are created immediately during initialization based on
            the `plot_config`. Data histories are initialized as empty and will be populated
            through subsequent `update()` calls. Font scaling is automatically applied
            for multi-row layouts to maintain readability.
        """
        if render_mode == "rgb_array":
            rows = plot_config.rows
            font_scale = 1 + (rows - 1) * 0.3
        else:
            font_scale = 1.0
        self.plot_config = plot_config
        self.lines = []
        self.fig, self.axes = create_figure(self.plot_config, font_scale)
        self.first_update = True

    def reset(self):
        """Reset the plotter to its initial state without closing the figure.

        This method clears all accumulated data histories and resets all plot lines
        to empty state. The figure remains open and ready for new data. If in human
        render mode, the display is refreshed to show the cleared state.

        Note:
            This method is useful for starting a new simulation run while keeping
            the same plotter instance and figure window.
        """
        for line in self.lines:
            line.set_xdata([])
            line.set_ydata([])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def update(self, current_state: xr.DataTree, t: float):
        """Update the visualization with new simulation data from TORAX output.

        This method processes the latest simulation state and updates all plot lines
        with new data points.

        The method performs data validation, coordinate transformation, and line
        updates while maintaining TORAX native plotting conventions and styling.

        Args:
            current_state (xarray.DataTree): Complete simulation output from TORAX,
                containing profiles and scalars datasets with plasma variables
                (temperatures, densities, currents, etc.).
            t (float): Current simulation time in seconds. Used for
                time series data accumulation and display labeling.

        Raises:
            ValueError: If required variables specified in `plot_config` are missing
                from the `current_state` `DataTree`.

        Note:
            On first call, this method initializes all plot lines and formatting.
            Subsequent calls efficiently update existing lines with new data points.
            The method automatically handles data unit conversions as defined in
            the TORAX plotting system.
        """
        plotdata = load_data(current_state)

        validate_plotdata(plotdata, self.plot_config)

        update_lines(
            self.lines, self.axes, self.plot_config, plotdata, t, self.first_update
        )

        if self.first_update is False:
            plotruns_lib.format_plots(self.plot_config, plotdata, None, self.axes)
        if self.first_update is True:
            self.first_update = False

    def render_frame(self, t: float | None = None):
        """Render and display a single frame for interactive visualization.

        This method completes the visualization pipeline by updating axis limits and legends
        , and refreshing the display for interactive
        viewing. It is designed for ``"human"`` render mode where users observe the
        simulation in real-time through matplotlib windows.

        Args:
            t (float): Current simulation time in seconds to display
                in the figure title. Formats as "t = {value:.3f}".
        """
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            ax.relim()
            ax.autoscale_view()
            if not ax.get_legend():
                ax.legend()

        self.fig.suptitle(f"t = {t:.3f}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the visualization figure and release all associated resources.

        This method properly closes the matplotlib figure and releases memory
        resources associated with the visualization. It should be called when
        the plotter is no longer needed.
        """
        plt.close(self.fig)

    def render_rgb_array(self, t: float | None = None) -> np.ndarray:
        """Render the current visualization state as RGB array for video recording.

        This method captures the current plot state as a RGB image
        suitable for video generation and programmatic processing. It performs
        the same axis scaling and legend management as ``render_frame()`` but outputs
        a numpy array instead of displaying interactively.

        Args:
            t (float): Current simulation time in seconds to display
                in the figure title. Formats as ``"t = {value:.3f}"``.

        Returns:
            numpy.ndarray: RGB image array with shape ``(height, width, 3)`` and dtype
                ``uint8``. Values are in the range ``[0, 255]`` representing RGB color
                channels. The array can be directly used by video encoding
                libraries or saved as image files.
        """
        # Update the plot without displaying it
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            ax.relim()
            ax.autoscale_view()
            if not ax.get_legend():
                ax.legend()
        if t is not None:
            self.fig.suptitle(f"t = {t:.3f}")

        # Draw to canvas without showing
        self.fig.canvas.draw()

        # Convert to RGB array using modern matplotlib API
        buf = self.fig.canvas.buffer_rgba()
        buf = np.asarray(buf).copy()  # Make a copy to avoid reference issues
        # Convert RGBA to RGB by dropping alpha channel
        buf = buf[:, :, :3]

        return buf
