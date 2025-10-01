"""Visualization utilities for gymtorax simulation results.

Features:
- GIF creation for time evolution of variables
- Real-time rendering of variables during simulation

All functions accept variable names as in DEFAULT_BOUNDS (see observation_handler.py).
"""

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


def process_plot_config(plot_config: str) -> plotruns_lib.FigureProperties:
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
        axes (list): List of matplotlib axes objects
        lines (list): List of matplotlib line objects for plotting
    """

    def __init__(
        self,
        plot_config: plotruns_lib.FigureProperties,
        render_mode: str | None = None,
    ):
        """Initialize the real-time plotter with configuration and display settings.

        Sets up the plotter with the specified configuration, initializes empty data
        histories for accumulation, and creates the matplotlib figure structure.
        The plotter is ready to receive data updates immediately after initialization.

        Args:
            plot_config (plotruns_lib.FigureProperties): Configuration object defining
                the plot layout, variables to display, axis properties, and styling options

        Note:
            The figure and axes are created immediately during initialization based on
            the plot_config. Data histories are initialized as empty and will be populated
            through subsequent update() calls.
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

    def update(self, current_state: xr.DataTree, t: float = None):
        """Update the plot with the current state datatree.

        Args:
            current_state: xarray.DataTree containing simulation output (see torax_plot_extensions.load_data)
            t (float, optional): Current simulation time.
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
        """Render a single frame of the visualization.

        This method updates the display by rescaling axes, adding legends if missing,
        and optionally updating the figure title with the current time. In human
        render mode, the display is refreshed immediately.

        Args:
            t (float, optional): Current simulation time to display in figure title.
                If None, no time is shown in the title.
        """
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            ax.relim()
            ax.autoscale_view()
            if not ax.get_legend():
                ax.legend()
        if t is not None:
            self.fig.suptitle(f"t = {t:.3f}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the visualization figure and release resources.

        This method properly closes the matplotlib figure to free up memory
        and resources. Should be called when the plotter is no longer needed.
        """
        plt.close(self.fig)

    def render_rgb_array(self, t: float | None = None) -> np.ndarray:
        """Render the current frame as RGB array for video recording.

        Args:
            t: Current simulation time for title display

        Returns:
            np.ndarray: RGB array of shape (height, width, 3) with values in [0, 255]
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
