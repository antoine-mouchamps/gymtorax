"""Visualization utilities for gymtorax simulation results.

Features:
- GIF creation for time evolution of variables
- Real-time rendering of variables during simulation

All functions accept variable names as in DEFAULT_BOUNDS (see observation_handler.py).
"""

import dataclasses
import logging

import matplotlib
import xarray as xr
from matplotlib.ticker import ScalarFormatter

try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torax._src.plotting import plotruns_lib

from gymtorax.torax_wrapper.torax_plot_extensions import _load_data

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class Plotter:
    """Real-time plotter using TORAX plot style and axis conventions.

    This class provides real-time visualization capabilities for TORAX plasma
    simulation data with support for both spatial profiles and time series plots.

    The plotter can operate in different render modes and supports live
    visualization during simulation runs. It automatically handles figure layout,
    axis scaling, and legend management based on the provided configuration.

    Attributes:
        plot_config (plotruns_lib.FigureProperties): Configuration for plot layout and styling
        xlabel_time_unit (str): Time unit label for x-axis on time series plots
        xlabel_space (str): Space unit label for x-axis on spatial plots
        time_history (list): Accumulated time points for time series
        scalar_histories (dict): Accumulated scalar data keyed by variable name
        render_mode (str): Rendering mode ("human", "rgb_array", etc.)
        fig (matplotlib.figure.Figure): Main figure object
        axes (list): List of matplotlib axes objects
        lines (list): List of matplotlib line objects for plotting
    """

    def __init__(
        self,
        plot_config: plotruns_lib.FigureProperties,
        render_mode: str = "human",
        xlabel_time_unit: str = "[s]",
        xlabel_space: str = "Normalized radius [-]",
    ):
        """Initialize the real-time plotter with configuration and display settings.

        Sets up the plotter with the specified configuration, initializes empty data
        histories for accumulation, and creates the matplotlib figure structure.
        The plotter is ready to receive data updates immediately after initialization.

        Args:
            plot_config (plotruns_lib.FigureProperties): Configuration object defining
                the plot layout, variables to display, axis properties, and styling options
            render_mode (str, optional): Rendering mode for display behavior.
                Options: "human" (interactive display), "rgb_array" (programmatic access).
                Defaults to "human"
            xlabel_time_unit (str, optional): Unit label for time axis on time series plots.
                Defaults to "[s]"
            xlabel_space (str, optional): Label for spatial coordinate axis on profile plots.
                Defaults to "Normalized radius [-]"

        Note:
            The figure and axes are created immediately during initialization based on
            the plot_config. Data histories are initialized as empty and will be populated
            through subsequent update() calls.
        """
        self.plot_config = plot_config
        self.xlabel_time_unit = xlabel_time_unit
        self.xlabel_space = xlabel_space
        # For time series accumulation
        self.time_history = []
        self.scalar_histories = {}  # key: variable name, value: list
        self.render_mode = render_mode
        # Setup figure, axes, and lines
        self.fig, self.axes, self.lines = self._setup_figure_and_lines()

        self.render_mode = render_mode

    def reset(self):
        """Reset the plotter to its initial state without closing the figure.

        This method clears all accumulated data histories and resets all plot lines
        to empty state. The figure remains open and ready for new data. If in human
        render mode, the display is refreshed to show the cleared state.

        Note:
            This method is useful for starting a new simulation run while keeping
            the same plotter instance and figure window.
        """
        self.time_history = []
        self.scalar_histories = {}
        for line in self.lines:
            line.set_xdata([])
            line.set_ydata([])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        if self.render_mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def update(self, current_state: xr.DataTree, t: float = None):
        """Update the plot with the current state datatree.

        Args:
            current_state: xr.DataTree containing simulation output (see torax_plot_extensions._load_data)
            t (float, optional): Current simulation time. If provided, added to time history.
        """
        plotdata = _load_data(current_state)

        if t is not None:
            self.time_history.append(t)
        line_idx = 0
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            # Spatial plot
            if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                for i, attr in enumerate(cfg.attrs):
                    y = getattr(plotdata, attr, None)
                    # y should be 2D: (time, space), get last time step
                    if y is not None and y.ndim == 2:
                        y_arr = y[-1]
                        x = plotruns_lib.get_rho(plotdata, attr)
                        if x is not None and x.ndim == 1 and len(x) == len(y_arr):
                            self.lines[line_idx].set_xdata(x)
                            self.lines[line_idx].set_ydata(y_arr)
                        else:
                            logger.warning(
                                f"[ToraxStyleRealTimePlotter] Warning: Cannot plot spatial variable '{attr}': x shape {None if x is None else x.shape}, y shape {y_arr.shape}"
                            )
                    else:
                        logger.warning(
                            f"[ToraxStyleRealTimePlotter] Warning: Cannot plot spatial variable '{attr}': missing or incompatible y."
                        )
                    line_idx += 1
            # Time series plot
            elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                for attr in cfg.attrs:
                    val = getattr(plotdata, attr, None)
                    # val should be 1D: (time,)
                    if attr not in self.scalar_histories:
                        self.scalar_histories[attr] = []
                    if val is not None and val.ndim == 1:
                        self.scalar_histories[attr].append(val[-1])
                        self.lines[line_idx].set_xdata(self.time_history)
                        self.lines[line_idx].set_ydata(self.scalar_histories[attr])
                    line_idx += 1

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
            self.fig.suptitle(f"t = {t:.3f} {self.xlabel_time_unit}")

        if self.render_mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        """Close the visualization figure and release resources.

        This method properly closes the matplotlib figure to free up memory
        and resources. Should be called when the plotter is no longer needed.
        """
        plt.close(self.fig)

    def _setup_figure_and_lines(self):
        """Create figure, axes, and line objects with proper labels and layout.

        This internal method sets up the matplotlib figure structure based on the
        plot configuration. It creates axes in a grid layout with proper spacing
        and margins, initializes plot lines for each variable, and sets up labels
        and legends.

        Returns:
            tuple: A tuple:
                - fig (matplotlib.figure.Figure): The main figure object
                - axes (list): List of matplotlib axes objects
                - lines (list): List of matplotlib line objects for plotting
        """
        fig, axes, slider_ax = plotruns_lib.create_figure(self.plot_config)
        # Flatten axes to 1D list if needed
        if isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        elif isinstance(axes, list):
            axes = axes
        else:
            axes = [axes]
        # Hide slider axis if it exists
        if slider_ax is not None:
            slider_ax.set_visible(False)
        lines = []
        # Arrange subplots in a grid with margins around and between subplots
        nrows = getattr(self.plot_config, "rows", 1)
        ncols = getattr(self.plot_config, "cols", len(axes))
        margin = (
            self.plot_config.margin if hasattr(self.plot_config, "margin") else 0.07
        )
        spacing = (
            self.plot_config.spacing if hasattr(self.plot_config, "spacing") else 0.07
        )
        subplot_width = (1.0 - 2 * margin - (ncols - 1) * spacing) / ncols
        subplot_height = (1.0 - 2 * margin - (nrows - 1) * spacing) / nrows

        for idx, (ax, cfg) in enumerate(zip(axes, self.plot_config.axes)):
            if hasattr(cfg, "ylabel"):
                ax.set_ylabel(cfg.ylabel)
            if hasattr(cfg, "plot_type"):
                if cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                    ax.set_xlabel("Time, " + self.xlabel_time_unit)
                elif cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                    ax.set_xlabel(self.xlabel_space)
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                for i, attr in enumerate(cfg.attrs):
                    x0 = []
                    y0 = []
                    (line,) = ax.plot(x0, y0, label=cfg.labels[i])
                    lines.append(line)
            elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                for i, attr in enumerate(cfg.attrs):
                    xdata = self.time_history if len(self.time_history) > 0 else []
                    ydata = self.scalar_histories.get(attr, [])
                    (line,) = ax.plot(xdata, ydata, label=cfg.labels[i])
                    lines.append(line)

            # Compute subplot position
            row = idx // ncols
            col = idx % ncols
            left = margin + col * (subplot_width + spacing)
            bottom = 1.0 - margin - (row + 1) * subplot_height - row * spacing
            ax.set_position([left, bottom, subplot_width, subplot_height])
            legend_fontsize = (
                cfg.legend_fontsize
                if cfg.legend_fontsize is not None
                else self.plot_config.default_legend_fontsize
            )
            ax.legend(fontsize=legend_fontsize)
            # Make empty axes invisible
            for ax in axes[len(self.plot_config.axes) :]:
                ax.set_visible(False)
        return fig, axes, lines


@dataclasses.dataclass
class FigureProperties(plotruns_lib.FigureProperties):
    """Configuration for figure properties.

    Mandatory args:
        rows (int): The number of rows in the figure.
        cols (int): The number of columns in the figure.
        axes (tuple[plotruns_lib.PlotProperties, ...]): List of the PlotProperties for each subplot.

    Facultative args:
        figure_size_factor = 5.0
        tick_fontsize: int = 10
        axes_fontsize: int = 10
        title_fontsize: int = 16
        default_legend_fontsize: int = 10
        colors: tuple[str, ...] = ('r', 'b', 'g', 'm', 'y', 'c')
    """

    margin: float = 0.05
    spacing: float = 0.05
