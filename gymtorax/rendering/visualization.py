"""Visualization utilities for gymtorax simulation results.

Features:
- GIF creation for time evolution of variables
- Real-time rendering of variables during simulation

All functions accept variable names as in DEFAULT_BOUNDS (see observation_handler.py).
"""

import logging
import matplotlib

import xarray as xr

import matplotlib.pyplot as plt
import numpy as np
from torax._src.plotting import plotruns_lib

from ..torax_wrapper import create_figure, get_line_at_time, load_data, validate_plotdata

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
        fig (matplotlib.figure.Figure): Main figure object
        axes (list): List of matplotlib axes objects
        lines (list): List of matplotlib line objects for plotting
    """

    def __init__(
        self,
        plot_config: plotruns_lib.FigureProperties,
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
        self.plot_config = plot_config
        self.lines = []
        self.fig, self.axes = create_figure(self.plot_config, 1)
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

        line_idx = 0
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            line_idx_color = 0
            cfg.include_first_timepoint = True # I don't know why, but it is needed...

            if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                for attr, label in zip(cfg.attrs, cfg.labels):
                    data = getattr(plotdata, attr)
                    # if cfg.suppress_zero_values and np.all(data == 0):
                    #     continue
                    
                    rho = plotruns_lib.get_rho(plotdata, attr)
                    if self.first_update is True:
                        (line,) = ax.plot(
                            rho,
                            data[0, :],
                            self.plot_config.colors[line_idx_color % len(self.plot_config.colors)],
                            label=label,
                        )
                        self.lines.append(line)
                        line_idx_color += 1
                    else:
                        self.lines[line_idx].set_xdata(rho)
                        self.lines[line_idx].set_ydata(data[0, :])
                    line_idx += 1

            elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                for attr, label in zip(cfg.attrs, cfg.labels):
                    data = getattr(plotdata, attr)

                    if self.first_update is True:
                        # if cfg.suppress_zero_values and np.all(data == 0):
                        #     continue
                        # EXACT same logic as get_lines() - plot entire time series
                        (line,) = ax.plot(
                            plotdata.t,
                            data,  # Plot entire time series (same as get_lines)
                            self.plot_config.colors[line_idx_color % len(self.plot_config.colors)],
                            label=label,
                        )
                        self.lines.append(line)
                        line_idx_color += 1
                    else:
                        xdata = self.lines[line_idx].get_xdata()
                        ydata = self.lines[line_idx].get_ydata()
                        self.lines[line_idx].set_xdata(np.append(xdata, t))
                        self.lines[line_idx].set_ydata(np.append(ydata, data))
                    line_idx += 1
            else:
                raise ValueError(f"Unknown plot type: {cfg.plot_type}")
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
