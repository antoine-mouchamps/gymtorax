"""
Visualization utilities for gymtorax simulation results.

Features:
- GIF creation for time evolution of variables
- Real-time rendering of variables during simulation

All functions accept variable names as in DEFAULT_BOUNDS (see observation_handler.py).
"""
import dataclasses
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any, Optional
import sys
from torax._src.plotting import plotruns_lib
import logging
import xarray as xr
from torax._src.output_tools import output
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
class ToraxStyleRealTimePlotter:
    """
    Real-time plotter using Torax plot style and axis conventions.
    It also deals with data accumulation for GIF creation.
    """
    def __init__(self, plot_config: plotruns_lib.FigureProperties, render_mode: str = "human", xlabel_time_unit: str = "[s]", xlabel_space: str = "Normalized radius [-]"):

        self.plot_config = plot_config
        self.xlabel_time_unit = xlabel_time_unit
        self.xlabel_space = xlabel_space
        # Data histories must be initialized before calling _setup_figure_and_lines
        # For time series accumulation
        self.time_history = []
        self.scalar_histories = {}  # key: variable name, value: list
        # For spatial data accumulation
        self.spatial_histories = {}  # key: variable name, value: list of y arrays
        self.spatial_x_histories = {}  # key: variable name, value: list of x arrays
        self.render_mode = render_mode
        # Setup figure, axes, and lines
        self.fig, self.axes, self.lines = self._setup_figure_and_lines()
        
        self.render_mode = render_mode

    def reset(self):
        """
        Reset the plotter to its initial state without closing the figure.
        """
        self.time_history = []
        self.scalar_histories = {}
        self.spatial_histories = {}
        self.spatial_x_histories = {}
        # Clear all lines
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

    def update(self, current_state: dict, action_input: dict, t: float = None):
        """
        Update the plot with the current state.
        current_state: dict with keys 'profiles' and 'scalars'.
        t: current time (float)
        """
        profiles = current_state.get("profiles", {}) 
        scalars = current_state.get("scalars", {})
        if action_input is not None:
            profiles.update(action_input.get("profiles", {}))
            scalars.update(action_input.get("scalars", {}))

        if t is not None:
            self.time_history.append(t)
        line_idx = 0
        for ax, cfg in zip(self.axes, self.plot_config.axes):
            # Spatial plot
            if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                for attr in cfg.attrs:
                    x = None
                    y = profiles.get(attr)
                    y_arr = np.asarray(y) if y is not None else None
                    if y_arr is not None and y_arr.ndim == 1:
                        x = np.linspace(0, 1, len(y_arr))
                    # Store spatial data for GIF
                    if attr not in self.spatial_histories:
                        self.spatial_histories[attr] = []
                        self.spatial_x_histories[attr] = []
                    if y is not None and x is not None:
                        self.spatial_histories[attr].append(np.copy(y_arr))
                        self.spatial_x_histories[attr].append(np.copy(x)) #I think it won't change with time
                    # Check for valid x and y
                    if x is not None and y is not None:
                        x_arr = np.asarray(x)
                        y_arr = np.asarray(y)
                        if (
                            x_arr.ndim == 1 and y_arr.ndim == 1 and
                            len(x_arr) == len(y_arr) and len(x_arr) > 0
                        ):
                            self.lines[line_idx].set_xdata(x_arr)
                            self.lines[line_idx].set_ydata(y_arr)
                        else:
                            logger.warning(f"[ToraxStyleRealTimePlotter] Warning: Cannot plot spatial variable '{attr}': x shape {x_arr.shape}, y shape {y_arr.shape}")
                    else:
                        logger.warning(f"[ToraxStyleRealTimePlotter] Warning: Cannot plot spatial variable '{attr}': missing or incompatible x or y. Available keys: {list(profiles.keys())}")
                    line_idx += 1
            # Time series plot
            elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                for attr in cfg.attrs:
                    val = scalars.get(attr)
                    if attr not in self.scalar_histories:
                        self.scalar_histories[attr] = []
                    if val is not None:
                        self.scalar_histories[attr].append(val)
                        self.lines[line_idx].set_xdata(self.time_history)
                        self.lines[line_idx].set_ydata(self.scalar_histories[attr])
                    line_idx += 1

    def render_frame(self, t: float | None = None):
        """
        Render a single frame of the visualization.
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

    def save_gif(self, filename: str, interval: float, frame_step: int):
        """
        Create a GIF using the stored spatial and time series data.
        Both spatial and time series variables are shown (spatial animated, time series static).

        Args:
            filename: Output GIF filename
            interval: Delay between frames in ms
            frame_step: Save every Nth frame (default 1 = all frames)
        """
        import matplotlib.animation as animation
        fig, axes, lines = self._setup_figure_and_lines()
        nframes = len(self.time_history)
        if frame_step < 1 or frame_step > nframes or frame_step is None:
            logger.warning(f"[ToraxStyleRealTimePlotter] Warning: Invalid frame_step {frame_step}, resetting to 1.")
            frame_step = 1
        frame_indices = list(range(0, nframes, frame_step))
        # Ensure the last frame is included
        if nframes > 0 and (frame_indices == [] or frame_indices[-1] != nframes - 1):
            frame_indices.append(nframes - 1)

        def animate(idx):
            i = frame_indices[idx]
            line_idx = 0
            for ax, cfg in zip(axes, self.plot_config.axes):
                if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                    for attr in cfg.attrs:
                        if attr in self.spatial_x_histories and attr in self.spatial_histories:
                            if i < len(self.spatial_x_histories[attr]) and i < len(self.spatial_histories[attr]):
                                x = self.spatial_x_histories[attr][i]
                                y = self.spatial_histories[attr][i]
                                lines[line_idx].set_xdata(x)
                                lines[line_idx].set_ydata(y)
                        line_idx += 1
                elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                    for attr in cfg.attrs:
                        xdata = self.time_history
                        ydata = self.scalar_histories.get(attr, [])
                        lines[line_idx].set_xdata(xdata)
                        lines[line_idx].set_ydata(ydata)
                        line_idx += 1
                ax.relim()
                ax.autoscale_view()
            fig.suptitle(f"t = {self.time_history[i]:.3f} {self.xlabel_time_unit}")
            return lines
        logger.debug(f" Saving GIF to {filename} with {len(frame_indices)} frames (step={frame_step}).")
        ani = animation.FuncAnimation(fig, animate, frames=len(frame_indices), blit=True, interval=interval)
        ani.save(filename, writer='pillow')
        logger.debug(f" Finished saving GIF to {filename} with {len(frame_indices)} frames.")
        plt.close(fig)

    def close(self):
        """
        Close the visualization figure.
        """
        plt.close(self.fig)

    def _setup_figure_and_lines(self):
        """
        Helper to create figure, axes, and lines with correct labels for both real-time and GIF.
        Returns: fig, axes (list), lines (list)
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
        nrows = getattr(self.plot_config, 'rows', 1)
        ncols = getattr(self.plot_config, 'cols', len(axes))
        margin = self.plot_config.margin if hasattr(self.plot_config, 'margin') else 0.05
        spacing = self.plot_config.spacing if hasattr(self.plot_config, 'spacing') else 0.05
        subplot_width = (1.0 - 2 * margin - (ncols - 1) * spacing) / ncols
        subplot_height = (1.0 - 2 * margin - (nrows - 1) * spacing) / nrows
        
        for idx, (ax, cfg) in enumerate(zip(axes, self.plot_config.axes)):
            if hasattr(cfg, 'ylabel'):
                ax.set_ylabel(cfg.ylabel)
            if hasattr(cfg, 'plot_type'):
                if cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                    ax.set_xlabel("Time, " + self.xlabel_time_unit)
                elif cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                    ax.set_xlabel(self.xlabel_space)
            if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                for i, attr in enumerate(cfg.attrs):
                    # Always create a line, even if no data yet
                    if attr in self.spatial_x_histories and len(self.spatial_x_histories[attr]) > 0 and len(self.spatial_histories[attr]) > 0:
                        x0 = self.spatial_x_histories[attr][0]
                        y0 = self.spatial_histories[attr][0]
                    else:
                        x0 = []
                        y0 = []
                    line, = ax.plot(x0, y0, label=cfg.labels[i])
                    lines.append(line)
            elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                for i, attr in enumerate(cfg.attrs):
                    xdata = self.time_history if len(self.time_history) > 0 else []
                    ydata = self.scalar_histories.get(attr, [])
                    line, = ax.plot(xdata, ydata, label=cfg.labels[i])
                    lines.append(line)
            
            # Compute subplot position
            row = idx // ncols
            col = idx % ncols
            left = margin + col * (subplot_width + spacing)
            bottom = 1.0 - margin - (row + 1) * subplot_height - row * spacing
            ax.set_position([left, bottom, subplot_width, subplot_height])
            legend_fontsize = cfg.legend_fontsize if cfg.legend_fontsize is not None else self.plot_config.default_legend_fontsize
            ax.legend(fontsize=legend_fontsize)
        return fig, axes, lines

def save_gif_from_nc(nc_path, fig_properties, filename="output.gif", interval=100, frame_step=1):
    """
    Create a GIF from a .nc (NetCDF) file using a FigureProperties object to select and layout variables.
    Uses plotruns_lib.load_data for loading.

    Args:
        nc_path: Path to the .nc file.
        fig_properties: FigureProperties instance (see PlotProperties_temporal/spatial) to select variables and layout.
        filename: Output GIF filename.
        interval: Delay between frames in ms.
        frame_step: Save every Nth frame (default 1 = all frames)
    """
    datatree = output.load_state_file(nc_path)
    # Get time from root dataset
    time = datatree['time'].values if 'time' in datatree else datatree['/'].ds['time'].values
    fig, axes, slider_ax = plotruns_lib.create_figure(fig_properties)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    elif isinstance(axes, list):
        axes = axes
    else:
        axes = [axes]
    if slider_ax is not None:
        slider_ax.set_visible(False)
    lines = []
    # Setup lines for each axis and variable
    for ax, cfg in zip(axes, fig_properties.axes):
        if hasattr(cfg, 'ylabel'):
            ax.set_ylabel(cfg.ylabel)
        if hasattr(cfg, 'plot_type'):
            if cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
                ax.set_xlabel("Time, [s]")
            elif cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
                ax.set_xlabel("Normalized radius [-]")
        if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
            for i, attr in enumerate(cfg.attrs):
                ydata = datatree['/profiles'][attr].values
                x0 = np.linspace(0, 1, ydata.shape[1])
                y0 = ydata[0] if ydata.ndim == 2 else []
                line, = ax.plot(x0, y0, label=cfg.labels[i])
                lines.append((line, attr, 'spatial'))
        elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
            for i, attr in enumerate(cfg.attrs):
                ydata = datatree['/scalars'][attr].values
                xdata = time if len(time) > 0 else []
                y0 = ydata[:1] if ydata.ndim == 1 else []
                line, = ax.plot(xdata[:1], y0, label=cfg.labels[i])
                lines.append((line, attr, 'temporal'))
        ax.legend()
    nframes = len(time)
    frame_indices = list(range(0, nframes, frame_step))
    if nframes > 0 and (frame_indices == [] or frame_indices[-1] != nframes - 1):
        frame_indices.append(nframes - 1)

    def animate(idx):
        i = frame_indices[idx]
        for (line, attr, kind) in lines:
            if kind == 'spatial':
                ydata = datatree['/profiles'][attr].values
                x = np.linspace(0, 1, ydata.shape[1])
                line.set_xdata(x)
                line.set_ydata(ydata[i])
            elif kind == 'temporal':
                ydata = datatree['/scalars'][attr].values
                xdata = time[:i+1]
                y = ydata[:i+1] if ydata.ndim == 1 else []
                line.set_xdata(xdata)
                line.set_ydata(y)
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        fig.suptitle(f"t = {time[i]:.3f} [s]")
        return [l[0] for l in lines]

    ani = animation.FuncAnimation(fig, animate, frames=len(frame_indices), blit=True, interval=interval)
    ani.save(filename, writer='pillow')
    plt.close(fig)

@dataclasses.dataclass
class PlotProperties_spatial(plotruns_lib.PlotProperties):
    """
    Configuration for spatial plots.
    Mandatory args:
        attrs (tuple[str, ...]): The attributes/variables to plot.
        labels (tuple[str, ...]): The labels for each attribute.
        ylabel (str): The label for the y-axis.
        
    Facultative args:
        legend_fontsize: int | None = None  # None reverts to default matplotlib value
    """  
    plot_type: plotruns_lib.PlotType = plotruns_lib.PlotType.SPATIAL

@dataclasses.dataclass
class PlotProperties_temporal(plotruns_lib.PlotProperties):
    """
    Configuration for temporal (time series) plots.
    Mandatory args:
        attrs (tuple[str, ...]): The attributes/variables to plot.
        labels (tuple[str, ...]): The labels for each attribute.
        ylabel (str): The label for the y-axis.
        
    Facultative args:
        legend_fontsize: int | None = None  # None reverts to default matplotlib value
    """
    plot_type: plotruns_lib.PlotType = plotruns_lib.PlotType.TIME_SERIES

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