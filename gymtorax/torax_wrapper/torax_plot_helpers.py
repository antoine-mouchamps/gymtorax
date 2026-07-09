"""TORAX plotting helper functions for visualization.

This module provides utilities for creating matplotlib figures and updating plots
with TORAX simulation data. The functions are designed to work with TORAX
plotting system while supporting both static image generation and real-time
visualization updates.

Key functions:
    - `create_figure()`: Sets up matplotlib figure with TORAX styling and font scaling
    - `update_lines()`: Updates plot lines with simulation data (spatial profiles or time series)
    - `validate_plotdata()`: Ensures plot configuration matches available data attributes
    - `load_data()`: Processes TORAX `DataTree` output into `PlotData` format with unit conversions
    - `format_plots()`: Applies axis labels, limits and legends (matplotlib version)

All of these functions are adapted from TORAX ``plotruns_lib`` module, with modifications
to be able to apply them in the GymTORAX environments. Since TORAX v1.4, the upstream
``plotruns_lib`` is plotly-based; the matplotlib-oriented logic (figure creation and
plot formatting) is therefore maintained here, adapted from TORAX v1.0.
"""

import logging

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torax._src.plotting import plotruns_lib

# Set up logger for this module
logger = logging.getLogger(__name__)

# Font scaling constants
FONT_SCALE_BASE = 1.0  # Base scaling factor
FONT_SCALE_PER_ROW = 0.3  # Additional scaling per row

# Matplotlib figure defaults, kept from the TORAX v1.0 matplotlib-based
# FigureProperties. The plotly-based FigureProperties (TORAX >= 1.4) no longer
# carries these fields.
FIGURE_SIZE_FACTOR = 5.0
DEFAULT_LEGEND_FONTSIZE = 10


def create_figure(plot_config: plotruns_lib.FigureProperties, font_scale: float = 1):
    """Create matplotlib figure with TORAX styling and configurable font scaling.

    Sets up a matplotlib figure using TORAX plot configuration, applies matplotlib
    RC settings for consistent styling, and creates a grid of subplots. Font sizes
    are scaled according to the `font_scale` parameter and applied to the `plot_config`
    object in-place. As side effects, this function modifies matplotlib global RC
    settings for tick, axes, legend, and figure fonts, and modifies axes
    ``legend_fontsize`` in-place.

    Args:
        plot_config (plotruns_lib.FigureProperties): TORAX plot configuration
            containing subplot layout (`rows`, `cols`), font sizes,
            and axes configurations. Modified in-place to apply font scaling.
        font_scale (float): Multiplier for all font sizes. Applied to
            tick labels, axis labels, titles, and legend fonts. Defaults to ``1.0``.

    Returns:
        tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
            - fig (matplotlib.figure.Figure): Figure object
            - axes (list[matplotlib.axes.Axes]): list of axes in row-major order (left-to-right, top-to-bottom).
    """
    # Same matplotlib RC settings as the TORAX v1.0 plotting code, but with
    # scaling and mapped to the TORAX >= 1.4 FigureProperties field names
    matplotlib.rc("xtick", labelsize=plot_config.tick_size * font_scale)
    matplotlib.rc("ytick", labelsize=plot_config.tick_size * font_scale)
    matplotlib.rc("axes", labelsize=plot_config.subplot_title_size * font_scale)
    matplotlib.rc("figure", titlesize=plot_config.title_size * font_scale)
    matplotlib.rc("legend", fontsize=DEFAULT_LEGEND_FONTSIZE * font_scale)

    # Scale the font size of per-axis legends
    for ax_cfg in plot_config.axes:
        if ax_cfg.legend_fontsize is not None:
            ax_cfg.legend_fontsize *= font_scale

    # Calculate font scaling based on rows and columns
    rows = plot_config.rows
    cols = plot_config.cols

    # EXACT same figure size calculation as original
    fig = plt.figure(
        figsize=(
            cols * FIGURE_SIZE_FACTOR,
            rows * FIGURE_SIZE_FACTOR,
        ),
        constrained_layout=True,
    )

    # Create GridSpec without slider row (no extra height ratio for slider)
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    # Create axes exactly as original - simple grid layout
    axes = []
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    return fig, axes


def update_lines(lines, axes, plot_config, plotdata, t, first_update):
    """Update or create plot lines with simulation data.

    As side effects, this function sets ``cfg.include_first_timepoint = True``
    on each axis config, and for `TIME_SERIES` on subsequent updates, appends
    data to existing line coordinates.

    Args:
        lines (list): Existing matplotlib `Line2D` objects. Empty on first call.
        axes (list): Matplotlib axes objects matching `plot_config` layout.
        plot_config (plotruns_lib.FigureProperties): Defines subplot configurations,
            each with `plot_type`, `attrs` (variable names), `labels`, and `colors`.
        plotdata (plotruns_lib.PlotData): Simulation data with plasma variables.
        t (float): Current simulation time (used for `TIME_SERIES` updates).
        first_update (bool): If `True`, creates new lines; if `False`, updates existing.

    Returns:
        list: Updated list of `Line2D` objects for future calls.

    Raises:
        ValueError: If `plot_type` is not `SPATIAL` or `TIME_SERIES`.

    Note:
        Uses ``plotruns_lib.get_rho()`` to determine x-coordinate for spatial plots.
        Color cycling follows each axis config ``colors`` list with modulo indexing.
    """
    line_idx = 0
    for ax, cfg in zip(axes, plot_config.axes):
        line_idx_color = 0
        cfg.include_first_timepoint = True  # I don't know why, but it is needed...
        # Colors are now per-axis (plotly hex strings, also valid for matplotlib)
        colors = tuple(cfg.colors)

        if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
            for attr, label in zip(cfg.attrs, cfg.labels):
                data = getattr(plotdata, attr)
                # if cfg.suppress_zero_values and np.all(data == 0):
                #     continue

                rho = plotruns_lib.get_rho(plotdata, attr)
                if first_update is True:
                    (line,) = ax.plot(
                        rho,
                        data[0, :],
                        color=colors[line_idx_color % len(colors)],
                        label=label,
                    )
                    lines.append(line)
                    line_idx_color += 1
                else:
                    lines[line_idx].set_xdata(rho)
                    lines[line_idx].set_ydata(data[0, :])
                line_idx += 1

        elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
            for attr, label in zip(cfg.attrs, cfg.labels):
                data = getattr(plotdata, attr)

                if first_update is True:
                    # if cfg.suppress_zero_values and np.all(data == 0):
                    #     continue
                    # EXACT same logic as get_lines() - plot entire time series
                    (line,) = ax.plot(
                        plotdata.t,
                        data,  # Plot entire time series (same as get_lines)
                        color=colors[line_idx_color % len(colors)],
                        label=label,
                    )
                    lines.append(line)
                    line_idx_color += 1
                else:
                    xdata = lines[line_idx].get_xdata()
                    ydata = lines[line_idx].get_ydata()
                    lines[line_idx].set_xdata(np.append(xdata, t))
                    lines[line_idx].set_ydata(np.append(ydata, data))
                line_idx += 1
        else:
            raise ValueError(f"Unknown plot type: {cfg.plot_type}")
    return lines


def validate_plotdata(
    plotdata: plotruns_lib.PlotData, plot_config: plotruns_lib.FigureProperties
):
    """Check that all plot configuration attributes exist in plotdata.

    Since TORAX >= 1.4, `PlotData` resolves attributes dynamically from the
    output datasets (with zero-filling for known optional variables), so this
    check simply attempts attribute access for each configured variable.

    Args:
        plotdata (plotruns_lib.PlotData): Data object to check.
        plot_config (plotruns_lib.FigureProperties): Plot configuration with
            axes definitions. Each axis config has an ``attrs`` list of variable names.

    Raises:
        ValueError: If any attribute in ``plot_config.axes[*].attrs`` is not found
            in `plotdata`. Error message identifies the missing attribute name.
    """
    for cfg in plot_config.axes:
        for attr in cfg.attrs:
            try:
                getattr(plotdata, attr)
            except AttributeError:
                raise ValueError(
                    f"Attribute '{attr}' in plot_config does not exist in PlotData"
                )


def load_data(data_tree: xr.DataTree) -> plotruns_lib.PlotData:
    r"""Convert TORAX DataTree output to PlotData with unit transformations.

    Delegates to the TORAX plotting library, which applies unit conversions to
    match TORAX plotting conventions (A/m² → MA/m², W → MW, m⁻³ → 10²⁰ m⁻³,
    etc.) and wraps the result in a `PlotData` object with dynamic variable
    access (missing optional variables are zero-filled on access).

    Args:
        data_tree (xarray.DataTree): TORAX simulation output.

    Returns:
        plotruns_lib.PlotData: Object with plasma variables in plotting units.
    """
    # pylint: disable-next=protected-access
    return plotruns_lib._data_tree_to_plot_data(data_tree)


def format_plots(
    plot_config: plotruns_lib.FigureProperties,
    plotdata: plotruns_lib.PlotData,
    axes: list,
):
    """Set up plot formatting: axis labels, y-limits, and legends.

    Adapted from the TORAX v1.0 matplotlib-based ``plotruns_lib.format_plots``
    (removed upstream in the plotly rewrite), without the second-run comparison
    data.

    Args:
        plot_config (plotruns_lib.FigureProperties): Plot configuration with
            axes definitions.
        plotdata (plotruns_lib.PlotData): Simulation data used to compute
            percentile-based y-axis limits.
        axes (list): Matplotlib axes objects matching `plot_config` layout.
    """

    def get_limit(plotdata, attrs, percentile, include_first_timepoint):
        """Gets the limit for a set of attributes based a histogram percentile."""
        if include_first_timepoint:
            values = np.concatenate(
                [getattr(plotdata, attr).flatten() for attr in attrs]
            )
        else:
            values = np.concatenate(
                [getattr(plotdata, attr)[1:, :].flatten() for attr in attrs]
            )
        return np.percentile(values, percentile)

    for ax, cfg in zip(axes, plot_config.axes):
        if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
            ax.set_xlabel("Normalized radius")
        elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
            ax.set_xlabel("Time [s]")
        else:
            raise ValueError(f"Unknown plot type: {cfg.plot_type}")
        ax.set_ylabel(cfg.ylabel)

        # Get limits for y-axis based on percentile values.
        # 0.0 or 100.0 are special cases for simple min/max values.
        ymin = get_limit(
            plotdata, cfg.attrs, cfg.lower_percentile, cfg.include_first_timepoint
        )
        ymax = get_limit(
            plotdata, cfg.attrs, cfg.upper_percentile, cfg.include_first_timepoint
        )

        lower_bound = ymin / 1.05 if ymin > 0 else ymin * 1.05

        # Guard against empty data
        if ymax != 0 or ymin != 0:  # Check for meaningful data range
            if cfg.ylim_min_zero:
                ax.set_ylim([min(lower_bound, 0), ymax * 1.05])
            else:
                ax.set_ylim([lower_bound, ymax * 1.05])

            ax.legend(fontsize=cfg.legend_fontsize)
