"""
Enhanced TORAX Plotting Functions - PNG and GIF Generation

This module provides functions similar to plotruns_lib.plot_run() but for generating
PNG images and animated GIFs instead of interactive plots. Uses the EXACT same
processing logic, spacing, and matplotlib configurations as the original.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from PIL import Image
import io
import inspect
from os import path
from typing import Optional, List, Sequence, Any

from torax._src.plotting import plotruns_lib


# Font scaling constants
FONT_SCALE_BASE = 1.0  # Base scaling factor
FONT_SCALE_PER_ROW = 0.5  # Additional scaling per row
FONT_SCALE_PER_COL = 0.8  # Additional scaling per column
MIN_FONT_SCALE = 0.5  # Minimum font scale to maintain readability


def create_figure(plot_config: plotruns_lib.FigureProperties):
    """
    Create figure without slider subplot (modified version of plotruns_lib.create_figure).
    Returns only fig and axes, no slider_ax.
    """
    # Calculate font scaling based on rows and columns
    rows = plot_config.rows
    cols = plot_config.cols

    font_scale = FONT_SCALE_BASE + (rows - 1) * FONT_SCALE_PER_ROW

    # EXACT same matplotlib RC settings as original, but with scaling
    matplotlib.rc("xtick", labelsize=int(plot_config.tick_fontsize * font_scale))
    matplotlib.rc("ytick", labelsize=int(plot_config.tick_fontsize * font_scale))
    matplotlib.rc("axes", labelsize=int(plot_config.axes_fontsize * font_scale))
    matplotlib.rc("figure", titlesize=int(plot_config.title_fontsize * font_scale))
    matplotlib.rc(
        "legend", fontsize=int(plot_config.default_legend_fontsize * font_scale)
    )

    # Additional settings for Agg backend compatibility
    matplotlib.rc(
        "font", size=int(plot_config.axes_fontsize * font_scale)
    )  # Base font size

    # EXACT same figure size calculation as original

    fig = plt.figure(
        figsize=(
            cols * plot_config.figure_size_factor,
            rows * plot_config.figure_size_factor,
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


def plot_run_to_png(
    plot_config: plotruns_lib.FigureProperties,
    outfile: str,
    time_indices: Optional[List[int]] = None,
    output_dir: str = "png_plots",
) -> List[str]:
    """
    Generate PNG images from TORAX simulation data, using EXACT same logic as plot_run().

    Args:
        plot_config: FigureProperties object defining the plot layout and content
        outfile: Path to the TORAX simulation output file (.nc)
        time_indices: List of time indices to plot. If None, plots 5 representative times
        output_dir: Directory to save PNG files

    Returns:
        List of generated PNG file paths
    """
    # EXACT same validation as plot_run()
    if not path.exists(outfile):
        raise ValueError(f"File {outfile} does not exist.")

    plotdata1 = plotruns_lib.load_data(outfile)
    plotdata2 = None  # No comparison for PNG generation

    # EXACT same attribute validation as plot_run()
    plotdata_fields = set(plotdata1.__dataclass_fields__)
    plotdata_properties = {
        name
        for name, _ in inspect.getmembers(
            type(plotdata1), lambda o: isinstance(o, property)
        )
    }
    plotdata_attrs = plotdata_fields.union(plotdata_properties)
    for cfg in plot_config.axes:
        for attr in cfg.attrs:
            if attr not in plotdata_attrs:
                raise ValueError(
                    f"Attribute '{attr}' in plot_config does not exist in PlotData"
                )

    # Select time indices if not provided
    if time_indices is None:
        n_times = len(plotdata1.t)
        time_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    png_files = []

    print(f"📊 Creating {len(time_indices)} PNG plots...")

    for i, time_idx in enumerate(time_indices):
        if time_idx >= len(plotdata1.t):
            print(f"Warning: time_idx {time_idx} exceeds data length, skipping")
            continue

        time_val = plotdata1.t[time_idx]

        # Create figure without slider using our custom function
        fig, axes = create_figure(plot_config)

        # EXACT same title handling as plot_run()
        title_lines = [f"(1)={outfile} - t = {time_val:.3f} s"]
        fig.suptitle("\n".join(title_lines))

        # EXACT same line generation as plot_run(), but at specific time
        lines1 = _get_lines_at_time(plot_config, plotdata1, axes, time_idx)

        # EXACT same plot formatting as plot_run()
        plotruns_lib.format_plots(plot_config, plotdata1, plotdata2, axes)

        # Save PNG
        filename = f"{output_dir}/torax_t_{time_val:.3f}s_{i + 1:02d}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        png_files.append(filename)
        plt.close(fig)

        print(f"  ✅ Saved: {filename}")

    print(f"✅ Generated {len(png_files)} PNG files in '{output_dir}/' directory")
    return png_files


def plot_run_to_gif(
    plot_config: plotruns_lib.FigureProperties,
    outfile: str,
    outfile2: str | None = None,
    gif_filename: str = "torax_evolution.gif",
    n_frames: int = 50,
    duration: int = 200,
    optimize: bool = True,
    frame_skip: int = 1,
) -> str:
    """
    Generate animated GIF from TORAX simulation data, using EXACT same logic as plot_run().

    Args:
        plot_config: FigureProperties object defining the plot layout and content
        outfile: Path to the TORAX simulation output file (.nc)
        outfile2: Optional path to second file for comparison
        gif_filename: Output GIF filename
        n_frames: Maximum number of frames in the animation
        duration: Duration per frame in milliseconds
        optimize: Whether to optimize the GIF file size
        frame_skip: Skip every N frames from the original data (default=1, no skip)

    Returns:
        Path to the generated GIF file
    """
    # EXACT same validation as plot_run()
    name, ext = path.splitext(outfile)
    if ext != "":
        if ext.lower() != ".nc":
            raise ValueError(f"Expected .nc file, got {ext} in {outfile}")
    else:
        outfile += ".nc"  # Ensure .nc extension if missing
    if not path.exists(outfile):
        raise ValueError(f"File {outfile} does not exist.")

    if outfile2 is not None and not path.exists(outfile2):
        raise ValueError(f"File {outfile2} does not exist.")

    plotdata1 = plotruns_lib.load_data(outfile)
    plotdata2 = plotruns_lib.load_data(outfile2) if outfile2 else None

    # EXACT same attribute validation as plot_run()
    plotdata_fields = set(plotdata1.__dataclass_fields__)
    plotdata_properties = {
        name
        for name, _ in inspect.getmembers(
            type(plotdata1), lambda o: isinstance(o, property)
        )
    }
    plotdata_attrs = plotdata_fields.union(plotdata_properties)
    for cfg in plot_config.axes:
        for attr in cfg.attrs:
            if attr not in plotdata_attrs:
                raise ValueError(
                    f"Attribute '{attr}' in plot_config does not exist in PlotData"
                )

    # Select time indices for animation with frame_skip
    n_times = len(plotdata1.t)

    # Apply frame_skip to reduce the data
    available_indices = list(range(0, n_times, frame_skip))
    actual_frames = min(n_frames, len(available_indices))

    # Select evenly spaced frames from the skipped data
    if actual_frames == len(available_indices):
        time_indices = available_indices
    else:
        selected_indices = np.linspace(
            0, len(available_indices) - 1, actual_frames, dtype=int
        )
        time_indices = [available_indices[i] for i in selected_indices]

    print(
        f"🎬 Creating animated GIF with {actual_frames} frames (frame_skip={frame_skip})..."
    )

    frames = []

    for frame_idx, time_idx in enumerate(time_indices):
        time_val = plotdata1.t[time_idx]

        # Create figure without slider using our custom function
        fig, axes = create_figure(plot_config)

        # EXACT same title handling as plot_run()
        title_lines = [f"(1)={outfile} - t = {time_val:.3f} s"]
        if outfile2:
            title_lines.append(f"(2)={outfile2}")
        fig.suptitle("\n".join(title_lines))

        # EXACT same line generation as plot_run(), but at specific time
        lines1 = _get_lines_at_time(plot_config, plotdata1, axes, time_idx)
        if plotdata2:
            lines2 = _get_lines_at_time(
                plot_config, plotdata2, axes, time_idx, comp_plot=True
            )

        # EXACT same plot formatting as plot_run()
        plotruns_lib.format_plots(plot_config, plotdata1, plotdata2, axes)

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

        # Progress indicator
        if (frame_idx + 1) % 2 == 0:
            print(f"  📸 Generated {frame_idx + 1}/{actual_frames} frames", end="\r")
    print()

    # Save animated GIF
    print(f"🎥 Saving animated GIF: {gif_filename}")

    frames[0].save(
        gif_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=optimize,
    )

    print(f"✅ Animated GIF saved: {gif_filename}")
    return gif_filename


def _apply_font_scaling_to_config(plot_config: plotruns_lib.FigureProperties):
    """Apply font scaling to legend_fontsize in each plot config."""
    rows = plot_config.rows
    cols = plot_config.cols

    font_scale = FONT_SCALE_BASE + (rows - 1) * FONT_SCALE_PER_ROW
    font_scale = max(font_scale, MIN_FONT_SCALE)

    # Update legend_fontsize for each axis config
    for cfg in plot_config.axes:
        if cfg.legend_fontsize is not None:
            cfg.legend_fontsize = int(cfg.legend_fontsize * font_scale)
        else:
            # Use default_legend_fontsize if legend_fontsize is None
            cfg.legend_fontsize = int(plot_config.default_legend_fontsize * font_scale)


def _get_lines_at_time(
    plot_config: plotruns_lib.FigureProperties,
    plotdata1: plotruns_lib.PlotData,
    axes: List[Any],
    time_idx: int,
    comp_plot: bool = False,
) -> List[Any]:
    """
    Generate lines at specific time index using EXACT same logic as get_lines() from plot_run.
    This replicates the exact behavior of plotruns_lib.get_lines() but for a specific time.
    """
    lines = []
    # Same logic as get_lines() - handle comparison plot suffix and dashing
    suffix = f" ({1 if not comp_plot else 2})"
    dashed = "--" if comp_plot else ""

    for ax, cfg in zip(axes, plot_config.axes):
        line_idx = 0  # Reset color selection cycling for each plot (same as get_lines)

        if cfg.plot_type == plotruns_lib.PlotType.SPATIAL:
            for attr, label in zip(cfg.attrs, cfg.labels):
                data = getattr(plotdata1, attr)
                if cfg.suppress_zero_values and np.all(data == 0):
                    continue

                # EXACT same rho calculation as get_lines()
                rho = plotruns_lib.get_rho(plotdata1, attr)

                # Plot data at specific time instead of time zero
                (line,) = ax.plot(
                    rho,
                    data[time_idx, :],  # Use specified time_idx instead of 0
                    plot_config.colors[line_idx % len(plot_config.colors)] + dashed,
                    label=f"{label}{suffix}",
                )
                lines.append(line)
                line_idx += 1

        elif cfg.plot_type == plotruns_lib.PlotType.TIME_SERIES:
            for attr, label in zip(cfg.attrs, cfg.labels):
                data = getattr(plotdata1, attr)
                if cfg.suppress_zero_values and np.all(data == 0):
                    continue

                # EXACT same logic as get_lines() - plot entire time series
                _ = ax.plot(
                    plotdata1.t,
                    data,  # Plot entire time series (same as get_lines)
                    plot_config.colors[line_idx % len(plot_config.colors)] + dashed,
                    label=f"{label}{suffix}",
                )
                line_idx += 1
        else:
            raise ValueError(f"Unknown plot type: {cfg.plot_type}")

    return lines
