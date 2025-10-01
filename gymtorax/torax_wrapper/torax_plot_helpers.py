"""Enhanced TORAX Plotting Functions - PNG and GIF Generation.

This module provides functions similar to plotruns_lib.plot_run() but for generating
PNG images and animated GIFs instead of interactive plots. Uses the EXACT same
processing logic, spacing, and matplotlib configurations as the original.
"""

import inspect
import io
import logging
from typing import Any

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image
from torax._src.output_tools import output
from torax._src.plotting import plotruns_lib

# Set up logger for this module
logger = logging.getLogger(__name__)

# Font scaling constants
FONT_SCALE_BASE = 1.0  # Base scaling factor
FONT_SCALE_PER_ROW = 0.3  # Additional scaling per row


def create_figure(plot_config: plotruns_lib.FigureProperties):
    """Create figure without slider subplot.

    Returns only fig and axes, no slider_ax.
    """
    # Calculate font scaling based on rows and columns
    rows = plot_config.rows
    cols = plot_config.cols

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


def plot_run_to_gif(
    plot_config: plotruns_lib.FigureProperties,
    data_tree: xr.DataTree,
    data_tree_2: xr.DataTree | None = None,
    gif_filename: str = "torax_evolution.gif",
    n_frames: int = 50,
    duration: int = 200,
    optimize: bool = True,
    frame_skip: int = 1,
) -> str:
    """Generate animated GIF from TORAX simulation data.

    Args:
        plot_config: FigureProperties object defining the plot layout and content
        data_tree: Datatree of the TORAX simulation output file
        data_tree_2: Optional Datatree of the second file for comparison
        gif_filename: Output GIF filename
        n_frames: Maximum number of frames in the animation
        duration: Duration per frame in milliseconds
        optimize: Whether to optimize the GIF file size
        frame_skip: Skip every N frames from the original data (default=1, no skip)

    Returns:
        Path to the generated GIF file
    """
    plotdata1 = load_data(data_tree)
    plotdata2 = load_data(data_tree_2) if data_tree_2 else None

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

    logger.info(
        f"🎬 Creating animated GIF with {actual_frames} frames"
        + f"(frame_skip={frame_skip})..."
    )

    frames = []

    # Calculate font scaling based on rows
    rows = plot_config.rows

    font_scale = FONT_SCALE_BASE + (rows - 1) * FONT_SCALE_PER_ROW

    # EXACT same matplotlib RC settings as original, but with scaling
    matplotlib.rc("xtick", labelsize=plot_config.tick_fontsize * font_scale)
    matplotlib.rc("ytick", labelsize=plot_config.tick_fontsize * font_scale)
    matplotlib.rc("axes", labelsize=plot_config.axes_fontsize * font_scale)
    matplotlib.rc("figure", titlesize=plot_config.title_fontsize * font_scale)

    # Scale the font size of legend
    plot_config.default_legend_fontsize *= font_scale
    for ax_cfg in plot_config.axes:
        if ax_cfg.legend_fontsize is not None:
            ax_cfg.legend_fontsize *= font_scale

    for frame_idx, time_idx in enumerate(time_indices):
        time_val = plotdata1.t[time_idx]

        # Create figure without slider using our custom function
        fig, axes = create_figure(plot_config)

        # EXACT same title handling as plot_run()
        title_lines = [f"(1)={''} - t = {time_val:.3f} s"]
        if data_tree_2:
            title_lines.append(f"(2)={''}")
        fig.suptitle("\n".join(title_lines))

        # EXACT same line generation as plot_run(), but at specific time
        _ = get_line_at_time(plot_config, plotdata1, axes, time_idx)
        if plotdata2:
            _ = get_line_at_time(
                plot_config, plotdata2, axes, time_idx, comp_plot=True
            )

        # EXACT same plot formatting as plot_run()
        plotruns_lib.format_plots(plot_config, plotdata1, plotdata2, axes)

        # Convert plot to image
        buf = io.BytesIO()
        fig.canvas.draw()
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
            logger.info(f"  📸 Generated {frame_idx + 1}/{actual_frames} frames")
    logger.info("")

    # Save animated GIF
    logger.info(f"🎥 Saving animated GIF: {gif_filename}")

    frames[0].save(
        gif_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=optimize,
    )

    logger.info(f"✅ Animated GIF saved: {gif_filename}")
    return gif_filename


def get_line_at_time(
    plot_config: plotruns_lib.FigureProperties,
    plotdata1: plotruns_lib.PlotData,
    axes: list[Any],
    time_idx: int,
    comp_plot: bool = False,
) -> list[Any]:
    """Generate lines at specific time index in the same way as in TORAX native.

    This replicates the exact behavior of plotruns_lib.get_lines() but for a specific
    time.
    """
    lines = []
    # Same logic as get_lines() - handle comparison plot suffix and dashing
    suffix = f" ({1 if not comp_plot else 2})"
    dashed = "--" if comp_plot else ""

    for ax, cfg in zip(axes, plot_config.axes):
        line_idx = 0  # Reset color selection cycling for each plot

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


# COPY PASTE FROM TORAX, but without the filename as argument
def load_data(data_tree: xr.DataTree) -> plotruns_lib.PlotData:
    """Loads an xr.Dataset from a file, handling potential coordinate name changes."""
    # Handle potential time coordinate name variations
    time = data_tree[output.TIME].to_numpy()

    def get_optional_data(ds, key, grid_type):
        if grid_type.lower() not in ["cell", "face"]:
            raise ValueError(
                f'grid_type for {key} must be either "cell" or "face", got {grid_type}'
            )
        if key in ds:
            return ds[key].to_numpy()
        else:
            return (
                np.zeros((len(time), len(ds[output.RHO_CELL_NORM])))
                if grid_type == "cell"
                else np.zeros((len(time), len(ds[output.RHO_FACE_NORM].to_numpy())))
            )

    def _transform_data(ds: xr.Dataset):
        """Transforms data in-place to the desired units."""
        # TODO(b/414755419)
        ds = ds.copy()

        transformations = {
            output.J_TOTAL: 1e6,  # A/m^2 to MA/m^2
            output.J_OHMIC: 1e6,  # A/m^2 to MA/m^2
            output.J_BOOTSTRAP: 1e6,  # A/m^2 to MA/m^2
            output.J_EXTERNAL: 1e6,  # A/m^2 to MA/m^2
            "j_generic_current": 1e6,  # A/m^2 to MA/m^2
            output.I_BOOTSTRAP: 1e6,  # A to MA
            output.IP_PROFILE: 1e6,  # A to MA
            "j_ecrh": 1e6,  # A/m^2 to MA/m^2
            "p_icrh_i": 1e6,  # W/m^3 to MW/m^3
            "p_icrh_e": 1e6,  # W/m^3 to MW/m^3
            "p_generic_heat_i": 1e6,  # W/m^3 to MW/m^3
            "p_generic_heat_e": 1e6,  # W/m^3 to MW/m^3
            "p_ecrh_e": 1e6,  # W/m^3 to MW/m^3
            "p_alpha_i": 1e6,  # W/m^3 to MW/m^3
            "p_alpha_e": 1e6,  # W/m^3 to MW/m^3
            "p_ohmic_e": 1e6,  # W/m^3 to MW/m^3
            "p_bremsstrahlung_e": 1e6,  # W/m^3 to MW/m^3
            "p_cyclotron_radiation_e": 1e6,  # W/m^3 to MW/m^3
            "p_impurity_radiation_e": 1e6,  # W/m^3 to MW/m^3
            "ei_exchange": 1e6,  # W/m^3 to MW/m^3
            "P_ohmic_e": 1e6,  # W to MW
            "P_aux_total": 1e6,  # W to MW
            "P_alpha_total": 1e6,  # W to MW
            "P_bremsstrahlung_e": 1e6,  # W to MW
            "P_cyclotron_e": 1e6,  # W to MW
            "P_ecrh": 1e6,  # W to MW
            "P_radiation_e": 1e6,  # W to MW
            "I_ecrh": 1e6,  # A to MA
            "I_aux_generic": 1e6,  # A to MA
            "W_thermal_total": 1e6,  # J to MJ
            output.N_E: 1e20,  # m^-3 to 10^{20} m^-3
            output.N_I: 1e20,  # m^-3 to 10^{20} m^-3
            output.N_IMPURITY: 1e20,  # m^-3 to 10^{20} m^-3
        }

        for var_name, scale in transformations.items():
            if var_name in ds:
                ds[var_name] /= scale

        return ds

    data_tree = xr.map_over_datasets(_transform_data, data_tree)
    profiles_dataset = data_tree.children[output.PROFILES].dataset
    scalars_dataset = data_tree.children[output.SCALARS].dataset
    dataset = data_tree.dataset

    return plotruns_lib.PlotData(
        T_i=profiles_dataset[output.T_I].to_numpy(),
        T_e=profiles_dataset[output.T_E].to_numpy(),
        n_e=profiles_dataset[output.N_E].to_numpy(),
        n_i=profiles_dataset[output.N_I].to_numpy(),
        n_impurity=profiles_dataset[output.N_IMPURITY].to_numpy(),
        Z_impurity=profiles_dataset[output.Z_IMPURITY].to_numpy(),
        psi=profiles_dataset[output.PSI].to_numpy(),
        v_loop=profiles_dataset[output.V_LOOP].to_numpy(),
        j_total=profiles_dataset[output.J_TOTAL].to_numpy(),
        j_ohmic=profiles_dataset[output.J_OHMIC].to_numpy(),
        j_bootstrap=profiles_dataset[output.J_BOOTSTRAP].to_numpy(),
        j_external=profiles_dataset[output.J_EXTERNAL].to_numpy(),
        j_ecrh=get_optional_data(profiles_dataset, "j_ecrh", "cell"),
        j_generic_current=get_optional_data(
            profiles_dataset, "j_generic_current", "cell"
        ),
        q=profiles_dataset[output.Q].to_numpy(),
        magnetic_shear=profiles_dataset[output.MAGNETIC_SHEAR].to_numpy(),
        chi_turb_i=profiles_dataset[output.CHI_TURB_I].to_numpy(),
        chi_turb_e=profiles_dataset[output.CHI_TURB_E].to_numpy(),
        D_turb_e=profiles_dataset[output.D_TURB_E].to_numpy(),
        V_turb_e=profiles_dataset[output.V_TURB_E].to_numpy(),
        rho_norm=dataset[output.RHO_NORM].to_numpy(),
        rho_cell_norm=dataset[output.RHO_CELL_NORM].to_numpy(),
        rho_face_norm=dataset[output.RHO_FACE_NORM].to_numpy(),
        p_icrh_i=get_optional_data(profiles_dataset, "p_icrh_i", "cell"),
        p_icrh_e=get_optional_data(profiles_dataset, "p_icrh_e", "cell"),
        p_generic_heat_i=get_optional_data(
            profiles_dataset, "p_generic_heat_i", "cell"
        ),
        p_generic_heat_e=get_optional_data(
            profiles_dataset, "p_generic_heat_e", "cell"
        ),
        p_ecrh_e=get_optional_data(profiles_dataset, "p_ecrh_e", "cell"),
        p_alpha_i=get_optional_data(profiles_dataset, "p_alpha_i", "cell"),
        p_alpha_e=get_optional_data(profiles_dataset, "p_alpha_e", "cell"),
        p_ohmic_e=get_optional_data(profiles_dataset, "p_ohmic_e", "cell"),
        p_bremsstrahlung_e=get_optional_data(
            profiles_dataset, "p_bremsstrahlung_e", "cell"
        ),
        p_cyclotron_radiation_e=get_optional_data(
            profiles_dataset, "p_cyclotron_radiation_e", "cell"
        ),
        p_impurity_radiation_e=get_optional_data(
            profiles_dataset, "p_impurity_radiation_e", "cell"
        ),
        ei_exchange=profiles_dataset["ei_exchange"].to_numpy(),  # ion heating/sink
        Q_fusion=scalars_dataset["Q_fusion"].to_numpy(),  # pylint: disable=invalid-name
        s_gas_puff=get_optional_data(profiles_dataset, "s_gas_puff", "cell"),
        s_generic_particle=get_optional_data(
            profiles_dataset, "s_generic_particle", "cell"
        ),
        s_pellet=get_optional_data(profiles_dataset, "s_pellet", "cell"),
        Ip_profile=profiles_dataset[output.IP_PROFILE].to_numpy()[:, -1],
        I_bootstrap=scalars_dataset[output.I_BOOTSTRAP].to_numpy(),
        I_aux_generic=scalars_dataset["I_aux_generic"].to_numpy(),
        I_ecrh=scalars_dataset["I_ecrh"].to_numpy(),
        P_ohmic_e=scalars_dataset["P_ohmic_e"].to_numpy(),
        P_auxiliary=scalars_dataset["P_aux_total"].to_numpy(),
        P_alpha_total=scalars_dataset["P_alpha_total"].to_numpy(),
        P_sink=scalars_dataset["P_bremsstrahlung_e"].to_numpy()
        + scalars_dataset["P_radiation_e"].to_numpy()
        + scalars_dataset["P_cyclotron_e"].to_numpy(),
        P_bremsstrahlung_e=scalars_dataset["P_bremsstrahlung_e"].to_numpy(),
        P_radiation_e=scalars_dataset["P_radiation_e"].to_numpy(),
        P_cyclotron_e=scalars_dataset["P_cyclotron_e"].to_numpy(),
        T_e_volume_avg=scalars_dataset["T_e_volume_avg"].to_numpy(),
        T_i_volume_avg=scalars_dataset["T_i_volume_avg"].to_numpy(),
        n_e_volume_avg=scalars_dataset["n_e_volume_avg"].to_numpy(),
        n_i_volume_avg=scalars_dataset["n_i_volume_avg"].to_numpy(),
        W_thermal_total=scalars_dataset["W_thermal_total"].to_numpy(),
        q95=scalars_dataset["q95"].to_numpy(),
        t=time,
    )
