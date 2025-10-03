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

All of these functions are adapted from TORAX ``plotruns_lib`` module, with modifications
to be able to apply them in the GymTORAX environments.
"""

import inspect
import logging

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torax._src.output_tools import output
from torax._src.plotting import plotruns_lib

# Set up logger for this module
logger = logging.getLogger(__name__)

# Font scaling constants
FONT_SCALE_BASE = 1.0  # Base scaling factor
FONT_SCALE_PER_ROW = 0.3  # Additional scaling per row


def create_figure(plot_config: plotruns_lib.FigureProperties, font_scale: float = 1):
    """Create matplotlib figure with TORAX styling and configurable font scaling.

    Sets up a matplotlib figure using TORAX plot configuration, applies matplotlib
    RC settings for consistent styling, and creates a grid of subplots. Font sizes
    are scaled according to the `font_scale` parameter and applied to the `plot_config`
    object in-place. As side effects, this function modifies matplotlib global RC
    settings for tick, axes, and figure fonts, and modifies
    ``plot_config.default_legend_fontsize`` and axes ``legend_fontsize`` in-place.

    Args:
        plot_config (plotruns_lib.FigureProperties): TORAX plot configuration
            containing subplot layout (`rows`, `cols`), font sizes, figure size factor,
            and axes configurations. Modified in-place to apply font scaling.
        font_scale (float): Multiplier for all font sizes. Applied to
            tick labels, axis labels, titles, and legend fonts. Defaults to ``1.0``.

    Returns:
        tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
            - fig (matplotlib.figure.Figure): Figure object
            - axes (list[matplotlib.axes.Axes]): list of axes in row-major order (left-to-right, top-to-bottom).
    """
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
        Color cycling follows ``plot_config.colors`` list with modulo indexing.
    """
    line_idx = 0
    for ax, cfg in zip(axes, plot_config.axes):
        line_idx_color = 0
        cfg.include_first_timepoint = True  # I don't know why, but it is needed...

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
                        plot_config.colors[line_idx_color % len(plot_config.colors)],
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
                        plot_config.colors[line_idx_color % len(plot_config.colors)],
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

    Uses introspection to find all available attributes in the `PlotData` object
    (both dataclass fields and properties), then verifies that every attribute
    name listed in the plot configuration ``axes.attrs`` lists exists.

    Args:
        plotdata (plotruns_lib.PlotData): Data object to check.
        plot_config (plotruns_lib.FigureProperties): Plot configuration with
            axes definitions. Each axis config has an ``attrs`` list of variable names.

    Raises:
        ValueError: If any attribute in ``plot_config.axes[*].attrs`` is not found
            in `plotdata`. Error message identifies the missing attribute name.
    """
    # EXACT same attribute validation as plot_run()
    plotdata_fields = set(plotdata.__dataclass_fields__)
    plotdata_properties = {
        name
        for name, _ in inspect.getmembers(
            type(plotdata), lambda o: isinstance(o, property)
        )
    }
    plotdata_attrs = plotdata_fields.union(plotdata_properties)
    for cfg in plot_config.axes:
        for attr in cfg.attrs:
            if attr not in plotdata_attrs:
                raise ValueError(
                    f"Attribute '{attr}' in plot_config does not exist in PlotData"
                )


def load_data(data_tree: xr.DataTree) -> plotruns_lib.PlotData:
    r"""Convert TORAX DataTree output to PlotData with unit transformations.

    Extracts time coordinate and applies unit conversions to match TORAX plotting
    conventions (A/m² → MA/m², W → MW, m⁻³ → 10²⁰ m⁻³, etc.). Handles hierarchical
    `DataTree` structure by extracting from ``profiles/`` and ``scalars/`` branches.

    Args:
        data_tree (xarray.DataTree): TORAX simulation output.

    Returns:
        plotruns_lib.PlotData: Object with plasma variables in plotting units.
    """
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
