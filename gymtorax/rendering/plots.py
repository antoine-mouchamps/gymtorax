import torax._src.plotting.plotruns_lib as plotruns_lib

from . import visualization as viz

# =====================================
# Example of custom figures
# =====================================

main_prop_fig = viz.FigureProperties(
    rows=2,
    cols=3,
    axes=(
        plotruns_lib.PlotProperties(
            attrs=(
                "Ip_profile",
                "I_ecrh",
                "I_bootstrap",
                "I_aux_generic",
            ),
            labels=(
                "Plasma Current",
                "ECRH Current",
                "Bootstrap Current",
                "Auxiliary Current",
            ),
            ylabel="Current, [A]",
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
        ),
        plotruns_lib.PlotProperties(
            attrs=("n_e", "n_i"),
            labels=("Electron Density", "Ion Density"),
            ylabel="Density, [1/m^3]",
            plot_type=plotruns_lib.PlotType.SPATIAL,
        ),
        plotruns_lib.PlotProperties(
            attrs=("T_i", "T_e"),
            labels=("Ion Temperature", "Electron Temperature"),
            ylabel="Temperature, [keV]",
            plot_type=plotruns_lib.PlotType.SPATIAL,
        ),
        plotruns_lib.PlotProperties(
            attrs=("Q_fusion",),
            labels=("Fusion Power gain",),
            ylabel="Power gain, [-]",
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
        ),
        plotruns_lib.PlotProperties(
            attrs=("q", "magnetic_shear"),
            labels=("Safety factor", "Magnetic shear"),
            ylabel="[-]",
            plot_type=plotruns_lib.PlotType.SPATIAL,
        ),
    ),
    default_legend_fontsize=10,
    tick_fontsize=10,
    axes_fontsize=10,
    margin=0.07,
    spacing=0.07,
)
