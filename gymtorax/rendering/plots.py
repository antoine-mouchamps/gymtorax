import gymtorax.rendering.visualization as viz 
# =====================================
# Examples of custom figures
# =====================================


sources_fig = viz.FigureProperties(rows=3, cols=3,
                    axes=(viz.PlotProperties_temporal(attrs=('P_ecrh_e',), labels=('ECRH power',), ylabel="Power, [W]"),
                            viz.PlotProperties_temporal(attrs=('P_radiation_e','P_bremsstrahlung_e', 'P_cyclotron_e'), labels=('Total sink power under radiation','P_bremsstrahlung_e', 'P_cyclotron_e'), ylabel="Sink power, [W]"),
                            viz.PlotProperties_temporal(attrs=('P_icrh_total',), labels=('Total ICRH power',), ylabel="Power, [W]"),
                            viz.PlotProperties_temporal(attrs=('P_aux_generic_total','P_aux_generic_e', 'P_aux_generic_i'), labels=('Total', 'Electron', 'Ion'), ylabel="Auxiliary heating power, [W]"),
                            viz.PlotProperties_temporal(attrs=('P_ei_exchange_i',), labels=('EI exchange to ions',), ylabel="Power, [W]"),
                            viz.PlotProperties_temporal(attrs=('P_ohmic_e',), labels=('Ohmic heating power',), ylabel="Power, [W]"),
                            viz.PlotProperties_temporal(attrs=('Q_fusion',), labels=('Fusion power gain',), ylabel="Ratio, [-]"),
                            viz.PlotProperties_temporal(attrs=('Ip', 'I_ecrh', 'I_bootstrap', 'I_aux_generic',), labels=('Plasma Current', 'ECRH Current', 'Bootstrap Current', 'Auxiliary Current'), ylabel="Current, [A]"),
                            viz.PlotProperties_spatial(attrs=('T_i', 'T_e'), labels=('Ion Temperature', 'Electron Temperature'), ylabel="Temperature, [keV]")
                            ),
                    default_legend_fontsize = 8,
                    tick_fontsize = 8,
                    axes_fontsize = 8,
                    margin = 0.07,
                    spacing = 0.07
)

main_prop_fig = viz.FigureProperties(rows=2, cols=3, 
                    axes=(viz.PlotProperties_temporal(attrs=('Ip', 'I_ecrh', 'I_bootstrap', 'I_aux_generic',), labels=('Plasma Current', 'ECRH Current', 'Bootstrap Current', 'Auxiliary Current'), ylabel="Current, [A]"),
                            viz.PlotProperties_spatial(attrs=('n_e', 'n_i'), labels=('Electron Density', 'Ion Density'), ylabel="Density, [1/m^3]"),
                            viz.PlotProperties_spatial(attrs=('T_i', 'T_e'), labels=('Ion Temperature', 'Electron Temperature'), ylabel="Temperature, [keV]"),
                            viz.PlotProperties_temporal(attrs=('Q_fusion',), labels=('Fusion Power gain',), ylabel="Power gain, [-]"),
                            viz.PlotProperties_temporal(attrs=('beta_N', 'beta_pol', 'beta_tor'), labels=('Beta normalized', 'Beta poloidal', 'Beta toroidal'), ylabel="Beta, [-]"),
                            viz.PlotProperties_spatial(attrs=('q','magnetic_shear'), labels=('Safety factor','Magnetic shear'), ylabel="[-]")
                            ),
                    default_legend_fontsize = 8,
                    tick_fontsize = 8,
                    axes_fontsize = 8,
                    margin = 0.07,
                    spacing = 0.07
                   )
