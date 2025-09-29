import importlib
import logging

from torax._src.plotting import plotruns_lib

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
