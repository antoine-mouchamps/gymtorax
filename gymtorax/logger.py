import logging


def setup_logging(level=logging.WARNING, logfile=None):
    """
    Setup global logging configuration.
    All modules (including external libs using the root logger) will share this config.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING, ...).
        logfile (str, optional): If provided, logs will also be written to this file.
    """
    
    handlers = [logging.StreamHandler()]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="w"))

    logging.basicConfig(
        level=level,
        # format="[%(asctime)s] %(levelname)s: %(message)s",
        # datefmt="%H:%M:%S",
        handlers=handlers,
        force=True  # overwrite existing config (important for Jupyter/rl loops)
    )