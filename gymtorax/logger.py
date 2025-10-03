import logging


def setup_logging(level=logging.WARNING, log_file=None, suppress_external=True):
    """Setup global logging configuration with optional external library suppression.

    When using DEBUG level, external libraries (JAX, TORAX, etc.) can generate
    overwhelming amounts of log messages. This function allows to debug the
    gymtorax code while keeping external libraries at WARNING level or higher.

    Args:
        level (int): Logging level for gymtorax modules (e.g., :data:`logging.DEBUG`,
            :data:`logging.INFO`, :data:`logging.WARNING`, ...).
        log_file (str or None): If provided, logs will also be written to this file.
        suppress_external (bool): If ``True`` and ``level=DEBUG``, suppress verbose output
            from external libraries (JAX, TORAX, TensorFlow, etc.) by setting them
            to ``WARNING`` level. Default: ``True``.

    Example:
        >>> # Debug gymtorax only, suppress external libraries
        >>> setup_logging(level=logging.DEBUG)
        >>>
        >>> # Debug everything including external libraries
        >>> setup_logging(level=logging.DEBUG, suppress_external=False)
        >>>
        >>> # Normal usage (unchanged behavior)
        >>> setup_logging(level=logging.WARNING)
    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=level,
        # format="[%(asctime)s] %(levelname)s: %(message)s",
        # datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,  # overwrite existing config (important for Jupyter/rl loops)
    )

    # If DEBUG level requested and suppression enabled, quiet external libraries
    if level == logging.DEBUG and suppress_external:
        external_libs = [
            "jax",
            "torax",
            "tensorflow",
            "tf",
            "numpy",
            "matplotlib",
            "h5netcdf",
            "h5py",
            "xarray",
            "sklearn",
            "pandas",
            "absl",
            "etils",
            "chex",
            "optax",
            "flax",
            "PIL",
        ]
        for lib in external_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)

    # Ensure gymtorax modules use the requested level
    logging.getLogger("gymtorax").setLevel(level)
