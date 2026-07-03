import logging


def setup_logging(level=logging.WARNING, log_file=None, suppress_external=True):
    """Setup global logging configuration with optional external library suppression.

    External libraries (JAX, TORAX, etc.) can generate overwhelming amounts of log
    messages. In particular, TORAX emits INFO logs on every internal ``run_loop``
    call, which happens once per action step and is pure noise here. This function
    lets gymtorax log at any level while keeping external libraries at ``WARNING``.

    Args:
        level (int): Logging level for gymtorax modules (e.g., :data:`logging.DEBUG`,
            :data:`logging.INFO`, :data:`logging.WARNING`, ...).
        log_file (str or None): If provided, logs will also be written to this file.
        suppress_external (bool): If ``True``, suppress verbose output from external
            libraries (JAX, TORAX, TensorFlow, etc.) by setting them to ``WARNING``
            level, regardless of the gymtorax ``level``. Default: ``True``.

    Example:
        >>> # gymtorax at INFO, external libraries quieted (no per-step TORAX noise)
        >>> setup_logging(level=logging.INFO)
        >>>
        >>> # Debug everything including external libraries
        >>> setup_logging(level=logging.DEBUG, suppress_external=False)
        >>>
        >>> # Normal usage
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

    # Quiet external libraries
    if suppress_external:
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
