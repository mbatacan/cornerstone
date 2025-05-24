import logging


def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with the specified name.

    This function ensures that each logger is only configured once, even if called multiple times.
    It sets up a stream handler that outputs log messages to the console with a consistent format.
    The logger's level is set to INFO by default.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    # Only add a handler if the logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)
    return logger
