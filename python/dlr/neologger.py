"""Logging API"""
import logging

def create_logger(log_level=logging.DEBUG, log_to_console=True, log_file=None):
    """Create logger.

    Parameters
    ----------
    log_level : int, optional
        Logging level.

    log_to_console : bool, optional
        Whether to log to console.

    log_file : str, optional
        Logging file.

    Returns
    -------
    logger : logger
        Created logger object
    """
    logger = logging.getLogger(__name__ + "_logger")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(log_level)
        logger.propagate = False

    return logger
