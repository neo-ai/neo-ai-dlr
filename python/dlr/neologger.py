"""Logging API"""
import logging

def create_logger(log_file=None, log_level=logging.DEBUG, verbose=True):
    """Create logger.

    Parameters
    ----------
    log_file : str, optional
        Logging file.

    log_level : int, optional
        Logging level.

    verbose : bool, optional
        Whether to log to console.

    Returns
    -------
    logger : logger
        Created logger object
    """
    logger = logging.getLogger(__name__ + "_logger")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(log_level)
        logger.propagate = False

    return logger
