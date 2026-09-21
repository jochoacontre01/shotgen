import logging
import os
import sys
import time
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that routes log records through tqdm.write()
    to prevent terminal progress bar corruption during active progress display.
    Falls back to sys.stderr if tqdm is unavailable.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=sys.stderr)
            else:
                sys.stderr.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str = "shotgen") -> logging.Logger:
    """
    Retrieves or configures a logger with standard formatting and tqdm-friendly output.
    The log level is determined by the `SHOTGEN_LOG_LEVEL` environment variable (defaulting to INFO).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_str = os.getenv("SHOTGEN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(level)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


@contextmanager
def log_execution_time(logger: logging.Logger, task_name: str, level: int = logging.INFO):
    """
    Context manager to log the start, completion, and elapsed execution time of a significant task.
    """
    logger.log(level, f"Starting task: '{task_name}'...")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.log(level, f"Completed task: '{task_name}' in {elapsed:.4f}s.")
