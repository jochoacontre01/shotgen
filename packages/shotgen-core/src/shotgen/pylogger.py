import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_active_file_handlers: list[logging.Handler] = []


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

    if not logger.handlers:
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

    # Attach any registered file handlers if not already present
    for fh in _active_file_handlers:
        if fh not in logger.handlers:
            logger.addHandler(fh)

    return logger


def add_file_handler(
    log_path: str | Path,
    logger: logging.Logger | None = None,
) -> logging.FileHandler:
    """
    Attaches a FileHandler with standard formatting to the specified logger,
    or to the root 'shotgen' logger and all active shotgen loggers if logger is None.
    Also registers the handler globally so any subsequent loggers inherit it.
    """
    path = Path(log_path)
    handler = logging.FileHandler(str(path), mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    if handler not in _active_file_handlers:
        _active_file_handlers.append(handler)

    targets: list[logging.Logger] = []
    if logger is not None:
        targets.append(logger)
    else:
        targets.append(get_logger("shotgen"))

    # Also attach to existing shotgen.* loggers
    for log_name, obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(obj, logging.Logger) and (log_name == "shotgen" or log_name.startswith("shotgen.")):
            if obj not in targets:
                targets.append(obj)

    for tgt in targets:
        if tgt and handler not in tgt.handlers:
            tgt.addHandler(handler)

    return handler


def setup_file_logging(
    dataset_dir: str | Path,
    script_name: str,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Configures file logging for a migration script, saving logs to:
    `<dataset_dir>/<script_name>_%Y%m%d_%H%M%S.log`

    If dataset_dir is read-only or not writable, falls back to:
    `/tmp/<dataset_name>_<script_name>_%Y%m%d_%H%M%S.log`
    with a warning.

    Returns the Path of the configured log file.
    """
    clean_script = Path(script_name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_script}_{timestamp}.log"

    dataset_path = Path(dataset_dir)
    target_path = dataset_path / filename

    # Test write permissions; fallback to /tmp if read-only or unwritable
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8"):
            pass
    except (OSError, PermissionError) as e:
        fallback_dir = Path("/tmp")
        dataset_name = dataset_path.name or "dataset"
        target_path = fallback_dir / f"{dataset_name}_{clean_script}_{timestamp}.log"
        fallback_msg = (
            f"Dataset directory '{dataset_dir}' is read-only or not writable ({e}); "
            f"falling back to log file at '{target_path}'."
        )
        if logger:
            logger.warning(fallback_msg)
        else:
            sys.stderr.write(f"[WARNING] {fallback_msg}\n")

    add_file_handler(target_path, logger=logger)

    effective_logger = logger if logger is not None else get_logger(clean_script)
    effective_logger.info("Session log initialized: %s", target_path)
    return target_path


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
