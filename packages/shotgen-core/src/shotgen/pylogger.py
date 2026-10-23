import logging
import os
import shlex
import subprocess
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


def get_code_version() -> str:
    """
    Returns the current git commit hash, branch, and dirty status if available,
    or falls back to package version / unknown.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        dirty = "-dirty" if status else ""
        return f"{commit} ({branch}{dirty})"
    except Exception:
        try:
            import shotgen
            return getattr(shotgen, "__version__", "unknown")
        except Exception:
            return "unknown"


def get_invoked_command() -> str:
    """
    Returns the exact command-line string invoked to execute the current process.
    """
    if not sys.argv:
        return "unknown"
    exe = Path(sys.executable).name if sys.executable else "python"
    first = sys.argv[0]
    if first.endswith(".py") or first.endswith(".pyc"):
        return f"{exe} " + " ".join(shlex.quote(a) for a in sys.argv)
    elif first == "-c":
        return f"{exe} -c ..."
    else:
        cmd_name = Path(first).name
        rest = " " + " ".join(shlex.quote(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else ""
        return f"{cmd_name}{rest}"


def setup_file_logging(
    dataset_dir: str | Path,
    script_name: str,
    logger: logging.Logger | None = None,
    timestamp: str | None = None,
    command: str | None = None,
) -> Path:
    """
    Configures file logging for a migration script, saving logs to:
    `<dataset_dir>/%Y%m%d_%H%M%S_<script_name>.log`

    If dataset_dir is read-only or not writable, falls back to:
    `/tmp/<dataset_name>_%Y%m%d_%H%M%S_<script_name>.log`
    with a warning.

    Writes the code version and invoked command at the very top of the log file.

    Returns the Path of the configured log file.
    """
    clean_script = Path(script_name).stem
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{clean_script}.log"

    dataset_path = Path(dataset_dir)
    target_path = dataset_path / filename

    code_version = get_code_version()
    invoked_cmd = command if command is not None else get_invoked_command()

    def _write_header(p: Path):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            is_new = not p.exists() or p.stat().st_size == 0
            with open(p, "a", encoding="utf-8") as f:
                if is_new:
                    f.write(f"# Code Version: {code_version}\n")
                    f.write(f"# Command: {invoked_cmd}\n")
                    f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Script: {clean_script}\n\n")
        except Exception:
            pass

    # Test write permissions; fallback to /tmp if read-only or unwritable
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8"):
            pass
        _write_header(target_path)
    except (OSError, PermissionError) as e:
        fallback_dir = Path("/tmp")
        dataset_name = dataset_path.name or "dataset"
        target_path = fallback_dir / f"{dataset_name}_{timestamp}_{clean_script}.log"
        fallback_msg = (
            f"Dataset directory '{dataset_dir}' is read-only or not writable ({e}); "
            f"falling back to log file at '{target_path}'."
        )
        if logger:
            logger.warning(fallback_msg)
        else:
            sys.stderr.write(f"[WARNING] {fallback_msg}\n")
        _write_header(target_path)

    add_file_handler(target_path, logger=logger)

    effective_logger = logger if logger is not None else get_logger(clean_script)
    effective_logger.info("Session log initialized: %s | Code version: %s | Command: %s", target_path, code_version, invoked_cmd)
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
