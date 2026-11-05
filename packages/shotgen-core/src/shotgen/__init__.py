from .evaluation import (
    contrast_to_noise_ratio,
    signal_to_noise_ratio,
    structural_similarity_index,
)
from .io import SegyIO, convert_npy_to_segy
from .migration import KirchhoffMigration, KirchhoffModel, load_dataset_dir
from .models import GeoModel
from .pylogger import (
    TqdmLoggingHandler,
    add_file_handler,
    get_code_version,
    get_invoked_command,
    get_logger,
    log_array_stats,
    log_execution_time,
    setup_file_logging,
)
from .runner import run_gpu_simulation
from .sampleshot import (
    LoadShotRecord,
    ShotRecord,
    configure_devito_device,
    detect_device,
)
from .utils import generate_simulation_dir_name, setup_matplotlib_style

__all__ = [
    "GeoModel",
    "KirchhoffMigration",
    "KirchhoffModel",
    "LoadShotRecord",
    "SegyIO",
    "ShotRecord",
    "TqdmLoggingHandler",
    "add_file_handler",
    "configure_devito_device",
    "contrast_to_noise_ratio",
    "convert_npy_to_segy",
    "detect_device",
    "generate_simulation_dir_name",
    "get_code_version",
    "get_invoked_command",
    "get_logger",
    "load_dataset_dir",
    "log_array_stats",
    "log_execution_time",
    "map_coordinate_to_index",
    "run_gpu_simulation",
    "setup_file_logging",
    "setup_matplotlib_style",
    "signal_to_noise_ratio",
    "structural_similarity_index",
]


def __getattr__(name):
    if name in ("ReverseTimeMigration", "ReverseTimeMigrationGPU"):
        if name == "ReverseTimeMigrationGPU":
            from .migration.pytorch_rtm import ReverseTimeMigrationGPU
            return ReverseTimeMigrationGPU
        else:
            try:
                from shotgen_gpu.engine.rtm import ReverseTimeMigration
                return ReverseTimeMigration
            except ImportError:
                raise ImportError(
                    "ReverseTimeMigration requires 'shotgen-gpu'. "
                    "Please install shotgen-gpu via `pip install -e packages/shotgen-gpu`."
                )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
