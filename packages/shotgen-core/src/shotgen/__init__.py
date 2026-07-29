from .sampleshot import ShotRecord, LoadShotRecord, detect_device, configure_devito_device
from .io import SegyIO
from .evaluation import contrast_to_noise_ratio, signal_to_noise_ratio, structural_similarity_index
from .models import GeoModel
from .migration import KirchhoffMigration, KirchhoffModel, load_dataset_dir
from .runner import run_gpu_simulation

from .utils import generate_simulation_dir_name

__all__ = [
    "ShotRecord",
    "LoadShotRecord",
    "SegyIO",
    "contrast_to_noise_ratio",
    "signal_to_noise_ratio",
    "structural_similarity_index",
    "detect_device",
    "configure_devito_device",
    "GeoModel",
    "KirchhoffMigration",
    "KirchhoffModel",
    "load_dataset_dir",
    "run_gpu_simulation",
    "map_coordinate_to_index",
    "generate_simulation_dir_name",
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
