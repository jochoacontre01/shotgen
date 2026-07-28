from .kirchhoff import KirchhoffMigration, KirchhoffModel, load_dataset_dir
from .pytorch_rtm import ReverseTimeMigrationGPU

__all__ = [
    "KirchhoffMigration",
    "KirchhoffModel",
    "load_dataset_dir",
    "ReverseTimeMigrationGPU",
]


def __getattr__(name):
    if name == "ReverseTimeMigration":
        try:
            from shotgen_gpu.engine.rtm import ReverseTimeMigration
            return ReverseTimeMigration
        except ImportError:
            raise ImportError(
                "ReverseTimeMigration requires 'shotgen-gpu'. "
                "Please install shotgen-gpu via `pip install -e packages/shotgen-gpu`."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
