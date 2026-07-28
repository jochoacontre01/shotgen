import subprocess
import sys
import json
import tempfile
from pathlib import Path


def run_gpu_simulation(config_file: str, cli: bool = False) -> None:
    """Executes shotgen-gpu in a fresh OS subprocess to isolate C/CUDA runtimes."""
    cmd = [sys.executable, "-m", "shotgen_gpu.cli", "--config", str(config_file)]
    if cli:
        cmd.append("--cli")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GPU Simulation process failed:\n{result.stderr}")
    print(result.stdout)


def run_gpu_simulation_dict(cfg: dict, cli: bool = False) -> dict:
    """Helper to run a GPU simulation from a dictionary config via subprocess bridge."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        run_gpu_simulation(tmp_path, cli=cli or cfg.get("cli", False))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
