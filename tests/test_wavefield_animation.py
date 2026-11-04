import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from wavefield_animation import (
    main,
)


def test_wavefield_animation_end_to_end(tmp_path):
    """
    Smoke test running wavefield simulation and animation generation on a
    small synthetic velocity model, testing both MP4 and NPZ output generation.
    """
    vel = 2000.0 * np.ones((40, 30), dtype=np.float32)
    vel_path = tmp_path / "synthetic_vel.npy"
    np.save(vel_path, vel)

    cfg = {
        "task": "simulation",
        "velocity_model": str(vel_path),
        "dx": 10.0,
        "dz": 10.0,
        "n_sources": 2,
        "n_receivers": 4,
        "ms": 50.0,
        "f0": 20.0,
        "src_origin": [100.0, 10.0],
        "rec_origin": [120.0, 10.0],
        "origin": [0.0, 0.0],
        "group_offset": 20.0,
        "shot_offset": 50.0,
        "gather": "common shot",
        "smooth": 2,
        "snr": None,
        "fd_order": 4,
        "n_damping": 10,
        "engine": "devito",
        "device": "cpu",
        "fs_ms": 5.0,
    }

    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    out_dir = tmp_path / "simulations"

    test_args = [
        "wavefield_animation.py",
        "--config", str(config_path),
        "--source", "0",
        "--step", "5.0",
        "--ms", "40.0",
        "--device", "cpu",
        "--save-numpy",
        "--output-dir", str(out_dir),
    ]

    with patch.object(sys, "argv", test_args):
        main()

    # Find created dataset directory
    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1, "Expected exactly one dataset folder created in simulations"
    dataset_dir = subdirs[0]

    # Check MP4 exists and is non-empty
    mp4_file = dataset_dir / "source_0.mp4"
    assert mp4_file.exists(), f"Expected {mp4_file} to exist"
    assert mp4_file.stat().st_size > 0, "Generated MP4 file is empty"

    # Check NumPy array container exists and verify keys
    npz_file = dataset_dir / "wavefield_arrays" / "source_0.npz"
    assert npz_file.exists(), f"Expected {npz_file} to exist"
    data = np.load(npz_file)
    assert "wavefield" in data
    assert "time" in data
    assert "source_idx" in data
    assert data["source_idx"] == 0
    assert data["wavefield"].shape[1:] == (40, 30)
