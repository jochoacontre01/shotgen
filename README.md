# Shotgen

A Python-based two-package monorepo framework for simulating **2D acoustic wave propagation**, generating synthetic **shot records**, and performing **seismic depth migration** (Kirchhoff PSDM and Reverse-Time Migration). This project leverages `Devito` (OpenACC/CUDA), `PyLops`, `DeepWave` (PyTorch), and `scikit-fmm`.

---

## Overview

**Shotgen** provides tools for:
- Creating synthetic geological velocity models (`GeoModel`) and loading standard seismic benchmarks (Marmousi-II, BP 2004 Salt, Sigsbee2A, 3D/2D Overthrust)
- Simulating 2D acoustic wave propagation on CPU (`PyLops`) and GPU (`Devito` / OpenACC)
- Reading and writing standardized SEG-Y (`SegyIO`) and HDF5 seismic datasets
- Performing Pre-Stack Depth Migration (Kirchhoff PSDM via Fast Marching Method and Reverse-Time Migration via Devito & PyTorch/Deepwave)
- Visualizing velocity models, shot gathers, migrated depth sections, and wavefield propagation animations

---

## Architecture & Repository Structure

The repository is organized as a **two-package monorepo** (`shotgen-core` and `shotgen-gpu`) to strictly isolate GNU OpenMP (`libgomp`) and NVIDIA HPC SDK OpenACC/OpenMP (`libnvomp`) runtimes:

```text
shotgen/
├── AGENTS.md                          # Agent instructions & engineering standards
├── README.md                          # Repository overview
├── USER_GUIDE.md                      # Detailed architecture and workflow guide
├── pyproject.toml                     # Monorepo workspace metadata
├── packages/
│   ├── shotgen-core/                  # CPU / PyTorch / Deepwave / Kirchhoff / I/O
│   │   ├── pyproject.toml
│   │   └── src/shotgen/
│   │       ├── cli.py                 # Core CLI entrypoint (`shotgen`)
│   │       ├── evaluation.py          # Image quality metrics (SNR, CNR, SSIM)
│   │       ├── geometry/              # Grid geometry and coordinate-index mappers
│   │       ├── io/                    # SEG-Y and HDF5 I/O routines (SegyIO, video)
│   │       ├── migration/             # Kirchhoff PSDM & PyTorch/Deepwave RTM
│   │       ├── models.py              # Geological velocity models (GeoModel)
│   │       ├── pylogger.py            # Unified logging and provenance utilities
│   │       ├── runner.py              # Subprocess bridge for isolated GPU runs
│   │       ├── sampleshot.py          # ShotRecord forward modeling & LoadShotRecord
│   │       └── utils.py               # Signal processing and directory naming helpers
│   └── shotgen-gpu/                   # Devito / OpenACC / NVHPC Stencils
│       ├── pyproject.toml
│       └── src/shotgen_gpu/
│           ├── cli.py                 # GPU CLI entrypoint (`shotgen-gpu`)
│           ├── config.py              # Device affinity & PureNvidiaCompiler overrides
│           └── engine/                # Devito acoustic wave solvers & RTM operators
├── examples/                          # Reference workflows and JSON simulation configs
│   ├── configs/                       # Simulation configuration files (.json)
│   ├── cmp_stacking.py                # Common Mid-Point stacking workflow
│   ├── evaluate_array.py              # Migrated image evaluation script
│   ├── kirchhoff_migration.py         # Parallel Kirchhoff PSDM workflow
│   ├── plot_shotrecord.py             # Shot gather visualization utility
│   ├── rtm_debug_sources.py           # Per-source RTM contribution debugger
│   ├── rtm_from_data.py               # Reverse-Time Migration workflow
│   ├── rtm_wavefield_animation.py     # Forward/adjoint RTM wavefield animation
│   ├── run_born_simulation.py         # Subprocess GPU simulation example
│   └── wavefield_animation.py         # Source wavefield propagation animation
├── aux/                               # Auxiliary plotting and SEG-Y conversion scripts
├── tests/                             # Pytest verification suite
├── assets/                            # Geological and SEG-Y velocity models
└── data/                              # Generated shot records and migrated outputs
```

---

## Installation

### Prerequisites

- Python >= 3.11 (default 3.12 in the `msc` Conda environment)
- NVIDIA HPC SDK (`nvc` / `nvc++`) for GPU OpenACC acceleration with `shotgen-gpu`

### Editable Installation

From the repository root, install both packages in editable mode:

```bash
conda activate msc
pip install -e packages/shotgen-core -e packages/shotgen-gpu
```

### Verify Installation

```bash
python -c "import shotgen; print('shotgen-core:', shotgen.__file__)"
python -c "import shotgen_gpu; print('shotgen-gpu:', shotgen_gpu.__file__)"
```

---

## Quick Start

### 1. Run a Wavefield Simulation from a Config File

```bash
# CPU / PyLops or bridged simulation via shotgen-core
shotgen --config examples/configs/test_config.json

# Isolated GPU / Devito simulation via shotgen-gpu
shotgen-gpu --config examples/configs/test_config.json -v
```

### 2. Run Depth Migration on a Saved Dataset

```bash
# Kirchhoff Pre-Stack Depth Migration (CPU multiprocessing)
python examples/kirchhoff_migration.py data/<dataset_dir> -c

# Reverse-Time Migration (RTM)
python examples/rtm_from_data.py data/<dataset_dir> -c
```

### 3. Run the Test Suite

```bash
pytest tests/
```

For full API examples, dataset format specifications, and migration parameters, see [USER_GUIDE.md](USER_GUIDE.md).

---

## Citation & Author

**Author:** Jesus Ochoa (Memorial University of Newfoundland)  
**Email:** jochoacontre@mun.ca  
**Version:** 1.0.1
