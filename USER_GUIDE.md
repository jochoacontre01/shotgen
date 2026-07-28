# `shotgen` Monorepo Usage & Architecture Guide

Welcome to the refactored **`shotgen`** monorepo! This document details the architectural design, installation, simulation workflows (CPU & GPU), parameter configurations, migration algorithms, and output dataset formats.

---

## 1. Architecture Overview

The repository is structured as a two-package monorepo to strictly isolate C-runtime libraries, CUDA contexts, and Python dependency graphs:

```text
shotgen/
├── pyproject.toml                     # Monorepo workspace configuration
├── README.md                          # Repository overview
├── USER_GUIDE.md                      # Comprehensive user guide
├── packages/
│   ├── shotgen-core/                  # CPU / PyTorch / Kirchhoff / I/O
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── shotgen/
│   │           ├── __init__.py
│   │           ├── io/                # SEGY & HDF5 I/O routines (SegyIO, generate_video)
│   │           ├── migration/         # Kirchhoff PSDM & PyTorch/Deepwave RTM
│   │           ├── geometry/          # Grid geometry & index mappers
│   │           ├── models.py          # Velocity model generators (GeoModel)
│   │           ├── evaluation.py      # Quality metrics (SNR, CNR, SSIM)
│   │           ├── sampleshot.py      # ShotRecord (PyLops CPU engine) & LoadShotRecord
│   │           └── runner.py          # Subprocess launcher for isolated GPU tasks
│   └── shotgen-gpu/                   # Devito / OpenACC / NVHPC Stencils
│       ├── pyproject.toml
│       └── src/
│           └── shotgen_gpu/
│               ├── __init__.py
│               ├── engine/            # Devito operator wrappers & solvers
│               ├── cli.py             # CLI entrypoint (`shotgen-gpu`)
│               └── config.py          # Device affinity & PureNvidiaCompiler overrides
└── examples/
    ├── kirchhoff_migration.py         # Kirchhoff Migration using shotgen-core
    ├── run_born_simulation.py         # Subprocess GPU execution example
    └── test_config.json               # JSON configuration example for shotgen-gpu CLI
```

### Key Technical Benefits of the Split:
- **Symbol Isolation**: Physical process-level separation prevents symbol interposition between GNU OpenMP (`libgomp.so`) and NVIDIA HPC SDK OpenMP/OpenACC (`libnvomp.so`).
- **Clean Contexts**: Resolves `RTLD_GLOBAL` deadlocks when PyTorch and Devito attempt to claim CUDA contexts simultaneously.
- **Independent Dependencies**: `shotgen-core` depends on `torch`, `deepwave`, `pylops`, `segyio`, and `scikit-fmm`. `shotgen-gpu` depends on `devito`, `sympy`, and `segyio`.

---

## 2. Installation & Environment Setup

Install both sub-packages in editable mode inside your Python / Conda environment:

```bash
# Navigate to the repository root directory
cd shotgen/

# Install shotgen-core and shotgen-gpu in editable mode
pip install -e packages/shotgen-core -e packages/shotgen-gpu
```

---

## 3. Generating Shot Records

### A. CPU Simulation (`engine="pylops"`)
CPU simulations use matrix-free linear wave equation operators via `PyLops` and multi-threaded CPU workers:

```python
import numpy as np
from shotgen import ShotRecord, GeoModel

# 1. Initialize Velocity Model (nx=100, nz=80)
geo = GeoModel(nx=100, nz=80, v_base=2500.0)
vp = geo.layered()

# 2. Configure ShotRecord for CPU PyLops simulation
shot_rec = ShotRecord(
    nx=100,
    nz=80,
    dx=10.0,
    dz=10.0,
    n_sources=4,
    n_receivers=16,
    f0=25.0,                  # Peak wavelet frequency (Hz)
    src_origin=(100.0, 0.0),   # First source coordinate (x, z) in meters
    rec_origin=(50.0, 0.0),    # First receiver coordinate (x, z) in meters
    group_offset=20.0,         # Receiver spacing in meters
    shot_offset=50.0,          # Source step spacing in meters
    gather="common shot",      # Survey gather type ('common shot' or 'common midpoint')
    smooth=5,                  # Gaussian smoothing sigma for background model v0
    engine="pylops",           # CPU PyLops engine
    device="cpu"
)

# 3. Attach model and run simulation (150 ms)
shot_rec.set_model(vp)
shot_data = shot_rec.run(ms=150)
print(f"CPU Shot Data shape: {shot_data.shape}") # Output shape: (n_sources, n_receivers, n_time)
```

---

### B. GPU Simulation (`engine="devito"` via Process Subprocess Bridge)
GPU simulations execute JIT-compiled Devito OpenACC wave equations inside an isolated OS process:

```python
from shotgen import ShotRecord, GeoModel

# 1. Setup Model
geo = GeoModel(nx=100, nz=80, v_base=2500.0)
vp = geo.layered()

# 2. Configure ShotRecord with engine="devito"
shot_rec = ShotRecord(
    nx=100,
    nz=80,
    dx=10.0,
    dz=10.0,
    n_sources=4,
    n_receivers=16,
    f0=25.0,
    src_origin=(100.0, 0.0),
    rec_origin=(50.0, 0.0),
    group_offset=20.0,
    shot_offset=50.0,
    gather="common shot",
    smooth=5,
    engine="devito",           # Delegates to shotgen-gpu in isolated process
    device="auto"              # Automatically selects CUDA GPU if available
)

shot_rec.set_model(vp)
shot_data = shot_rec.run(ms=150)
```

---

### C. Headless GPU Simulation via `shotgen-gpu` CLI
You can execute GPU wave propagation directly from the command line using a JSON parameter configuration file.

#### Example Config (`examples/test_config.json`):
```json
{
  "task": "simulation",
  "nx": 60,
  "nz": 40,
  "dx": 10.0,
  "dz": 10.0,
  "n_sources": 2,
  "n_receivers": 6,
  "f0": 30.0,
  "ms": 150.0,
  "fd_order": 4,
  "n_damping": 20,
  "smooth": 5.0,
  "v_base": 2500.0,
  "device": "auto",
  "output_dir": "data/gpu_simulation_test"
}
```

#### Running the CLI:
```bash
shotgen-gpu --config examples/test_config.json
```
or via python:
```bash
python3 -m shotgen_gpu.cli --config examples/test_config.json
```

---

## 4. Dataset Saving Location & File Formats

When you call `shot_rec.save_shot(name="data/my_dataset")`, `shotgen` creates a directory containing standardized SEG-Y and HDF5 files:

```text
data/my_dataset/
├── traces.segy             # Seismic shot gathers & spatial trace headers
├── velocity_model.segy     # True 2D P-wave velocity grid (nx, nz)
├── smooth_velocity.segy   # Smooth background velocity grid (nx, nz)
└── metadata.h5            # Time axis, source wavelet, f0, grid origin
```

### File Specifications:
1. **`traces.segy`** (SEG-Y format):
   - Contains all trace waveforms stored as IEEE 32-bit floats.
   - Trace headers include: `SourceX`, `SourceY`, `GroupX`, `GroupY`, `CDP`, `CDP_X`, `CDP_Y`, `offset`, and sample rate `Interval` in microseconds.
2. **`velocity_model.segy` & `smooth_velocity.segy`** (SEG-Y format):
   - 2D grid matrix encoded as SEG-Y traces (where each column is a trace).
3. **`metadata.h5`** (HDF5 format):
   - `time`: 1D array of time sample points in seconds.
   - `wavelet`: 1D array of source Ricker wavelet amplitudes.
   - `f0`: Peak frequency scalar value (Hz).
   - `origin`: Array `[origin_x, origin_z]` in meters.

---

## 5. Loading Saved Shot Records

Load pre-recorded datasets for analysis or migration using `LoadShotRecord`:

```python
from shotgen import LoadShotRecord

dataset = LoadShotRecord("data/my_dataset")

print("Number of shots:", dataset.nshots)
print("Shots tensor shape:", dataset.shots.shape)     # (n_sources, n_receivers, n_time)
print("Velocity model shape:", dataset.velocity_model.shape)
print("Sources coordinates:", dataset.sources.shape)
print("Receivers coordinates:", dataset.receivers.shape)

# Visualize shot gather
dataset.plot()
```

---

## 6. Migration Algorithms & Execution

`shotgen` provides three migration algorithms catering to CPU ray traveltimes, Devito OpenACC finite-difference RTM, and PyTorch automatic differentiation RTM.

### A. Kirchhoff Pre-Stack Depth Migration (PSDM) — CPU (`shotgen-core`)
Uses the Fast Marching Method (`scikit-fmm`) for CPU traveltime calculation:

```python
from shotgen.migration import KirchhoffMigration

# 1. Initialize Kirchhoff PSDM using a saved dataset directory
migrator = KirchhoffMigration(
    dataset_dir="data/my_dataset",
    spacing=(10.0, 10.0)      # (dx, dz) grid spacing in meters
)

# 2. Run migration
image = migrator.run()

# 3. Save migrated depth image
import numpy as np
np.save("data/kirchhoff_image.npy", image)
```

---

### B. Devito Reverse Time Migration (RTM) — GPU/OpenACC (`shotgen-gpu`)
Solves adjoint acoustic wave equations via Devito GPU acceleration:

```python
from shotgen_gpu.engine.rtm import ReverseTimeMigration

# 1. Initialize RTM with dataset directory
rtm = ReverseTimeMigration(
    dataset_dir="data/my_dataset",
    nbl=40,                   # Absorbing boundary layers
    space_order=8,            # Spatial finite-difference order
    device="cuda"             # Target CUDA GPU
)

# 2. Run RTM
migrated_image = rtm.run(save_wavefield=False)

# 3. Save RTM image
import numpy as np
np.save("data/rtm_image.npy", migrated_image)
```

---

### C. PyTorch Born Inversion RTM — PyTorch/Deepwave (`shotgen-core`)
Uses PyTorch and Deepwave scalar Born operators:

```python
from shotgen.migration import ReverseTimeMigrationGPU

rtm_torch = ReverseTimeMigrationGPU(
    dataset_dir="data/my_dataset"
)

# Run optimization for 5 epochs
scatter_map = rtm_torch.run(epochs=5)
```

---

## 7. Migration Output Formats

Migration algorithms return 2D numpy arrays of shape `(nx, nz)` corresponding to physical subsurface depth images.

- Output files are saved as standard NumPy array binary files (`.npy`) or embedded into HDF5 output files (`.h5`).
- Results can be rendered using standard matplotlib / cmocean plotting:

```python
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from scipy.ndimage import laplace

image = np.load("data/kirchhoff_image.npy")

plt.figure(figsize=(10, 6))
# Apply Laplace filter to highlight sharp impedance interfaces
plt.imshow(laplace(image.T), cmap=cmocean.cm.balance_r, aspect="auto")
plt.colorbar(label="Reflection Amplitude")
plt.xlabel("Horizontal Distance (cells)")
plt.ylabel("Depth (cells)")
plt.title("Depth Migrated Subsurface Image")
plt.tight_layout()
plt.show()
```
